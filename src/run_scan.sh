#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 8 ]]; then
  echo "Usage: $0 JOB_ID MODEL_VERSION DATA_SOURCE SCAN_TYPE FLAG INPUT_S3_URI OUTPUT_S3_PREFIX STATUS_S3_PREFIX" >&2
  exit 2
fi

JOB_ID="$1"
MODEL_VERSION="$2"
DATA_SOURCE="$3"
SCAN_TYPE="$4"      # url|domain
FLAG="$5"           # mal|clean|unknown
INPUT_S3_URI="$6"
OUTPUT_S3_PREFIX="$7"
STATUS_S3_PREFIX="$8"

WORK_DIR="/tmp/urlmodel/${JOB_ID}"
mkdir -p "${WORK_DIR}"
STATUS_LOCAL="${WORK_DIR}/status.json"
LOCAL_INPUT="${WORK_DIR}/input.txt"

write_status() {
  echo "$1" > "${STATUS_LOCAL}"
  aws s3 cp "${STATUS_LOCAL}" "${STATUS_S3_PREFIX%/}/${JOB_ID}.json" --region eu-west-2 >/dev/null
}

write_status "{\"job_id\":\"${JOB_ID}\",\"status\":\"RUNNING\"}"

NAME=""
PORT=""
IMAGE="registry.example.com/spoke/sai-url:model-version-${MODEL_VERSION}"
OUTPUT_FILE="${MODEL_VERSION}_${DATA_SOURCE}_${SCAN_TYPE}_${FLAG}.csv"

cleanup() {
  if [[ -n "${NAME}" ]]; then
    docker rm -f "${NAME}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

# unique container name
for i in $(seq 0 999); do
  CAND="urlmodel${i}"
  if ! docker ps -a --format '{{.Names}}' | grep -qx "${CAND}"; then
    NAME="${CAND}"
    break
  fi
done

# unique local port
for p in $(seq 8700 9800); do
  if ! ss -ltn "( sport = :${p} )" | grep -q ":${p}"; then
    PORT="${p}"
    break
  fi
done

if [[ -z "${NAME}" || -z "${PORT}" ]]; then
  write_status "{\"job_id\":\"${JOB_ID}\",\"status\":\"FAILED\",\"error\":\"No free container name or port\"}"
  exit 1
fi

{
  docker pull "${IMAGE}"

  docker run -d \
    --name "${NAME}" \
    -p 127.0.0.1:${PORT}:8080 \
    -e WORKERS=1 \
    -e THREADS=0 \
    -e SYSTEM=internal \
    "${IMAGE}"

  # wait up to 90s for model API readiness on the actual scoring endpoint
  READY=0
  for _ in $(seq 1 90); do
    if python3 - <<'PY' "${PORT}"
import hashlib
import json
import sys
import urllib.error
import urllib.request

port = int(sys.argv[1])
sample = "http://example.com"
payload = {
    "source": "inline",
    "dataFormat": "raw",
    "samples": [
        {
            "sha256": hashlib.sha256(sample.encode()).hexdigest(),
            "data": sample,
        }
    ],
}
url = f"http://127.0.0.1:{port}/v1/ml/reports?fields=report"
data = json.dumps(payload).encode("utf-8")
req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")

try:
    with urllib.request.urlopen(req, timeout=5) as resp:
        sys.exit(0 if resp.status < 500 else 1)
except urllib.error.HTTPError as err:
    # 4xx means server is reachable and has started.
    sys.exit(0 if err.code < 500 else 1)
except Exception:
    sys.exit(1)
PY
    then
      READY=1
      break
    fi
    sleep 1
  done
  if [[ "${READY}" -ne 1 ]]; then
    write_status "{\"job_id\":\"${JOB_ID}\",\"status\":\"FAILED\",\"error\":\"Container did not become ready\"}"
    exit 1
  fi

  aws s3 cp "${INPUT_S3_URI}" "${LOCAL_INPUT}" --region eu-west-2

  cd /home/ubuntu/efs/urlmodel

  python3 - <<'PY' "${LOCAL_INPUT}" "${MODEL_VERSION}" "${PORT}" "${OUTPUT_FILE}"
import sys
from scan_urls import scan_single_urls_list

input_path, model_version, port, output_file = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
with open(input_path, "r", encoding="utf-8") as f:
    url_list = [x.strip() for x in f if x.strip()]
scan_single_urls_list(url_list, model_version, port, output_file)
PY

  OUT_URI="${OUTPUT_S3_PREFIX%/}/${OUTPUT_FILE}"
  aws s3 cp "${OUTPUT_FILE}" "${OUT_URI}" --region eu-west-2
  rm -f "${OUTPUT_FILE}"

  write_status "{\"job_id\":\"${JOB_ID}\",\"status\":\"SUCCEEDED\",\"output_file\":\"${OUTPUT_FILE}\",\"output_s3_uri\":\"${OUT_URI}\"}"
} || {
  write_status "{\"job_id\":\"${JOB_ID}\",\"status\":\"FAILED\",\"error\":\"Scan execution failed\"}"
  exit 1
}