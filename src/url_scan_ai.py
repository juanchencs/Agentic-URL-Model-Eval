import os
import time
import json
from pathlib import Path
import boto3
import requests
import re
from collections import Counter
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone

try:
    from IPython.display import display, Markdown, HTML
except Exception:
    display = None
    Markdown = None
    HTML = None
    
    
#!/usr/bin/env python3
THRESHOLD = 30.0
REGION = os.getenv("AWS_REGION", "eu-west-2")
API_BASE = os.getenv("URLMODEL_API_BASE", "https://example.execute-api.example-region.amazonaws.com/prod")
BUCKET = os.getenv("URLMODEL_BUCKET", "example-bucket")
INPUT_PREFIX = "mlmodels/urlmodel/input"
API_KEY = os.getenv("URLMODEL_API_KEY", "")
 
# Credentials:
# - Prefer IAM role / default AWS credential chain (recommended)
# - URLMODEL_API_KEY should be provided via environment variable
 

s3 = boto3.client("s3", region_name=REGION)


def upload_input(url_txt_path: str, model_version: str, data_source: str, scan_type: str, flag: str) -> str:
    stamp = int(time.time())
    key = f"{INPUT_PREFIX}/{model_version}_{data_source}_{scan_type}_{flag}_{stamp}.txt"
    s3.upload_file(url_txt_path, BUCKET, key)
    return f"s3://{BUCKET}/{key}"


def start_scan(model_version: str, data_source: str, scan_type: str, flag: str, input_s3_uri: str) -> dict:
    payload = {
        "model_version": model_version,
        "data_source": data_source,
        "type": scan_type,      # "url" | "domain"
        "flag": flag,           # "mal" | "clean" | "unknown"
        "input_s3_uri": input_s3_uri,
    }
    headers = {"x-api-key": API_KEY, "Content-Type": "application/json"}
    r = requests.post(f"{API_BASE}/start-scan", json=payload, headers=headers, timeout=60)
    r.raise_for_status()
    return r.json()


def poll_status(job_id: str, command_id: str, interval_sec: int = 15, timeout_sec: int = 7200) -> dict:
    headers = {"x-api-key": API_KEY}
    start_ts = time.time()
    while True:
        r = requests.get(
            f"{API_BASE}/scan-status",
            params={"job_id": job_id, "command_id": command_id},
            headers=headers,
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
        status = data.get("status", "UNKNOWN")
        print(f"[scan-status] {status} :: {json.dumps(data)}")

        if status in ("SUCCEEDED", "FAILED"):
            return data

        if time.time() - start_ts > timeout_sec:
            raise TimeoutError("Polling timed out")

        time.sleep(interval_sec)


def download_output(output_s3_uri: str, local_dir: str = ".") -> str:
    path_no_scheme = output_s3_uri.replace("s3://", "", 1)
    out_bucket, out_key = path_no_scheme.split("/", 1)
    local_dir_path = Path(local_dir)
    local_dir_path.mkdir(parents=True, exist_ok=True)
    local_path = str(local_dir_path / Path(out_key).name)
    s3.download_file(out_bucket, out_key, local_path)
    return local_path


def summarize_with_bedrock(client, model_id: str, prompt: str) -> str:
    """
    Prefer Bedrock Converse API, fallback to invoke_model for older boto3.
    """
    try:
        resp = client.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"temperature": 0.2, "maxTokens": 900},
        )
        return resp["output"]["message"]["content"][0]["text"]
    except AttributeError:
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 900,
            "temperature": 0.2,
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": prompt}]}
            ],
        }
        resp = client.invoke_model(
            modelId=model_id,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )
        out = json.loads(resp["body"].read())
        return "".join([c.get("text", "") for c in out.get("content", []) if isinstance(c, dict)])


def analyze_and_plot(csv_path: str, model_version: str, out_dir: str = "./analysis_output") -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    score_col = f"score{model_version}"
    if score_col not in df.columns:
        raise ValueError(f"Missing expected score column: {score_col}")

    url_col = "url" if "url" in df.columns else df.columns[0]
    df["flagged"] = df[score_col] >= 30
    df["url_length"] = df[url_col].astype(str).str.len()

    words = []
    for u in df[url_col].astype(str):
        words.extend(re.findall(r"[a-zA-Z]{4,}", u.lower()))
    stop = {"http", "https", "www", "com", "net", "org", "html", "php", "index"}
    keywords = Counter([w for w in words if w not in stop]).most_common(10)

    total = int(len(df))
    flagged = int(df["flagged"].sum())
    flagged_rate = round((flagged / total * 100.0), 1) if total else 0.0
    avg_score = round(float(df[score_col].mean()), 1) if total else 0.0

    # 1) ML score distribution
    plt.figure(figsize=(8, 5))
    plt.hist(df[score_col], bins=15, color="#2563eb", edgecolor="white")
    plt.axvline(30, color="#ef4444", linestyle="--", label="threshold=30")
    plt.title(f"ML Score Distribution ({score_col})")
    plt.xlabel("Score")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    ml_chart = out / "chart_ml_score_distribution.png"
    plt.savefig(ml_chart, dpi=140)
    plt.close()

    # 2) URL length distribution
    plt.figure(figsize=(8, 5))
    plt.hist(df["url_length"], bins=15, color="#16a34a", edgecolor="white")
    plt.title("URL Length Distribution")
    plt.xlabel("URL Length")
    plt.ylabel("Count")
    plt.tight_layout()
    len_chart = out / "chart_url_length_distribution.png"
    plt.savefig(len_chart, dpi=140)
    plt.close()

    # 3) Top suspicious keywords
    plt.figure(figsize=(9, 5))
    if keywords:
        labels = [k for k, _ in keywords][::-1]
        values = [v for _, v in keywords][::-1]
        plt.barh(labels, values, color="#9333ea")
    plt.title("Top Suspicious Keywords in URLs")
    plt.xlabel("Frequency")
    plt.ylabel("Keyword")
    plt.tight_layout()
    kw_chart = out / "chart_top_suspicious_keywords.png"
    plt.savefig(kw_chart, dpi=140)
    plt.close()

    return {
        "model_version": model_version,
        "score_col": score_col,
        "url_col": url_col,
        "total_urls": total,
        "flagged_urls": flagged,
        "flagged_rate_pct": flagged_rate,
        "avg_score": avg_score,
        "top_keywords": dict(keywords),
        "charts": {
            "ml_score_distribution": str(ml_chart),
            "url_length_distribution": str(len_chart),
            "top_keywords": str(kw_chart),
        },
    }


def build_prompt(metrics: dict) -> str:
    mv = metrics["model_version"]
    return f"""
Analyze the data in OUTPUT_FILE from a URL ML scan.

Rules:
- score{mv} >= 30 means the URL is flagged as malicious.
- Keep output concise, practical, and analyst-friendly.

Computed metrics:
- Total URLs: {metrics['total_urls']}
- Flagged URLs: {metrics['flagged_urls']}
- Flagged URL Rate: {metrics['flagged_rate_pct']}%
- Average Risk Score: {metrics['avg_score']} out of 100
- Top suspicious keywords: {json.dumps(metrics['top_keywords'])}

Return markdown with this exact structure:

## Summary of Scanning Results
- Flagged URL Rate: ...
- Average Risk Score: ...
- Recommendation: ...

Use concrete numbers from the metrics above.
"""


def format_summary_markdown(summary_text: str, model_version: str, out_dir: str) -> str:
    return f"""


{summary_text.strip()}

---
**Interpretation notes**
- `score{model_version} >= 30` is treated as malicious.
- Charts are saved under `{out_dir}`.
""".strip() + "\n"


def render_summary(summary_text: str, summary_path: Path, model_version: str, out_dir: str) -> None:
    md_text = format_summary_markdown(summary_text, model_version, out_dir)
    summary_path.write_text(md_text, encoding="utf-8")
    print(f"Saved summary: {summary_path}")

    if display and Markdown and HTML:
 
        display(Markdown(md_text))
    else:
        print("\n" + md_text)

        


def cleanup_old_s3_objects(
    bucket: str = "example-bucket",
    prefixes: tuple[str, ...] = (
        "mlmodels/urlmodel/input/",
        "mlmodels/urlmodel/output/",
    ),
    days: int = 7,
    region: str = "eu-west-2",
) -> dict:
    s3 = boto3.client("s3", region_name=region)
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    deleted = 0
    scanned = 0

    for prefix in prefixes:
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            contents = page.get("Contents", [])
            to_delete = []

            for obj in contents:
                scanned += 1
                if obj["LastModified"] < cutoff:
                    to_delete.append({"Key": obj["Key"]})

            # S3 delete_objects max 1000 keys/request
            for i in range(0, len(to_delete), 1000):
                chunk = to_delete[i:i+1000]
                if chunk:
                    s3.delete_objects(
                        Bucket=bucket,
                        Delete={"Objects": chunk, "Quiet": True},
                    )
                    deleted += len(chunk)

    return {"scanned": scanned, "deleted": deleted, "cutoff_utc": cutoff.isoformat()}
 
    
    
    
ALLOWED_TYPES = {"url", "domain"}
ALLOWED_FLAGS = {"mal", "clean", "unknown"}


def expected_csv_path(output_dir: Path, model_version: str, data_source: str, scan_type: str, flag: str) -> Path:
    return output_dir / f"{model_version}_{data_source}_{scan_type}_{flag}.csv"


def run_single_model_scan(
    model_version: str,
    input_url: str,
    data_source: str,
    scan_type: str,
    flag: str,
    output_dir: Path,
) -> Path:
    print(f"\n=== Running scan for model {model_version} ===")
    input_s3_uri = upload_input(input_url, model_version, data_source, scan_type, flag)
    print(f"Uploaded input: {input_s3_uri}")

    start_resp = start_scan(model_version, data_source, scan_type, flag, input_s3_uri)
    job_id = start_resp.get("job_id")
    command_id = start_resp.get("command_id")
    if not job_id or not command_id:
        raise RuntimeError(f"Missing job_id/command_id from start_scan response: {start_resp}")

    final_status = poll_status(job_id, command_id, interval_sec=15)
    if final_status.get("status") != "SUCCEEDED":
        raise RuntimeError(f"Scan failed for model {model_version}: {final_status}")

    downloaded_path = Path(download_output(final_status["output_s3_uri"], local_dir=str(output_dir)))
    target_path = expected_csv_path(output_dir, model_version, data_source, scan_type, flag)

    # Normalize file name exactly to {MODEL_VERSION}_{DATA_SOURCE}_{TYPE}_{FLAG}.csv
    if downloaded_path.resolve() != target_path.resolve():
        target_path.write_bytes(downloaded_path.read_bytes())
    print(f"Saved result: {target_path}")
    return target_path


def pick_join_keys(df_old: pd.DataFrame, df_new: pd.DataFrame) -> list[str]:
    preferred = ["url", "domain", "sha256"]
    keys = [k for k in preferred if k in df_old.columns and k in df_new.columns]
    if keys:
        return keys

    fallback = [c for c in df_old.columns if c in df_new.columns and not c.startswith("score")]
    if fallback:
        return fallback[:1]

    raise ValueError("Could not determine join key(s) to combine old/new model results.")


def combine_outputs(
    csv_new: Path,
    csv_old: Path,
    model_version_new: str,
    model_version_old: str,
    data_source: str,
    scan_type: str,
    flag: str,
    output_dir: Path,
) -> Path:
    df_new = pd.read_csv(csv_new)
    df_old = pd.read_csv(csv_old)

    score_new = f"score{model_version_new}"
    score_old = f"score{model_version_old}"
    if score_new not in df_new.columns:
        raise ValueError(f"Missing column {score_new} in {csv_new}")
    if score_old not in df_old.columns:
        raise ValueError(f"Missing column {score_old} in {csv_old}")

    join_keys = pick_join_keys(df_old, df_new)
    merged = pd.merge(
        df_new[join_keys + [score_new]],
        df_old[join_keys + [score_old]],
        on=join_keys,
        how="inner",
    ).drop_duplicates(subset=join_keys, keep="first")

    combined_csv = output_dir / f"{data_source}_{scan_type}_{flag}.csv"
    merged.to_csv(combined_csv, index=False)
    print(f"Combined result: {combined_csv}")
    return combined_csv


def run_pipeline(
    model_version_new: str,
    model_version_old: str,
    input_url: str,
    data_source: str,
    scan_type: str,
    flag: str,
    model_id: str,
    output_results: str,
) -> dict:
    if scan_type not in ALLOWED_TYPES:
        raise ValueError(f"TYPE must be one of {sorted(ALLOWED_TYPES)}")
    if flag not in ALLOWED_FLAGS:
        raise ValueError(f"FLAG must be one of {sorted(ALLOWED_FLAGS)}")

    output_dir = Path(output_results)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_new = run_single_model_scan(
        model_version=model_version_new,
        input_url=input_url,
        data_source=data_source,
        scan_type=scan_type,
        flag=flag,
        output_dir=output_dir,
    )
    csv_old = run_single_model_scan(
        model_version=model_version_old,
        input_url=input_url,
        data_source=data_source,
        scan_type=scan_type,
        flag=flag,
        output_dir=output_dir,
    )

    combined_csv = combine_outputs(
        csv_new=csv_new,
        csv_old=csv_old,
        model_version_new=model_version_new,
        model_version_old=model_version_old,
        data_source=data_source,
        scan_type=scan_type,
        flag=flag,
        output_dir=output_dir,
    )

    result = {
        "model_version_new": model_version_new,
        "model_version_old": model_version_old,
        "input_url": input_url,
        "data_source": data_source,
        "type": scan_type,
        "flag": flag,
        "model_id": model_id,
        "output_results": output_results,
        "new_model_csv": str(csv_new),
        "old_model_csv": str(csv_old),
        "combined_csv": str(combined_csv),
    }
    print("\nPipeline completed.")
    for k, v in result.items():
        print(f"{k}: {v}")
    return result