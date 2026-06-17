#!/usr/bin/env python3
import re
import time
from pathlib import Path
import pandas as pd
import url_scan_ai

#!/usr/bin/env python3
 

ALLOWED_TYPES = {"url", "domain"}
ALLOWED_FLAGS = {"clean", "mal", "unknown"}
INPUT_NAME_RE = re.compile(r"^(?P<data_source>.+)_(?P<scan_type>url|domain)_(?P<flag>clean|mal|unknown)\.txt$")

POLL_START_DELAY_SEC = 6
POLL_MAX_ATTEMPTS = 5
BATCH_SIZE = 5000


def parse_input_filename(input_path: Path) -> tuple[str, str, str] | None:
    m = INPUT_NAME_RE.match(input_path.name)
    if not m:
        return None
    return m.group("data_source"), m.group("scan_type"), m.group("flag")


def collect_input_list(input_folder: Path) -> list[tuple[Path, str, str, str]]:
    input_list = []
    for p in sorted(input_folder.glob("*.txt")):
        parsed = parse_input_filename(p)
        if parsed is None:
            continue
        data_source, scan_type, flag = parsed
        input_list.append((p, data_source, scan_type, flag))
    return input_list


def expected_model_csv(output_dir: Path, model_version: str, data_source: str, scan_type: str, flag: str) -> Path:
    return output_dir / f"{model_version}_{data_source}_{scan_type}_{flag}.csv"


def expected_combined_csv(output_dir: Path, data_source: str, scan_type: str, flag: str) -> Path:
    return output_dir / f"{data_source}_{scan_type}_{flag}.csv"


def read_input_items(input_file: Path) -> list[str]:
    with input_file.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def build_batch_files(items: list[str], batch_size: int, batch_dir: Path) -> list[Path]:
    batch_dir.mkdir(parents=True, exist_ok=True)
    batch_files: list[Path] = []
    for idx in range(0, len(items), batch_size):
        chunk = items[idx : idx + batch_size]
        batch_path = batch_dir / f"batch_{idx // batch_size + 1}.txt"
        batch_path.write_text("\n".join(chunk) + "\n", encoding="utf-8")
        batch_files.append(batch_path)
    return batch_files


def run_single_scan(model_version: str, input_s3_uri: str, data_source: str, scan_type: str, flag: str, output_dir: Path) -> Path:
    start_resp = url_scan_ai.start_scan(model_version, data_source, scan_type, flag, input_s3_uri)
    job_id = start_resp.get("job_id")
    command_id = start_resp.get("command_id")
    if not job_id or not command_id:
        raise RuntimeError(f"Missing job_id/command_id in start_scan response: {start_resp}")

    time.sleep(POLL_START_DELAY_SEC)
    final_status = None
    for attempt in range(1, POLL_MAX_ATTEMPTS + 1):
        try:
            final_status = url_scan_ai.poll_status(job_id, command_id, interval_sec=15)
            break
        except Exception as exc:
            if attempt == POLL_MAX_ATTEMPTS:
                raise RuntimeError(
                    f"Polling failed for model {model_version} after {POLL_MAX_ATTEMPTS} attempts"
                ) from exc
            wait_sec = 10 * attempt
            print(
                f"poll_status transient error for model {model_version} "
                f"(attempt {attempt}/{POLL_MAX_ATTEMPTS}): {exc}. Retrying in {wait_sec}s..."
            )
            time.sleep(wait_sec)

    if final_status is None:
        raise RuntimeError(f"Polling did not return a final status for model {model_version}")
    if final_status.get("status") != "SUCCEEDED":
        raise RuntimeError(f"Scan failed for model {model_version}: {final_status}")

    downloaded = Path(url_scan_ai.download_output(final_status["output_s3_uri"], local_dir=str(output_dir)))
    target = expected_model_csv(output_dir, model_version, data_source, scan_type, flag)
    if downloaded.resolve() != target.resolve():
        target.write_bytes(downloaded.read_bytes())
    return target


def run_model_with_batches(
    model_version: str,
    input_file: Path,
    data_source: str,
    scan_type: str,
    flag: str,
    output_dir: Path,
) -> Path:
    items = read_input_items(input_file)
    if not items:
        raise ValueError(f"Input file has no URLs/domains: {input_file}")

    # Small file: one upload + one scan
    if len(items) <= BATCH_SIZE:
        print(f"  model {model_version}: 1 batch")
        input_s3_uri = url_scan_ai.upload_input(str(input_file), model_version, data_source, scan_type, flag)
        return run_single_scan(model_version, input_s3_uri, data_source, scan_type, flag, output_dir)

    # Large file: split and run batch by batch
    batch_root = output_dir / f".tmp_batches_{data_source}_{scan_type}_{flag}_{model_version}"
    work_root = output_dir / f".tmp_outputs_{data_source}_{scan_type}_{flag}_{model_version}"
    batch_files = build_batch_files(items, BATCH_SIZE, batch_root)
    work_root.mkdir(parents=True, exist_ok=True)

    print(f"  model {model_version}: {len(batch_files)} batches")
    frames = []
    total = len(batch_files)
    for idx, batch_file in enumerate(batch_files, start=1):
        print(f"    batch {idx}/{total}")
        batch_s3_uri = url_scan_ai.upload_input(str(batch_file), model_version, data_source, scan_type, flag)
        batch_csv = run_single_scan(model_version, batch_s3_uri, data_source, scan_type, flag, work_root)
        frames.append(pd.read_csv(batch_csv))

    merged = pd.concat(frames, ignore_index=True)
    target = expected_model_csv(output_dir, model_version, data_source, scan_type, flag)
    merged.to_csv(target, index=False)

    # Cleanup temporary batch files and temporary outputs
    for p in batch_files:
        if p.exists():
            p.unlink()
    if batch_root.exists():
        batch_root.rmdir()

    for p in work_root.glob("*"):
        if p.is_file():
            p.unlink()
    if work_root.exists():
        work_root.rmdir()

    return target


def pick_join_keys(df_old: pd.DataFrame, df_new: pd.DataFrame) -> list[str]:
    preferred = ["url", "domain", "sha256"]
    keys = [k for k in preferred if k in df_old.columns and k in df_new.columns]
    if keys:
        return keys
    fallback = [c for c in df_old.columns if c in df_new.columns and not c.startswith("score")]
    if fallback:
        return fallback[:1]
    raise ValueError("Could not determine join key(s) for combining outputs.")


def combine_two_models(
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
        raise ValueError(f"Missing {score_new} in {csv_new}")
    if score_old not in df_old.columns:
        raise ValueError(f"Missing {score_old} in {csv_old}")

    join_keys = pick_join_keys(df_old, df_new)
    merged = pd.merge(
        df_new[join_keys + [score_new]],
        df_old[join_keys + [score_old]],
        on=join_keys,
        how="inner",
    ).drop_duplicates(subset=join_keys, keep="first")

    out_csv = expected_combined_csv(output_dir, data_source, scan_type, flag)
    merged.to_csv(out_csv, index=False)
    return out_csv


def process_input_file(
    input_file: Path,
    data_source: str,
    scan_type: str,
    flag: str,
    model_version_new: str,
    model_version_old: str,
    output_dir: Path,
) -> Path:
    print(f"\n=== {input_file.name} ===")
    print(f"  DATA_SOURCE={data_source} TYPE={scan_type} FLAG={flag}")

    csv_new = run_model_with_batches(
        model_version=model_version_new,
        input_file=input_file,
        data_source=data_source,
        scan_type=scan_type,
        flag=flag,
        output_dir=output_dir,
    )
    print(f"  saved: {csv_new}")

    csv_old = run_model_with_batches(
        model_version=model_version_old,
        input_file=input_file,
        data_source=data_source,
        scan_type=scan_type,
        flag=flag,
        output_dir=output_dir,
    )
    print(f"  saved: {csv_old}")

    combined_csv = combine_two_models(
        csv_new=csv_new,
        csv_old=csv_old,
        model_version_new=model_version_new,
        model_version_old=model_version_old,
        data_source=data_source,
        scan_type=scan_type,
        flag=flag,
        output_dir=output_dir,
    )
    print(f"  combined: {combined_csv}")

    if not combined_csv.exists():
        raise RuntimeError(f"Combined CSV missing; not removing intermediate files: {combined_csv}")

    if csv_new.exists():
        csv_new.unlink()
        print(f"  removed: {csv_new}")
    if csv_old.exists():
        csv_old.unlink()
        print(f"  removed: {csv_old}")

    return combined_csv


def run_pipeline(model_version_new: str, model_version_old: str, input_folder: str, output_folder: str) -> list[Path]:
    input_dir = Path(input_folder)
    output_dir = Path(output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise FileNotFoundError(f"INPUT_FOLDER not found: {input_dir}")

    input_list = collect_input_list(input_dir)
    if not input_list:
        raise ValueError(
            f"No valid input .txt files found in {input_dir}. "
            "Expected names like {DATA_SOURCE}_{TYPE}_{FLAG}.txt"
        )

    print(f"Found {len(input_list)} valid input files.")
    outputs: list[Path] = []
    for idx, (input_file, data_source, scan_type, flag) in enumerate(input_list, start=1):
        print(f"\n--- File {idx}/{len(input_list)} ---")
        out_csv = process_input_file(
            input_file=input_file,
            data_source=data_source,
            scan_type=scan_type,
            flag=flag,
            model_version_new=model_version_new,
            model_version_old=model_version_old,
            output_dir=output_dir,
        )
        outputs.append(out_csv)

    print("\nCompleted all input files.")
    for p in outputs:
        print(f"- {p}")
    return outputs

 
if __name__ == "__main__":
    # Set values here directly.
    MODEL_VERSION_NEW = "123456" # placeholder, not the actual model version
    MODEL_VERSION_OLD = "654321" # placeholder, not the actual model version
    OUTPUT_FOLDER= "/home/ubuntu/efs/urlmodel/data/output_data/"  # keep private,  not the actual folder path
    INPUT_FOLDER = "/home/ubuntu/efs/urlmodel/data/input_data/"
     

    run_pipeline(
        model_version_new=MODEL_VERSION_NEW,
        model_version_old=MODEL_VERSION_OLD,
        input_folder=INPUT_FOLDER,
        output_folder=OUTPUT_FOLDER,
    )