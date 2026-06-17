#!/usr/bin/env python3
"""
ai_2models10datasets.py

Compares two URL/domain ML model outputs across all dataset CSVs in OUTPUT_FOLDER.

Workflow:
1) Discover valid dataset files named {DATA_SOURCE}_{TYPE}_{FLAG}.csv
2) Validate required score columns:
   - score{MODEL_VERSION_NEW}  (MLP1)
   - score{MODEL_VERSION_OLD}  (ML)
3) Compute per-dataset comparison table (counts + percentages)
4) Use LangChain + Bedrock with tool-calling/function-calling to generate bullet summaries
5) Build Confluence-compatible HTML page(s) under OUTPUT_FOLDER
6) (Optional) Create/update Confluence subpage and attach dataset CSVs as links
7) Generate a project report HTML (summary / workflow / architecture / scheduling / cost)
"""

from __future__ import annotations

import base64
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
import tomllib
import callmodel


THRESHOLD = 30.0
DATASET_FILE_RE = re.compile(
    r"^(?P<data_source>.+)_(?P<scan_type>url|domain)_(?P<flag>clean|mal|unknown)\.csv$"
)

DEFAULT_PARENT_PAGE_ID = " 1234567"
DEFAULT_SPACE_KEY = "global"
DEFAULT_CONFLUENCE_BASE = "https://your-domain.atlassian.net/wiki"

DATASET_REGISTRY: dict[str, dict[str, Any]] = {}
OVERALL_REGISTRY: dict[str, Any] = {}
REVIEW_REGISTRY: dict[str, dict[str, Any]] = {}


@dataclass
class DatasetResult:
    file_path: Path
    file_name: str
    data_source: str
    scan_type: str
    flag: str
    total_rows: int
    model_version_new: str
    model_version_old: str
    winner_text: str
    table_counts: dict[str, int]
    table_percent: dict[str, float]
    llm_summary_bullets: str
    review_result: dict[str, Any]


def parse_dataset_filename(path: Path) -> tuple[str, str, str] | None:
    match = DATASET_FILE_RE.match(path.name)
    if not match:
        return None
    return match.group("data_source"), match.group("scan_type"), match.group("flag")


def compute_table(
    df: pd.DataFrame, model_version_new: str, model_version_old: str
) -> tuple[dict[str, int], dict[str, float]]:
    col_new = f"score{model_version_new}"
    col_old = f"score{model_version_old}"

    new_convicted = df[col_new] >= THRESHOLD
    old_convicted = df[col_old] >= THRESHOLD
    total = len(df)

    counts = {
        "BOTH NOT CONVICTED": int((~new_convicted & ~old_convicted).sum()),
        "BOTH CONVICTED": int((new_convicted & old_convicted).sum()),
        "ONLY ML": int((~new_convicted & old_convicted).sum()),
        "ONLY MLP1": int((new_convicted & ~old_convicted).sum()),
        "TOTAL CONVICTED ML": int(old_convicted.sum()),
        "TOTAL CONVICTED MLP1": int(new_convicted.sum()),
    }
    percentages = {
        key: (value / total * 100.0 if total else 0.0) for key, value in counts.items()
    }
    return counts, percentages


def decide_winner(flag: str, counts: dict[str, int]) -> str:
    ml = counts["TOTAL CONVICTED ML"]
    mlp1 = counts["TOTAL CONVICTED MLP1"]

    if ml == mlp1:
        return "MLP1 and ML have the same performance."
    if flag == "clean":
        return "ML wins (fewer FPs)." if mlp1 > ml else "MLP1 wins (fewer FPs)."
    if flag == "mal":
        return "MLP1 wins (fewer FNs)." if mlp1 > ml else "ML wins (fewer FNs)."
    return "Unknown-label dataset: no strict winner rule; review manually."


def pct_str(value: float) -> str:
    return f"{value:.2f}%"


@tool
def get_dataset_context(dataset_key: str) -> str:
    """Return JSON context for one dataset key."""
    if dataset_key not in DATASET_REGISTRY:
        return json.dumps({"error": f"unknown dataset_key {dataset_key}"})
    return json.dumps(DATASET_REGISTRY[dataset_key], ensure_ascii=False)


@tool
def get_overall_summary_context(_: str = "overall") -> str:
    """Return JSON context for overall metrics across datasets."""
    return json.dumps(OVERALL_REGISTRY, ensure_ascii=False)


@tool
def get_review_context(dataset_key: str) -> str:
    """Return JSON review context for one dataset key."""
    if dataset_key not in REVIEW_REGISTRY:
        return json.dumps({"error": f"unknown dataset_key {dataset_key}"})
    return json.dumps(REVIEW_REGISTRY[dataset_key], ensure_ascii=False)


def normalize_bullets(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    output = []
    for line in lines:
        candidate = re.sub(r"^\s*[-*•]+\s*", "", line).strip()
        lower = candidate.lower()
        if lower.startswith("here is") or lower.startswith("here are"):
            continue
        if line.startswith("- ") or line.startswith("* "):
            output.append(f"- {line[2:].strip()}")
        elif line.startswith("• "):
            output.append(f"- {line[2:].strip()}")
        else:
            output.append(f"- {line}")
    return "\n".join(output) if output else "- (no summary)"


def normalize_review_json_text(text: str) -> str:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate)
        candidate = re.sub(r"\s*```$", "", candidate)
        candidate = candidate.strip()
    if candidate.startswith("{") and candidate.endswith("}"):
        return candidate

    start = candidate.find("{")
    end = candidate.rfind("}")
    if start != -1 and end != -1 and end > start:
        return candidate[start : end + 1]
    return "{}"


def parse_review_result(text: str) -> dict[str, Any]:
    default_result = {
        "quality_score": 1,
        "approved": False,
        "feedback": "Reviewer output parse failed.",
    }
    try:
        raw = json.loads(normalize_review_json_text(text))
    except Exception:
        return default_result

    quality_score = raw.get("quality_score", 1)
    approved = raw.get("approved", False)
    feedback = str(raw.get("feedback", "")).strip() or "No feedback provided."

    try:
        quality_score = int(quality_score)
    except Exception:
        quality_score = 1
    quality_score = max(1, min(10, quality_score))
    approved = bool(approved)
    return {
        "quality_score": quality_score,
        "approved": approved,
        "feedback": feedback,
    }


def run_tool_calling_agent(
    llm: ChatBedrockConverse,
    system_prompt: str,
    user_prompt: str,
    tools: list[Any],
) -> str:
    bound = llm.bind_tools(tools)
    messages: list[Any] = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    for _ in range(6):
        ai_msg = bound.invoke(messages)
        messages.append(ai_msg)
        tool_calls = getattr(ai_msg, "tool_calls", None) or []

        if not tool_calls:
            content = ai_msg.content
            if isinstance(content, list):
                text_parts = [p.get("text", "") for p in content if isinstance(p, dict)]
                return "\n".join([t for t in text_parts if t.strip()]).strip()
            return str(content).strip()

        for tc in tool_calls:
            name = tc["name"]
            args = tc.get("args", {})
            if name == "get_dataset_context":
                payload = get_dataset_context.invoke(args)
            elif name == "get_overall_summary_context":
                payload = get_overall_summary_context.invoke(args)
            else:
                payload = json.dumps({"error": f"unsupported tool {name}"})
            messages.append(ToolMessage(content=payload, tool_call_id=tc["id"]))

    return "- Unable to generate summary within tool-calling loop limit."


def analyze_dataset_with_agent(
    llm: ChatBedrockConverse,
    dataset_key: str,
    model_version_new: str,
    model_version_old: str,
) -> str:
    system_prompt = (
        "You are Analyst AGENT, an ML engineer reviewing model performance. "
        "Always call get_dataset_context first, then return concise bullet points only. "
        "Do not include any intro sentence (for example: 'Here is my analysis'). "
        "Start directly with bullets."
    )
    user_prompt = (
        f"Analyze dataset key `{dataset_key}`.\n"
        f"- MLP1 model version is {model_version_new}\n"
        f"- ML model version is {model_version_old}\n"
        "Return no more than 4 concise bullet points with numeric evidence and winner conclusion. "
        "Do not use markdown tables or pipe symbols."
    )
    text = run_tool_calling_agent(
        llm=llm,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        tools=[get_dataset_context],
    )
    return normalize_bullets(text)


def detect_dataset_anomalies(counts: dict[str, int], total_rows: int) -> list[str]:
    anomalies: list[str] = []
    if total_rows <= 0:
        anomalies.append("Dataset has zero rows.")
        return anomalies

    if not anomalies:
        anomalies.append("No major anomalies detected.")
    return anomalies


def review_dataset_with_agent(llm: ChatBedrockConverse, dataset_key: str) -> dict[str, Any]:
    system_prompt = (
        "You are REVIEWER AGENT. "
        "You must call get_review_context and get_dataset_context before final answer. "
        "Review dimensions: factual accuracy, completeness. "
        "Output ONLY JSON with exact keys: "
        '{"quality_score": 1-10, "approved": bool, "feedback": "..."}'
    )
    user_prompt = (
        f"Review dataset `{dataset_key}`.\n"
        "Review dimensions:\n"
        "1) Factual accuracy — Are the numbers consistent with the data?\n"
        "2) Clarity — Is the winner conclusion clear and supported?\n"
       
        "Return ONLY JSON. Do not return markdown or extra text."
    )
    text = run_tool_calling_agent(
        llm=llm,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        tools=[get_review_context, get_dataset_context],
    )
    return parse_review_result(text)


def analyze_overall_with_agent(llm: ChatBedrockConverse) -> str:
    system_prompt = (
        "You are an ML engineer preparing an executive summary. "
        "Always call get_overall_summary_context and then produce concise bullets. "
        "Do not include any intro sentence (for example: 'Here is my analysis'). "
        "Start directly with bullets."
    )
    user_prompt = (
        "Create an overall summary across all datasets. "
        "Focus on FP vs FN tradeoffs and where each model wins. "
        "Return no more than 4 concise bullet points. "
        "Do not use markdown tables or pipe symbols."
    )
    text = run_tool_calling_agent(
        llm=llm,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        tools=[get_overall_summary_context],
    )
    return normalize_bullets(text)


def html_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def strip_markdown_symbols(line: str) -> str:
    line = line.strip()
    line = re.sub(r"^\s*[-*•]+\s*", "", line)
    line = re.sub(r"^\s*#+\s*", "", line)
    line = line.replace("**", "").replace("__", "").replace("`", "")
    line = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1", line)
    line = line.replace("|", " ")
    line = re.sub(r"\s{2,}", " ", line)
    return line.strip()


def is_markdown_noise_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    # Markdown table rows/separators and horizontal rules.
    if stripped.startswith("|") and stripped.endswith("|"):
        return True
    if stripped.count("|") >= 2:
        return True
    if re.fullmatch(r"[\|\-\:\s]+", stripped):
        return True
    if re.fullmatch(r"[-_*]{3,}", stripped):
        return True
    return False


def bullet_text_to_html_list(text: str, max_items: int = 4) -> str:
    items = []
    for raw_line in text.splitlines():
        if is_markdown_noise_line(raw_line):
            continue
        clean = strip_markdown_symbols(raw_line)
        if clean:
            items.append(f"<li>{html_escape(clean)}</li>")
        if len(items) >= max_items:
            break
    if not items:
        items.append("<li>(no summary)</li>")
    return "<ul>" + "".join(items) + "</ul>"


def counts_percent_table_html(counts: dict[str, int], percent: dict[str, float]) -> str:
    cols = list(counts.keys())
    header = "".join(f"<th>{html_escape(col)}</th>" for col in cols)
    row_count = "".join(f"<td>{counts[col]}</td>" for col in cols)
    row_pct = "".join(f"<td>{pct_str(percent[col])}</td>" for col in cols)
    return f"""
<table border="1" cellspacing="0" cellpadding="6">
  <thead><tr>{header}</tr></thead>
  <tbody>
    <tr>{row_count}</tr>
    <tr>{row_pct}</tr>
  </tbody>
</table>
""".strip()


def overall_dataset_table_html(results: list[DatasetResult]) -> str:
    rows = []
    for result in results:
        ml = result.table_counts["TOTAL CONVICTED ML"]
        mlp1 = result.table_counts["TOTAL CONVICTED MLP1"]
        ml_pct = result.table_percent["TOTAL CONVICTED ML"]
        mlp1_pct = result.table_percent["TOTAL CONVICTED MLP1"]
        rows.append(
            f"<tr>"
            f"<td>{html_escape(result.file_name)}</td>"
            f"<td>{result.total_rows}</td>"
            f"<td>{ml} ({pct_str(ml_pct)})</td>"
            f"<td>{mlp1} ({pct_str(mlp1_pct)})</td>"
            f"<td>{html_escape(result.winner_text)}</td>"
            f"</tr>"
        )

    if not rows:
        rows.append('<tr><td colspan="5">No datasets found.</td></tr>')

    return f"""
<table border="1" cellspacing="0" cellpadding="6">
  <thead>
    <tr>
      <th>Dataset</th>
      <th>Rows</th>
      <th>TOTAL CONVICTED ML</th>
      <th>TOTAL CONVICTED MLP1</th>
      <th>Winner</th>
    </tr>
  </thead>
  <tbody>
    {''.join(rows)}
  </tbody>
</table>
""".strip()


def build_overall_counts(results: list[DatasetResult]) -> tuple[dict[str, int], dict[str, float]]:
    agg = {
        "BOTH NOT CONVICTED": 0,
        "BOTH CONVICTED": 0,
        "ONLY ML": 0,
        "ONLY MLP1": 0,
        "TOTAL CONVICTED ML": 0,
        "TOTAL CONVICTED MLP1": 0,
    }
    total_rows = 0
    for result in results:
        total_rows += result.total_rows
        for key in agg:
            agg[key] += result.table_counts[key]
    pct = {k: (v / total_rows * 100.0 if total_rows else 0.0) for k, v in agg.items()}
    return agg, pct


def build_confluence_html(
    model_version_new: str,
    model_version_old: str,
    results: list[DatasetResult],
    overall_counts: dict[str, int],
    overall_pct: dict[str, float],
    overall_summary_bullets: str,
    attachment_map: dict[str, str] | None = None,
) -> str:
    attachment_map = attachment_map or {}

    def section_for_dataset(result: DatasetResult) -> str:
        title = f"{result.data_source} – {result.scan_type} - {result.flag}"
        winner = result.winner_text
        table_html = counts_percent_table_html(result.table_counts, result.table_percent)
        summary_html = bullet_text_to_html_list(result.llm_summary_bullets, max_items=4)
        attach_link = attachment_map.get(result.file_name, "")
        attach_html = (
            f'<p><a href="{html_escape(attach_link)}">{html_escape(result.file_name)}</a></p>'
            if attach_link
            else ""
        )
        return f"""
<h3>{html_escape(title)}</h3>
<p><b>{html_escape(winner)}</b></p>
{attach_html}
{table_html}
{summary_html}
""".strip()

    dataset_sections = "".join(section_for_dataset(r) for r in results)

    overall_table = overall_dataset_table_html(results)
    overall_summary_html = bullet_text_to_html_list(overall_summary_bullets, max_items=4)

    return f"""
<h1>URL Model review-FP FN {html_escape(model_version_new)} VS {html_escape(model_version_old)}</h1>

<h2>Overall Summary</h2>
{overall_table}
{overall_summary_html}

<h2>Dataset Comparison</h2>
{dataset_sections}

<h2>Spot Checking</h2>
<p></p>
""".strip()


def read_wiki_credentials() -> tuple[str, str]:
    pyproject_path = Path(__file__).resolve().parent / "pyproject.toml"
    if not pyproject_path.exists():
        raise FileNotFoundError("pyproject.toml not found for Confluence credentials.")
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    cfg = data.get("tool", {}).get("urlmodel", {})
    email = os.getenv("CONFLUENCE_EMAIL") or cfg.get("confluence_email", "")
    token = os.getenv("CONFLUENCE_TOKEN") or cfg.get("confluence_token", "")
    if not token:
        raise ValueError("Confluence token missing (set CONFLUENCE_TOKEN or tool.urlmodel.confluence_token).")
    return email, token


def _auth_headers(email: str, token: str) -> dict[str, str]:
    if email:
        b64 = base64.b64encode(f"{email}:{token}".encode("utf-8")).decode("utf-8")
        return {"Authorization": f"Basic {b64}"}
    return {"Authorization": f"Bearer {token}"}


def confluence_upsert_subpage(
    base_url: str,
    space_key: str,
    parent_page_id: str,
    title: str,
    storage_html: str,
    email: str,
    token: str,
) -> str:
    headers = {"Content-Type": "application/json", **_auth_headers(email, token)}
    # Restrict lookup to child pages under the specified parent.
    search_url = f"{base_url}/rest/api/content/search"
    escaped_title = title.replace('"', '\\"')
    cql = (
        f'type=page AND space="{space_key}" '
        f"AND ancestor={parent_page_id} "
        f'AND title="{escaped_title}"'
    )
    resp = requests.get(
        search_url,
        params={"cql": cql, "expand": "version"},
        headers=headers,
        timeout=60,
    )
    resp.raise_for_status()
    matches = resp.json().get("results", [])

    body_value = {"value": storage_html, "representation": "storage"}
    if matches:
        page = matches[0]
        page_id = page["id"]
        version = page["version"]["number"] + 1
        payload = {
            "id": page_id,
            "type": "page",
            "title": title,
            "space": {"key": space_key},
            "version": {"number": version},
            "body": {"storage": body_value},
        }
        update_url = f"{base_url}/rest/api/content/{page_id}"
        update_resp = requests.put(update_url, headers=headers, data=json.dumps(payload), timeout=60)
        update_resp.raise_for_status()
        return page_id

    payload = {
        "type": "page",
        "title": title,
        "space": {"key": space_key},
        "ancestors": [{"id": parent_page_id}],
        "body": {"storage": body_value},
    }
    create_url = f"{base_url}/rest/api/content"
    create_resp = requests.post(create_url, headers=headers, data=json.dumps(payload), timeout=60)
    create_resp.raise_for_status()
    return create_resp.json()["id"]


def confluence_upload_attachment(
    base_url: str,
    page_id: str,
    file_path: Path,
    email: str,
    token: str,
) -> str:
    url = f"{base_url}/rest/api/content/{page_id}/child/attachment"
    headers = {"X-Atlassian-Token": "no-check", **_auth_headers(email, token)}

    with file_path.open("rb") as handle:
        files = {"file": (file_path.name, handle, "text/csv")}
        resp = requests.post(url, headers=headers, files=files, timeout=120)
        resp.raise_for_status()
    return f"{base_url}/download/attachments/{page_id}/{file_path.name}"


def discover_datasets(
    output_folder: Path, model_version_new: str, model_version_old: str
) -> list[tuple[Path, str, str, str]]:
    discovered: list[tuple[Path, str, str, str]] = []
    col_new = f"score{model_version_new}"
    col_old = f"score{model_version_old}"

    for path in sorted(output_folder.glob("*.csv")):
        parsed = parse_dataset_filename(path)
        if not parsed:
            continue
        data_source, scan_type, flag = parsed
        try:
            df = pd.read_csv(path, nrows=50)
        except Exception:
            continue
        if col_new in df.columns and col_old in df.columns:
            discovered.append((path, data_source, scan_type, flag))
    return discovered


def run_pipeline(
    model_version_new: str,
    model_version_old: str,
    output_folder: str,
    model_id: str,
    confluence_publish: bool = False,
    confluence_base_url: str = DEFAULT_CONFLUENCE_BASE,
    confluence_space_key: str = DEFAULT_SPACE_KEY,
    confluence_parent_page_id: str = DEFAULT_PARENT_PAGE_ID,
) -> dict[str, Any]:
    out_dir = Path(output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    discovered = discover_datasets(out_dir, model_version_new, model_version_old)
    if not discovered:
        raise ValueError(
            "No valid input CSVs found. Need files named {DATA_SOURCE}_{TYPE}_{FLAG}.csv "
            f"with both score{model_version_new} and score{model_version_old} columns."
        )

    input_list = [p.name for (p, _ds, _t, _f) in discovered]
    print(f"INPUT_LIST ({len(input_list)} datasets): {input_list}")

    llm = ChatBedrockConverse(
        model=model_id,
        client=callmodel.get_bedrock_client(),
        temperature=0.2,
        max_tokens=900,
    )

    results: list[DatasetResult] = []
    for path, data_source, scan_type, flag in discovered:
        df = pd.read_csv(path)
        counts, pct = compute_table(df, model_version_new, model_version_old)
        winner = decide_winner(flag, counts)
        dataset_key = path.name

        DATASET_REGISTRY[dataset_key] = {
            "file_name": path.name,
            "data_source": data_source,
            "type": scan_type,
            "flag": flag,
            "model_version_new": model_version_new,
            "model_version_old": model_version_old,
            "threshold": THRESHOLD,
            "row_count": len(df),
            "comparison_counts": counts,
            "comparison_percentages": {k: round(v, 4) for k, v in pct.items()},
            "winner": winner,
        }

        bullets = analyze_dataset_with_agent(
            llm=llm,
            dataset_key=dataset_key,
            model_version_new=model_version_new,
            model_version_old=model_version_old,
        )
        REVIEW_REGISTRY[dataset_key] = {
            "file_name": path.name,
            "model_version_new": model_version_new,
            "model_version_old": model_version_old,
            "row_count": len(df),
            "winner": winner,
            "comparison_counts": counts,
            "comparison_percentages": {k: round(v, 4) for k, v in pct.items()},
            "detected_anomalies": detect_dataset_anomalies(counts, len(df)),
            "analyst_summary_bullets": bullets,
        }
        review_result = review_dataset_with_agent(llm=llm, dataset_key=dataset_key)

        results.append(
            DatasetResult(
                file_path=path,
                file_name=path.name,
                data_source=data_source,
                scan_type=scan_type,
                flag=flag,
                total_rows=len(df),
                model_version_new=model_version_new,
                model_version_old=model_version_old,
                winner_text=winner,
                table_counts=counts,
                table_percent=pct,
                llm_summary_bullets=bullets,
                review_result=review_result,
            )
        )

    overall_counts, overall_pct = build_overall_counts(results)
    OVERALL_REGISTRY.update(
        {
            "model_version_new": model_version_new,
            "model_version_old": model_version_old,
            "overall_counts": overall_counts,
            "overall_percentages": {k: round(v, 4) for k, v in overall_pct.items()},
            "dataset_count": len(results),
            "datasets": [r.file_name for r in results],
        }
    )

    overall_summary = analyze_overall_with_agent(llm)

    page_title = f"URL Model review-FP FN {model_version_new} VS {model_version_old}"
    local_page_html = build_confluence_html(
        model_version_new=model_version_new,
        model_version_old=model_version_old,
        results=results,
        overall_counts=overall_counts,
        overall_pct=overall_pct,
        overall_summary_bullets=overall_summary,
        attachment_map={},
    )
    page_path = out_dir / f"url_model_review_{model_version_new}_vs_{model_version_old}.html"
    page_path.write_text(local_page_html, encoding="utf-8")

    confluence_page_id = ""
    confluence_error = ""
    if confluence_publish:
        try:
            email, token = read_wiki_credentials()
            confluence_page_id = confluence_upsert_subpage(
                base_url=confluence_base_url,
                space_key=confluence_space_key,
                parent_page_id=confluence_parent_page_id,
                title=page_title,
                storage_html=local_page_html,
                email=email,
                token=token,
            )

            attachment_map: dict[str, str] = {}
            for result in results:
                attachment_map[result.file_name] = confluence_upload_attachment(
                    base_url=confluence_base_url,
                    page_id=confluence_page_id,
                    file_path=result.file_path,
                    email=email,
                    token=token,
                )

            page_with_links = build_confluence_html(
                model_version_new=model_version_new,
                model_version_old=model_version_old,
                results=results,
                overall_counts=overall_counts,
                overall_pct=overall_pct,
                overall_summary_bullets=overall_summary,
                attachment_map=attachment_map,
            )
            confluence_upsert_subpage(
                base_url=confluence_base_url,
                space_key=confluence_space_key,
                parent_page_id=confluence_parent_page_id,
                title=page_title,
                storage_html=page_with_links,
                email=email,
                token=token,
            )
            print(f"Confluence subpage published/updated: page_id={confluence_page_id}")
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "unknown"
            body = exc.response.text[:500] if exc.response is not None else str(exc)
            confluence_error = (
                f"Confluence publish failed (HTTP {status}). "
                f"Check Confluence token permissions for space '{confluence_space_key}'. "
                f"Response: {body}"
            )
            print(f"WARNING: {confluence_error}")

    return {
        "datasets_processed": len(results),
        "page_title": page_title,
        "local_confluence_html": str(page_path),
        "confluence_page_id": confluence_page_id,
        "published_to_confluence": bool(confluence_page_id),
        "confluence_error": confluence_error,
        "review_results": {r.file_name: r.review_result for r in results},
    }


def main(CONFLUENCE_PUBLISH: bool = False) -> None:
    MODEL_VERSION_NEW = "123456"  # alias: MLP1 (placeholder)
    MODEL_VERSION_OLD = "654321"  # alias: ML (placeholder)
    OUTPUT_FOLDER = "/home/ubuntu/efs/urlmodel/data/output_data/"
    MODEL_ID = os.getenv("FOUNDATION_MODEL_ID", "<FOUNDATION_MODEL_ID>")
 
    
    CONFLUENCE_BASE_URL = "https://aaaaa.atlassian.net/wiki"
    CONFLUENCE_SPACE_KEY = "global"
    CONFLUENCE_PARENT_PAGE_ID = "1234567"
    result = run_pipeline(
        model_version_new=MODEL_VERSION_NEW,
        model_version_old=MODEL_VERSION_OLD,
        output_folder=OUTPUT_FOLDER,
        model_id=MODEL_ID,
        confluence_publish=CONFLUENCE_PUBLISH,
        confluence_base_url=CONFLUENCE_BASE_URL,
        confluence_space_key=CONFLUENCE_SPACE_KEY,
        confluence_parent_page_id=CONFLUENCE_PARENT_PAGE_ID,
    )
    print("Done.")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
