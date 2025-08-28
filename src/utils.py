
import re
import json
import csv

XML_FENCE_RE = re.compile(r'```(?:xml)?\n(.*?)```', re.DOTALL)

def extract_xml_from_markdown(text: str) -> str:
    m = XML_FENCE_RE.search(text or "")
    if m:
        return m.group(1).strip()
    return (text or "").strip()

def save_metrics(metrics: dict, filename: str):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

def save_metrics_csv(metrics: dict, filename: str):
    with open(filename, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        for k, v in metrics.items():
            writer.writerow([k, v])
