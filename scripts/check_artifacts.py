"""Check report structure, cross-file consistency, geometry and local links."""

import csv
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import tomllib
import pdfplumber
from pypdf import PdfReader
from ozon_dimensioner import __version__

ROOT = Path(__file__).resolve().parents[1]
summary = json.loads((ROOT / "results/summary.json").read_text())
measurements = json.loads((ROOT / "results/measurements.json").read_text())
rows = list(csv.DictReader((ROOT / "results/benchmark.csv").open()))
assert len(rows) == len(measurements) == summary["total_cases"]
assert sum(p["status"] == "review" for p in measurements) == summary["review_count"]
assert len({p["measurement_id"] for p in measurements}) == len(measurements)
for p in measurements:
    assert p["data_origin"] == "synthetic"
    if p["status"] == "review":
        assert p["dimensions_mm"] is None and p["reasons"]
    if p["status"] == "ok":
        assert p["dimensions_mm"] and not p["reasons"]
for root in [ROOT / "README.md", ROOT / "docs/report.md", ROOT / "docs/acceptance.md"]:
    for target in re.findall(r"\]\(([^)]+)\)", root.read_text()):
        if target.startswith(("http:", "https:", "#")):
            continue
        assert (root.parent / target.split("#")[0]).exists(), (root, target)
xml = ET.parse(ROOT / "results/tests.xml").getroot()
suites = xml.findall("testsuite") if xml.tag == "testsuites" else [xml]
assert sum(int(s.get("failures", 0)) + int(s.get("errors", 0)) for s in suites) == 0
pdf = ROOT / "output/pdf/ozon_dimensioning_report.pdf"
reader = PdfReader(pdf)
assert reader.metadata.author == "Kr1ny"
texts = []
page_checks = []
with pdfplumber.open(pdf) as doc:
    for i, page in enumerate(doc.pages):
        text = page.extract_text() or ""
        texts.append(text)
        assert len(text) > 600, f"Orphan/empty page {i + 1}"
        assert "{{" not in text and "\ufffd" not in text
        outside = [
            c
            for c in page.chars
            if c["x0"] < 35
            or c["x1"] > page.width - 35
            or c["top"] < 10
            or c["bottom"] > page.height - 10
        ]
        assert not outside, f"Text outside page margins: {i + 1}"
        page_checks.append(
            {"page": i + 1, "characters": len(text), "text_inside_margins": True}
        )
full = "\n".join(texts)
for term in [
    "Gocator 2490",
    "DFS60B-S1CC01000",
    "MIC-770",
    "Хаусдорфа",
    "SLSQP",
    "review",
    "800",
    "583",
]:
    assert term in full, term
assert str(summary["reference_pass_count"]) in full
geometry = json.loads((ROOT / "results/geometry_crosscheck.json").read_text())
assert geometry["passed"] and len(geometry["runs"]) == geometry["cases"]
assert geometry["configuration"] == "minimum_box defaults, identical to pipeline"
assert summary["accepted_reference_failure_count"] == 0
package = tomllib.loads((ROOT / "pyproject.toml").read_text())
assert package["project"]["version"] == __version__
assert all(p["algorithm_version"] == __version__ for p in measurements)
assert re.search(r"Алгоритм:\s*" + re.escape(__version__) + r"\b", full)
for group in [
    "isolated_reflection",
    "reflection_cluster",
    "missing_sensor_ids",
    "single_view",
    "invalid_sensor_ids",
    "missing_attestations",
    "undetected_encoder_fault",
]:
    cases = [p for p in measurements if p["case_group"] == group]
    assert cases and all(p["status"] == "review" for p in cases)
delivery = json.loads((ROOT / "results/integration_demo.json").read_text())
assert delivery["passed"] and delivery["first_delivery"]["delivered"] == len(
    measurements
)
assert delivery["repeat_delivery"] == {"delivered": 0, "pending": 0}
assert delivery["receiver_duplicate_confirmed"] and delivery["conflict_status"] == 409
assert (
    summary["nominal_accepted_count"] + summary["nominal_review_count"]
    == summary["reference_evaluable_cases"]
)
result = {
    "pages": len(reader.pages),
    "automated_tests": sum(int(s.get("tests", 0)) for s in suites),
    "synthetic_cases": len(rows),
    "local_links_valid": True,
    "status_consistency": True,
    "pdf_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
    "page_checks": page_checks,
}
(ROOT / "results/artifact_audit.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({k: v for k, v in result.items() if k != "page_checks"}, indent=2))
