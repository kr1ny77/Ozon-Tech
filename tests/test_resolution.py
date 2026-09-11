import json
from pathlib import Path
import subprocess
import sys
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import pytest
from ozon_dimensioner.resolution import resolve_review


@pytest.fixture
def original():
    return {
        "measurement_id": "demo-040",
        "item_id": "item-1",
        "status": "review",
        "dimensions_mm": None,
        "data_origin": "synthetic",
    }


@pytest.fixture
def record():
    path = (
        Path(__file__).resolve().parents[1] / "docs/examples/reference_measurement.json"
    )
    return json.loads(path.read_text())


def test_resolution_preserves_original_and_identifies_source(original, record):
    before = original.copy()
    result = resolve_review(original, record)
    assert original == before
    assert result["status"] == "ok"
    assert result["item_id"] == original["item_id"]
    assert result["resolution_of"] == original["measurement_id"]
    assert result["measurement_id"] != original["measurement_id"]
    assert result["obb_dimensions_sorted_mm"] == [200, 100, 40]
    assert result["height_z_mm"] is None and result["obb_axes"] is None
    assert result["data_origin"] == "synthetic"


@pytest.mark.parametrize(
    "changes",
    [
        {"resolution_of": "wrong-item"},
        {"measurement_id": "demo-040"},
        {"protocol_id": ""},
        {"minimum_box_verified": False},
        {"method": "manual_cuboid", "shape": "irregular"},
        {"uncertainty_mm": [50, 50, 50]},
        {"obb_dimensions_sorted_mm": [100, 200, 40]},
        {"data_origin": "physical"},
        {"measured_at": "2026-09-09T00:03:00"},
    ],
)
def test_incomplete_or_mismatched_reference_is_rejected(original, record, changes):
    with pytest.raises(ValueError):
        resolve_review(original, {**record, **changes})


def test_cli_resolution_and_receiver_lifecycle(tmp_path, original, record):
    source = tmp_path / "original.json"
    reference = tmp_path / "reference.json"
    output = tmp_path / "resolved.json"
    source.write_text(json.dumps([original]))
    reference.write_text(json.dumps(record))
    subprocess.run(
        [
            sys.executable,
            "-m",
            "ozon_dimensioner.cli",
            "resolve",
            str(source),
            str(reference),
            "--output",
            str(output),
        ],
        check=True,
    )
    result = json.loads(output.read_text())
    script = Path(__file__).resolve().parents[1] / "scripts/wms_demo_server.py"
    server = subprocess.Popen(
        [sys.executable, str(script), "--port", "0"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    try:
        line = server.stdout.readline().strip()
        assert line.startswith("WMS demo: ")
        url = line.removeprefix("WMS demo: ")

        def post(payload):
            request = Request(
                url,
                data=json.dumps(payload).encode(),
                headers={
                    "Content-Type": "application/json",
                    "Idempotency-Key": payload["measurement_id"],
                },
                method="POST",
            )
            with urlopen(request, timeout=2) as response:
                return json.load(response)

        with pytest.raises(HTTPError) as missing:
            post(result)
        assert missing.value.code == 409
        post(original)
        assert post(result)["resolution_of"] == original["measurement_id"]
        assert post(result)["duplicate"]
        with pytest.raises(HTTPError) as conflict:
            post({**result, "measurement_id": "second-reference"})
        assert conflict.value.code == 409
    finally:
        server.terminate()
        server.wait(timeout=5)
        server.stdout.close()


def test_physical_review_rejects_synthetic_reference(original, record):
    with pytest.raises(ValueError, match="origins must match"):
        resolve_review({**original, "data_origin": "physical"}, record)


def test_physical_reference_retains_physical_origin(original, record):
    result = resolve_review(
        {**original, "data_origin": "physical"}, {**record, "data_origin": "physical"}
    )
    assert result["data_origin"] == "physical"
