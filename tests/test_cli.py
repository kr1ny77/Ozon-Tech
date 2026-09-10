import json
import subprocess
import sys
import numpy as np
import pytest
from ozon_dimensioner.simulation import simulate, cuboid
from ozon_dimensioner import __version__


@pytest.mark.parametrize("attest", [True, False])
def test_measure_cli_requires_explicit_qualification(tmp_path, attest):
    data = simulate(cuboid((80, 50, 30)))
    source = tmp_path / "scan.npz"
    target = tmp_path / "measurement.json"
    np.savez(source, points=data["points"], sensor_ids=data["sensor_ids"])
    cmd = [
        sys.executable,
        "-m",
        "ozon_dimensioner.cli",
        "measure",
        str(source),
        "--output",
        str(target),
        "--measurement-id",
        "cli-1",
        "--calibration-id",
        "test-v1",
        "--expected-profiles",
        str(data["expected_profiles"]),
        "--received-profiles",
        str(data["received_profiles"]),
    ]
    if attest:
        cmd += ["--qualified", "--calibration-valid", "--motion-valid"]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    payload = json.loads(target.read_text())
    assert payload["status"] == ("ok" if attest else "review")
    assert (payload["dimensions_mm"] is not None) == attest
    assert payload["algorithm_version"] == __version__
