"""A whole-pipeline job stops before training when the camera solve disagrees with the photos' GPS."""
import json

import pytest

from splatworker.pipeline import PipelineError, gps_gate


def test_gate_stops_on_a_failed_gps_check(tmp_path):
    (tmp_path / "gps_check.json").write_text(json.dumps({"ok": False, "badFraction": 0.84, "medianM": 17.0,
                                                          "extentM": 211.0}))
    with pytest.raises(PipelineError, match="84 % of them"):
        gps_gate(tmp_path)


def test_gate_passes_a_good_check_or_no_gps(tmp_path):
    gps_gate(tmp_path)                              # photos without GPS: no file, nothing to check
    (tmp_path / "gps_check.json").write_text(json.dumps({"ok": True, "badFraction": 0.01}))
    gps_gate(tmp_path)
