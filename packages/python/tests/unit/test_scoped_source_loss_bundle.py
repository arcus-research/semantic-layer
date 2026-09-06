from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from semantic_layer import CaptureSource, initialize, reset_capture_for_tests
from semantic_layer.validation import validate_artifact


@pytest.mark.parametrize(
    "mode",
    ["valid", "absent", "empty", "malformed", "mixed", "unresolved", "omitted", "mixed_omitted"],
)
def test_seals_source_loss_scope_from_public_receipts(tmp_path: Path, mode: str) -> None:
    reset_capture_for_tests()
    requests: list[str] = []

    class Source(CaptureSource):
        metadata = {
            "name": "fixture", "seam": "callback",
            "identity_domain": "fixture", "coverage": [],
        }

        def install(self, sink: Any) -> Any:
            root = sink.open_trace({
                "name": "run", "semantic": {"type": "agent.run", "name": "run"},
            })
            assert root.accepted and root.identity is not None
            for name in ["first", "second", "unrelated"]:
                receipt = sink.record({
                    "kind": "model", "phase": "start", "name": name,
                    "trace": root.identity, "parent_record_id": root.record_id,
                    "native": None, "semantic": {"type": "model.request"},
                })
                assert receipt.accepted
                requests.append(receipt.record_id)
            omitted = sink.record({
                "kind": "log", "phase": "event", "name": "redundant",
                "trace": root.identity, "parent_record_id": root.record_id,
                "native": None, "semantic": {"type": "capture.redundant"},
            })
            assert omitted.accepted
            refs = {
                "omitted": [omitted.record_id],
                "mixed_omitted": [requests[0], omitted.record_id],
                "valid": [requests[1], requests[0], requests[1]], "empty": [],
                "malformed": requests[0], "mixed": [requests[0], 42],
                "unresolved": [requests[0], "future-source-record"],
            }
            semantic = {
                "type": "capture.gap", "reason": "fixture_missing_evidence",
                "detail": "The callback omitted provider evidence.", "count": 3,
            }
            if mode != "absent":
                semantic["affects_refs"] = refs[mode]
            receipt = sink.record({
                "kind": "unknown", "phase": "gap", "name": "gap",
                "trace": root.identity, "parent_record_id": root.record_id,
                "native": None, "semantic": semantic,
            })
            assert receipt.accepted
            sink.record({
                "kind": "lifecycle", "phase": "end", "name": "run",
                "trace": root.identity, "parent_record_id": root.record_id,
                "native": None,
                "semantic": {"type": "agent.run", "status": "succeeded"},
            })

            class Lifecycle:
                def deactivate(self) -> None:
                    pass

                def drain(self) -> None:
                    pass

            return Lifecycle()

    try:
        capture = initialize(output=tmp_path, service_name="scoped-source-loss")
        capture.install_source(Source())
        closed = capture.shutdown()
        artifact = Path(closed.artifact_path)
        rows = [json.loads(line) for line in (artifact / "trace.jsonl").read_text().splitlines()]
        losses = [row for row in rows if row["kind"] == "loss"]
        original = next(
            row for row in losses if row["data"]["reason"] == "fixture_missing_evidence"
        )
        assert original["data"]["count"] == 3
        assert original["data"]["detail"] == "The callback omitted provider evidence."
        assert [link for link in original.get("links", []) if link["type"] == "affects"] == (
            [{"type": "affects", "record": requests[1]}, {"type": "affects", "record": requests[0]}]
            if mode == "valid" else []
        )
        invalid = mode in ["malformed", "mixed", "unresolved", "omitted", "mixed_omitted"]
        assert len(losses) == (2 if invalid else 1)
        if invalid:
            assert losses[1]["data"]["reason"] == "unresolved_affected_ref"
        assert closed.losses["fixture_missing_evidence"] == 3
        report = validate_artifact(artifact)
        assert report.valid, report.issues
    finally:
        reset_capture_for_tests()
