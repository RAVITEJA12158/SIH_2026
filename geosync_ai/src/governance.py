"""Local prototype controls for review routing and evidence traceability.

This is intentionally not an authentication, authorisation, or tamper-proof
audit system. It demonstrates the governance contract a production API and
PostGIS-backed deployment must enforce.
"""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    import pandas as pd


ROLE_ACTIONS = {
    "analyst": {"view"},
    "reviewer": {"view", "recommend"},
    "approver": {"view", "recommend", "approve", "reject"},
    "auditor": {"view", "audit"},
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as artifact:
        for block in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_artifact_manifest(paths: Iterable[Path], output_path: Path) -> dict:
    """Hash generated evidence files so the review package can reference them."""
    artifacts = []
    for path in paths:
        if path.exists():
            artifacts.append({
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            })
    manifest = {
        "generated_at": utc_now(),
        "prototype_notice": "Hashes provide demonstration traceability only; this local file is not an immutable audit ledger.",
        "artifacts": artifacts,
    }
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def append_audit_event(audit_path: Path, event_type: str, actor: str, details: dict) -> dict:
    """Append a structured local audit event for the demonstration workflow."""
    event = {
        "timestamp": utc_now(),
        "event_type": event_type,
        "actor": actor,
        "details": details,
        "prototype_notice": "Local append-only demonstration log; production requires protected, immutable audit storage.",
    }
    with audit_path.open("a", encoding="utf-8") as log:
        log.write(json.dumps(event, sort_keys=True) + "\n")
    return event


def create_review_queue(best_matches: pd.DataFrame, conflicts: dict, output_path: Path) -> dict:
    """Create an evidence-first, human-authorised review queue.

    Even a high-confidence result is only *eligible* for an automated
    progression under a future approved policy. It is never an official change
    in this prototype.
    """
    cases = []
    for row in best_matches.itertuples(index=False):
        tier = str(row.confidence_tier)
        route = "eligible_for_authorized_workflow" if tier == "high" else "manual_review_required"
        cases.append({
            "case_id": f"GS-{int(row.extracted_idx):04d}",
            "extracted_feature_index": int(row.extracted_idx),
            "cadastral_candidate_index": int(row.gov_idx),
            "confidence": round(float(row.confidence), 3),
            "confidence_tier": tier,
            "prototype_route": route,
            "evidence": {
                "iou": round(float(row.iou), 3),
                "centroid_distance_m": round(float(row.centroid_dist_m), 3),
                "area_ratio": round(float(row.area_ratio), 3),
                "attribute_score": round(float(row.attribute_score), 3),
                "cross_source_support_count": int(row.cross_source_support_count),
                "extraction_method": str(row.extraction_method),
                "extraction_confidence": round(float(row.extraction_confidence), 3),
            },
            "allowed_actions": ["recommend"],
            "final_decision": None,
        })
    queue = {
        "generated_at": utc_now(),
        "prototype_notice": (
            "This queue contains AI-assisted proposals only. No record may be "
            "changed without an authenticated, authorised official workflow."
        ),
        "role_action_policy": ROLE_ACTIONS,
        "cases": cases,
        "conflicts": conflicts,
    }
    output_path.write_text(json.dumps(queue, indent=2), encoding="utf-8")
    return queue


def record_review_decision(
    queue: dict, case_id: str, actor: str, role: str, action: str, rationale: str,
) -> dict:
    """Apply an in-memory demo decision after role-policy validation.

    Callers must persist the returned event in protected storage in a real
    deployment. This function does not authenticate an identity.
    """
    if action not in ROLE_ACTIONS.get(role, set()):
        raise PermissionError(f"Role '{role}' is not allowed to '{action}'.")
    if action not in {"approve", "reject", "recommend"}:
        raise ValueError("Review action must be recommend, approve, or reject.")
    case = next((item for item in queue["cases"] if item["case_id"] == case_id), None)
    if case is None:
        raise KeyError(f"Unknown case: {case_id}")
    event = {"timestamp": utc_now(), "actor": actor, "role": role, "action": action, "rationale": rationale}
    case["final_decision"] = event
    return event
