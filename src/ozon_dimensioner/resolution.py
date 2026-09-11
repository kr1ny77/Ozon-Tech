"""Complete a review event using an independently measured reference record."""

from datetime import datetime
import numpy as np
from . import __version__
from .geometry import tolerance


def resolve_review(original, record):
    """Validate the reference protocol and issue a new immutable WMS event.

    Instrument qualification and minimum-box verification are external procedures.
    This function checks the submitted record, identity and uncertainty budget.
    """
    if original.get("status") != "review" or original.get("dimensions_mm") is not None:
        raise ValueError("Only unresolved review measurements can be completed")
    for key in (
        "measurement_id",
        "resolution_of",
        "operator_id",
        "instrument_id",
        "protocol_id",
        "measured_at",
    ):
        if not isinstance(record.get(key), str) or not record[key].strip():
            raise ValueError(f"Missing reference field: {key}")
    if record["resolution_of"] != original.get("measurement_id"):
        raise ValueError("Reference record belongs to a different measurement")
    if record["measurement_id"] == record["resolution_of"]:
        raise ValueError("A resolution requires a new measurement_id")
    method = record.get("method")
    if method not in ("manual_cuboid", "reference_3d"):
        raise ValueError("Unsupported reference measurement method")
    if method == "manual_cuboid" and record.get("shape") != "cuboid":
        raise ValueError("Manual edge measurement requires a verified cuboid")
    if record.get("minimum_box_verified") is not True:
        raise ValueError("Minimum-box verification is required")
    measured_at = datetime.fromisoformat(record["measured_at"].replace("Z", "+00:00"))
    if measured_at.utcoffset() is None:
        raise ValueError("Reference timestamp must include a time zone")
    origin = record.get("data_origin")
    if origin not in ("synthetic", "physical"):
        raise ValueError("Reference data_origin must be explicit")
    if (
        original.get("data_origin") in ("synthetic", "physical")
        and origin != original["data_origin"]
    ):
        raise ValueError("Reference and original data origins must match")
    dimensions = np.asarray(record.get("obb_dimensions_sorted_mm"), dtype=float)
    uncertainty = np.asarray(record.get("uncertainty_mm"), dtype=float)
    if dimensions.shape != (3,) or uncertainty.shape != (3,):
        raise ValueError("Three dimensions and three uncertainties are required")
    if (
        not np.isfinite(dimensions).all()
        or np.any(dimensions <= 0)
        or np.any(np.diff(dimensions) > 0)
    ):
        raise ValueError("Reference dimensions must be finite, positive and descending")
    if not np.isfinite(uncertainty).all() or np.any(uncertainty <= 0):
        raise ValueError("Reference uncertainties must be positive and finite")
    lower = np.maximum(dimensions - uncertainty, np.finfo(float).tiny)
    if np.any(uncertainty > tolerance(lower)):
        raise ValueError("Reference uncertainty exceeds the dimension tolerance")
    return {
        "schema_version": "1.1",
        "measurement_id": record["measurement_id"],
        "resolution_of": record["resolution_of"],
        "item_id": original.get("item_id"),
        "units": "mm",
        "status": "ok",
        "reasons": [],
        "dimension_convention": "obb_edges_descending",
        "dimensions_mm": dict(zip(("length", "width", "height"), dimensions.tolist())),
        "obb_dimensions_sorted_mm": dimensions.tolist(),
        "height_z_mm": None,
        "obb_axes": None,
        "candidate_box": None,
        "algorithm_version": __version__,
        "measured_at": record["measured_at"],
        "data_origin": origin,
        "measurement_method": method,
        "reference_evidence": {
            key: record[key]
            for key in (
                "operator_id",
                "instrument_id",
                "protocol_id",
                "minimum_box_verified",
            )
        },
        "uncertainty_mm": uncertainty.tolist(),
    }
