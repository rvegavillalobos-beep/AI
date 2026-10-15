"""Domain constants shared by the analytics (colours live in ``rickhard.charts.theme``)."""

from __future__ import annotations

CORNERS: tuple[str, ...] = ("FL", "FR", "RL", "RR")
CORNER_LABELS = {
    "FL": "Front Left",
    "FR": "Front Right",
    "RL": "Rear Left",
    "RR": "Rear Right",
}
CORNER_INDEX = {1: "FL", 2: "FR", 3: "RL", 4: "RR"}
AXES: tuple[str, ...] = ("X", "Y")
DEV_COLS: tuple[str, ...] = tuple(f"{c}_{a}" for c in CORNERS for a in AXES)
Z_COLS: tuple[str, ...] = tuple(f"{c}_Z" for c in CORNERS)

BATTERY_TYPES: tuple[str, ...] = ("Type S", "Type M")

# Nominal corner coordinates [mm] in the CAD reference system (left corners
# carry negative Y). Same values as the original RICKHARDV app.
DEFAULT_NOMINALS: dict[str, dict[str, tuple[float, float]]] = {
    "Type S": {
        "FL": (2290.48, -559.4),
        "FR": (2290.48, 558.9),
        "RL": (997.28, -559.4),
        "RR": (997.28, 511.1),
    },
    "Type M": {
        "FL": (2290.48, -559.4),
        "FR": (2290.48, 558.9),
        "RL": (609.31, -583.3),
        "RR": (609.31, 535.0),
    },
}

# Squareness root-cause thresholds (unchanged from the original app).
ANGULAR_DEV_TOL = 0.15  # deg
DIM_DELTA_TOL = 0.8  # mm
SQ_CAUSE_ORDER: tuple[str, ...] = (
    "Parallelogram Tilt", "Trapezoidal Width", "Trapezoidal Length", "Combined Asymmetry",
)

STATUSES = ("PASS", "FAIL", "INCOMPLETE")

# Error-anatomy modes, in the order they are orthogonalised.
MODES: tuple[str, ...] = ("Tx", "Ty", "Rot", "ScaleX", "ScaleY", "Shear", "TrapX", "TrapY")
MODE_LABELS = {
    "Tx": "Shift X",
    "Ty": "Shift Y",
    "Rot": "Yaw rotation",
    "ScaleX": "Length stretch",
    "ScaleY": "Width stretch",
    "Shear": "Parallelogram (shear)",
    "TrapX": "Trapezoid (length L/R)",
    "TrapY": "Trapezoid (width F/R)",
}
MODE_FAMILY = {
    "Tx": "Placement", "Ty": "Placement", "Rot": "Placement",
    "ScaleX": "Size", "ScaleY": "Size",
    "Shear": "Shape", "TrapX": "Shape", "TrapY": "Shape",
}
RIGID_MODES = ("Tx", "Ty", "Rot")
SHAPE_MODES = ("ScaleX", "ScaleY", "Shear", "TrapX", "TrapY")

# Failure mechanisms (one per NOK first run).
MECHANISMS: tuple[str, ...] = (
    "Shift", "Yaw", "Shift + yaw", "Shape", "Placement + shape", "Height (Z)", "Missing corner",
)
MECHANISM_HELP = {
    "Shift": "The module is shifted in X/Y: a placement error a conveyor adjustment can fix.",
    "Yaw": "The module is rotated: a placement error (guides, side pushers).",
    "Shift + yaw": "Shift and rotation together push the corner out (placement).",
    "Shape": "The module itself is not square: no placement can fix it.",
    "Placement + shape": "Placement and shape errors add up on the failing corner.",
    "Height (Z)": "X and Y are in tolerance; the corner height (Z) is not.",
    "Missing corner": "A corner was not measured in the first run.",
}
