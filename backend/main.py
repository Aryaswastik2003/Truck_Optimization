# backend/main.py
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Optional, Tuple
from math import floor
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# allow frontend served from file:// or a dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Models ---
class BoxInput(BaseModel):
    box_type: str
    external_length_mm: float = Field(..., gt=0)
    external_width_mm: float = Field(..., gt=0)
    external_height_mm: float = Field(..., gt=0)
    max_payload_kg: Optional[float] = None
    quantity: Optional[int] = None  # optional limit of how many boxes available

class TruckSpec(BaseModel):
    name: str
    internal_length_mm: float
    internal_width_mm: float
    internal_height_mm: float
    payload_kg: Optional[float] = None

class OptimizeRequest(BaseModel):
    boxes: List[BoxInput]
    trucks: Optional[List[TruckSpec]] = None
    route_source: Optional[str] = None
    route_dest: Optional[str] = None
    route_types: Optional[List[str]] = None  # highway / semi-urban / village

class TruckResult(BaseModel):
    truck_name: str
    units_per_truck: int
    box_counts_by_type: dict
    cube_utilisation_pct: float
    payload_used_pct: Optional[float] = None
    notes: List[str] = []

# --- Default trucks (3 sample trucks like PRD) ---
DEFAULT_TRUCKS = [
    TruckSpec(name="Small Truck (LPT)", internal_length_mm=4500, internal_width_mm=2000, internal_height_mm=2000, payload_kg=2000),
    TruckSpec(name="Medium Truck (19ft)", internal_length_mm=8000, internal_width_mm=2300, internal_height_mm=2300, payload_kg=5000),
    TruckSpec(name="Large Truck (Container-like)", internal_length_mm=10000, internal_width_mm=2400, internal_height_mm=2400, payload_kg=10000),
]

# --- Utilities: try all rotations of a box (6 permutations for cuboid) ---
def rotations_of(dim: Tuple[float,float,float]):
    L,W,H = dim
    perms = set()
    for a,b,c in [(L,W,H),(L,H,W),(W,L,H),(W,H,L),(H,L,W),(H,W,L)]:
        perms.add((round(a,6),round(b,6),round(c,6)))
    return list(perms)

# Compute how many units of a *single* box (with one rotation) fit into truck by simple grid packing:
def units_fit_for_rotation(truck_dims, box_dims):
    tL,tW,tH = truck_dims
    bL,bW,bH = box_dims
    if bL <= 0 or bW <= 0 or bH <= 0:
        return 0
    nL = int(tL // bL)
    nW = int(tW // bW)
    nPerLayer = nL * nW
    nLayers = int(tH // bH)
    return nPerLayer * nLayers

# For a given box type, choose best rotation that yields maximum units in the truck:
def best_units_for_box(truck: TruckSpec, box: BoxInput):
    truck_dims = (truck.internal_length_mm, truck.internal_width_mm, truck.internal_height_mm)
    box_dims = (box.external_length_mm, box.external_width_mm, box.external_height_mm)
    best = 0
    best_rot = None
    for rot in rotations_of(box_dims):
        u = units_fit_for_rotation(truck_dims, rot)
        if u > best:
            best = u
            best_rot = rot
    # if user provided quantity, cap
    if box.quantity is not None:
        best = min(best, box.quantity)
    return best, best_rot

# compute volume util
def truck_volume_utilisation(truck: TruckSpec, box_list_counts, box_inputs):
    truck_vol = truck.internal_length_mm * truck.internal_width_mm * truck.internal_height_mm
    used_vol = 0
    for box_type, count in box_list_counts.items():
        # find box dims
        matching = next((b for b in box_inputs if b.box_type == box_type), None)
        if matching:
            vol = matching.external_length_mm * matching.external_width_mm * matching.external_height_mm
            used_vol += vol * count
    if truck_vol <= 0:
        return 0.0
    return 100.0 * used_vol / truck_vol

# payload usage
def payload_usage_pct(truck: TruckSpec, box_list_counts, box_inputs):
    if truck.payload_kg is None:
        return None
    # If box provides max_payload_kg, assume filled with that payload per box; else unknown -> skip
    total_payload = 0.0
    for box_type, count in box_list_counts.items():
        matching = next((b for b in box_inputs if b.box_type == box_type), None)
        if matching and matching.max_payload_kg is not None:
            total_payload += matching.max_payload_kg * count
    if total_payload == 0:
        return None
    pct = 100.0 * total_payload / truck.payload_kg if truck.payload_kg > 0 else None
    return pct

# Optimize over the provided trucks: for each truck compute how many units fit for all boxes (simple per-type greedy)
@app.post("/api/optimize", response_model=List[TruckResult])
def optimize(req: OptimizeRequest):
    trucks = req.trucks if req.trucks else DEFAULT_TRUCKS
    results = []
    for truck in trucks:
        counts_by_type = {}
        notes = []
        total_units = 0
        # For each box type compute best units fitting (independent packing per type)
        # NOTE: this is a pragmatic approach: we compute per-box-type packing (not mixed interleaved packing).
        # For better packing when mixing types, more complex solver needed.
        for box in req.boxes:
            units, best_rot = best_units_for_box(truck, box)
            counts_by_type[box.box_type] = units
            total_units += units
            if units == 0:
                notes.append(f"No {box.box_type} fit with available truck internal dims or orientation.")
            else:
                notes.append(f"{units} units of {box.box_type} (best rot {best_rot})")
        cube_pct = truck_volume_utilisation(truck, counts_by_type, req.boxes)
        payload_pct = payload_usage_pct(truck, counts_by_type, req.boxes)
        results.append(TruckResult(
            truck_name=truck.name,
            units_per_truck=total_units,
            box_counts_by_type=counts_by_type,
            cube_utilisation_pct=round(cube_pct,2),
            payload_used_pct=round(payload_pct,2) if payload_pct is not None else None,
            notes=notes
        ))
    # sort by highest cube utilisation then units
    results.sort(key=lambda r: (r.cube_utilisation_pct, r.units_per_truck), reverse=True)
    return results

    # ✅ Health check for frontend
@app.get("/api/health")
def health_check():
    return {"status": "ok"}
from fastapi import Request

@app.post("/api/recommend")
async def recommend(request: Request):
    data = await request.json()

    # --- Convert part_dimensions → BoxInput ---
    part_dims = data.get("part_dimensions", {})
    if not part_dims or not all(part_dims.get(k, 0) > 0 for k in ["length", "width", "height"]):
        return {"error": "Invalid or missing part_dimensions (length, width, height must be > 0)"}

    box = BoxInput(
        box_type="CustomBox",
        external_length_mm=part_dims["length"],
        external_width_mm=part_dims["width"],
        external_height_mm=part_dims["height"],
        max_payload_kg=data.get("part_weight"),
        quantity=data.get("quantity_per_month")
    )

    # --- Convert custom_trucks → TruckSpec list ---
    trucks = []
    for t in data.get("custom_trucks", []) or []:
        trucks.append(
            TruckSpec(
                name=t.get("name", "Custom Truck"),
                internal_length_mm=t.get("length", 0),
                internal_width_mm=t.get("width", 0),
                internal_height_mm=t.get("height", 0),
                payload_kg=t.get("max_payload")
            )
        )

    # --- Extract route details ---
    route_details = data.get("route_details", {})
    route_source = route_details.get("source_city")
    route_dest = route_details.get("destination_city")
    route_types = list(route_details.get("route_types", {}).keys()) if route_details.get("route_types") else None

    # --- Build OptimizeRequest ---
    req = OptimizeRequest(
        boxes=[box],
        trucks=trucks if trucks else None,
        route_source=route_source,
        route_dest=route_dest,
        route_types=route_types
    )

    return optimize(req)



