# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Tuple, Dict, Any
from fastapi.middleware.cors import CORSMiddleware
import logging
from copy import deepcopy

# Attempt to import py3dbp; if not installed we will return friendly error.
try:
    from py3dbp import Packer, Bin, Item
except Exception:
    Packer = None
    Bin = None
    Item = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("truck_packer")

app = FastAPI(title="3D Truck Packing API")

# allow frontend served from file:// or a dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Data models
# ----------------------------
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

# ----------------------------
# Default trucks
# ----------------------------
DEFAULT_TRUCKS = [
    TruckSpec(name="22 ft Truck", internal_length_mm=6700, internal_width_mm=2440, internal_height_mm=2440, payload_kg=7500),
    TruckSpec(name="32 ft Single Axle", internal_length_mm=9750, internal_width_mm=2440, internal_height_mm=2440, payload_kg=9000),
    TruckSpec(name="32 ft Multi Axle", internal_length_mm=9750, internal_width_mm=2440, internal_height_mm=2440, payload_kg=16000),
]

# ----------------------------
# Helpers
# ----------------------------
def compute_volume_mm_from_dims(l: float, w: float, h: float) -> float:
    return l * w * h

def compute_volume_mm(box: BoxInput) -> float:
    return compute_volume_mm_from_dims(box.external_length_mm, box.external_width_mm, box.external_height_mm)

def _make_bin_for_truck(truck: TruckSpec) -> Bin:
    return Bin(
        truck.name,
        truck.internal_length_mm,
        truck.internal_width_mm,
        truck.internal_height_mm,
        truck.payload_kg if truck.payload_kg is not None else 0,
    )

def _collect_summary_from_packer(packer: Packer, req_boxes: List[BoxInput], truck: TruckSpec, total_items_created: int) -> Dict[str, Any]:
    if not hasattr(packer, "bins") or len(packer.bins) == 0:
        return None

    b = packer.bins[0]
    fitted = getattr(b, "items", []) or []
    unfitted = getattr(b, "unfitted_items", []) or []

    counts_by_type: Dict[str, int] = {}
    used_volume_mm3 = 0.0
    used_payload_kg = 0.0
    placements = []

    for it in fitted:
        item_name = getattr(it, "name", None) or str(it)
        counts_by_type[item_name] = counts_by_type.get(item_name, 0) + 1

        # Get dimensions from the item
        w = getattr(it, "width", None) or getattr(it, "w", None)
        h = getattr(it, "height", None) or getattr(it, "h", None)
        d = getattr(it, "depth", None) or getattr(it, "d", None)
        weight = getattr(it, "weight", None) or getattr(it, "mass", None) or 0.0

        # Fallback to original box dimensions if item doesn't have them
        if (w is None or h is None or d is None):
            matching = next((bx for bx in req_boxes if bx.box_type == item_name), None)
            if matching:
                w = w or matching.external_length_mm
                h = h or matching.external_width_mm
                d = d or matching.external_height_mm
                weight = weight or (matching.max_payload_kg or 0.0)

        try:
            used_volume_mm3 += float(w) * float(h) * float(d)
        except Exception:
            pass
        try:
            used_payload_kg += float(weight)
        except Exception:
            pass

        # Get position and rotation information
        pos = getattr(it, "position", None) or getattr(it, "pos", None)
        rotation = getattr(it, "rotation_type", None) or getattr(it, "rotation", None)
        
        # Enhanced placement data for 3D visualization
        placement_data = {
            "name": item_name,
            "position": list(pos) if pos is not None else [0, 0, 0],
            "rotation": rotation if rotation is not None else 0,
            "dims_mm": [float(w) if w else 0, float(h) if h else 0, float(d) if d else 0],
            "weight_kg": float(weight) if weight else 0.0,
        }
        
        # Add corner coordinates for better 3D positioning
        if pos is not None and w is not None and h is not None and d is not None:
            x, y, z = pos
            placement_data["corners"] = {
                "min": [float(x), float(y), float(z)],
                "max": [float(x + w), float(y + h), float(z + d)]
            }
            placement_data["center"] = [
                float(x + w/2), 
                float(y + h/2), 
                float(z + d/2)
            ]
        
        placements.append(placement_data)

    truck_vol_mm3 = truck.internal_length_mm * truck.internal_width_mm * truck.internal_height_mm
    cube_util_pct = (100.0 * used_volume_mm3 / truck_vol_mm3) if truck_vol_mm3 > 0 else 0.0
    payload_pct = (100.0 * used_payload_kg / truck.payload_kg) if (truck.payload_kg and truck.payload_kg > 0) else None

    unfitted_counts: Dict[str, int] = {}
    for it in unfitted:
        item_name = getattr(it, "name", None) or str(it)
        unfitted_counts[item_name] = unfitted_counts.get(item_name, 0) + 1

    return {
        "truck_name": b.name if hasattr(b, "name") else truck.name,
        "truck_dimensions": {
            "length_mm": truck.internal_length_mm,
            "width_mm": truck.internal_width_mm,
            "height_mm": truck.internal_height_mm,
            "volume_mm3": truck_vol_mm3
        },
        "units_packed_total": sum(counts_by_type.values()),
        "box_counts_by_type": counts_by_type,
        "unfitted_counts": unfitted_counts,
        "cube_utilisation_pct": round(cube_util_pct, 4),
        "payload_used_kg": round(used_payload_kg, 3),
        "payload_used_pct": round(payload_pct, 4) if payload_pct is not None else None,
        "placements_sample": placements[:500],  # Increased limit for better 3D viz
        "total_placements": len(placements),
        "notes": [
            f"Total items considered/created: {total_items_created}.",
            f"Successfully placed: {len(placements)} items.",
            "Tried multiple insertion orders (volume/weight/density/round-robin) and returned best found."
        ],
    }

def _prepare_box_sequences(truck: TruckSpec, boxes: List[BoxInput], cap_per_type: int = 5000) -> Dict[str, List[Tuple[BoxInput, int]]]:
    truck_vol = truck.internal_length_mm * truck.internal_width_mm * truck.internal_height_mm
    per_box_info = []
    for bx in boxes:
        vol = compute_volume_mm(bx)
        if vol <= 0:
            continue
        theoretical_max = int(truck_vol // vol) if vol > 0 else 1
        if theoretical_max <= 0:
            theoretical_max = 1
        q = bx.quantity if bx.quantity is not None else min(theoretical_max, cap_per_type)
        per_box_info.append({
            "box": bx,
            "volume": vol,
            "weight": bx.max_payload_kg or 0.0,
            "theoretical_max": theoretical_max,
            "q": q
        })

    sequences: Dict[str, List[Tuple[BoxInput, int]]] = {}

    # 1. largest volume first
    seq = sorted(per_box_info, key=lambda x: x["volume"], reverse=True)
    sequences["volume_desc"] = [(p["box"], p["q"]) for p in seq]

    # 2. smallest volume first
    seq = sorted(per_box_info, key=lambda x: x["volume"])
    sequences["volume_asc"] = [(p["box"], p["q"]) for p in seq]

    # 3. heaviest first
    seq = sorted(per_box_info, key=lambda x: x["weight"], reverse=True)
    sequences["weight_desc"] = [(p["box"], p["q"]) for p in seq]

    # 4. weight density (weight/volume) first
    seq = sorted(per_box_info, key=lambda x: (x["weight"] / x["volume"]) if x["volume"] > 0 else 0.0, reverse=True)
    sequences["weight_density_desc"] = [(p["box"], p["q"]) for p in seq]

    # 5. round-robin single-unit mixing (to encourage mixed packing)
    rr: List[Tuple[BoxInput, int]] = []
    qleft = {p["box"].box_type: p["q"] for p in per_box_info}
    total_cap = min(sum(qleft.values()), 5000)
    count_added = 0
    any_left = True
    while any_left and count_added < total_cap:
        any_left = False
        for p in per_box_info:
            btype = p["box"].box_type
            if qleft.get(btype, 0) > 0 and count_added < total_cap:
                rr.append((p["box"], 1))
                qleft[btype] -= 1
                count_added += 1
                any_left = True
    sequences["round_robin_single_units"] = rr

    # 6. hybrid: heavy-first then small
    heavy_then_small = sorted(per_box_info, key=lambda x: x["weight"], reverse=True) + sorted(per_box_info, key=lambda x: x["volume"])
    seen = set()
    hybrid = []
    for p in heavy_then_small:
        btype = p["box"].box_type
        if btype not in seen:
            seen.add(btype)
            hybrid.append((p["box"], p["q"]))
    sequences["hybrid_heavy_then_small"] = hybrid

    return sequences

def _run_packing_for_sequence(truck: TruckSpec, sequence: List[Tuple[BoxInput, int]]) -> Tuple[Optional[Dict[str, Any]], int]:
    packer = Packer()
    truck_bin = _make_bin_for_truck(truck)
    packer.add_bin(truck_bin)

    total_items_created = 0
    for box, cnt in sequence:
        if cnt <= 0:
            continue
        safe_cnt = min(cnt, 5000)
        for _ in range(safe_cnt):
            packer.add_item(Item(
                box.box_type,
                box.external_length_mm,
                box.external_width_mm,
                box.external_height_mm,
                box.max_payload_kg if box.max_payload_kg is not None else 0.0
            ))
            total_items_created += 1

    try:
        # try the modern signature first
        packer.pack(distribute_items=True)
    except TypeError:
        try:
            packer.pack(bigger_first=True)
        except Exception as e:
            logger.exception("Packer.pack failed: %s", e)
            return None, total_items_created
    except Exception as e:
        logger.exception("Packer.pack unexpected error: %s", e)
        return None, total_items_created

    summary = _collect_summary_from_packer(packer, [b for b, _ in sequence], truck, total_items_created)
    return summary, total_items_created

# ----------------------------
# Main packing endpoint
# ----------------------------
@app.post("/api/optimize")
def optimize(req: OptimizeRequest):
    logger.info("DEBUG: /api/optimize called — incoming keys: %s", list(req.dict().keys()) if hasattr(req, "dict") else "no-req-dict")

    if Packer is None:
        raise HTTPException(status_code=500, detail="py3dbp not installed. Please run `pip install py3dbp` on the server.")

    trucks = req.trucks if req.trucks else DEFAULT_TRUCKS
    results: List[Dict[str, Any]] = []

    for truck in trucks:
        sequences = _prepare_box_sequences(truck, req.boxes, cap_per_type=5000)

        best_summary = None
        best_units = -1
        best_cube = -1.0
        best_created = 0

        for name, seq in sequences.items():
            try:
                summary, created = _run_packing_for_sequence(truck, seq)
            except Exception as e:
                logger.exception("Error running strategy %s: %s", name, e)
                summary, created = None, 0

            if not summary:
                continue

            units = summary.get("units_packed_total", 0)
            cube = summary.get("cube_utilisation_pct", 0.0)
            payload_pct = summary.get("payload_used_pct", None)

            if payload_pct is not None and payload_pct > 100.0 + 1e-6:
                logger.info("Strategy %s produced payload >100%%, skipping", name)
                continue

            if (units > best_units) or (units == best_units and cube > best_cube):
                best_summary = deepcopy(summary)
                best_units = units
                best_cube = cube
                best_created = created

        if best_summary is None:
            seq_default = [(b, (int((truck.internal_length_mm * truck.internal_width_mm * truck.internal_height_mm) // compute_volume_mm(b)) if b.quantity is None else b.quantity)) for b in req.boxes]
            summary, created = _run_packing_for_sequence(truck, seq_default)
            best_summary = summary or {
                "truck_name": truck.name,
                "truck_dimensions": {
                    "length_mm": truck.internal_length_mm,
                    "width_mm": truck.internal_width_mm,
                    "height_mm": truck.internal_height_mm,
                    "volume_mm3": truck.internal_length_mm * truck.internal_width_mm * truck.internal_height_mm
                },
                "units_packed_total": 0,
                "box_counts_by_type": {},
                "unfitted_counts": {},
                "cube_utilisation_pct": 0.0,
                "payload_used_kg": 0.0,
                "payload_used_pct": 0.0,
                "placements_sample": [],
                "total_placements": 0,
                "notes": ["No bin result returned by packer (fallback)."]
            }
            best_created = created

        if best_summary and isinstance(best_summary, dict):
            notes = best_summary.get("notes", [])
            if not notes or "Total items considered/created" not in str(notes[0]):
                notes.insert(0, f"Total items considered/created: {best_created}.")
            best_summary["notes"] = notes
            results.append(best_summary)
        else:
            results.append({
                "truck_name": truck.name,
                "truck_dimensions": {
                    "length_mm": truck.internal_length_mm,
                    "width_mm": truck.internal_width_mm,
                    "height_mm": truck.internal_height_mm,
                    "volume_mm3": truck.internal_length_mm * truck.internal_width_mm * truck.internal_height_mm
                },
                "units_packed_total": 0,
                "box_counts_by_type": {},
                "unfitted_counts": {},
                "cube_utilisation_pct": 0.0,
                "payload_used_kg": 0.0,
                "payload_used_pct": 0.0,
                "placements_sample": [],
                "total_placements": 0,
                "notes": ["No valid packing produced."]
            })

    results.sort(key=lambda r: (r.get("cube_utilisation_pct", 0.0), r.get("units_packed_total", 0)), reverse=True)
    return results