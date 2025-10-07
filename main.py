""" Real-World 3D Truck Optimization System - FastAPI Backend
Practical bin packing with accurate results and verification
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Tuple, Any
from dataclasses import dataclass
import numpy as np
from enum import Enum
import itertools
import time
import logging

# ==================== Tunable Fill Factor ====================
VOLUME_FILL_FACTOR = 0.90  # Adjust this later to control how tightly the truck is filled

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="3D Truck Loading Optimization API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== Shipment & Cost Data ====================
CITY_DISTANCES_KM = {
    frozenset(("Mumbai", "Delhi")): 1420,
    frozenset(("Mumbai", "Bangalore")): 985,
    frozenset(("Mumbai", "Kolkata")): 1940,
    frozenset(("Mumbai", "Chennai")): 1335,
    frozenset(("Mumbai", "Hyderabad")): 710,
    frozenset(("Delhi", "Bangalore")): 2150,
    frozenset(("Delhi", "Kolkata")): 1530,
    frozenset(("Delhi", "Chennai")): 2210,
    frozenset(("Delhi", "Hyderabad")): 1580,
    frozenset(("Bangalore", "Kolkata")): 1870,
    frozenset(("Bangalore", "Chennai")): 350,
    frozenset(("Bangalore", "Hyderabad")): 570,
    frozenset(("Kolkata", "Chennai")): 1670,
    frozenset(("Kolkata", "Hyderabad")): 1480,
    frozenset(("Chennai", "Hyderabad")): 630,
}

COST_MODEL = {
    "22 ft Truck": {"base_rate": 4000, "rate_per_km": 55},
    "32 ft Single Axle": {"base_rate": 5000, "rate_per_km": 65},
    "32 ft Multi Axle": {"base_rate": 6000, "rate_per_km": 75},
}

# ==================== Data Models ====================
class BoxInput(BaseModel):
    box_type: str
    external_length_mm: Optional[float] = Field(None)
    external_width_mm: Optional[float] = Field(None)
    external_height_mm: Optional[float] = Field(None)
    max_payload_kg: float
    quantity: Optional[int] = None

class TruckInput(BaseModel):
    name: str
    internal_length_mm: float
    internal_width_mm: float
    internal_height_mm: float
    payload_kg: float

class OptimizationRequest(BaseModel):
    boxes: List[BoxInput]
    trucks: List[TruckInput]
    source_city: Optional[str] = None
    destination_city: Optional[str] = None

class BoxPlacement(BaseModel):
    type: str
    dims_mm: List[float]
    pos_mm: List[float]
    rotation: str
    corners: Dict[str, List[float]]
    weight_kg: float

class TruckDimensions(BaseModel):
    length_mm: float
    width_mm: float
    height_mm: float
    volume_mm3: float
    payload_kg: float

class TruckResult(BaseModel):
    truck_name: str
    truck_dimensions: TruckDimensions
    units_packed_total: int
    cube_utilisation_pct: float
    payload_used_kg: float
    payload_used_pct: float
    estimated_cost: Optional[float] = None
    box_counts_by_type: Dict[str, int]
    unfitted_counts: Dict[str, int]
    placements_sample: List[BoxPlacement]
    verification_passed: bool
    verification_details: List[str]

# ==================== Core Optimization Engine ====================
@dataclass
class Box:
    type: str
    length: float
    width: float
    height: float
    weight: float
    id: int

    @property
    def volume(self):
        return self.length * self.width * self.height

    def get_rotations(self):
        """Deterministic rotations (L,H,W permutations)"""
        rotations = [
            (self.length, self.height, self.width),
            (self.length, self.width, self.height),
            (self.width, self.height, self.length),
            (self.width, self.length, self.height),
            (self.height, self.length, self.width),
            (self.height, self.width, self.length),
        ]
        seen, unique = set(), []
        for r in rotations:
            if r not in seen:
                seen.add(r)
                unique.append(r)
        return unique

@dataclass
class Placement:
    box: Box
    x: float
    y: float
    z: float
    length: float
    width: float
    height: float
    rotation_idx: int

    @property
    def x_max(self): return self.x + self.length
    @property
    def y_max(self): return self.y + self.height
    @property
    def z_max(self): return self.z + self.width

    def intersects(self, other: 'Placement') -> bool:
        TOL = 0.1
        return not (
            self.x_max <= other.x + TOL or other.x_max <= self.x + TOL or
            self.y_max <= other.y + TOL or other.y_max <= self.y + TOL or
            self.z_max <= other.z + TOL or other.z_max <= self.z + TOL
        )

class Space:
    def __init__(self, x, y, z, length, width, height):
        self.x, self.y, self.z = x, y, z
        self.length, self.width, self.height = length, width, height

    @property
    def volume(self): return self.length * self.width * self.height
    @property
    def x_max(self): return self.x + self.length
    @property
    def y_max(self): return self.y + self.height
    @property
    def z_max(self): return self.z + self.width

    def can_fit(self, l, w, h, tol=0.01):  # Increased tolerance for small boxes
        """Small tolerance to avoid false negatives"""
        return (l <= self.length + tol) and (w <= self.width + tol) and (h <= self.height + tol)

    def split(self, placement: Placement) -> List['Space']:
        """Guillotine-like split, avoids micro gaps"""
        new_spaces = []

        # Right
        if self.length - placement.length > 0.0001:
            new_spaces.append(Space(self.x + placement.length, self.y, self.z,
                                    self.length - placement.length, self.width, self.height))
        # Front
        if self.width - placement.width > 0.0001:
            new_spaces.append(Space(self.x, self.y, self.z + placement.width,
                                    placement.length, self.width - placement.width, self.height))
        # Top
        if self.height - placement.height > 0.0001:
            new_spaces.append(Space(self.x, self.y + placement.height, self.z,
                                    placement.length, placement.width, self.height - placement.height))
        return [s for s in new_spaces if s.volume > 1]

class TruckPacker:
    def __init__(self, truck_length, truck_width, truck_height, max_weight):
        self.truck_length, self.truck_width, self.truck_height = truck_length, truck_width, truck_height
        self.max_weight = max_weight
        self.placements, self.spaces = [], [Space(0, 0, 0, truck_length, truck_width, truck_height)]
        self.total_weight = 0
        self.SUPPORT_THRESHOLD = 0.8

    def _merge_spaces(self):
        """Merge adjacent spaces along X/Z to reduce fragmentation"""
        merged, used = [], [False]*len(self.spaces)
        for i, a in enumerate(self.spaces):
            if used[i]:
                continue
            cur, changed = a, True
            while changed:
                changed = False
                for j, b in enumerate(self.spaces):
                    if i == j or used[j]:
                        continue
                    if (abs(cur.y - b.y) < 0.001 and abs(cur.z - b.z) < 0.001 and
                        abs(cur.width - b.width) < 0.001 and abs(cur.height - b.height) < 0.001):
                        if abs(cur.x + cur.length - b.x) < 0.001:
                            cur = Space(cur.x, cur.y, cur.z, cur.length + b.length, cur.width, cur.height)
                            used[j], changed = True, True
                        elif abs(b.x + b.length - cur.x) < 0.001:
                            cur = Space(b.x, b.y, b.z, cur.length + b.length, cur.width, cur.height)
                            used[j], changed = True, True
                    if (abs(cur.x - b.x) < 0.001 and abs(cur.y - b.y) < 0.001 and
                        abs(cur.length - b.length) < 0.001 and abs(cur.height - b.height) < 0.001):
                        if abs(cur.z + cur.width - b.z) < 0.001:
                            cur = Space(cur.x, cur.y, cur.z, cur.length, cur.width + b.width, cur.height)
                            used[j], changed = True, True
                        elif abs(b.z + b.width - cur.z) < 0.001:
                            cur = Space(cur.x, cur.y, b.z, cur.length, cur.width + b.width, cur.height)
                            used[j], changed = True, True
            merged.append(cur)
            used[i] = True
        self.spaces = merged

    def pack_boxes(self, boxes: List[Box]) -> Tuple[List[Placement], List[Box]]:
        unpacked = []
        sorted_boxes = sorted(boxes, key=lambda b: b.volume, reverse=True)
        for box in sorted_boxes:
            if not self._try_pack_box(box):
                unpacked.append(box)
        # second pass fill for small boxes
        if unpacked:
            unpacked = self.fill_remaining(unpacked)
        return self.placements, unpacked

    def _is_supported(self, placement: Placement) -> bool:
        if abs(placement.y) < 0.1:
            return True
        total_support_area, box_base_area = 0.0, placement.length * placement.width
        for p in self.placements:
            if abs(p.y_max - placement.y) < 0.1:
                overlap_x_min, overlap_x_max = max(placement.x, p.x), min(placement.x_max, p.x_max)
                overlap_z_min, overlap_z_max = max(placement.z, p.z), min(placement.z_max, p.z_max)
                overlap_l = max(0.0, overlap_x_max - overlap_x_min)
                overlap_w = max(0.0, overlap_z_max - overlap_z_min)
                total_support_area += overlap_l * overlap_w
        if box_base_area <= 0:
            return True
        return (total_support_area / box_base_area) >= self.SUPPORT_THRESHOLD

    def _try_pack_box(self, box: Box) -> bool:
        if self.total_weight + box.weight > self.max_weight:
            return False
        sorted_spaces = sorted(self.spaces, key=lambda s: (s.y, s.x, s.z, -s.volume))
        rotations = box.get_rotations()
        for space in sorted_spaces:
            for rotation_idx, (l, h, w) in enumerate(rotations):
                if space.can_fit(l, w, h):
                    test_placement = Placement(box, space.x, space.y, space.z, l, w, h, rotation_idx)
                    if not any(test_placement.intersects(p) for p in self.placements) and self._is_supported(test_placement):
                        self._place_box(test_placement, space)
                        return True
        return False

    def _place_box(self, placement: Placement, used_space: Space):
        self.placements.append(placement)
        self.total_weight += placement.box.weight
        new_spaces = []
        for space in self.spaces:
            new_spaces.extend(space.split(placement) if space == used_space else [space])
        self.spaces = [s for s in new_spaces if not (
            s.x >= placement.x - 0.1 and s.x_max <= placement.x_max + 0.1 and
            s.y >= placement.y - 0.1 and s.y_max <= placement.y_max + 0.1 and
            s.z >= placement.z - 0.1 and s.z_max <= placement.z_max + 0.1)]
        self._merge_spaces()
        N_KEEP = 1000  # increase free space tracking for small boxes
        self.spaces.sort(key=lambda s: (s.y, s.x, s.z, -s.volume))
        if len(self.spaces) > N_KEEP:
            self.spaces = self.spaces[:N_KEEP]

    def fill_remaining(self, unpacked: List[Box]):
        """Second pass: fill small gaps with relaxed support"""
        original_threshold = self.SUPPORT_THRESHOLD
        for box in sorted(unpacked, key=lambda b: b.volume):
            if box.volume < 1e6:  # very small boxes (< 1,000,000 mm³)
                self.SUPPORT_THRESHOLD = 0.0  # ignore support
            else:
                self.SUPPORT_THRESHOLD = 0.2
            if not self._try_pack_box(box):
                continue
        self.SUPPORT_THRESHOLD = original_threshold
        remaining = [b for b in unpacked if b.id not in {p.box.id for p in self.placements}]
        return remaining

    def get_utilization(self) -> float:
        truck_volume = self.truck_length * self.truck_width * self.truck_height
        used_volume = sum(p.length * p.height * p.width for p in self.placements)
        return (used_volume / truck_volume) * 100 if truck_volume > 0 else 0

    


    def verify_packing(self) -> Tuple[bool, List[str]]:
        issues = []
        if self.total_weight > self.max_weight + 0.1:
            issues.append(f"Weight exceeds limit: {self.total_weight:.0f} > {self.max_weight:.0f} kg")
        for p in self.placements:
            if p.x_max > self.truck_length + 0.1 or p.y_max > self.truck_height + 0.1 or p.z_max > self.truck_width + 0.1:
                issues.append(f"Box {p.box.type} (ID: {p.box.id}) exceeds truck boundaries.")
        for i, p1 in enumerate(self.placements):
            for p2 in self.placements[i+1:]:
                if p1.intersects(p2):
                    issues.append(f"Overlap detected between boxes {p1.box.id} and {p2.box.id}")
        for p in self.placements:
            if not self._is_supported(p):
                issues.append(f"Box {p.box.type} (ID: {p.box.id}) at y={p.y} is not supported.")
        return len(issues) == 0, issues

# ==================== API Endpoints ====================
@app.post("/api/optimize", response_model=List[TruckResult])
async def optimize_loading(request: OptimizationRequest):
    try:
        results = []
        for truck in request.trucks:
            logger.info(f"Optimizing for truck: {truck.name}")
            calculated_cost = None
            if request.source_city and request.destination_city and request.source_city != request.destination_city:
                distance_key = frozenset((request.source_city, request.destination_city))
                distance = CITY_DISTANCES_KM.get(distance_key)
                cost_params = COST_MODEL.get(truck.name)
                if distance and cost_params:
                    calculated_cost = cost_params["base_rate"] + (distance * cost_params["rate_per_km"])
                    logger.info(f"Cost for {truck.name} from {request.source_city} to {request.destination_city}: INR {calculated_cost:.2f}")

            all_boxes, box_id_counter = [], 0
            for box_config in request.boxes:
                if box_config.quantity is None:
                    truck_volume = truck.internal_length_mm * truck.internal_width_mm * truck.internal_height_mm
                    box_volume = box_config.external_length_mm * box_config.external_width_mm * box_config.external_height_mm
                    max_by_volume = int(truck_volume / box_volume * VOLUME_FILL_FACTOR) if box_volume > 0 else 0
                    max_by_weight = int(truck.payload_kg / box_config.max_payload_kg) if box_config.max_payload_kg > 0 else 0
                    quantity = min(max_by_volume, max_by_weight, 1000)
                else:
                    quantity = box_config.quantity
                for _ in range(quantity):
                    all_boxes.append(Box(
                        type=box_config.box_type,
                        length=box_config.external_length_mm,
                        width=box_config.external_width_mm,
                        height=box_config.external_height_mm,
                        weight=box_config.max_payload_kg,
                        id=box_id_counter))
                    box_id_counter += 1

            packer = TruckPacker(truck.internal_length_mm, truck.internal_width_mm, truck.internal_height_mm, truck.payload_kg)
            packed_placements, unpacked_boxes = packer.pack_boxes(all_boxes)

            box_counts, unfitted_counts = {}, {}
            for p in packed_placements:
                box_counts[p.box.type] = box_counts.get(p.box.type, 0) + 1
            for box in unpacked_boxes:
                unfitted_counts[box.type] = unfitted_counts.get(box.type, 0) + 1

            placements_sample, rotation_names = [], ["LWH", "LHW", "WLH", "WHL", "HLW", "HWL"]
            for p in packed_placements[:min(len(packed_placements), 1500)]:
                placements_sample.append(BoxPlacement(
                    type=p.box.type,
                    dims_mm=[p.length, p.height, p.width],
                    pos_mm=[p.x, p.y, p.z],
                    rotation=rotation_names[p.rotation_idx],
                    corners={"min": [p.x, p.y, p.z], "max": [p.x_max, p.y_max, p.z_max]},
                    weight_kg=p.box.weight))

            is_valid, verification_issues = packer.verify_packing()
            utilization = packer.get_utilization()
            total_weight = sum(p.box.weight for p in packed_placements)
            weight_utilization = (total_weight / truck.payload_kg * 100) if truck.payload_kg > 0 else 0

            results.append(TruckResult(
                truck_name=truck.name,
                truck_dimensions=TruckDimensions(
                    length_mm=truck.internal_length_mm,
                    width_mm=truck.internal_width_mm,
                    height_mm=truck.internal_height_mm,
                    volume_mm3=truck.internal_length_mm * truck.internal_width_mm * truck.internal_height_mm,
                    payload_kg=truck.payload_kg),
                units_packed_total=len(packed_placements),
                cube_utilisation_pct=round(utilization, 2),
                payload_used_kg=round(total_weight, 2),
                payload_used_pct=round(weight_utilization, 2),
                estimated_cost=calculated_cost,
                box_counts_by_type=box_counts,
                unfitted_counts=unfitted_counts,
                placements_sample=placements_sample,
                verification_passed=is_valid,
                verification_details=verification_issues if not is_valid else ["All checks passed"]
            ))

            logger.info(f"Truck {truck.name}: Packed {len(packed_placements)} boxes, {utilization:.1f}% utilization")

        return results
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Optimization error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "timestamp": time.time()}

@app.get("/")
async def root():
    return {
        "name": "3D Truck Loading Optimization API",
        "version": "1.1.0",
        "features": ["3D Bin Packing", "Cost Estimation", "Gravity-Aware Placement"],
        "endpoints": {"optimize": "/api/optimize", "health": "/api/health", "docs": "/docs"}
    }

if __name__ == "__main__":
    import uvicorn
    print("Starting Real-World 3D Truck Optimization Server...")
    print("API will be available at http://localhost:8000")
    print("Documentation at http://localhost:8000/docs")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
