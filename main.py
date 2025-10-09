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
VOLUME_FILL_FACTOR = 1  # Adjust this later to control how tightly the truck is filled

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

class PerformanceGrade(BaseModel):
    grade: str
    color: str
    description: str
    percentile: str

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
    performance_grade: PerformanceGrade
    limiting_factor: str

# ==================== Performance Grading ====================
def calculate_performance_grade(utilization: float) -> PerformanceGrade:
    """Grade the packing performance based on industry standards"""
    if utilization >= 95:
        return PerformanceGrade(
            grade="Excellent",
            color="#28a745",
            description="Top 10% - Outstanding optimization",
            percentile="Top 10%"
        )
    elif utilization >= 90:
        return PerformanceGrade(
            grade="Very Good",
            color="#20c997",
            description="Top 25% - Above industry average",
            percentile="Top 25%"
        )
    elif utilization >= 85:
        return PerformanceGrade(
            grade="Good",
            color="#17a2b8",
            description="Industry standard performance",
            percentile="Average"
        )
    elif utilization >= 75:
        return PerformanceGrade(
            grade="Fair",
            color="#ffc107",
            description="Room for improvement",
            percentile="Below Average"
        )
    else:
        return PerformanceGrade(
            grade="Low",
            color="#dc3545",
            description="Consider alternative loading strategy",
            percentile="Low"
        )

def determine_limiting_factor(volume_util: float, weight_util: float) -> str:
    """Determine what's limiting the packing"""
    diff = abs(volume_util - weight_util)
    
    if diff < 10:
        return "Balanced (Volume and Weight)"
    elif weight_util > volume_util + 10:
        return "Weight Limited (Truck full by weight before volume)"
    else:
        return "Volume Limited (Physical space constraints)"

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

    def can_fit(self, l, w, h, tol=0.5):
        """Tolerance to avoid false negatives on small boxes"""
        return (l <= self.length + tol) and (w <= self.width + tol) and (h <= self.height + tol)

    def split(self, placement: Placement) -> List['Space']:
        """Guillotine-like split, avoids micro gaps"""
        new_spaces = []
        MIN_DIMENSION = 30.0

        # Right space
        remaining_length = self.length - placement.length
        if remaining_length > MIN_DIMENSION:
            new_spaces.append(Space(
                self.x + placement.length, self.y, self.z,
                remaining_length, self.width, self.height
            ))
        
        # Front space
        remaining_width = self.width - placement.width
        if remaining_width > MIN_DIMENSION:
            new_spaces.append(Space(
                self.x, self.y, self.z + placement.width,
                placement.length, remaining_width, self.height
            ))
        
        # Top space
        remaining_height = self.height - placement.height
        if remaining_height > MIN_DIMENSION:
            new_spaces.append(Space(
                self.x, self.y + placement.height, self.z,
                placement.length, placement.width, remaining_height
            ))
        
        MIN_VOLUME = 27000
        return [s for s in new_spaces if s.volume >= MIN_VOLUME]

class TruckPacker:
    def __init__(self, truck_length, truck_width, truck_height, max_weight):
        self.truck_length, self.truck_width, self.truck_height = truck_length, truck_width, truck_height
        self.max_weight = max_weight
        self.placements, self.spaces = [], [Space(0, 0, 0, truck_length, truck_width, truck_height)]
        self.total_weight = 0
        self.SUPPORT_THRESHOLD = 0.5

    def _merge_spaces(self):
        """Merge adjacent spaces along X/Z to reduce fragmentation"""
        if len(self.spaces) < 2:
            return
            
        merged, used = [], [False]*len(self.spaces)
        TOL = 1.0
        
        for i, a in enumerate(self.spaces):
            if used[i]:
                continue
            cur, changed = a, True
            while changed:
                changed = False
                for j, b in enumerate(self.spaces):
                    if i == j or used[j]:
                        continue
                    
                    if (abs(cur.y - b.y) < TOL and abs(cur.z - b.z) < TOL and
                        abs(cur.width - b.width) < TOL and abs(cur.height - b.height) < TOL):
                        if abs(cur.x + cur.length - b.x) < TOL:
                            cur = Space(cur.x, cur.y, cur.z, cur.length + b.length, cur.width, cur.height)
                            used[j], changed = True, True
                        elif abs(b.x + b.length - cur.x) < TOL:
                            cur = Space(b.x, b.y, b.z, cur.length + b.length, cur.width, cur.height)
                            used[j], changed = True, True
                    
                    if (abs(cur.x - b.x) < TOL and abs(cur.y - b.y) < TOL and
                        abs(cur.length - b.length) < TOL and abs(cur.height - b.height) < TOL):
                        if abs(cur.z + cur.width - b.z) < TOL:
                            cur = Space(cur.x, cur.y, cur.z, cur.length, cur.width + b.width, cur.height)
                            used[j], changed = True, True
                        elif abs(b.z + b.width - cur.z) < TOL:
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
        
        if unpacked:
            unpacked = self._fill_remaining(unpacked)
        
        return self.placements, unpacked

    def _is_supported(self, placement: Placement) -> bool:
        """Check if box has adequate support below it"""
        if abs(placement.y) < 0.1:
            return True
        
        total_support_area = 0.0
        box_base_area = placement.length * placement.width
        
        if box_base_area <= 0:
            return True
        
        for p in self.placements:
            if abs(p.y_max - placement.y) < 1.0:
                overlap_x_min = max(placement.x, p.x)
                overlap_x_max = min(placement.x_max, p.x_max)
                overlap_z_min = max(placement.z, p.z)
                overlap_z_max = min(placement.z_max, p.z_max)
                
                overlap_l = max(0.0, overlap_x_max - overlap_x_min)
                overlap_w = max(0.0, overlap_z_max - overlap_z_min)
                total_support_area += overlap_l * overlap_w
        
        support_ratio = total_support_area / box_base_area
        return support_ratio >= self.SUPPORT_THRESHOLD

    def _try_pack_box(self, box: Box) -> bool:
        """Try to pack a single box into available spaces"""
        if self.total_weight + box.weight > self.max_weight:
            return False
        
        sorted_spaces = sorted(self.spaces, key=lambda s: (s.y, s.x, s.z, -s.volume))
        rotations = box.get_rotations()
        
        for space in sorted_spaces:
            for rotation_idx, (l, h, w) in enumerate(rotations):
                if space.can_fit(l, w, h):
                    test_placement = Placement(box, space.x, space.y, space.z, l, w, h, rotation_idx)
                    
                    if not any(test_placement.intersects(p) for p in self.placements):
                        if self._is_supported(test_placement):
                            self._place_box(test_placement, space)
                            return True
        
        return False

    def _place_box(self, placement: Placement, used_space: Space):
        """Place a box and update available spaces"""
        self.placements.append(placement)
        self.total_weight += placement.box.weight
        
        new_spaces = []
        for space in self.spaces:
            if space == used_space:
                new_spaces.extend(space.split(placement))
            else:
                new_spaces.append(space)
        
        TOL = 0.5
        self.spaces = [s for s in new_spaces if not (
            s.x >= placement.x - TOL and s.x_max <= placement.x_max + TOL and
            s.y >= placement.y - TOL and s.y_max <= placement.y_max + TOL and
            s.z >= placement.z - TOL and s.z_max <= placement.z_max + TOL
        )]
        
        self._merge_spaces()
        
        N_KEEP = 20000
        self.spaces.sort(key=lambda s: (s.y, s.x, s.z, -s.volume))
        if len(self.spaces) > N_KEEP:
            self.spaces = self.spaces[:N_KEEP]

    def _fill_remaining(self, unpacked: List[Box]) -> List[Box]:
        """Second pass: fill small gaps with relaxed support requirements"""
        original_threshold = self.SUPPORT_THRESHOLD
        still_unpacked = []
        
        for box in sorted(unpacked, key=lambda b: b.volume):
            if box.volume < 10e6:
                self.SUPPORT_THRESHOLD = 0.0
            else:
                self.SUPPORT_THRESHOLD = 0.2
            
            if not self._try_pack_box(box):
                still_unpacked.append(box)
        
        self.SUPPORT_THRESHOLD = original_threshold
        return still_unpacked

    def get_utilization(self) -> float:
        """Calculate volume utilization percentage"""
        truck_volume = self.truck_length * self.truck_width * self.truck_height
        if truck_volume <= 0:
            return 0
        
        used_volume = sum(p.length * p.height * p.width for p in self.placements)
        return (used_volume / truck_volume) * 100

    def verify_packing(self) -> Tuple[bool, List[str]]:
        """Verify packing integrity"""
        issues = []
        
        if self.total_weight > self.max_weight + 1.0:
            issues.append(f"Weight exceeds limit: {self.total_weight:.0f} > {self.max_weight:.0f} kg")
        
        for p in self.placements:
            if (p.x_max > self.truck_length + 1.0 or 
                p.y_max > self.truck_height + 1.0 or 
                p.z_max > self.truck_width + 1.0):
                issues.append(f"Box {p.box.type} (ID: {p.box.id}) exceeds truck boundaries.")
        
        for i, p1 in enumerate(self.placements):
            for p2 in self.placements[i+1:]:
                if p1.intersects(p2):
                    issues.append(f"Overlap detected between boxes {p1.box.id} and {p2.box.id}")
        
        original_threshold = self.SUPPORT_THRESHOLD
        self.SUPPORT_THRESHOLD = 0.3
        for p in self.placements:
            if not self._is_supported(p):
                issues.append(f"Box {p.box.type} (ID: {p.box.id}) at y={p.y:.0f} has insufficient support.")
        self.SUPPORT_THRESHOLD = original_threshold
        
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
                    logger.info(f"Cost for {truck.name}: INR {calculated_cost:.2f}")

            MAX_TOTAL_BOXES_TO_PROCESS = 10000
            all_boxes = []
            box_id_counter = 0
            
            box_quantities = []
            for box_config in request.boxes:
                if box_config.quantity is None:
                    truck_volume = truck.internal_length_mm * truck.internal_width_mm * truck.internal_height_mm
                    box_volume = box_config.external_length_mm * box_config.external_width_mm * box_config.external_height_mm
                    max_by_volume = int(truck_volume / box_volume * VOLUME_FILL_FACTOR) if box_volume > 0 else 0
                    max_by_weight = int(truck.payload_kg / box_config.max_payload_kg) if box_config.max_payload_kg > 0 else 0
                    quantity_cap = 20000
                    quantity = min(max_by_volume, max_by_weight, quantity_cap)
                else:
                    quantity = box_config.quantity
                
                box_quantities.append((box_config, quantity))

            boxes_to_process_count = 0
            for box_config, quantity in box_quantities:
                num_to_add = min(quantity, MAX_TOTAL_BOXES_TO_PROCESS - boxes_to_process_count)
                
                for _ in range(num_to_add):
                    all_boxes.append(Box(
                        type=box_config.box_type,
                        length=box_config.external_length_mm,
                        width=box_config.external_width_mm,
                        height=box_config.external_height_mm,
                        weight=box_config.max_payload_kg,
                        id=box_id_counter
                    ))
                    box_id_counter += 1
                
                boxes_to_process_count += num_to_add
                if boxes_to_process_count >= MAX_TOTAL_BOXES_TO_PROCESS:
                    logger.warning(f"Hit max processing limit of {MAX_TOTAL_BOXES_TO_PROCESS} boxes")
                    break

            packer = TruckPacker(
                truck.internal_length_mm, 
                truck.internal_width_mm, 
                truck.internal_height_mm, 
                truck.payload_kg
            )
            packed_placements, unpacked_boxes = packer.pack_boxes(all_boxes)
            
            box_counts = {}
            for p in packed_placements:
                box_counts[p.box.type] = box_counts.get(p.box.type, 0) + 1
            
            unfitted_counts = {}
            
            for box in unpacked_boxes:
                unfitted_counts[box.type] = unfitted_counts.get(box.type, 0) + 1
            
            for box_config, total_quantity in box_quantities:
                packed_count = box_counts.get(box_config.box_type, 0)
                unfitted_from_packer = unfitted_counts.get(box_config.box_type, 0)
                processed_count = packed_count + unfitted_from_packer
                
                if processed_count < total_quantity:
                    unprocessed_count = total_quantity - processed_count
                    unfitted_counts[box_config.box_type] = unfitted_counts.get(box_config.box_type, 0) + unprocessed_count

            placements_sample = []
            rotation_names = ["LHW", "LWH", "WHL", "WLH", "HLW", "HWL"]
            for p in packed_placements[:min(len(packed_placements), 10000)]:
                placements_sample.append(BoxPlacement(
                    type=p.box.type,
                    dims_mm=[p.length, p.height, p.width],
                    pos_mm=[p.x, p.y, p.z],
                    rotation=rotation_names[p.rotation_idx] if p.rotation_idx < len(rotation_names) else "LWH",
                    corners={
                        "min": [p.x, p.y, p.z],
                        "max": [p.x_max, p.y_max, p.z_max]
                    },
                    weight_kg=p.box.weight
                ))

            is_valid, verification_issues = packer.verify_packing()
            
            utilization = packer.get_utilization()
            total_weight = sum(p.box.weight for p in packed_placements)
            weight_utilization = (total_weight / truck.payload_kg * 100) if truck.payload_kg > 0 else 0

            # NEW: Calculate performance grade and limiting factor
            performance_grade = calculate_performance_grade(utilization)
            limiting_factor = determine_limiting_factor(utilization, weight_utilization)

            results.append(TruckResult(
                truck_name=truck.name,
                truck_dimensions=TruckDimensions(
                    length_mm=truck.internal_length_mm,
                    width_mm=truck.internal_width_mm,
                    height_mm=truck.internal_height_mm,
                    volume_mm3=truck.internal_length_mm * truck.internal_width_mm * truck.internal_height_mm,
                    payload_kg=truck.payload_kg
                ),
                units_packed_total=len(packed_placements),
                cube_utilisation_pct=round(utilization, 2),
                payload_used_kg=round(total_weight, 2),
                payload_used_pct=round(weight_utilization, 2),
                estimated_cost=calculated_cost,
                box_counts_by_type=box_counts,
                unfitted_counts=unfitted_counts,
                placements_sample=placements_sample,
                verification_passed=is_valid,
                verification_details=verification_issues if not is_valid else ["All checks passed"],
                performance_grade=performance_grade,
                limiting_factor=limiting_factor
            ))

            logger.info(f"Truck {truck.name}: Packed {len(packed_placements)} boxes, {utilization:.1f}% utilization, Grade: {performance_grade.grade}")

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
        "version": "1.3.0",
        "features": ["3D Bin Packing", "Cost Estimation", "Gravity-Aware Placement", "Enhanced Small Box Support", "Performance Grading"],
        "endpoints": {"optimize": "/api/optimize", "health": "/api/health", "docs": "/docs"}
    }

if __name__ == "__main__":
    import uvicorn
    print("Starting Real-World 3D Truck Optimization Server...")
    print("API will be available at http://localhost:8000")
    print("Documentation at http://localhost:8000/docs")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)