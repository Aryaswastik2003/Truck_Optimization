"""
Real-World 3D Truck Optimization System - FastAPI Backend
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

# Approximate road distances between major Indian cities in KM
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

# Cost model: Base Rate (INR) + Rate per KM (INR)
COST_MODEL = {
    "22 ft Truck": {"base_rate": 4000, "rate_per_km": 55},
    "32 ft Single Axle": {"base_rate": 5000, "rate_per_km": 65},
    "32 ft Multi Axle": {"base_rate": 6000, "rate_per_km": 75},
}


# ==================== Data Models ====================

class BoxInput(BaseModel):
    box_type: str
    # Dimensions are optional in the schema because 'pp' (custom) boxes will provide them per-request.
    external_length_mm: Optional[float] = Field(
        None, description="Length in mm. Required for 'pp' (custom) boxes."
    )
    external_width_mm: Optional[float] = Field(
        None, description="Width in mm. Required for 'pp' (custom) boxes."
    )
    external_height_mm: Optional[float] = Field(
        None, description="Height in mm. Required for 'pp' (custom) boxes."
    )
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
    """Represents a box with dimensions and weight"""
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
        """Get all valid 90-degree rotations of the box (L, H, W)"""
        # (Length, Height, Width) where Height is along Y-axis
        return list(set([
            (self.length, self.height, self.width),
            (self.length, self.width, self.height),
            (self.width, self.height, self.length),
            (self.width, self.length, self.height),
            (self.height, self.length, self.width),
            (self.height, self.width, self.length),
        ]))

@dataclass
class Placement:
    """Represents a placed box in the truck"""
    box: Box
    x: float
    y: float
    z: float
    length: float
    width: float
    height: float
    rotation_idx: int
    
    @property
    def x_max(self):
        return self.x + self.length
    
    @property
    def y_max(self):
        return self.y + self.height
    
    @property
    def z_max(self):
        return self.z + self.width
    
    def intersects(self, other: 'Placement') -> bool:
        """Check if this placement intersects with another"""
        # Add a tiny tolerance (0.1mm) to prevent false positives from floating point arithmetic
        TOL = 0.1 
        return not (
            self.x_max <= other.x + TOL or other.x_max <= self.x + TOL or
            self.y_max <= other.y + TOL or other.y_max <= self.y + TOL or
            self.z_max <= other.z + TOL or other.z_max <= self.z + TOL
        )

class Space:
    """Represents an empty space in the truck"""
    def __init__(self, x, y, z, length, width, height):
        self.x = x
        self.y = y
        self.z = z
        self.length = length
        self.width = width
        self.height = height
    
    @property
    def volume(self):
        return self.length * self.width * self.height

    @property
    def x_max(self):
        return self.x + self.length

    @property
    def y_max(self):
        return self.y + self.height

    @property
    def z_max(self):
        return self.z + self.width
    
    def can_fit(self, l, w, h):
        """Check if dimensions can fit in this space"""
        return l <= self.length and w <= self.width and h <= self.height
    
    def split(self, placement: Placement) -> List['Space']:
        """Split this space after placing a box (simplified approach)"""
        new_spaces = []
        
        # Space to the right (X-axis split)
        if self.length > placement.length:
            new_spaces.append(Space(
                self.x + placement.length, self.y, self.z,
                self.length - placement.length, self.width, self.height
            ))
        
        # Space in front (Z-axis split)
        if self.width > placement.width:
            new_spaces.append(Space(
                self.x, self.y, self.z + placement.width,
                placement.length, self.width - placement.width, self.height
            ))

        # Space on top (Y-axis split)
        if self.height > placement.height:
             new_spaces.append(Space(
                self.x, self.y + placement.height, self.z,
                placement.length, placement.width, self.height - placement.height
            ))
        
        # Filter out spaces with zero or negative volume (using a small tolerance)
        return [s for s in new_spaces if s.volume > 1]


class TruckPacker:
    """Advanced 3D bin packing algorithm for truck loading with gravity constraint"""
    
    def __init__(self, truck_length, truck_width, truck_height, max_weight):
        self.truck_length = truck_length
        self.truck_width = truck_width
        self.truck_height = truck_height
        self.max_weight = max_weight
        self.placements: List[Placement] = []
        self.spaces: List[Space] = []
        self.total_weight = 0
        
        # Initialize with the entire truck as one space
        self.spaces.append(Space(0, 0, 0, truck_length, truck_width, truck_height))
        # Constant for gravity check, slightly lower threshold might improve packing consistency
        self.SUPPORT_THRESHOLD = 0.8
    
    def pack_boxes(self, boxes: List[Box]) -> Tuple[List[Placement], List[Box]]:
        unpacked = []
        
        # First-Fit Decreasing (FFD) strategy
        sorted_boxes = sorted(boxes, key=lambda b: b.volume, reverse=True)
        
        for box in sorted_boxes:
            if not self._try_pack_box(box):
                unpacked.append(box)
        
        return self.placements, unpacked

    def _is_supported(self, placement: Placement) -> bool:
        """Check if a placement is physically supported by the floor or other boxes."""
        # Rule 1: Box is on the floor (Y-axis = 0)
        if abs(placement.y) < 0.1:
            return True

        # Rule 2: Box is supported by other boxes
        total_support_area = 0.0
        box_base_area = placement.length * placement.width

        # Find all boxes directly underneath the new placement
        for p in self.placements:
            # Check if p is directly below with tolerance (Y_max of p matches Y of new placement)
            if abs(p.y_max - placement.y) < 0.1: 
                # Calculate intersection area in the X-Z plane
                overlap_x_min = max(placement.x, p.x)
                overlap_x_max = min(placement.x_max, p.x_max)
                overlap_z_min = max(placement.z, p.z)
                overlap_z_max = min(placement.z_max, p.z_max)

                overlap_l = max(0.0, overlap_x_max - overlap_x_min)
                overlap_w = max(0.0, overlap_z_max - overlap_z_min)
                
                total_support_area += overlap_l * overlap_w

        # Check if the support area meets the threshold
        if box_base_area <= 0:
            return True # Avoid division by zero for degenerate boxes
            
        return (total_support_area / box_base_area) >= self.SUPPORT_THRESHOLD

    def _try_pack_box(self, box: Box) -> bool:
        """Try to pack a single box using a gravity-aware, first-fit strategy."""
        if self.total_weight + box.weight > self.max_weight:
            return False
        
        # --- IMPROVED: Strategic Space Sorting ---
        # Prioritize: 1. Lowest Y (Stability/Floor), 2. Lowest X (Corner), 3. Largest Volume (to fit better)
        sorted_spaces = sorted(self.spaces, key=lambda s: (s.y, s.x, s.z, -s.volume))

        for space in sorted_spaces:
            for rotation in box.get_rotations():
                # l: length (x), h: height (y), w: width (z)
                l, h, w = rotation
                
                if space.can_fit(l, w, h):
                    # Create a potential placement anchored to the space's corner
                    test_placement = Placement(
                        box, space.x, space.y, space.z,
                        l, w, h, box.get_rotations().index(rotation)
                    )
                    
                    # Check for collisions AND physical support
                    if (not any(test_placement.intersects(p) for p in self.placements) and 
                        self._is_supported(test_placement)):
                        
                        self._place_box(test_placement, space)
                        return True
        return False

    def _place_box(self, placement: Placement, used_space: Space):
        """Place a box, update spaces, and maintain sorted space list."""
        self.placements.append(placement)
        self.total_weight += placement.box.weight
        
        new_spaces = []
        for space in self.spaces:
            if space == used_space:
                new_spaces.extend(space.split(placement))
            else:
                # Basic copy of existing spaces
                new_spaces.append(space) 
        
        # Remove spaces fully contained within the new placement (cleanup)
        self.spaces = [s for s in new_spaces if not (
            s.x >= placement.x - 0.1 and s.x_max <= placement.x_max + 0.1 and
            s.y >= placement.y - 0.1 and s.y_max <= placement.y_max + 0.1 and
            s.z >= placement.z - 0.1 and s.z_max <= placement.z_max + 0.1
        )]
        
        # --- IMPROVED: Less Aggressive Pruning ---
        # Sort by strategic key, and keep more spaces (e.g., 500 instead of 200)
        N_KEEP = 500
        self.spaces.sort(key=lambda s: (s.y, s.x, s.z, -s.volume))
        if len(self.spaces) > N_KEEP:
            # Keep the most promising spaces (lowest Y, lowest X, largest volume)
            self.spaces = self.spaces[:N_KEEP]
    
    def get_utilization(self) -> float:
        """Calculate volume utilization percentage"""
        truck_volume = self.truck_length * self.truck_width * self.truck_height
        used_volume = sum(p.length * p.height * p.width for p in self.placements) # Note: order changed to match L,H,W
        return (used_volume / truck_volume) * 100 if truck_volume > 0 else 0
    
    def verify_packing(self) -> Tuple[bool, List[str]]:
        """Verify the packing is valid"""
        issues = []
        
        # Check weight constraint
        if self.total_weight > self.max_weight + 0.1: # Added tolerance
            issues.append(f"Weight exceeds limit: {self.total_weight:.0f} > {self.max_weight:.0f} kg")
        
        # Check boundaries
        for p in self.placements:
            if p.x_max > self.truck_length + 0.1 or p.y_max > self.truck_height + 0.1 or p.z_max > self.truck_width + 0.1:
                issues.append(f"Box {p.box.type} (ID: {p.box.id}) exceeds truck boundaries.")
        
        # Check for overlaps
        for i, p1 in enumerate(self.placements):
            for p2 in self.placements[i+1:]:
                if p1.intersects(p2):
                    issues.append(f"Overlap detected between boxes {p1.box.id} and {p2.box.id}")

        # Verify support for all boxes
        for p in self.placements:
            if not self._is_supported(p):
                 issues.append(f"Box {p.box.type} (ID: {p.box.id}) at y={p.y} is not supported.")

        return len(issues) == 0, issues

# ==================== API Endpoints (No changes needed here) ====================

@app.post("/api/optimize", response_model=List[TruckResult])
async def optimize_loading(request: OptimizationRequest):
    """
    Optimize box loading across multiple trucks
    """
    try:
        results = []
        
        for truck in request.trucks:
            logger.info(f"Optimizing for truck: {truck.name}")
            
            # --- Cost Calculation (Skipped for brevity, no change) ---
            calculated_cost = None
            if request.source_city and request.destination_city and request.source_city != request.destination_city:
                distance_key = frozenset((request.source_city, request.destination_city))
                distance = CITY_DISTANCES_KM.get(distance_key)
                
                cost_params = COST_MODEL.get(truck.name)
                
                if distance and cost_params:
                    calculated_cost = cost_params["base_rate"] + (distance * cost_params["rate_per_km"])
                    logger.info(f"Cost for {truck.name} from {request.source_city} to {request.destination_city}: INR {calculated_cost:.2f}")

            # Prepare boxes (Skipped for brevity, no change)
            all_boxes = []
            box_id_counter = 0
            
            for box_config in request.boxes:
                # Determine if this is a PP/custom box type (case-insensitive)
                is_pp_custom = isinstance(box_config.box_type, str) and box_config.box_type.strip().lower().startswith("pp")
                
                # Validation logic (Skipped for brevity, no change)
                if is_pp_custom:
                    if (box_config.external_length_mm is None or box_config.external_width_mm is None or box_config.external_height_mm is None):
                        raise HTTPException(
                            status_code=400,
                            detail=f"Box type '{box_config.box_type}' is a PP/custom box — you must provide external_length_mm, external_width_mm, and external_height_mm for custom boxes."
                        )
                    if box_config.external_length_mm <= 0 or box_config.external_width_mm <= 0 or box_config.external_height_mm <= 0:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Custom PP box dimensions must be positive numbers."
                        )
                
                else:
                    if (box_config.external_length_mm is None or box_config.external_width_mm is None or box_config.external_height_mm is None):
                        raise HTTPException(
                            status_code=400,
                            detail=f"Box type '{box_config.box_type}' requires external_length_mm, external_width_mm, and external_height_mm to be set."
                        )
                    if box_config.external_length_mm <= 0 or box_config.external_width_mm <= 0 or box_config.external_height_mm <= 0:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Box dimensions must be positive numbers."
                        )
                
                # Calculate quantity if not specified (Skipped for brevity, no change)
                if box_config.quantity is None:
                    truck_volume = truck.internal_length_mm * truck.internal_width_mm * truck.internal_height_mm
                    box_volume = box_config.external_length_mm * box_config.external_width_mm * box_config.external_height_mm
                    max_by_volume = int(truck_volume / box_volume * 0.85) if box_volume > 0 else 0
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
                        id=box_id_counter
                    ))
                    box_id_counter += 1
            
            # Pack boxes
            packer = TruckPacker(
                truck.internal_length_mm,
                truck.internal_width_mm,
                truck.internal_height_mm,
                truck.payload_kg
            )
            
            # --- Packing happens here using the improved logic ---
            packed_placements, unpacked_boxes = packer.pack_boxes(all_boxes)
            
            # Count boxes by type (Skipped for brevity, no change)
            box_counts = {}
            for p in packed_placements:
                box_counts[p.box.type] = box_counts.get(p.box.type, 0) + 1
            
            unfitted_counts = {}
            for box in unpacked_boxes:
                unfitted_counts[box.type] = unfitted_counts.get(box.type, 0) + 1
            
            # Prepare placements for response (sample for visualization) (Skipped for brevity, no change)
            placements_sample = []
            sample_size = min(len(packed_placements), 1500)
            rotation_names = ["LWH", "LHW", "WLH", "WHL", "HLW", "HWL"] # Matches the Box.get_rotations logic order
            for p in packed_placements[:sample_size]:
                # rotation_names index is now based on the list produced by Box.get_rotations()
                placements_sample.append(BoxPlacement(
                    type=p.box.type,
                    dims_mm=[p.length, p.height, p.width], # Note: height is now p.height (y-axis)
                    pos_mm=[p.x, p.y, p.z],
                    rotation=rotation_names[p.rotation_idx],
                    corners={
                        "min": [p.x, p.y, p.z],
                        "max": [p.x_max, p.y_max, p.z_max]
                    },
                    weight_kg=p.box.weight
                ))
            
            # Verify packing
            is_valid, verification_issues = packer.verify_packing()
            
            # Calculate metrics
            utilization = packer.get_utilization()
            total_weight = sum(p.box.weight for p in packed_placements)
            weight_utilization = (total_weight / truck.payload_kg * 100) if truck.payload_kg > 0 else 0
            
            # Create result (Skipped for brevity, no change)
            result = TruckResult(
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
                verification_details=verification_issues if not is_valid else ["All checks passed"]
            )
            
            results.append(result)
            logger.info(f"Truck {truck.name}: Packed {len(packed_placements)} boxes, {utilization:.1f}% utilization")
        
        return results
    
    except HTTPException:
        # Re-raise HTTPExceptions so FastAPI returns the right status codes
        raise
    except Exception as e:
        logger.error(f"Optimization error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": time.time()}

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "3D Truck Loading Optimization API",
        "version": "1.1.0",
        "features": ["3D Bin Packing", "Cost Estimation", "Gravity-Aware Placement"],
        "endpoints": {
            "optimize": "/api/optimize",
            "health": "/api/health",
            "docs": "/docs"
        }
    }

# ==================== Running Instructions ====================

if __name__ == "__main__":
    import uvicorn
    print("Starting Real-World 3D Truck Optimization Server...")
    print("API will be available at http://localhost:8000")
    print("Documentation at http://localhost:8000/docs")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)