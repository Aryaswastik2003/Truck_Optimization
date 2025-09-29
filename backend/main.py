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

# ==================== Data Models ====================

class BoxInput(BaseModel):
    box_type: str
    external_length_mm: float
    external_width_mm: float
    external_height_mm: float
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
        """Get all valid 90-degree rotations of the box"""
        return [
            (self.length, self.width, self.height),  # Original
            (self.length, self.height, self.width),  # Rotate around X
            (self.width, self.length, self.height),  # Rotate around Z
            (self.width, self.height, self.length),  # Rotate around X then Z
            (self.height, self.length, self.width),  # Rotate around Y
            (self.height, self.width, self.length),  # Rotate around Y then Z
        ]

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
        return not (
            self.x_max <= other.x or other.x_max <= self.x or
            self.y_max <= other.y or other.y_max <= self.y or
            self.z_max <= other.z or other.z_max <= self.z
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
    
    def can_fit(self, l, w, h):
        """Check if dimensions can fit in this space"""
        return l <= self.length and w <= self.width and h <= self.height
    
    def split(self, placement: Placement) -> List['Space']:
        """Split this space after placing a box"""
        new_spaces = []
        
        # New space to the right of the placed box
        if self.x + self.length > placement.x_max:
            new_spaces.append(Space(
                placement.x_max, self.y, self.z,
                self.x + self.length - placement.x_max,
                self.width, self.height
            ))
            
        # New space in front of the placed box
        if self.z + self.width > placement.z_max:
            new_spaces.append(Space(
                self.x, self.y, placement.z_max,
                placement.length, self.width - placement.width,
                self.height
            ))

        # New space on top of the placed box
        if self.y + self.height > placement.y_max:
             new_spaces.append(Space(
                self.x, placement.y_max, self.z,
                placement.length, self.width, self.height - placement.height
            ))
        
        return [s for s in new_spaces if s.volume > 1000]  # Filter tiny spaces

class TruckPacker:
    """Advanced 3D bin packing algorithm for truck loading"""
    
    def __init__(self, truck_length, truck_width, truck_height, max_weight):
        self.truck_length = truck_length
        self.truck_width = truck_width
        self.truck_height = truck_height
        self.max_weight = max_weight
        self.placements: List[Placement] = []
        self.spaces: List[Space] = []
        self.total_weight = 0
        
        # Initialize with the entire truck as one space
        self.spaces.append(Space(0, 0, 0, truck_length, truck_height, truck_width))
    
    def pack_boxes(self, boxes: List[Box]) -> Tuple[List[Placement], List[Box]]:
        """Pack boxes into the truck using best-fit strategy"""
        unpacked = []
        
        # Sort boxes by volume (largest first) for better packing
        sorted_boxes = sorted(boxes, key=lambda b: b.volume, reverse=True)
        
        for box in sorted_boxes:
            if not self._try_pack_box(box):
                unpacked.append(box)
        
        return self.placements, unpacked
    
    def _try_pack_box(self, box: Box) -> bool:
        """Try to pack a single box"""
        if self.total_weight + box.weight > self.max_weight:
            return False
        
        best_placement = None
        best_space_idx = -1
        best_waste = float('inf')
        
        # Try all rotations and spaces
        for rotation in box.get_rotations():
            l, h, w = rotation
            
            for i, space in enumerate(self.spaces):
                if space.can_fit(l, w, h):
                    # Calculate wasted space (smaller is better)
                    waste = space.volume - (l * w * h)
                    
                    # Prefer lower positions (stability)
                    stability_score = space.y * 0.001
                    total_score = waste + stability_score
                    
                    if total_score < best_waste:
                        # Check for collisions
                        test_placement = Placement(
                            box, space.x, space.y, space.z,
                            l, w, h, box.get_rotations().index(rotation)
                        )
                        
                        if not any(test_placement.intersects(p) for p in self.placements):
                            best_placement = test_placement
                            best_space_idx = i
                            best_waste = total_score
        
        if best_placement:
            self._place_box(best_placement, best_space_idx)
            return True
        
        return False
    
    def _place_box(self, placement: Placement, space_idx: int):
        """Place a box and update spaces"""
        self.placements.append(placement)
        self.total_weight += placement.box.weight
        
        # Split the used space
        space = self.spaces[space_idx]
        new_spaces = space.split(placement)
        
        # Remove the used space and add new ones
        del self.spaces[space_idx]
        self.spaces.extend(new_spaces)
        
        # Merge adjacent spaces if possible
        self._merge_spaces()
        
        # Sort spaces by position (back to front, bottom to top)
        self.spaces.sort(key=lambda s: (s.y, s.x, s.z))
    
    def _merge_spaces(self):
        """Merge adjacent spaces to create larger packing areas"""
        merged = True
        while merged:
            merged = False
            for i in range(len(self.spaces)):
                for j in range(i + 1, len(self.spaces)):
                    if self._can_merge(self.spaces[i], self.spaces[j]):
                        # Merge spaces[j] into spaces[i]
                        self.spaces[i] = self._merge_two_spaces(self.spaces[i], self.spaces[j])
                        del self.spaces[j]
                        merged = True
                        break
                if merged:
                    break
    
    def _can_merge(self, s1: Space, s2: Space) -> bool:
        """Check if two spaces can be merged"""
        # Check if spaces are adjacent and aligned
        # X-axis adjacency
        if (s1.x + s1.length == s2.x and s1.y == s2.y and s1.z == s2.z and
            s1.height == s2.height and s1.width == s2.width):
            return True
        # Y-axis adjacency
        if (s1.y + s1.height == s2.y and s1.x == s2.x and s1.z == s2.z and
            s1.length == s2.length and s1.width == s2.width):
            return True
        # Z-axis adjacency
        if (s1.z + s1.width == s2.z and s1.x == s2.x and s1.y == s2.y and
            s1.length == s2.length and s1.height == s2.height):
            return True
        return False
    
    def _merge_two_spaces(self, s1: Space, s2: Space) -> Space:
        """Merge two adjacent spaces"""
        x = min(s1.x, s2.x)
        y = min(s1.y, s2.y)
        z = min(s1.z, s2.z)
        length = max(s1.x + s1.length, s2.x + s2.length) - x
        height = max(s1.y + s1.height, s2.y + s2.height) - y
        width = max(s1.z + s1.width, s2.z + s2.width) - z
        return Space(x, y, z, length, width, height)
    
    def get_utilization(self) -> float:
        """Calculate volume utilization percentage"""
        truck_volume = self.truck_length * self.truck_width * self.truck_height
        used_volume = sum(p.length * p.width * p.height for p in self.placements)
        return (used_volume / truck_volume) * 100 if truck_volume > 0 else 0
    
    def verify_packing(self) -> Tuple[bool, List[str]]:
        """Verify the packing is valid"""
        issues = []
        
        # Check weight constraint
        if self.total_weight > self.max_weight:
            issues.append(f"Weight exceeds limit: {self.total_weight:.0f} > {self.max_weight:.0f} kg")
        
        # Check boundaries
        for p in self.placements:
            if p.x_max > self.truck_length + 0.01:
                issues.append(f"Box {p.box.type} exceeds length boundary")
            if p.y_max > self.truck_height + 0.01:
                issues.append(f"Box {p.box.type} exceeds height boundary")
            if p.z_max > self.truck_width + 0.01:
                issues.append(f"Box {p.box.type} exceeds width boundary")
        
        # Check for overlaps
        for i, p1 in enumerate(self.placements):
            for p2 in self.placements[i+1:]:
                if p1.intersects(p2):
                    issues.append(f"Overlap detected between boxes {p1.box.type} and {p2.box.type}")
        
        return len(issues) == 0, issues

# ==================== API Endpoints ====================

@app.post("/api/optimize", response_model=List[TruckResult])
async def optimize_loading(request: OptimizationRequest):
    """
    Optimize box loading across multiple trucks
    """
    try:
        results = []
        
        for truck in request.trucks:
            logger.info(f"Optimizing for truck: {truck.name}")
            
            # Prepare boxes
            all_boxes = []
            box_id_counter = 0
            
            for box_config in request.boxes:
                # Calculate quantity if not specified
                if box_config.quantity is None:
                    # Estimate max quantity based on volume
                    truck_volume = truck.internal_length_mm * truck.internal_width_mm * truck.internal_height_mm
                    box_volume = box_config.external_length_mm * box_config.external_width_mm * box_config.external_height_mm
                    max_by_volume = int(truck_volume / box_volume * 0.85)  # 85% theoretical max
                    max_by_weight = int(truck.payload_kg / box_config.max_payload_kg)
                    quantity = min(max_by_volume, max_by_weight, 1000)  # Cap at 1000
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
            
            packed_placements, unpacked_boxes = packer.pack_boxes(all_boxes)
            
            # Count boxes by type
            box_counts = {}
            for p in packed_placements:
                box_counts[p.box.type] = box_counts.get(p.box.type, 0) + 1
            
            unfitted_counts = {}
            for box in unpacked_boxes:
                unfitted_counts[box.type] = unfitted_counts.get(box.type, 0) + 1
            
            # Prepare placements for response (sample for visualization)
            placements_sample = []
            sample_size = min(len(packed_placements), 1500)  # Limit for performance
            for p in packed_placements[:sample_size]:
                rotation_names = ["LWH", "LHW", "WLH", "WHL", "HLW", "HWL"]
                placements_sample.append(BoxPlacement(
                    type=p.box.type,
                    dims_mm=[p.length, p.height, p.width],
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
            
            # Create result
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
                box_counts_by_type=box_counts,
                unfitted_counts=unfitted_counts,
                placements_sample=placements_sample,
                verification_passed=is_valid,
                verification_details=verification_issues if not is_valid else ["All checks passed"]
            )
            
            results.append(result)
            logger.info(f"Truck {truck.name}: Packed {len(packed_placements)} boxes, {utilization:.1f}% utilization")
        
        return results
    
    except Exception as e:
        logger.error(f"Optimization error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": time.time()}

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "3D Truck Loading Optimization API",
        "version": "1.0.0",
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
    uvicorn.run(app, host="0.0.0.0", port=8000)