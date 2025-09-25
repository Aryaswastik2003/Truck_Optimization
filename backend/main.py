from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Tuple, Dict, Any, Literal
from fastapi.middleware.cors import CORSMiddleware
import logging
import math
from dataclasses import dataclass
from copy import deepcopy
import random
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("real_world_truck_packer")

app = FastAPI(title="Real-World 3D Truck Packing API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Enhanced Data Models
# ----------------------------

class FragilityLevel(str, Enum):
    VERY_FRAGILE = "very_fragile"  # Glass, electronics
    FRAGILE = "fragile"           # Appliances, ceramics  
    NORMAL = "normal"             # Standard goods
    ROBUST = "robust"             # Heavy machinery, steel

class OrientationConstraint(str, Enum):
    ANY = "any"                   # Can be rotated any way
    UPRIGHT_ONLY = "upright_only" # "This side up"
    NO_INVERT = "no_invert"       # Can lay flat or upright, not inverted

class LoadingPriority(str, Enum):
    FIRST_OUT = "first_out"       # Must be accessible (first delivery)
    NORMAL = "normal"             # Standard loading
    LAST_OUT = "last_out"         # Can be buried (last delivery)

@dataclass
class LoadPoint:
    x: float
    y: float
    z: float
    weight: float

@dataclass
class PlacedItem:
    name: str
    position: Tuple[float, float, float]  # Bottom-left corner
    dimensions: Tuple[float, float, float]  # L, W, H
    weight: float
    rotation: int  # 0-5 for different orientations
    support_area: float  # Area actually supported by items below
    fragility: FragilityLevel
    orientation: OrientationConstraint
    priority: LoadingPriority

class EnhancedBoxInput(BaseModel):
    box_type: str
    length_mm: float = Field(..., gt=0)
    width_mm: float = Field(..., gt=0) 
    height_mm: float = Field(..., gt=0)
    weight_kg: float = Field(..., gt=0)
    quantity: int = Field(default=1, ge=1)
    
    # Real-world constraints
    fragility: FragilityLevel = FragilityLevel.NORMAL
    orientation_constraint: OrientationConstraint = OrientationConstraint.ANY
    max_stack_weight_kg: Optional[float] = None  # Max weight this can support on top
    loading_priority: LoadingPriority = LoadingPriority.NORMAL
    
    # Stacking and handling
    stackable: bool = True
    requires_bottom_support: bool = False  # Needs full bottom support
    min_support_percentage: float = Field(default=80.0, ge=50.0, le=100.0)
    
    @validator('min_support_percentage')
    def validate_support(cls, v):
        return max(50.0, min(100.0, v))

class RealWorldTruckSpec(BaseModel):
    name: str
    
    # Internal dimensions (mm)
    internal_length_mm: float
    internal_width_mm: float  
    internal_height_mm: float
    
    # Weight limits (kg)
    max_payload_kg: float
    front_axle_limit_kg: float
    rear_axle_limit_kg: float
    
    # Physical constraints
    wheelbase_mm: float  # Distance between front and rear axles
    loading_height_mm: float = Field(default=1200)  # Height from ground
    door_width_mm: Optional[float] = None
    door_height_mm: Optional[float] = None
    
    # Safety factors
    max_center_of_gravity_height_mm: float  # Max CG height for stability
    weight_distribution_tolerance: float = Field(default=0.10)  # ±10% from ideal
    
    # Operational constraints  
    tie_down_points: List[Tuple[float, float, float]] = []  # Anchor points for straps
    intrusions: List[Dict[str, float]] = []  # Wheel wells, equipment boxes

class OptimizeRequest(BaseModel):
    boxes: List[EnhancedBoxInput]
    truck: RealWorldTruckSpec
    route_stops: Optional[List[str]] = None  # For delivery sequence optimization

# ----------------------------
# Real-World Physics Engine
# ----------------------------

class RealWorldPhysicsEngine:
    def __init__(self, truck: RealWorldTruckSpec):
        self.truck = truck
        self.safety_margin = 0.95  # Use 95% of limits for safety
        
    def calculate_center_of_gravity(self, placed_items: List[PlacedItem]) -> Tuple[float, float, float]:
        """Calculate actual center of gravity of the load"""
        if not placed_items:
            return (0, 0, 0)
            
        total_weight = 0
        weighted_x = weighted_y = weighted_z = 0
        
        for item in placed_items:
            # CG is at center of each box
            item_cg_x = item.position[0] + item.dimensions[0] / 2
            item_cg_y = item.position[1] + item.dimensions[1] / 2  
            item_cg_z = item.position[2] + item.dimensions[2] / 2
            
            weighted_x += item_cg_x * item.weight
            weighted_y += item_cg_y * item.weight
            weighted_z += item_cg_z * item.weight
            total_weight += item.weight
            
        if total_weight == 0:
            return (0, 0, 0)
            
        return (weighted_x / total_weight, weighted_y / total_weight, weighted_z / total_weight)
    
    def calculate_axle_weights(self, placed_items: List[PlacedItem]) -> Tuple[float, float]:
        """Calculate weight on front and rear axles"""
        if not placed_items:
            return (0, 0)
            
        # Assume front axle is at position 0, rear axle at wheelbase distance
        front_axle_pos = 0
        rear_axle_pos = self.truck.wheelbase_mm
        
        total_front_weight = 0
        total_rear_weight = 0
        
        for item in placed_items:
            # Weight distribution based on item's longitudinal position
            item_center_x = item.position[0] + item.dimensions[0] / 2
            
            if item_center_x <= front_axle_pos:
                # Item is ahead of front axle (unusual but possible)
                total_front_weight += item.weight
            elif item_center_x >= rear_axle_pos:
                # Item is behind rear axle
                total_rear_weight += item.weight
            else:
                # Item is between axles - distribute weight
                distance_from_front = item_center_x - front_axle_pos
                distance_from_rear = rear_axle_pos - item_center_x
                total_distance = rear_axle_pos - front_axle_pos
                
                # Weight closer to rear axle puts more weight on rear
                rear_ratio = distance_from_front / total_distance
                front_ratio = distance_from_rear / total_distance
                
                total_front_weight += item.weight * front_ratio
                total_rear_weight += item.weight * rear_ratio
        
        return (total_front_weight, total_rear_weight)
    
    def check_stability(self, placed_items: List[PlacedItem]) -> Dict[str, Any]:
        """Comprehensive stability analysis"""
        cg_x, cg_y, cg_z = self.calculate_center_of_gravity(placed_items)
        front_weight, rear_weight = self.calculate_axle_weights(placed_items)
        
        issues = []
        warnings = []
        
        # Check center of gravity height
        max_cg_height = self.truck.max_center_of_gravity_height_mm * self.safety_margin
        if cg_z > max_cg_height:
            issues.append(f"Center of gravity too high: {cg_z:.0f}mm > {max_cg_height:.0f}mm")
        
        # Check axle weight limits
        max_front = self.truck.front_axle_limit_kg * self.safety_margin
        max_rear = self.truck.rear_axle_limit_kg * self.safety_margin
        
        if front_weight > max_front:
            issues.append(f"Front axle overloaded: {front_weight:.0f}kg > {max_front:.0f}kg")
        
        if rear_weight > max_rear:
            issues.append(f"Rear axle overloaded: {rear_weight:.0f}kg > {max_rear:.0f}kg")
        
        # Check weight distribution
        total_weight = front_weight + rear_weight
        if total_weight > 0:
            rear_percentage = rear_weight / total_weight
            ideal_rear_percentage = 0.6  # Typical for trucks
            tolerance = self.truck.weight_distribution_tolerance
            
            if abs(rear_percentage - ideal_rear_percentage) > tolerance:
                warnings.append(f"Weight distribution suboptimal: {rear_percentage*100:.1f}% on rear axle")
        
        # Check lateral center of gravity
        truck_centerline = self.truck.internal_width_mm / 2
        lateral_offset = abs(cg_y - truck_centerline)
        max_lateral_offset = self.truck.internal_width_mm * 0.1  # 10% of width
        
        if lateral_offset > max_lateral_offset:
            warnings.append(f"Load not centered laterally: {lateral_offset:.0f}mm offset")
        
        return {
            "stable": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "center_of_gravity": (cg_x, cg_y, cg_z),
            "axle_weights": (front_weight, rear_weight),
            "weight_distribution_rear_pct": (rear_weight / total_weight * 100) if total_weight > 0 else 0
        }

# ----------------------------
# Advanced Packing Algorithm
# ----------------------------

class RealWorldPacker:
    def __init__(self, truck: RealWorldTruckSpec):
        self.truck = truck
        self.physics = RealWorldPhysicsEngine(truck)
        self.placed_items: List[PlacedItem] = []
        
    def get_valid_orientations(self, box: EnhancedBoxInput) -> List[Tuple[float, float, float]]:
        """Get valid orientations based on constraints"""
        base_dims = (box.length_mm, box.width_mm, box.height_mm)
        orientations = []
        
        if box.orientation_constraint == OrientationConstraint.ANY:
            # All 6 orientations possible
            l, w, h = base_dims
            orientations = [
                (l, w, h),  # Original
                (l, h, w),  # Rotated around length axis  
                (w, l, h),  # Rotated around vertical axis
                (w, h, l),  # Rotated around width axis
                (h, l, w),  # Rotated around diagonal
                (h, w, l)   # Rotated around other diagonal
            ]
        elif box.orientation_constraint == OrientationConstraint.UPRIGHT_ONLY:
            # Only original orientation
            orientations = [base_dims]
        elif box.orientation_constraint == OrientationConstraint.NO_INVERT:
            # Can lay flat or upright, but not inverted
            l, w, h = base_dims
            orientations = [
                (l, w, h),  # Upright
                (l, h, w),  # Laid on side (width becomes height)
                (w, l, h),  # Rotated 90° but still upright
                (h, w, l)   # Laid flat (height becomes length)
            ]
        
        # Filter orientations that fit in truck
        valid_orientations = []
        for i, dims in enumerate(orientations):
            if (dims[0] <= self.truck.internal_length_mm and 
                dims[1] <= self.truck.internal_width_mm and 
                dims[2] <= self.truck.internal_height_mm):
                valid_orientations.append((dims, i))
        
        return valid_orientations
    
    def find_support_area(self, position: Tuple[float, float, float], 
                         dimensions: Tuple[float, float, float]) -> float:
        """Calculate how much of the bottom is supported by items below"""
        x, y, z = position
        l, w, h = dimensions
        
        if z == 0:  # On the floor
            return l * w
        
        bottom_area = l * w
        supported_area = 0
        
        # Check each placed item to see if it provides support
        for item in self.placed_items:
            item_top = item.position[2] + item.dimensions[2]
            
            # Must be directly below (within tolerance)
            if abs(item_top - z) > 1:  # 1mm tolerance
                continue
            
            # Calculate overlap area
            overlap_x = max(0, min(x + l, item.position[0] + item.dimensions[0]) - max(x, item.position[0]))
            overlap_y = max(0, min(y + w, item.position[1] + item.dimensions[1]) - max(y, item.position[1]))
            
            supported_area += overlap_x * overlap_y
        
        return min(supported_area, bottom_area)
    
    def can_place_item(self, box: EnhancedBoxInput, position: Tuple[float, float, float], 
                      dimensions: Tuple[float, float, float]) -> Tuple[bool, List[str]]:
        """Check if item can be placed at position with given dimensions"""
        x, y, z = position
        l, w, h = dimensions
        reasons = []
        
        # Check truck boundaries
        if x + l > self.truck.internal_length_mm:
            reasons.append("Exceeds truck length")
        if y + w > self.truck.internal_width_mm:
            reasons.append("Exceeds truck width")  
        if z + h > self.truck.internal_height_mm:
            reasons.append("Exceeds truck height")
        
        # Check collision with existing items
        for item in self.placed_items:
            if (x < item.position[0] + item.dimensions[0] and x + l > item.position[0] and
                y < item.position[1] + item.dimensions[1] and y + w > item.position[1] and
                z < item.position[2] + item.dimensions[2] and z + h > item.position[2]):
                reasons.append(f"Collision with {item.name}")
        
        # Check support requirements
        if box.requires_bottom_support or z > 0:
            support_area = self.find_support_area(position, dimensions)
            required_support = (l * w * box.min_support_percentage / 100.0)
            
            if support_area < required_support:
                reasons.append(f"Insufficient support: {support_area:.0f} < {required_support:.0f} mm²")
        
        # Check weight support of items below
        if z > 0:
            for item in self.placed_items:
                item_top = item.position[2] + item.dimensions[2]
                if abs(item_top - z) <= 1:  # This item would rest on existing item
                    # Find the corresponding box spec
                    existing_box = None
                    # In real implementation, you'd track this better
                    if item.weight > 0:  # Simplified check
                        max_stack_weight = getattr(item, 'max_stack_weight_kg', None)
                        if max_stack_weight and box.weight_kg > max_stack_weight:
                            reasons.append(f"Too heavy for item below: {box.weight_kg}kg > {max_stack_weight}kg")
        
        return len(reasons) == 0, reasons
    
    def find_placement_positions(self, box: EnhancedBoxInput, 
                                dimensions: Tuple[float, float, float]) -> List[Tuple[float, float, float]]:
        """Find all valid positions for placing an item"""
        positions = []
        l, w, h = dimensions
        
        # Start with floor positions
        positions.append((0, 0, 0))
        
        # Add positions based on existing items
        for item in self.placed_items:
            # Try placing on top
            top_z = item.position[2] + item.dimensions[2]
            positions.append((item.position[0], item.position[1], top_z))
            
            # Try placing adjacent (same level)
            same_z = item.position[2]
            # To the right
            positions.append((item.position[0] + item.dimensions[0], item.position[1], same_z))
            # Behind  
            positions.append((item.position[0], item.position[1] + item.dimensions[1], same_z))
            # To the right and behind
            positions.append((item.position[0] + item.dimensions[0], 
                           item.position[1] + item.dimensions[1], same_z))
        
        # Filter valid positions
        valid_positions = []
        for pos in positions:
            can_place, _ = self.can_place_item(box, pos, dimensions)
            if can_place:
                valid_positions.append(pos)
        
        # Sort by preference (lower first, then forward, then left)
        valid_positions.sort(key=lambda p: (p[2], p[0], p[1]))
        
        return valid_positions
    
    def place_item(self, box: EnhancedBoxInput) -> bool:
        """Attempt to place a single item"""
        valid_orientations = self.get_valid_orientations(box)
        
        if not valid_orientations:
            return False
        
        # Try each orientation
        for dimensions, rotation in valid_orientations:
            positions = self.find_placement_positions(box, dimensions)
            
            for position in positions:
                can_place, reasons = self.can_place_item(box, position, dimensions)
                
                if can_place:
                    # Calculate support area for the placed item
                    support_area = self.find_support_area(position, dimensions)
                    
                    placed_item = PlacedItem(
                        name=box.box_type,
                        position=position,
                        dimensions=dimensions,
                        weight=box.weight_kg,
                        rotation=rotation,
                        support_area=support_area,
                        fragility=box.fragility,
                        orientation=box.orientation_constraint,
                        priority=box.loading_priority
                    )
                    
                    self.placed_items.append(placed_item)
                    return True
        
        return False
    
    def pack_items(self, boxes: List[EnhancedBoxInput]) -> Dict[str, Any]:
        """Pack all items using real-world constraints"""
        self.placed_items = []
        
        # Sort boxes by priority and characteristics
        # 1. Heavy items first (for stability)
        # 2. Large items first (harder to place)
        # 3. Priority items (first delivery) accessible
        
        sorted_boxes = sorted(boxes, key=lambda b: (
            -b.weight_kg,  # Heavy first
            -(b.length_mm * b.width_mm * b.height_mm),  # Large first
            0 if b.loading_priority == LoadingPriority.FIRST_OUT else 1  # Priority items first
        ))
        
        # Expand boxes by quantity
        items_to_pack = []
        for box in sorted_boxes:
            for _ in range(box.quantity):
                items_to_pack.append(box)
        
        # Pack items
        packed_count = 0
        unpacked_items = []
        
        for item in items_to_pack:
            if self.place_item(item):
                packed_count += 1
            else:
                unpacked_items.append(item.box_type)
        
        # Analyze the result
        stability_analysis = self.physics.check_stability(self.placed_items)
        
        # Calculate utilization
        total_volume = sum(item.dimensions[0] * item.dimensions[1] * item.dimensions[2] 
                          for item in self.placed_items)
        truck_volume = (self.truck.internal_length_mm * 
                       self.truck.internal_width_mm * 
                       self.truck.internal_height_mm)
        volume_utilization = (total_volume / truck_volume) * 100 if truck_volume > 0 else 0
        
        total_weight = sum(item.weight for item in self.placed_items)
        weight_utilization = (total_weight / self.truck.max_payload_kg) * 100 if self.truck.max_payload_kg > 0 else 0
        
        return {
            "success": True,
            "packed_items": len(self.placed_items),
            "total_items": len(items_to_pack),
            "unpacked_items": unpacked_items,
            "volume_utilization_pct": round(volume_utilization, 2),
            "weight_utilization_pct": round(weight_utilization, 2),
            "total_weight_kg": round(total_weight, 2),
            "stability_analysis": stability_analysis,
            "placements": [
                {
                    "name": item.name,
                    "position_mm": item.position,
                    "dimensions_mm": item.dimensions,
                    "weight_kg": item.weight,
                    "rotation": item.rotation,
                    "support_area_mm2": round(item.support_area, 0)
                }
                for item in self.placed_items
            ],
            "loading_instructions": self._generate_loading_instructions()
        }
    
    def _generate_loading_instructions(self) -> List[str]:
        """Generate step-by-step loading instructions"""
        instructions = []
        
        # Sort by loading order (bottom to top, back to front for accessibility)
        loading_order = sorted(self.placed_items, 
                             key=lambda item: (item.position[2], -item.position[0]))
        
        for i, item in enumerate(loading_order, 1):
            x, y, z = item.position
            l, w, h = item.dimensions
            
            instruction = f"{i}. Load {item.name} at position ({x:.0f}, {y:.0f}, {z:.0f})mm"
            
            if z > 0:
                instruction += " (stack on previous items)"
            if item.fragility == FragilityLevel.VERY_FRAGILE:
                instruction += " - HANDLE WITH EXTREME CARE"
            elif item.fragility == FragilityLevel.FRAGILE:
                instruction += " - Handle carefully"
                
            instructions.append(instruction)
        
        return instructions

# ----------------------------
# API Endpoints
# ----------------------------

@app.post("/api/optimize-real-world")
def optimize_real_world(req: OptimizeRequest):
    """Real-world 3D truck optimization with physics and constraints"""
    logger.info("Real-world optimization request received")
    
    try:
        packer = RealWorldPacker(req.truck)
        result = packer.pack_items(req.boxes)
        
        # Add truck information
        result["truck_name"] = req.truck.name
        result["truck_specs"] = {
            "dimensions_mm": {
                "length": req.truck.internal_length_mm,
                "width": req.truck.internal_width_mm,
                "height": req.truck.internal_height_mm
            },
            "weight_limits_kg": {
                "max_payload": req.truck.max_payload_kg,
                "front_axle": req.truck.front_axle_limit_kg,
                "rear_axle": req.truck.rear_axle_limit_kg
            }
        }
        
        return result
        
    except Exception as e:
        logger.exception("Error in real-world optimization: %s", e)
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")

# Default truck specifications for Indian market
DEFAULT_REAL_TRUCKS = [
    RealWorldTruckSpec(
        name="Tata 407 (22 ft)",
        internal_length_mm=6700,
        internal_width_mm=2440,
        internal_height_mm=2440,
        max_payload_kg=7500,
        front_axle_limit_kg=2500,
        rear_axle_limit_kg=5000,
        wheelbase_mm=3800,
        max_center_of_gravity_height_mm=1800,
        loading_height_mm=1200
    ),
    RealWorldTruckSpec(
        name="Tata LPT 3118 (32 ft)",
        internal_length_mm=9750,
        internal_width_mm=2440,
        internal_height_mm=2440,
        max_payload_kg=16000,
        front_axle_limit_kg=6500,
        rear_axle_limit_kg=11500,
        wheelbase_mm=5200,
        max_center_of_gravity_height_mm=1900,
        loading_height_mm=1200
    )
]

@app.get("/api/default-trucks")
def get_default_trucks():
    """Get default truck specifications"""
    return [truck.dict() for truck in DEFAULT_REAL_TRUCKS]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)