"""
P-System Membrane Boundaries for Cognitive Encapsulation

Implements membrane systems (P-systems) for encapsulating recursive agentic grammars
and creating boundaries between different cognitive processes.
"""

import numpy as np
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from datetime import datetime, timezone
import json
import uuid
import threading
from collections import defaultdict

from .atomspace import AtomSpace
from .neural_symbolic_reasoning import NeuralSymbolicReasoningEngine


class MembraneType(Enum):
    """Types of cognitive membranes"""
    PERCEPTION = "perception"
    REASONING = "reasoning"
    MEMORY = "memory"
    ATTENTION = "attention"
    EXECUTION = "execution"
    COMMUNICATION = "communication"
    META_COGNITIVE = "meta_cognitive"


class PermeabilityLevel(Enum):
    """Levels of membrane permeability"""
    IMPERMEABLE = 0      # No exchange
    SELECTIVE = 1        # Selective exchange based on rules
    SEMI_PERMEABLE = 2   # Limited exchange
    PERMEABLE = 3        # Free exchange


@dataclass
class CognitiveObject:
    """Objects that exist within membranes"""
    id: str
    type: str
    content: Any
    energy_level: float = 1.0
    priority: float = 0.5
    creation_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MembraneRule:
    """Rules governing membrane behavior"""
    id: str
    condition: str  # Pattern or condition for rule activation
    action: str     # Action to take when condition is met
    priority: int   # Rule priority (higher = more important)
    permeability_change: Optional[PermeabilityLevel] = None
    object_transformation: Optional[Dict[str, Any]] = None


@dataclass
class Membrane:
    """P-System membrane for cognitive encapsulation"""
    id: str
    name: str
    type: MembraneType
    parent_id: Optional[str] = None
    children: Set[str] = field(default_factory=set)
    
    # Membrane properties
    permeability: PermeabilityLevel = PermeabilityLevel.SEMI_PERMEABLE
    capacity: int = 1000
    energy_level: float = 1.0
    
    # Contained objects and rules
    objects: Dict[str, CognitiveObject] = field(default_factory=dict)
    rules: Dict[str, MembraneRule] = field(default_factory=dict)
    
    # State tracking
    active: bool = True
    last_update: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    statistics: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize statistics"""
        self.statistics = {
            "object_count": 0,
            "rule_executions": 0,
            "exchanges_in": 0,
            "exchanges_out": 0,
            "transformations": 0
        }


class PSystemMembraneNetwork:
    """
    Network of P-System membranes for cognitive architecture encapsulation
    """
    
    def __init__(self, atomspace: AtomSpace):
        self.atomspace = atomspace
        self.membranes: Dict[str, Membrane] = {}
        self.root_membrane_id: Optional[str] = None
        self.execution_queue: List[Tuple[str, str, Any]] = []  # (membrane_id, rule_id, context)
        self.running = False
        self.execution_thread: Optional[threading.Thread] = None
        
        # Global statistics
        self.global_stats = {
            "total_membranes": 0,
            "total_objects": 0,
            "total_rules": 0,
            "execution_cycles": 0,
            "membrane_communications": 0
        }
    
    async def initialize_cognitive_architecture(self) -> str:
        """Initialize the standard cognitive architecture with membranes"""
        
        # Create root membrane (global cognitive space)
        root_id = await self.create_membrane(
            "root_cognitive_space",
            MembraneType.META_COGNITIVE,
            capacity=10000,
            permeability=PermeabilityLevel.PERMEABLE
        )
        self.root_membrane_id = root_id
        
        # Create perception membrane
        perception_id = await self.create_membrane(
            "perception_membrane",
            MembraneType.PERCEPTION,
            parent_id=root_id,
            capacity=1000,
            permeability=PermeabilityLevel.SELECTIVE
        )
        
        # Create reasoning membrane
        reasoning_id = await self.create_membrane(
            "reasoning_membrane", 
            MembraneType.REASONING,
            parent_id=root_id,
            capacity=2000,
            permeability=PermeabilityLevel.SEMI_PERMEABLE
        )
        
        # Create memory membrane
        memory_id = await self.create_membrane(
            "memory_membrane",
            MembraneType.MEMORY,
            parent_id=root_id,
            capacity=5000,
            permeability=PermeabilityLevel.SELECTIVE
        )
        
        # Create attention membrane
        attention_id = await self.create_membrane(
            "attention_membrane",
            MembraneType.ATTENTION,
            parent_id=root_id,
            capacity=500,
            permeability=PermeabilityLevel.SEMI_PERMEABLE
        )
        
        # Create execution membrane
        execution_id = await self.create_membrane(
            "execution_membrane",
            MembraneType.EXECUTION,
            parent_id=root_id,
            capacity=1000,
            permeability=PermeabilityLevel.PERMEABLE
        )
        
        # Create communication membrane
        communication_id = await self.create_membrane(
            "communication_membrane",
            MembraneType.COMMUNICATION,
            parent_id=root_id,
            capacity=500,
            permeability=PermeabilityLevel.PERMEABLE
        )
        
        # Set up basic rules for cognitive flow
        await self._setup_basic_cognitive_rules()
        
        # Start execution
        self.running = True
        self.execution_thread = threading.Thread(target=self._execute_membrane_system, daemon=True)
        self.execution_thread.start()
        
        return root_id
    
    async def create_membrane(self, name: str, membrane_type: MembraneType,
                            parent_id: Optional[str] = None,
                            capacity: int = 1000,
                            permeability: PermeabilityLevel = PermeabilityLevel.SEMI_PERMEABLE) -> str:
        """Create a new membrane"""
        membrane_id = f"membrane_{uuid.uuid4().hex[:8]}"
        
        membrane = Membrane(
            id=membrane_id,
            name=name,
            type=membrane_type,
            parent_id=parent_id,
            capacity=capacity,
            permeability=permeability
        )
        
        self.membranes[membrane_id] = membrane
        
        # Add to parent's children if parent exists
        if parent_id and parent_id in self.membranes:
            self.membranes[parent_id].children.add(membrane_id)
        
        self.global_stats["total_membranes"] += 1
        
        return membrane_id
    
    async def add_object_to_membrane(self, membrane_id: str, obj: CognitiveObject) -> bool:
        """Add an object to a membrane"""
        if membrane_id not in self.membranes:
            return False
        
        membrane = self.membranes[membrane_id]
        
        # Check capacity
        if len(membrane.objects) >= membrane.capacity:
            return False
        
        membrane.objects[obj.id] = obj
        membrane.statistics["object_count"] = len(membrane.objects)
        membrane.last_update = datetime.now(timezone.utc)
        
        self.global_stats["total_objects"] += 1
        
        return True
    
    async def add_rule_to_membrane(self, membrane_id: str, rule: MembraneRule) -> bool:
        """Add a rule to a membrane"""
        if membrane_id not in self.membranes:
            return False
        
        membrane = self.membranes[membrane_id]
        membrane.rules[rule.id] = rule
        membrane.last_update = datetime.now(timezone.utc)
        
        self.global_stats["total_rules"] += 1
        
        return True
    
    async def transfer_object(self, obj_id: str, from_membrane_id: str, 
                            to_membrane_id: str) -> bool:
        """Transfer an object between membranes"""
        
        # Check if both membranes exist
        if (from_membrane_id not in self.membranes or 
            to_membrane_id not in self.membranes):
            return False
        
        from_membrane = self.membranes[from_membrane_id]
        to_membrane = self.membranes[to_membrane_id]
        
        # Check if object exists in source membrane
        if obj_id not in from_membrane.objects:
            return False
        
        # Check permeability constraints
        if not self._check_transfer_permissions(from_membrane, to_membrane):
            return False
        
        # Check capacity in destination
        if len(to_membrane.objects) >= to_membrane.capacity:
            return False
        
        # Perform transfer
        obj = from_membrane.objects.pop(obj_id)
        to_membrane.objects[obj_id] = obj
        
        # Update statistics
        from_membrane.statistics["exchanges_out"] += 1
        from_membrane.statistics["object_count"] = len(from_membrane.objects)
        to_membrane.statistics["exchanges_in"] += 1
        to_membrane.statistics["object_count"] = len(to_membrane.objects)
        
        from_membrane.last_update = datetime.now(timezone.utc)
        to_membrane.last_update = datetime.now(timezone.utc)
        
        self.global_stats["membrane_communications"] += 1
        
        return True
    
    def _check_transfer_permissions(self, from_membrane: Membrane, to_membrane: Membrane) -> bool:
        """Check if transfer is allowed between membranes"""
        
        # Impermeable membranes don't allow transfer
        if (from_membrane.permeability == PermeabilityLevel.IMPERMEABLE or
            to_membrane.permeability == PermeabilityLevel.IMPERMEABLE):
            return False
        
        # Check if membranes are related (parent-child relationship allows easier transfer)
        if (from_membrane.parent_id == to_membrane.id or
            to_membrane.parent_id == from_membrane.id or
            from_membrane.parent_id == to_membrane.parent_id):
            return True
        
        # For non-related membranes, require higher permeability
        min_permeability = min(from_membrane.permeability.value, to_membrane.permeability.value)
        return min_permeability >= PermeabilityLevel.SEMI_PERMEABLE.value
    
    async def _setup_basic_cognitive_rules(self):
        """Set up basic rules for cognitive flow between membranes"""
        
        # Find membrane IDs
        perception_id = None
        reasoning_id = None
        memory_id = None
        attention_id = None
        execution_id = None
        
        for membrane_id, membrane in self.membranes.items():
            if membrane.type == MembraneType.PERCEPTION:
                perception_id = membrane_id
            elif membrane.type == MembraneType.REASONING:
                reasoning_id = membrane_id
            elif membrane.type == MembraneType.MEMORY:
                memory_id = membrane_id
            elif membrane.type == MembraneType.ATTENTION:
                attention_id = membrane_id
            elif membrane.type == MembraneType.EXECUTION:
                execution_id = membrane_id
        
        # Perception → Attention rule
        if perception_id and attention_id:
            await self.add_rule_to_membrane(perception_id, MembraneRule(
                id="perception_to_attention",
                condition="object_type:sensory_input AND priority > 0.5",
                action=f"transfer_to:{attention_id}",
                priority=10
            ))
        
        # Attention → Reasoning rule
        if attention_id and reasoning_id:
            await self.add_rule_to_membrane(attention_id, MembraneRule(
                id="attention_to_reasoning",
                condition="object_type:attended_input AND energy_level > 0.7",
                action=f"transfer_to:{reasoning_id}",
                priority=9
            ))
        
        # Reasoning → Memory rule
        if reasoning_id and memory_id:
            await self.add_rule_to_membrane(reasoning_id, MembraneRule(
                id="reasoning_to_memory",
                condition="object_type:inference_result AND priority > 0.6",
                action=f"transfer_to:{memory_id}",
                priority=8
            ))
        
        # Reasoning → Execution rule
        if reasoning_id and execution_id:
            await self.add_rule_to_membrane(reasoning_id, MembraneRule(
                id="reasoning_to_execution",
                condition="object_type:action_plan AND energy_level > 0.8",
                action=f"transfer_to:{execution_id}",
                priority=10
            ))
    
    def _execute_membrane_system(self):
        """Execute membrane system in separate thread"""
        while self.running:
            try:
                # Execute one cycle
                asyncio.run(self._execute_one_cycle())
                
                # Sleep for a short time
                import time
                time.sleep(0.1)
                
            except Exception as e:
                print(f"❌ Membrane execution error: {e}")
    
    async def _execute_one_cycle(self):
        """Execute one cycle of the membrane system"""
        
        # Process all membranes
        for membrane_id, membrane in self.membranes.items():
            if not membrane.active:
                continue
            
            # Execute rules for this membrane
            await self._execute_membrane_rules(membrane_id)
        
        self.global_stats["execution_cycles"] += 1
    
    async def _execute_membrane_rules(self, membrane_id: str):
        """Execute rules for a specific membrane"""
        membrane = self.membranes[membrane_id]
        
        # Sort rules by priority
        sorted_rules = sorted(membrane.rules.values(), key=lambda r: r.priority, reverse=True)
        
        for rule in sorted_rules:
            # Check if rule condition is met
            if await self._evaluate_rule_condition(membrane_id, rule):
                # Execute rule action
                await self._execute_rule_action(membrane_id, rule)
                membrane.statistics["rule_executions"] += 1
    
    async def _evaluate_rule_condition(self, membrane_id: str, rule: MembraneRule) -> bool:
        """Evaluate if a rule condition is met"""
        membrane = self.membranes[membrane_id]
        condition = rule.condition
        
        # Simple condition parsing (can be enhanced)
        if "object_type:" in condition:
            # Extract object type requirement
            object_type = condition.split("object_type:")[1].split(" ")[0]
            
            # Check if any objects match this type
            for obj in membrane.objects.values():
                if obj.type == object_type:
                    # Check additional conditions
                    if "priority >" in condition:
                        threshold = float(condition.split("priority >")[1].split(" ")[0])
                        if obj.priority > threshold:
                            return True
                    elif "energy_level >" in condition:
                        threshold = float(condition.split("energy_level >")[1].split(" ")[0])
                        if obj.energy_level > threshold:
                            return True
                    else:
                        return True
        
        return False
    
    async def _execute_rule_action(self, membrane_id: str, rule: MembraneRule):
        """Execute a rule action"""
        action = rule.action
        
        if action.startswith("transfer_to:"):
            target_membrane_id = action.split("transfer_to:")[1]
            
            # Find matching objects to transfer
            membrane = self.membranes[membrane_id]
            for obj_id, obj in list(membrane.objects.items()):
                # Transfer the first matching object
                await self.transfer_object(obj_id, membrane_id, target_membrane_id)
                break  # Transfer one object per rule execution
    
    async def inject_cognitive_object(self, membrane_type: MembraneType, 
                                    obj_type: str, content: Any,
                                    priority: float = 0.5, energy: float = 1.0) -> str:
        """Inject a cognitive object into the appropriate membrane"""
        
        # Find membrane of the specified type
        target_membrane_id = None
        for membrane_id, membrane in self.membranes.items():
            if membrane.type == membrane_type:
                target_membrane_id = membrane_id
                break
        
        if not target_membrane_id:
            raise ValueError(f"No membrane of type {membrane_type} found")
        
        # Create object
        obj = CognitiveObject(
            id=f"obj_{uuid.uuid4().hex[:8]}",
            type=obj_type,
            content=content,
            priority=priority,
            energy_level=energy
        )
        
        # Add to membrane
        success = await self.add_object_to_membrane(target_membrane_id, obj)
        
        if success:
            return obj.id
        else:
            raise RuntimeError(f"Failed to add object to membrane {target_membrane_id}")
    
    def get_membrane_state(self, membrane_id: str) -> Dict[str, Any]:
        """Get current state of a membrane"""
        if membrane_id not in self.membranes:
            return {}
        
        membrane = self.membranes[membrane_id]
        return {
            "id": membrane.id,
            "name": membrane.name,
            "type": membrane.type.value,
            "object_count": len(membrane.objects),
            "rule_count": len(membrane.rules),
            "capacity": membrane.capacity,
            "permeability": membrane.permeability.value,
            "energy_level": membrane.energy_level,
            "active": membrane.active,
            "statistics": membrane.statistics,
            "last_update": membrane.last_update.isoformat()
        }
    
    def get_system_state(self) -> Dict[str, Any]:
        """Get complete system state"""
        return {
            "global_statistics": self.global_stats,
            "total_membranes": len(self.membranes),
            "running": self.running,
            "membranes": {
                membrane_id: self.get_membrane_state(membrane_id)
                for membrane_id in self.membranes
            }
        }
    
    def stop(self):
        """Stop the membrane system"""
        self.running = False
        if self.execution_thread and self.execution_thread.is_alive():
            self.execution_thread.join(timeout=5)