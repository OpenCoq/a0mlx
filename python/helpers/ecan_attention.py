"""
ECAN (Economic Attention Allocation Network) for Meta-Cognitive Control

Implements attention allocation and economic resource management for cognitive agents
based on AtomSpace hypergraph structures and neural-symbolic reasoning.
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
# Import at function level to avoid circular import
# from .neural_symbolic_reasoning import NeuralSymbolicReasoningEngine, ReasoningStage, CognitiveKernel


class AttentionType(Enum):
    """Types of attention allocation"""
    PERCEPTUAL = "perceptual"
    COGNITIVE = "cognitive"
    EXECUTIVE = "executive"
    SOCIAL = "social"
    SELF_MONITORING = "self_monitoring"
    # MOSES-specific attention types
    MOSES_EVOLUTION = "moses_evolution"
    MOSES_FITNESS = "moses_fitness"
    MOSES_PROGRAM = "moses_program"


class ResourceType(Enum):
    """Types of cognitive resources"""
    MEMORY = "memory"
    PROCESSING = "processing"
    ATTENTION = "attention"
    BANDWIDTH = "bandwidth"
    ENERGY = "energy"


class MetaCognitiveState(Enum):
    """Meta-cognitive states for self-monitoring"""
    ANALYZING = "analyzing"
    LEARNING = "learning"
    ADAPTING = "adapting"
    OPTIMIZING = "optimizing"
    IDLE = "idle"
    SELF_MODIFYING = "self_modifying"


@dataclass
class AttentionUnit:
    """Represents a unit of attention with economic properties"""
    id: str
    atom_id: str  # Reference to AtomSpace atom
    attention_value: float  # Short-term importance
    importance: float  # Long-term importance
    confidence: float  # Confidence in importance
    resource_cost: float  # Cost to maintain attention
    last_accessed: datetime
    access_count: int = 0
    decay_rate: float = 0.01
    
    def update_attention(self, delta: float, access_time: datetime = None):
        """Update attention value with decay"""
        if access_time is None:
            access_time = datetime.now(timezone.utc)
        
        # Apply decay based on time since last access
        time_delta = (access_time - self.last_accessed).total_seconds()
        decay = self.decay_rate * time_delta
        
        # Update values
        self.attention_value = max(0.0, self.attention_value - decay + delta)
        self.last_accessed = access_time
        self.access_count += 1


@dataclass
class MOSESAttentionUnit(AttentionUnit):
    """Specialized attention unit for MOSES entities"""
    moses_entity_type: str = "program"  # program, population, fitness
    fitness_score: float = 0.0
    generation: int = 0
    complexity: int = 0
    evolution_priority: float = 0.0
    
    def update_from_moses_fitness(self, fitness_score: float, complexity: int = None, generation: int = None):
        """Update attention based on MOSES fitness metrics"""
        # Base attention on fitness score
        fitness_attention_delta = (fitness_score - self.fitness_score) * 2.0
        
        # Bonus for high-performing programs
        if fitness_score > 0.8:
            fitness_attention_delta *= 1.5
        
        # Penalty for high complexity (encourage efficiency)
        if complexity is not None and complexity > 10:
            complexity_penalty = (complexity - 10) * 0.1
            fitness_attention_delta -= complexity_penalty
        
        # Generational recency bonus
        if generation is not None and generation > self.generation:
            generation_bonus = (generation - self.generation) * 0.1
            fitness_attention_delta += generation_bonus
        
        # Update values
        self.fitness_score = fitness_score
        if complexity is not None:
            self.complexity = complexity
        if generation is not None:
            self.generation = generation
        
        # Apply attention update
        self.update_attention(fitness_attention_delta)
        
        # Update evolution priority based on fitness and recency
        self.evolution_priority = fitness_score * (1.0 + (generation * 0.1))


@dataclass
class ResourcePool:
    """Manages cognitive resource allocation"""
    resource_type: ResourceType
    total_capacity: float
    available_capacity: float
    allocation_history: List[Tuple[str, float, datetime]] = field(default_factory=list)
    
    def allocate(self, requester_id: str, amount: float) -> bool:
        """Attempt to allocate resources"""
        if self.available_capacity >= amount:
            self.available_capacity -= amount
            self.allocation_history.append((requester_id, amount, datetime.now(timezone.utc)))
            return True
        return False
    
    def release(self, amount: float):
        """Release allocated resources"""
        self.available_capacity = min(self.total_capacity, self.available_capacity + amount)


@dataclass
class SelfInspectionReport:
    """Report from self-inspection routine"""
    id: str
    timestamp: datetime
    cognitive_state: MetaCognitiveState
    attention_distribution: Dict[str, float]
    resource_utilization: Dict[str, float]
    performance_metrics: Dict[str, float]
    anomalies_detected: List[str]
    recommendations: List[str]
    hypergraph_snapshot: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "cognitive_state": self.cognitive_state.value,
            "attention_distribution": self.attention_distribution,
            "resource_utilization": self.resource_utilization,
            "performance_metrics": self.performance_metrics,
            "anomalies_detected": self.anomalies_detected,
            "recommendations": self.recommendations,
            "hypergraph_snapshot": self.hypergraph_snapshot
        }


class ECANAttentionSystem:
    """
    Economic Attention Allocation Network (ECAN) for meta-cognitive control
    
    Implements distributed attention allocation based on economic principles
    with self-monitoring and adaptive behavior modification.
    """
    
    def __init__(self, atomspace: AtomSpace, reasoning_engine=None):
        self.atomspace = atomspace
        self.reasoning_engine = reasoning_engine  # Can be None initially
        
        # Attention management
        self.attention_units: Dict[str, AttentionUnit] = {}
        self.attention_threshold = 0.1
        self.max_attention_units = 1000
        
        # Resource management
        self.resource_pools = {
            ResourceType.MEMORY: ResourcePool(ResourceType.MEMORY, 100.0, 100.0),
            ResourceType.PROCESSING: ResourcePool(ResourceType.PROCESSING, 100.0, 100.0),
            ResourceType.ATTENTION: ResourcePool(ResourceType.ATTENTION, 100.0, 100.0),
            ResourceType.BANDWIDTH: ResourcePool(ResourceType.BANDWIDTH, 100.0, 100.0),
            ResourceType.ENERGY: ResourcePool(ResourceType.ENERGY, 100.0, 100.0)
        }
        
        # Self-monitoring
        self.current_state = MetaCognitiveState.IDLE
        self.inspection_reports: List[SelfInspectionReport] = []
        self.self_modification_events: List[Dict[str, Any]] = []
        
        # Autonomy metrics tensor T_auto[a_levels, r_types, m_states]
        self.autonomy_levels = 5  # Different levels of autonomy
        self.num_resource_types = len(ResourceType)
        self.num_meta_states = len(MetaCognitiveState)
        self.autonomy_tensor = np.zeros((self.autonomy_levels, self.num_resource_types, self.num_meta_states))
        
        # Threading for periodic tasks
        self.inspection_thread: Optional[threading.Thread] = None
        self.running = False
        
        # Agent reliability tracking for attention modulation
        self.agent_reliability_scores: Dict[str, float] = {}
        self.agent_performance_history: Dict[str, List[float]] = defaultdict(list)
        self.max_attention_units = 1000  # Maximum number of attention units
        
        # MOSES-specific tracking
        self.moses_attention_units: Dict[str, MOSESAttentionUnit] = {}
        self.moses_fitness_history: List[Dict[str, Any]] = []
        self.evolution_metrics: Dict[str, float] = {
            "best_fitness_seen": 0.0,
            "generation_count": 0,
            "population_diversity": 0.0,
            "attention_on_best_programs": 0.0
        }
        
        # Resource allocation policies for MOSES
        self.moses_resource_policies = {
            "high_fitness_bonus": 0.3,  # Extra resources for high-fitness programs
            "complexity_penalty": 0.2,  # Resource penalty for complex programs
            "generation_decay": 0.05,   # Attention decay for old generations
            "diversity_bonus": 0.15     # Bonus for maintaining population diversity
        }
        
        # Statistics
        self.statistics = {
            "attention_allocations": 0,
            "resource_allocations": 0,
            "self_modifications": 0,
            "inspection_cycles": 0,
            "adaptation_events": 0,
            "attention_redistributions": 0
        }
    
    async def initialize(self):
        """Initialize the ECAN system"""
        # Initialize attention units from existing atoms
        await self._initialize_attention_units()
        
        # Start self-monitoring thread
        self.running = True
        self.inspection_thread = threading.Thread(target=self._periodic_inspection, daemon=True)
        self.inspection_thread.start()
    
    async def _initialize_attention_units(self):
        """Initialize attention units from atomspace"""
        # Get all atoms from atomspace using pattern matching
        atoms = await self.atomspace.pattern_match({}, limit=1000)
        
        for atom in atoms:
            attention_unit = AttentionUnit(
                id=f"au_{atom.id}",
                atom_id=atom.id,
                attention_value=atom.truth_value,
                importance=atom.confidence,
                confidence=atom.confidence,
                resource_cost=0.1,  # Base cost
                last_accessed=datetime.now(timezone.utc)
            )
            self.attention_units[attention_unit.id] = attention_unit
    
    def get_autonomy_tensor(self) -> np.ndarray:
        """Get current autonomy metrics tensor T_auto[a_levels, r_types, m_states]"""
        # Update tensor with current state
        self._update_autonomy_tensor()
        return self.autonomy_tensor.copy()
    
    def _update_autonomy_tensor(self):
        """Update autonomy tensor with current metrics"""
        # Reset tensor
        self.autonomy_tensor.fill(0.0)
        
        # Current state index
        state_idx = list(MetaCognitiveState).index(self.current_state)
        
        # Resource utilization
        for i, resource_type in enumerate(ResourceType):
            pool = self.resource_pools[resource_type]
            utilization = 1.0 - (pool.available_capacity / pool.total_capacity)
            
            # Distribute across autonomy levels based on utilization
            for level in range(self.autonomy_levels):
                if utilization > (level / self.autonomy_levels):
                    self.autonomy_tensor[level, i, state_idx] = utilization
    
    async def allocate_attention(self, atom_id: str, priority: float, 
                               requester_id: str) -> bool:
        """Allocate attention to a specific atom with dynamic modulation"""
        # Enhanced attention allocation with adaptive modulation
        modulated_priority = await self._modulate_attention_priority(priority, atom_id, requester_id)
        
        # Check if we have attention resources
        if not self.resource_pools[ResourceType.ATTENTION].allocate(requester_id, 1.0):
            return False
        
        # Find or create attention unit
        attention_unit = None
        for au in self.attention_units.values():
            if au.atom_id == atom_id:
                attention_unit = au
                break
        
        if attention_unit is None:
            # Create new attention unit
            attention_unit = AttentionUnit(
                id=f"au_{atom_id}_{uuid.uuid4().hex[:8]}",
                atom_id=atom_id,
                attention_value=modulated_priority,
                importance=modulated_priority,
                urgency=modulated_priority,
                confidence=min(1.0, modulated_priority),
                resource_cost=0.1,  # Base cost
                last_accessed=datetime.now(timezone.utc)
            )
            self.attention_units[attention_unit.id] = attention_unit
        else:
            # Update existing attention unit with modulated priority
            attention_unit.attention_value = max(attention_unit.attention_value, modulated_priority)
            attention_unit.importance = max(attention_unit.importance, modulated_priority)
            attention_unit.urgency = max(attention_unit.urgency, modulated_priority)
            attention_unit.last_accessed = datetime.now(timezone.utc)
        
        # Update statistics
        self.statistics["attention_allocations"] += 1
        
        # Trigger attention redistribution if needed
        if len(self.attention_units) > self.max_attention_units:
            await self._redistribute_attention()
        
        return True
    
    async def _modulate_attention_priority(self, base_priority: float, atom_id: str, requester_id: str) -> float:
        """Dynamically modulate attention priority based on context and history"""
        modulated_priority = base_priority
        
        # Historical performance modulation
        if requester_id in self.agent_reliability_scores:
            reliability = self.agent_reliability_scores[requester_id]
            modulated_priority *= (0.5 + reliability * 0.5)  # Scale by reliability
        
        # Urgency boost for critical tasks
        if base_priority > 0.9:
            modulated_priority *= 1.2
        
        # Diminishing returns for repeated attention requests
        recent_requests = sum(1 for au in self.attention_units.values() 
                            if au.atom_id == atom_id and 
                            (datetime.now(timezone.utc) - au.last_accessed).total_seconds() < 300)
        
        if recent_requests > 0:
            diminishing_factor = 1.0 / (1.0 + recent_requests * 0.1)
            modulated_priority *= diminishing_factor
        
        # Resource availability modulation
        attention_pool = self.resource_pools[ResourceType.ATTENTION]
        availability_ratio = attention_pool.available_capacity / attention_pool.total_capacity
        
        if availability_ratio < 0.2:  # Low availability
            modulated_priority *= 0.8
        elif availability_ratio > 0.8:  # High availability
            modulated_priority *= 1.1
        
        return min(1.0, max(0.0, modulated_priority))
    
    async def _redistribute_attention(self):
        """Redistribute attention when resources are constrained"""
        # Sort attention units by a composite score
        sorted_units = sorted(
            self.attention_units.values(),
            key=lambda au: (au.importance * au.urgency * au.confidence) - 
                          (datetime.now(timezone.utc) - au.last_accessed).total_seconds() / 3600,
            reverse=True
        )
        
        # Keep only the top attention units
        units_to_keep = sorted_units[:self.max_attention_units]
        
        # Remove excess units
        for unit in sorted_units[self.max_attention_units:]:
            if unit.id in self.attention_units:
                del self.attention_units[unit.id]
        
        # Update statistics
        self.statistics["attention_redistributions"] += 1
    
    async def adaptive_attention_allocation(self, task_salience: Dict[str, float], 
                                          resource_load: Dict[str, float]) -> Dict[str, float]:
        """Adaptive attention allocation based on task salience and resource load"""
        allocation_result = {}
        
        # Calculate attention budget based on resource load
        total_load = sum(resource_load.values())
        attention_budget = max(0.1, 1.0 - (total_load / len(resource_load)))
        
        # Sort tasks by salience
        sorted_tasks = sorted(task_salience.items(), key=lambda x: x[1], reverse=True)
        
        # Allocate attention proportionally
        total_salience = sum(task_salience.values())
        for task_id, salience in sorted_tasks:
            if total_salience > 0:
                allocation = (salience / total_salience) * attention_budget
                allocation_result[task_id] = allocation
                
                # Actually allocate attention
                await self.allocate_attention(task_id, allocation, "adaptive_system")
        
        return allocation_result
    
    async def self_modify(self, modification_type: str, parameters: Dict[str, Any]) -> bool:
        """Perform self-modification based on introspection"""
        modification_event = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": modification_type,
            "parameters": parameters,
            "success": False,
            "changes_made": []
        }
        
        try:
            if modification_type == "attention_threshold":
                old_threshold = self.attention_threshold
                new_threshold = parameters.get("new_threshold", self.attention_threshold)
                self.attention_threshold = max(0.01, min(1.0, new_threshold))
                modification_event["changes_made"].append(f"Attention threshold: {old_threshold} -> {self.attention_threshold}")
            
            elif modification_type == "resource_reallocation":
                # Redistribute resource capacities
                reallocation = parameters.get("reallocation", {})
                for resource_name, new_capacity in reallocation.items():
                    try:
                        resource_type = ResourceType(resource_name)
                        if resource_type in self.resource_pools:
                            old_capacity = self.resource_pools[resource_type].total_capacity
                            self.resource_pools[resource_type].total_capacity = max(1.0, new_capacity)
                            modification_event["changes_made"].append(f"{resource_name} capacity: {old_capacity} -> {new_capacity}")
                    except ValueError:
                        continue
            
            elif modification_type == "decay_rate_adjustment":
                new_decay_rate = parameters.get("new_decay_rate", 0.01)
                changes_count = 0
                for au in self.attention_units.values():
                    if au.decay_rate != new_decay_rate:
                        au.decay_rate = new_decay_rate
                        changes_count += 1
                modification_event["changes_made"].append(f"Updated decay rate for {changes_count} attention units")
            
            modification_event["success"] = True
            self.statistics["self_modifications"] += 1
            
        except Exception as e:
            modification_event["error"] = str(e)
        
        self.self_modification_events.append(modification_event)
        return modification_event["success"]
    
    async def inspect_system(self) -> SelfInspectionReport:
        """Perform comprehensive self-inspection"""
        report_id = str(uuid.uuid4())
        
        # Attention distribution
        attention_distribution = {}
        for au in self.attention_units.values():
            attention_distribution[au.atom_id] = au.attention_value
        
        # Resource utilization
        resource_utilization = {}
        for resource_type, pool in self.resource_pools.items():
            utilization = 1.0 - (pool.available_capacity / pool.total_capacity)
            resource_utilization[resource_type.value] = utilization
        
        # Performance metrics
        performance_metrics = {
            "attention_units_count": len(self.attention_units),
            "average_attention": np.mean(list(attention_distribution.values())) if attention_distribution else 0.0,
            "resource_efficiency": 1.0 - np.mean(list(resource_utilization.values())),
            "allocation_success_rate": self.statistics["attention_allocations"] / max(1, self.statistics["attention_allocations"])
        }
        
        # Anomaly detection
        anomalies = []
        if performance_metrics["average_attention"] < 0.1:
            anomalies.append("Low average attention levels detected")
        if performance_metrics["resource_efficiency"] < 0.3:
            anomalies.append("Poor resource efficiency detected")
        if len(self.attention_units) > self.max_attention_units * 0.9:
            anomalies.append("Approaching maximum attention units limit")
        
        # Recommendations
        recommendations = []
        if performance_metrics["average_attention"] < 0.2:
            recommendations.append("Consider increasing attention threshold")
        if performance_metrics["resource_efficiency"] < 0.5:
            recommendations.append("Optimize resource allocation strategies")
        if len(anomalies) > 0:
            recommendations.append("Investigate detected anomalies")
        
        # Hypergraph snapshot
        hypergraph_snapshot = {
            "atoms_count": len(await self.atomspace.pattern_match({}, limit=1000)),
            "attention_units": len(self.attention_units),
            "active_kernels": len(self.reasoning_engine.cognitive_kernels) if self.reasoning_engine else 0,
            "tensor_shape": self.autonomy_tensor.shape,
            "tensor_stats": {
                "mean": float(self.autonomy_tensor.mean()),
                "std": float(self.autonomy_tensor.std()),
                "max": float(self.autonomy_tensor.max())
            }
        }
        
        report = SelfInspectionReport(
            id=report_id,
            timestamp=datetime.now(timezone.utc),
            cognitive_state=self.current_state,
            attention_distribution=attention_distribution,
            resource_utilization=resource_utilization,
            performance_metrics=performance_metrics,
            anomalies_detected=anomalies,
            recommendations=recommendations,
            hypergraph_snapshot=hypergraph_snapshot
        )
        
        self.inspection_reports.append(report)
        self.statistics["inspection_cycles"] += 1
        
        # Trigger self-modification if needed
        if len(anomalies) > 0:
            await self._trigger_adaptive_modifications(report)
        
        return report
    
    async def _trigger_adaptive_modifications(self, report: SelfInspectionReport):
        """Trigger adaptive modifications based on inspection report"""
        if "Low average attention levels detected" in report.anomalies_detected:
            await self.self_modify("attention_threshold", {"new_threshold": self.attention_threshold * 0.8})
        
        if "Poor resource efficiency detected" in report.anomalies_detected:
            # Increase processing and memory capacity
            await self.self_modify("resource_reallocation", {
                "reallocation": {
                    "processing": self.resource_pools[ResourceType.PROCESSING].total_capacity * 1.2,
                    "memory": self.resource_pools[ResourceType.MEMORY].total_capacity * 1.1
                }
            })
        
        self.statistics["adaptation_events"] += 1
    
    def _periodic_inspection(self):
        """Periodic self-inspection routine (runs in separate thread)"""
        while self.running:
            try:
                # Run inspection every 30 seconds
                import time
                time.sleep(30)
                
                # Create event loop for async inspection
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                # Perform inspection
                report = loop.run_until_complete(self.inspect_system())
                
                # Log report (in real implementation, this would be logged properly)
                print(f"🔍 Self-inspection completed: {report.id}")
                
            except Exception as e:
                print(f"❌ Self-inspection error: {e}")
            finally:
                if 'loop' in locals():
                    loop.close()
    
    def stop(self):
        """Stop the ECAN system"""
        self.running = False
        if self.inspection_thread and self.inspection_thread.is_alive():
            self.inspection_thread.join(timeout=5)
    
    def update_agent_performance(self, agent_id: str, performance_score: float):
        """Update agent performance score for attention modulation"""
        # Clamp performance score to [0, 1]
        performance_score = max(0.0, min(1.0, performance_score))
        
        # Add to history
        self.agent_performance_history[agent_id].append(performance_score)
        
        # Keep only recent history (last 10 scores)
        if len(self.agent_performance_history[agent_id]) > 10:
            self.agent_performance_history[agent_id] = self.agent_performance_history[agent_id][-10:]
        
        # Calculate moving average as reliability score
        scores = self.agent_performance_history[agent_id]
        self.agent_reliability_scores[agent_id] = sum(scores) / len(scores)
    
    def get_agent_reliability(self, agent_id: str) -> float:
        """Get agent reliability score"""
        return self.agent_reliability_scores.get(agent_id, 0.5)  # Default to neutral
    
    async def reward_high_performance(self, agent_id: str, task_completion_quality: float):
        """Reward agent for high performance by increasing attention allocation priority"""
        if task_completion_quality > 0.8:
            # Update performance score
            self.update_agent_performance(agent_id, task_completion_quality)
            
            # Boost attention for this agent's future requests
            reliability = self.get_agent_reliability(agent_id)
            
            # Update attention units associated with this agent
            for au in self.attention_units.values():
                # In a real system, you'd track which agent created which attention units
                # For now, boost attention value based on reliability
                if reliability > 0.7:
                    au.attention_value = min(1.0, au.attention_value * 1.1)
                    au.importance = min(1.0, au.importance * 1.1)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics"""
        return {
            **self.statistics,
            "attention_units_count": len(self.attention_units),
            "active_reports": len(self.inspection_reports),
            "modification_events": len(self.self_modification_events),
            "current_state": self.current_state.value,
            "agent_reliability_scores": dict(self.agent_reliability_scores),
            "average_agent_reliability": sum(self.agent_reliability_scores.values()) / max(1, len(self.agent_reliability_scores)),
            "moses_attention_units": len(self.moses_attention_units),
            "moses_evolution_metrics": self.evolution_metrics,
            "resource_pools": {
                resource_type.value: {
                    "total_capacity": pool.total_capacity,
                    "available_capacity": pool.available_capacity,
                    "utilization": 1.0 - (pool.available_capacity / pool.total_capacity)
                }
                for resource_type, pool in self.resource_pools.items()
            }
        }
    
    async def allocate_moses_attention(self, atom_id: str, moses_entity_type: str,
                                     fitness_score: float = 0.0, complexity: int = 0,
                                     generation: int = 0, requester_id: str = "moses") -> bool:
        """Allocate attention specifically for MOSES entities with fitness-based prioritization"""
        try:
            # Calculate attention priority based on MOSES metrics
            attention_priority = await self._calculate_moses_attention_priority(
                fitness_score, complexity, generation, moses_entity_type
            )
            
            # Check resource allocation based on MOSES policies
            resource_cost = self._calculate_moses_resource_cost(
                fitness_score, complexity, moses_entity_type
            )
            
            if not self.resource_pools[ResourceType.ATTENTION].allocate(requester_id, resource_cost):
                return False
            
            # Create or update MOSES attention unit
            moses_unit_id = f"moses_{moses_entity_type}_{atom_id}"
            
            if moses_unit_id in self.moses_attention_units:
                # Update existing MOSES attention unit
                moses_unit = self.moses_attention_units[moses_unit_id]
                moses_unit.update_from_moses_fitness(fitness_score, complexity, generation)
            else:
                # Create new MOSES attention unit
                moses_unit = MOSESAttentionUnit(
                    id=moses_unit_id,
                    atom_id=atom_id,
                    attention_value=attention_priority,
                    importance=fitness_score,
                    confidence=min(1.0, fitness_score + 0.2),
                    resource_cost=resource_cost,
                    last_accessed=datetime.now(timezone.utc),
                    moses_entity_type=moses_entity_type,
                    fitness_score=fitness_score,
                    generation=generation,
                    complexity=complexity,
                    evolution_priority=attention_priority
                )
                self.moses_attention_units[moses_unit_id] = moses_unit
                
                # Also add to general attention units for compatibility
                self.attention_units[moses_unit_id] = moses_unit
            
            # Update evolution metrics
            await self._update_evolution_metrics(fitness_score, generation)
            
            # Apply resource allocation policies
            await self._apply_moses_resource_policies()
            
            # Update statistics
            self.statistics["attention_allocations"] += 1
            
            return True
            
        except Exception as e:
            print(f"Error allocating MOSES attention: {e}")
            return False
    
    async def _calculate_moses_attention_priority(self, fitness_score: float, complexity: int,
                                                generation: int, entity_type: str) -> float:
        """Calculate attention priority for MOSES entities"""
        base_priority = fitness_score
        
        # Entity type modifiers
        if entity_type == "program":
            base_priority *= 1.0  # Programs get full attention
        elif entity_type == "fitness":
            base_priority *= 0.8  # Fitness records get slightly less
        elif entity_type == "population":
            base_priority *= 1.2  # Populations get bonus attention
        
        # Fitness-based modifiers
        if fitness_score > 0.9:
            base_priority *= 1.5  # Exceptional programs get more attention
        elif fitness_score > 0.7:
            base_priority *= 1.2  # Good programs get bonus attention
        elif fitness_score < 0.3:
            base_priority *= 0.5  # Poor programs get less attention
        
        # Complexity modifier (simpler is better)
        if complexity > 15:
            base_priority *= 0.8  # Penalize overly complex programs
        elif complexity < 5:
            base_priority *= 1.1  # Bonus for simple, effective programs
        
        # Generation recency (newer generations get more attention)
        generation_bonus = min(0.3, generation * 0.02)
        base_priority += generation_bonus
        
        # Apply resource policy modifiers
        policy_modifier = 1.0
        if fitness_score > 0.8:
            policy_modifier += self.moses_resource_policies["high_fitness_bonus"]
        
        if complexity > 10:
            policy_modifier -= self.moses_resource_policies["complexity_penalty"]
        
        return min(1.0, max(0.0, base_priority * policy_modifier))
    
    def _calculate_moses_resource_cost(self, fitness_score: float, complexity: int, entity_type: str) -> float:
        """Calculate resource cost for MOSES attention allocation"""
        base_cost = 1.0
        
        # Higher fitness programs deserve more resources
        if fitness_score > 0.8:
            base_cost *= 1.5
        elif fitness_score < 0.3:
            base_cost *= 0.5
        
        # Complex programs cost more resources
        complexity_multiplier = 1.0 + (complexity * 0.05)
        base_cost *= complexity_multiplier
        
        # Entity type cost modifiers
        if entity_type == "population":
            base_cost *= 2.0  # Populations are more expensive
        elif entity_type == "fitness":
            base_cost *= 0.5  # Fitness records are cheaper
        
        return max(0.1, min(5.0, base_cost))  # Clamp between 0.1 and 5.0
    
    async def _update_evolution_metrics(self, fitness_score: float, generation: int):
        """Update MOSES evolution metrics"""
        if fitness_score > self.evolution_metrics["best_fitness_seen"]:
            self.evolution_metrics["best_fitness_seen"] = fitness_score
        
        if generation > self.evolution_metrics["generation_count"]:
            self.evolution_metrics["generation_count"] = generation
        
        # Calculate attention on best programs
        best_program_attention = sum(
            unit.attention_value for unit in self.moses_attention_units.values()
            if unit.fitness_score > 0.8
        )
        total_moses_attention = sum(
            unit.attention_value for unit in self.moses_attention_units.values()
        )
        
        if total_moses_attention > 0:
            self.evolution_metrics["attention_on_best_programs"] = (
                best_program_attention / total_moses_attention
            )
    
    async def _apply_moses_resource_policies(self):
        """Apply resource allocation policies for MOSES evolution"""
        try:
            # Policy 1: Boost high-fitness programs
            high_fitness_units = [
                unit for unit in self.moses_attention_units.values()
                if unit.fitness_score > 0.8
            ]
            for unit in high_fitness_units:
                unit.resource_cost *= (1.0 + self.moses_resource_policies["high_fitness_bonus"])
            
            # Policy 2: Apply generation decay
            current_generation = self.evolution_metrics["generation_count"]
            for unit in self.moses_attention_units.values():
                generation_age = current_generation - unit.generation
                if generation_age > 0:
                    decay_factor = 1.0 - (generation_age * self.moses_resource_policies["generation_decay"])
                    unit.attention_value *= max(0.1, decay_factor)
            
            # Policy 3: Maintain diversity by preventing attention monopoly
            attention_values = [unit.attention_value for unit in self.moses_attention_units.values()]
            if attention_values:
                max_attention = max(attention_values)
                avg_attention = sum(attention_values) / len(attention_values)
                
                # If there's too much concentration, redistribute
                if max_attention > avg_attention * 3:
                    diversity_penalty = self.moses_resource_policies["diversity_bonus"]
                    for unit in self.moses_attention_units.values():
                        if unit.attention_value > avg_attention * 2:
                            unit.attention_value *= (1.0 - diversity_penalty)
                        else:
                            unit.attention_value *= (1.0 + diversity_penalty * 0.5)
            
        except Exception as e:
            print(f"Error applying MOSES resource policies: {e}")
    
    async def get_moses_attention_visualization(self) -> Dict[str, Any]:
        """Get visualization data for MOSES attention allocation"""
        try:
            visualization_data = {
                "attention_distribution": {},
                "fitness_vs_attention": [],
                "generation_attention": {},
                "entity_type_breakdown": {},
                "resource_allocation": {},
                "evolution_timeline": []
            }
            
            # Attention distribution by entity type
            for entity_type in ["program", "population", "fitness"]:
                units = [u for u in self.moses_attention_units.values() 
                        if u.moses_entity_type == entity_type]
                total_attention = sum(u.attention_value for u in units)
                visualization_data["entity_type_breakdown"][entity_type] = {
                    "count": len(units),
                    "total_attention": total_attention,
                    "average_attention": total_attention / len(units) if units else 0
                }
            
            # Fitness vs attention correlation
            for unit in self.moses_attention_units.values():
                visualization_data["fitness_vs_attention"].append({
                    "fitness": unit.fitness_score,
                    "attention": unit.attention_value,
                    "complexity": unit.complexity,
                    "generation": unit.generation,
                    "entity_type": unit.moses_entity_type
                })
            
            # Generation-based attention
            generation_attention = defaultdict(float)
            for unit in self.moses_attention_units.values():
                generation_attention[unit.generation] += unit.attention_value
            visualization_data["generation_attention"] = dict(generation_attention)
            
            # Resource allocation by type
            for resource_type, pool in self.resource_pools.items():
                visualization_data["resource_allocation"][resource_type.value] = {
                    "total": pool.total_capacity,
                    "available": pool.available_capacity,
                    "utilization": 1.0 - (pool.available_capacity / pool.total_capacity)
                }
            
            # Evolution metrics timeline
            visualization_data["evolution_timeline"] = self.moses_fitness_history[-20:]  # Last 20 entries
            
            return visualization_data
            
        except Exception as e:
            print(f"Error generating MOSES attention visualization: {e}")
            return {}
    
    async def update_from_moses_generation(self, generation_stats: Dict[str, Any], 
                                         programs: List[Any] = None):
        """Update ECAN attention based on MOSES generation results"""
        try:
            # Record fitness history for visualization
            fitness_record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "generation": generation_stats.get("generation", 0),
                "best_fitness": generation_stats.get("best_fitness", 0.0),
                "average_fitness": generation_stats.get("average_fitness", 0.0),
                "population_size": generation_stats.get("population_size", 0)
            }
            self.moses_fitness_history.append(fitness_record)
            
            # Keep only recent history
            if len(self.moses_fitness_history) > 100:
                self.moses_fitness_history = self.moses_fitness_history[-100:]
            
            # Update attention for programs if provided
            if programs:
                for program in programs:
                    await self.allocate_moses_attention(
                        atom_id=program.id,
                        moses_entity_type="program",
                        fitness_score=program.fitness,
                        complexity=program.complexity,
                        generation=program.generation,
                        requester_id="moses_evolution"
                    )
            
            # Update global evolution metrics
            await self._update_evolution_metrics(
                generation_stats.get("best_fitness", 0.0),
                generation_stats.get("generation", 0)
            )
            
            # Trigger adaptive modifications if needed
            if generation_stats.get("best_fitness", 0.0) > 0.9:
                await self.self_modify("moses_high_performance", {
                    "fitness_threshold": generation_stats.get("best_fitness", 0.0),
                    "generation": generation_stats.get("generation", 0)
                })
            
        except Exception as e:
            print(f"Error updating from MOSES generation: {e}")