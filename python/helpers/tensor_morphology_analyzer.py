"""
Tensor Morphology Analysis System

Analyzes and catalogs tensor field morphology for all orchestration elements
including tasks, agents, skills, priorities, attention, and negotiation states.
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
import math

from .distributed_orchestrator import DistributedOrchestrator, AtomicSubtask, AgentCapability, TaskPriority
from .ecan_attention import ECANAttentionSystem, AttentionType, ResourceType
from .atomspace import AtomSpace


class TensorType(Enum):
    """Types of tensors in the cognitive system"""
    TASK_AGENT_PRIORITY = "task_agent_priority"           # T[n_tasks, n_agents, p_levels]
    ATTENTION_ALLOCATION = "attention_allocation"          # T[atoms, agents, attention_types]
    SKILL_COMPATIBILITY = "skill_compatibility"           # T[skills, agents, competence_levels]
    RESOURCE_UTILIZATION = "resource_utilization"         # T[resources, agents, time_steps]
    DEPENDENCY_GRAPH = "dependency_graph"                 # T[tasks, tasks, dependency_types]
    NEGOTIATION_STATE = "negotiation_state"               # T[agents, tasks, negotiation_phases]
    COGNITIVE_STATE = "cognitive_state"                   # T[agents, cognitive_dimensions, time]
    PERFORMANCE_METRICS = "performance_metrics"           # T[agents, metrics, evaluation_periods]


class MorphologyMetric(Enum):
    """Metrics for analyzing tensor morphology"""
    SPARSITY = "sparsity"                    # Fraction of zero elements
    RANK = "rank"                            # Tensor rank/dimensionality
    ENTROPY = "entropy"                      # Information entropy
    CLUSTERING = "clustering"                # Cluster coefficient
    CONNECTIVITY = "connectivity"            # Connection density
    SYMMETRY = "symmetry"                    # Symmetry measures
    STABILITY = "stability"                  # Temporal stability
    COMPLEXITY = "complexity"                # Structural complexity


@dataclass
class TensorShape:
    """Shape information for a tensor"""
    dimensions: Tuple[int, ...]
    element_count: int
    memory_size: int  # in bytes
    sparsity_ratio: float
    data_type: str
    
    def __post_init__(self):
        """Calculate derived properties"""
        self.element_count = math.prod(self.dimensions)
        # Estimate memory size (assuming float64)
        self.memory_size = self.element_count * 8


@dataclass
class MorphologyAnalysis:
    """Analysis results for tensor morphology"""
    tensor_id: str
    tensor_type: TensorType
    shape: TensorShape
    metrics: Dict[MorphologyMetric, float]
    patterns: List[str]  # Identified patterns
    anomalies: List[str]  # Detected anomalies
    recommendations: List[str]  # Optimization recommendations
    analysis_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class TensorRegistry:
    """Registry entry for a tensor"""
    id: str
    name: str
    tensor_type: TensorType
    shape: TensorShape
    creation_time: datetime
    last_update: datetime
    access_count: int = 0
    modification_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class TensorMorphologyAnalyzer:
    """
    System for analyzing tensor field morphology across cognitive architectures
    """
    
    def __init__(self, orchestrator: Optional[DistributedOrchestrator] = None,
                 ecan_system: Optional[ECANAttentionSystem] = None,
                 atomspace: Optional[AtomSpace] = None):
        self.orchestrator = orchestrator
        self.ecan_system = ecan_system
        self.atomspace = atomspace
        
        # Tensor registry and analysis storage
        self.tensor_registry: Dict[str, TensorRegistry] = {}
        self.tensor_data: Dict[str, np.ndarray] = {}
        self.morphology_analyses: Dict[str, List[MorphologyAnalysis]] = defaultdict(list)
        
        # Analysis configuration
        self.analysis_frequency = 60.0  # seconds
        self.pattern_threshold = 0.7
        self.anomaly_threshold = 0.3
        
        # Running state
        self.running = False
        self.analysis_thread: Optional[threading.Thread] = None
        
        # Statistics
        self.statistics = {
            "total_tensors": 0,
            "total_analyses": 0,
            "patterns_discovered": 0,
            "anomalies_detected": 0,
            "optimizations_suggested": 0
        }
        
        # Initialize built-in analysis functions
        self.morphology_analyzers = {
            MorphologyMetric.SPARSITY: self._analyze_sparsity,
            MorphologyMetric.RANK: self._analyze_rank,
            MorphologyMetric.ENTROPY: self._analyze_entropy,
            MorphologyMetric.CLUSTERING: self._analyze_clustering,
            MorphologyMetric.CONNECTIVITY: self._analyze_connectivity,
            MorphologyMetric.SYMMETRY: self._analyze_symmetry,
            MorphologyMetric.STABILITY: self._analyze_stability,
            MorphologyMetric.COMPLEXITY: self._analyze_complexity
        }
    
    async def initialize_tensor_analysis(self):
        """Initialize the tensor morphology analysis system"""
        print("🔍 Initializing Tensor Morphology Analysis System...")
        
        # Discover and register existing tensors
        await self._discover_system_tensors()
        
        # Start periodic analysis
        self.running = True
        self.analysis_thread = threading.Thread(target=self._periodic_analysis, daemon=True)
        self.analysis_thread.start()
        
        print(f"✅ Tensor Analysis initialized with {len(self.tensor_registry)} tensors")
    
    async def register_tensor(self, name: str, tensor_type: TensorType, 
                            tensor_data: np.ndarray, metadata: Dict[str, Any] = None) -> str:
        """Register a tensor for morphology analysis"""
        
        tensor_id = f"tensor_{uuid.uuid4().hex[:8]}"
        
        # Calculate shape information
        shape = TensorShape(
            dimensions=tensor_data.shape,
            element_count=tensor_data.size,
            memory_size=tensor_data.nbytes,
            sparsity_ratio=self._calculate_sparsity(tensor_data),
            data_type=str(tensor_data.dtype)
        )
        
        # Create registry entry
        registry_entry = TensorRegistry(
            id=tensor_id,
            name=name,
            tensor_type=tensor_type,
            shape=shape,
            creation_time=datetime.now(timezone.utc),
            last_update=datetime.now(timezone.utc),
            metadata=metadata or {}
        )
        
        # Store tensor and registry
        self.tensor_registry[tensor_id] = registry_entry
        self.tensor_data[tensor_id] = tensor_data.copy()
        
        self.statistics["total_tensors"] += 1
        
        # Perform initial analysis
        analysis = await self._analyze_tensor_morphology(tensor_id)
        self.morphology_analyses[tensor_id].append(analysis)
        
        return tensor_id
    
    async def update_tensor(self, tensor_id: str, new_data: np.ndarray):
        """Update a registered tensor with new data"""
        if tensor_id not in self.tensor_registry:
            raise ValueError(f"Tensor {tensor_id} not found in registry")
        
        # Update tensor data
        self.tensor_data[tensor_id] = new_data.copy()
        
        # Update registry
        registry = self.tensor_registry[tensor_id]
        registry.last_update = datetime.now(timezone.utc)
        registry.modification_count += 1
        
        # Update shape if changed
        if new_data.shape != registry.shape.dimensions:
            registry.shape = TensorShape(
                dimensions=new_data.shape,
                element_count=new_data.size,
                memory_size=new_data.nbytes,
                sparsity_ratio=self._calculate_sparsity(new_data),
                data_type=str(new_data.dtype)
            )
        
        # Trigger immediate analysis for significant changes
        await self._analyze_tensor_morphology(tensor_id)
    
    async def get_tensor_analysis(self, tensor_id: str) -> Optional[MorphologyAnalysis]:
        """Get the latest morphology analysis for a tensor"""
        if tensor_id in self.morphology_analyses and self.morphology_analyses[tensor_id]:
            return self.morphology_analyses[tensor_id][-1]
        return None
    
    async def get_system_morphology_report(self) -> Dict[str, Any]:
        """Generate comprehensive morphology report for the entire system"""
        
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "system_overview": {
                "total_tensors": len(self.tensor_registry),
                "total_elements": sum(reg.shape.element_count for reg in self.tensor_registry.values()),
                "total_memory": sum(reg.shape.memory_size for reg in self.tensor_registry.values()),
                "tensor_types": {}
            },
            "morphology_summary": {},
            "global_patterns": [],
            "system_recommendations": [],
            "statistics": self.statistics.copy()
        }
        
        # Analyze by tensor type
        type_groups = defaultdict(list)
        for tensor_id, registry in self.tensor_registry.items():
            type_groups[registry.tensor_type].append(tensor_id)
        
        # Generate summaries for each type
        for tensor_type, tensor_ids in type_groups.items():
            type_analysis = await self._analyze_tensor_type_group(tensor_type, tensor_ids)
            report["system_overview"]["tensor_types"][tensor_type.value] = {
                "count": len(tensor_ids),
                "total_elements": sum(self.tensor_registry[tid].shape.element_count for tid in tensor_ids),
                "average_sparsity": type_analysis.get("average_sparsity", 0.0)
            }
            report["morphology_summary"][tensor_type.value] = type_analysis
        
        # Identify global patterns
        report["global_patterns"] = await self._identify_global_patterns()
        
        # Generate system-wide recommendations
        report["system_recommendations"] = await self._generate_system_recommendations()
        
        return report
    
    async def _discover_system_tensors(self):
        """Discover tensors from connected systems"""
        
        # Discover orchestration tensors
        if self.orchestrator:
            await self._discover_orchestration_tensors()
        
        # Discover ECAN tensors
        if self.ecan_system:
            await self._discover_ecan_tensors()
        
        # Discover AtomSpace tensors
        if self.atomspace:
            await self._discover_atomspace_tensors()
    
    async def _discover_orchestration_tensors(self):
        """Discover tensors from the distributed orchestrator"""
        
        # Task-Agent-Priority tensor
        try:
            task_agent_priority_tensor = self.orchestrator.get_task_assignment_tensor()
            if task_agent_priority_tensor is not None:
                await self.register_tensor(
                    "task_agent_priority",
                    TensorType.TASK_AGENT_PRIORITY,
                    task_agent_priority_tensor,
                    {"source": "distributed_orchestrator", "description": "Task assignment probabilities"}
                )
        except:
            # Create a sample tensor if not available
            sample_tensor = np.random.rand(10, 5, 4)  # 10 tasks, 5 agents, 4 priority levels
            await self.register_tensor(
                "task_agent_priority_sample",
                TensorType.TASK_AGENT_PRIORITY,
                sample_tensor,
                {"source": "sample_data", "description": "Sample task assignment tensor"}
            )
        
        # Skill compatibility tensor
        if hasattr(self.orchestrator, '_registered_agents'):
            agents = list(self.orchestrator._registered_agents.keys())
            if agents:
                # Extract skill data
                all_skills = set()
                for agent_capability in self.orchestrator._registered_agents.values():
                    all_skills.update(agent_capability.skills)
                
                if all_skills:
                    skills = list(all_skills)
                    skill_tensor = np.zeros((len(skills), len(agents), 5))  # 5 competence levels
                    
                    # Fill with sample competence data
                    for i, agent_id in enumerate(agents):
                        agent_skills = self.orchestrator._registered_agents[agent_id].skills
                        for j, skill in enumerate(skills):
                            if skill in agent_skills:
                                # Random competence level for demonstration
                                competence_level = np.random.randint(0, 5)
                                skill_tensor[j, i, competence_level] = 1.0
                    
                    await self.register_tensor(
                        "skill_compatibility",
                        TensorType.SKILL_COMPATIBILITY,
                        skill_tensor,
                        {"skills": skills, "agents": agents}
                    )
        
        # Dependency graph tensor
        if hasattr(self.orchestrator, '_subtasks'):
            subtasks = list(self.orchestrator._subtasks.keys())
            if len(subtasks) > 1:
                dep_tensor = np.zeros((len(subtasks), len(subtasks), 3))  # 3 dependency types
                
                # Fill dependency information
                for i, task_id in enumerate(subtasks):
                    task = self.orchestrator._subtasks[task_id]
                    for dep_id in task.dependencies:
                        if dep_id in subtasks:
                            j = subtasks.index(dep_id)
                            dep_tensor[i, j, 0] = 1.0  # Direct dependency
                
                await self.register_tensor(
                    "dependency_graph",
                    TensorType.DEPENDENCY_GRAPH,
                    dep_tensor,
                    {"subtasks": subtasks}
                )
    
    async def _discover_ecan_tensors(self):
        """Discover tensors from the ECAN attention system"""
        
        # Attention allocation tensor
        try:
            attention_tensor = self.ecan_system.get_autonomy_tensor()
            await self.register_tensor(
                "ecan_autonomy",
                TensorType.ATTENTION_ALLOCATION,
                attention_tensor,
                {"source": "ecan_system", "description": "ECAN autonomy metrics"}
            )
        except:
            # Create sample attention tensor
            sample_tensor = np.random.rand(100, 5, 5)  # 100 atoms, 5 agents, 5 attention types
            await self.register_tensor(
                "attention_allocation_sample",
                TensorType.ATTENTION_ALLOCATION,
                sample_tensor,
                {"source": "sample_data"}
            )
        
        # Resource utilization tensor
        if hasattr(self.ecan_system, 'resource_pools'):
            resources = list(ResourceType)
            # Sample agents (in real implementation, get from actual system)
            agents = [f"agent_{i}" for i in range(5)]
            time_steps = 24  # Hours
            
            resource_tensor = np.random.rand(len(resources), len(agents), time_steps)
            
            await self.register_tensor(
                "resource_utilization",
                TensorType.RESOURCE_UTILIZATION,
                resource_tensor,
                {"resources": [r.value for r in resources], "agents": agents}
            )
    
    async def _discover_atomspace_tensors(self):
        """Discover tensors from the AtomSpace"""
        
        # For now, create a sample cognitive state tensor
        agents = [f"agent_{i}" for i in range(5)]
        cognitive_dims = ["perception", "reasoning", "memory", "attention", "action"]
        time_steps = 100
        
        cognitive_tensor = np.random.rand(len(agents), len(cognitive_dims), time_steps)
        
        await self.register_tensor(
            "cognitive_state",
            TensorType.COGNITIVE_STATE,
            cognitive_tensor,
            {"agents": agents, "dimensions": cognitive_dims}
        )
    
    async def _analyze_tensor_morphology(self, tensor_id: str) -> MorphologyAnalysis:
        """Perform comprehensive morphology analysis on a tensor"""
        
        if tensor_id not in self.tensor_data:
            raise ValueError(f"Tensor {tensor_id} not found")
        
        tensor = self.tensor_data[tensor_id]
        registry = self.tensor_registry[tensor_id]
        
        # Calculate all morphology metrics
        metrics = {}
        for metric_type, analyzer_func in self.morphology_analyzers.items():
            try:
                metrics[metric_type] = analyzer_func(tensor)
            except Exception as e:
                print(f"⚠️ Failed to analyze {metric_type} for tensor {tensor_id}: {e}")
                metrics[metric_type] = 0.0
        
        # Identify patterns
        patterns = self._identify_tensor_patterns(tensor, metrics)
        
        # Detect anomalies
        anomalies = self._detect_tensor_anomalies(tensor, metrics)
        
        # Generate recommendations
        recommendations = self._generate_tensor_recommendations(tensor, metrics, patterns, anomalies)
        
        # Create analysis result
        analysis = MorphologyAnalysis(
            tensor_id=tensor_id,
            tensor_type=registry.tensor_type,
            shape=registry.shape,
            metrics=metrics,
            patterns=patterns,
            anomalies=anomalies,
            recommendations=recommendations
        )
        
        self.statistics["total_analyses"] += 1
        if patterns:
            self.statistics["patterns_discovered"] += len(patterns)
        if anomalies:
            self.statistics["anomalies_detected"] += len(anomalies)
        if recommendations:
            self.statistics["optimizations_suggested"] += len(recommendations)
        
        return analysis
    
    # Morphology analysis functions
    def _analyze_sparsity(self, tensor: np.ndarray) -> float:
        """Analyze tensor sparsity (fraction of zero or near-zero elements)"""
        return self._calculate_sparsity(tensor)
    
    def _calculate_sparsity(self, tensor: np.ndarray) -> float:
        """Calculate sparsity ratio"""
        if tensor.size == 0:
            return 0.0
        
        # Count near-zero elements (threshold for numerical precision)
        threshold = 1e-10
        near_zero_count = np.sum(np.abs(tensor) < threshold)
        
        return near_zero_count / tensor.size
    
    def _analyze_rank(self, tensor: np.ndarray) -> float:
        """Analyze tensor rank (effective dimensionality)"""
        if tensor.ndim == 2:
            # For 2D tensors, use matrix rank
            return float(np.linalg.matrix_rank(tensor))
        else:
            # For higher-order tensors, estimate rank through unfolding
            # Unfold along the first mode
            unfolded = tensor.reshape(tensor.shape[0], -1)
            rank = np.linalg.matrix_rank(unfolded)
            # Normalize by maximum possible rank
            max_rank = min(unfolded.shape)
            return rank / max_rank if max_rank > 0 else 0.0
    
    def _analyze_entropy(self, tensor: np.ndarray) -> float:
        """Analyze information entropy of tensor"""
        # Flatten tensor and compute probability distribution
        flat_tensor = tensor.flatten()
        
        # Discretize values into bins
        n_bins = min(100, len(flat_tensor) // 10)
        if n_bins < 2:
            return 0.0
        
        hist, _ = np.histogram(flat_tensor, bins=n_bins)
        
        # Calculate probabilities
        probabilities = hist / np.sum(hist)
        
        # Remove zero probabilities
        probabilities = probabilities[probabilities > 0]
        
        # Calculate entropy
        entropy = -np.sum(probabilities * np.log2(probabilities))
        
        # Normalize by maximum possible entropy
        max_entropy = np.log2(n_bins)
        
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def _analyze_clustering(self, tensor: np.ndarray) -> float:
        """Analyze clustering coefficient of tensor structure"""
        if tensor.ndim != 2:
            # For higher-order tensors, flatten to 2D
            tensor = tensor.reshape(tensor.shape[0], -1)
        
        # Threshold tensor to create binary adjacency matrix
        threshold = np.mean(tensor) + np.std(tensor)
        adj_matrix = (tensor > threshold).astype(float)
        
        # Calculate clustering coefficient
        n = adj_matrix.shape[0]
        if n < 3:
            return 0.0
        
        clustering_sum = 0.0
        for i in range(n):
            neighbors = np.where(adj_matrix[i] > 0)[0]
            if len(neighbors) < 2:
                continue
            
            # Count triangles
            triangle_count = 0
            for j in neighbors:
                for k in neighbors:
                    if j < k and adj_matrix[j, k] > 0:
                        triangle_count += 1
            
            # Possible triangles
            possible_triangles = len(neighbors) * (len(neighbors) - 1) // 2
            
            if possible_triangles > 0:
                clustering_sum += triangle_count / possible_triangles
        
        return clustering_sum / n
    
    def _analyze_connectivity(self, tensor: np.ndarray) -> float:
        """Analyze connectivity density of tensor"""
        if tensor.ndim != 2:
            # For higher-order tensors, use first two dimensions
            tensor = tensor.reshape(tensor.shape[0], -1)
        
        # Calculate connection density
        threshold = np.mean(tensor)
        connections = np.sum(tensor > threshold)
        total_possible = tensor.size
        
        return connections / total_possible if total_possible > 0 else 0.0
    
    def _analyze_symmetry(self, tensor: np.ndarray) -> float:
        """Analyze symmetry properties of tensor"""
        if tensor.ndim == 2 and tensor.shape[0] == tensor.shape[1]:
            # For square matrices, check symmetry
            symmetry_error = np.mean(np.abs(tensor - tensor.T))
            max_value = np.max(np.abs(tensor))
            return 1.0 - (symmetry_error / max_value) if max_value > 0 else 1.0
        
        elif tensor.ndim > 2:
            # For higher-order tensors, check symmetry across dimensions
            symmetry_scores = []
            
            # Check pairwise dimension symmetries
            for dim1 in range(tensor.ndim):
                for dim2 in range(dim1 + 1, tensor.ndim):
                    # Transpose tensor along these dimensions
                    axes = list(range(tensor.ndim))
                    axes[dim1], axes[dim2] = axes[dim2], axes[dim1]
                    transposed = np.transpose(tensor, axes)
                    
                    # Compare shapes
                    if tensor.shape == transposed.shape:
                        diff = np.mean(np.abs(tensor - transposed))
                        max_val = np.max(np.abs(tensor))
                        score = 1.0 - (diff / max_val) if max_val > 0 else 1.0
                        symmetry_scores.append(score)
            
            return np.mean(symmetry_scores) if symmetry_scores else 0.0
        
        return 0.0  # No meaningful symmetry for non-square 2D tensors
    
    def _analyze_stability(self, tensor: np.ndarray) -> float:
        """Analyze temporal stability (requires historical data)"""
        # For this implementation, analyze structural stability
        # by examining variance across different dimensions
        
        if tensor.size == 0:
            return 0.0
        
        # Calculate coefficient of variation across each dimension
        stabilities = []
        
        for dim in range(tensor.ndim):
            # Collapse all other dimensions
            collapsed = np.mean(tensor, axis=tuple(i for i in range(tensor.ndim) if i != dim))
            
            if len(collapsed) > 1:
                mean_val = np.mean(collapsed)
                std_val = np.std(collapsed)
                cv = std_val / mean_val if mean_val > 0 else float('inf')
                stability = 1.0 / (1.0 + cv)  # Higher stability = lower coefficient of variation
                stabilities.append(stability)
        
        return np.mean(stabilities) if stabilities else 0.5
    
    def _analyze_complexity(self, tensor: np.ndarray) -> float:
        """Analyze structural complexity of tensor"""
        if tensor.size == 0:
            return 0.0
        
        # Combine multiple complexity measures
        complexity_measures = []
        
        # 1. Entropy-based complexity
        entropy = self._analyze_entropy(tensor)
        complexity_measures.append(entropy)
        
        # 2. Variance-based complexity
        variance = np.var(tensor)
        mean_abs = np.mean(np.abs(tensor))
        normalized_variance = variance / (mean_abs ** 2) if mean_abs > 0 else 0
        complexity_measures.append(min(1.0, normalized_variance))
        
        # 3. Rank-based complexity
        rank_ratio = self._analyze_rank(tensor)
        complexity_measures.append(rank_ratio)
        
        # 4. Sparsity-inverse complexity
        sparsity = self._analyze_sparsity(tensor)
        density_complexity = 1.0 - sparsity
        complexity_measures.append(density_complexity)
        
        return np.mean(complexity_measures)
    
    def _identify_tensor_patterns(self, tensor: np.ndarray, metrics: Dict[MorphologyMetric, float]) -> List[str]:
        """Identify patterns in tensor structure"""
        patterns = []
        
        # High sparsity pattern
        if metrics.get(MorphologyMetric.SPARSITY, 0) > 0.8:
            patterns.append("High sparsity - efficient compression possible")
        
        # Low rank pattern
        if metrics.get(MorphologyMetric.RANK, 1.0) < 0.3:
            patterns.append("Low rank structure - dimensionality reduction applicable")
        
        # High symmetry pattern
        if metrics.get(MorphologyMetric.SYMMETRY, 0) > 0.9:
            patterns.append("High symmetry - structural regularity present")
        
        # High clustering pattern
        if metrics.get(MorphologyMetric.CLUSTERING, 0) > 0.7:
            patterns.append("Strong clustering - modular structure detected")
        
        # High entropy pattern
        if metrics.get(MorphologyMetric.ENTROPY, 0) > 0.9:
            patterns.append("High entropy - rich information content")
        
        # Block structure detection (for 2D tensors)
        if tensor.ndim == 2 and tensor.shape[0] > 4 and tensor.shape[1] > 4:
            if self._detect_block_structure(tensor):
                patterns.append("Block structure - hierarchical organization")
        
        return patterns
    
    def _detect_block_structure(self, tensor: np.ndarray) -> bool:
        """Detect block structure in 2D tensor"""
        # Simple block detection using variance analysis
        n_blocks = 4
        rows_per_block = tensor.shape[0] // n_blocks
        cols_per_block = tensor.shape[1] // n_blocks
        
        if rows_per_block < 2 or cols_per_block < 2:
            return False
        
        block_variances = []
        
        for i in range(n_blocks):
            for j in range(n_blocks):
                r_start = i * rows_per_block
                r_end = min((i + 1) * rows_per_block, tensor.shape[0])
                c_start = j * cols_per_block
                c_end = min((j + 1) * cols_per_block, tensor.shape[1])
                
                block = tensor[r_start:r_end, c_start:c_end]
                block_variances.append(np.var(block))
        
        # If within-block variance is significantly less than overall variance
        within_block_var = np.mean(block_variances)
        overall_var = np.var(tensor)
        
        return within_block_var < 0.5 * overall_var
    
    def _detect_tensor_anomalies(self, tensor: np.ndarray, metrics: Dict[MorphologyMetric, float]) -> List[str]:
        """Detect anomalies in tensor structure"""
        anomalies = []
        
        # Extreme sparsity
        sparsity = metrics.get(MorphologyMetric.SPARSITY, 0)
        if sparsity > 0.95:
            anomalies.append("Extreme sparsity - possible data quality issue")
        elif sparsity < 0.01:
            anomalies.append("No sparsity - possible initialization issue")
        
        # Numerical instability
        if np.any(np.isnan(tensor)) or np.any(np.isinf(tensor)):
            anomalies.append("Numerical instability - NaN or Inf values detected")
        
        # Extreme values
        max_val = np.max(np.abs(tensor))
        if max_val > 1e6:
            anomalies.append("Extreme values - possible overflow or scaling issue")
        
        # Zero variance (constant tensor)
        if np.var(tensor) < 1e-10:
            anomalies.append("Zero variance - tensor is effectively constant")
        
        # Rank deficiency
        if metrics.get(MorphologyMetric.RANK, 1.0) < 0.1:
            anomalies.append("Severe rank deficiency - loss of information")
        
        return anomalies
    
    def _generate_tensor_recommendations(self, tensor: np.ndarray, metrics: Dict[MorphologyMetric, float],
                                       patterns: List[str], anomalies: List[str]) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        # Sparsity-based recommendations
        sparsity = metrics.get(MorphologyMetric.SPARSITY, 0)
        if sparsity > 0.5:
            recommendations.append("Consider sparse tensor representation for memory efficiency")
        
        # Rank-based recommendations
        rank_ratio = metrics.get(MorphologyMetric.RANK, 1.0)
        if rank_ratio < 0.5:
            recommendations.append("Apply low-rank decomposition for compression")
        
        # Clustering recommendations
        clustering = metrics.get(MorphologyMetric.CLUSTERING, 0)
        if clustering > 0.6:
            recommendations.append("Exploit modular structure for parallel processing")
        
        # Symmetry recommendations
        symmetry = metrics.get(MorphologyMetric.SYMMETRY, 0)
        if symmetry > 0.8:
            recommendations.append("Leverage symmetry for computational optimization")
        
        # Anomaly-based recommendations
        if anomalies:
            recommendations.append("Address detected anomalies before optimization")
        
        # Complexity-based recommendations
        complexity = metrics.get(MorphologyMetric.COMPLEXITY, 0)
        if complexity > 0.9:
            recommendations.append("High complexity - consider regularization techniques")
        elif complexity < 0.1:
            recommendations.append("Low complexity - may need feature enrichment")
        
        return recommendations
    
    async def _analyze_tensor_type_group(self, tensor_type: TensorType, tensor_ids: List[str]) -> Dict[str, Any]:
        """Analyze a group of tensors of the same type"""
        
        if not tensor_ids:
            return {}
        
        # Collect metrics from all tensors of this type
        all_metrics = defaultdict(list)
        all_shapes = []
        
        for tensor_id in tensor_ids:
            if tensor_id in self.morphology_analyses:
                latest_analysis = self.morphology_analyses[tensor_id][-1]
                for metric, value in latest_analysis.metrics.items():
                    all_metrics[metric].append(value)
                all_shapes.append(latest_analysis.shape)
        
        # Calculate aggregate statistics
        analysis = {
            "tensor_count": len(tensor_ids),
            "average_metrics": {},
            "metric_ranges": {},
            "common_patterns": [],
            "type_specific_insights": []
        }
        
        for metric, values in all_metrics.items():
            if values:
                analysis["average_metrics"][metric.value] = np.mean(values)
                analysis["metric_ranges"][metric.value] = {
                    "min": np.min(values),
                    "max": np.max(values),
                    "std": np.std(values)
                }
        
        # Identify common patterns
        pattern_counts = defaultdict(int)
        for tensor_id in tensor_ids:
            if tensor_id in self.morphology_analyses:
                latest_analysis = self.morphology_analyses[tensor_id][-1]
                for pattern in latest_analysis.patterns:
                    pattern_counts[pattern] += 1
        
        # Patterns that appear in > 50% of tensors
        common_threshold = len(tensor_ids) * 0.5
        analysis["common_patterns"] = [
            pattern for pattern, count in pattern_counts.items()
            if count > common_threshold
        ]
        
        return analysis
    
    async def _identify_global_patterns(self) -> List[str]:
        """Identify system-wide patterns across all tensors"""
        global_patterns = []
        
        # Cross-tensor correlation patterns
        if len(self.tensor_registry) > 1:
            # Analyze sparsity correlation
            sparsity_values = []
            for tensor_id in self.tensor_registry:
                if tensor_id in self.morphology_analyses:
                    latest = self.morphology_analyses[tensor_id][-1]
                    sparsity = latest.metrics.get(MorphologyMetric.SPARSITY, 0)
                    sparsity_values.append(sparsity)
            
            if len(sparsity_values) > 1:
                sparsity_std = np.std(sparsity_values)
                if sparsity_std < 0.1:
                    global_patterns.append("Consistent sparsity across system")
                elif sparsity_std > 0.4:
                    global_patterns.append("High sparsity variation - heterogeneous tensor types")
        
        # Memory usage patterns
        total_memory = sum(reg.shape.memory_size for reg in self.tensor_registry.values())
        if total_memory > 100_000_000:  # > 100MB
            global_patterns.append("High memory usage - consider tensor optimization")
        
        # Tensor type distribution
        type_counts = defaultdict(int)
        for registry in self.tensor_registry.values():
            type_counts[registry.tensor_type] += 1
        
        if len(type_counts) > 1:
            max_type_count = max(type_counts.values())
            total_tensors = sum(type_counts.values())
            if max_type_count / total_tensors > 0.7:
                dominant_type = max(type_counts, key=type_counts.get)
                global_patterns.append(f"System dominated by {dominant_type.value} tensors")
        
        return global_patterns
    
    async def _generate_system_recommendations(self) -> List[str]:
        """Generate system-wide optimization recommendations"""
        recommendations = []
        
        # Memory optimization
        total_memory = sum(reg.shape.memory_size for reg in self.tensor_registry.values())
        if total_memory > 50_000_000:  # > 50MB
            recommendations.append("Implement tensor compression for memory optimization")
        
        # Computational optimization
        high_complexity_count = 0
        for tensor_id in self.tensor_registry:
            if tensor_id in self.morphology_analyses:
                latest = self.morphology_analyses[tensor_id][-1]
                complexity = latest.metrics.get(MorphologyMetric.COMPLEXITY, 0)
                if complexity > 0.8:
                    high_complexity_count += 1
        
        if high_complexity_count > len(self.tensor_registry) * 0.5:
            recommendations.append("High system complexity - consider hierarchical processing")
        
        # Sparsity optimization
        high_sparsity_count = 0
        for tensor_id in self.tensor_registry:
            if tensor_id in self.morphology_analyses:
                latest = self.morphology_analyses[tensor_id][-1]
                sparsity = latest.metrics.get(MorphologyMetric.SPARSITY, 0)
                if sparsity > 0.6:
                    high_sparsity_count += 1
        
        if high_sparsity_count > len(self.tensor_registry) * 0.5:
            recommendations.append("System-wide sparse tensor implementation recommended")
        
        return recommendations
    
    def _periodic_analysis(self):
        """Periodic tensor analysis in separate thread"""
        while self.running:
            try:
                # Analyze all registered tensors
                for tensor_id in list(self.tensor_registry.keys()):
                    asyncio.run(self._analyze_tensor_morphology(tensor_id))
                
                # Sleep for configured interval
                import time
                time.sleep(self.analysis_frequency)
                
            except Exception as e:
                print(f"❌ Periodic analysis error: {e}")
                import time
                time.sleep(10)  # Short sleep on error
    
    def get_analysis_statistics(self) -> Dict[str, Any]:
        """Get analysis system statistics"""
        return {
            **self.statistics,
            "registered_tensors": len(self.tensor_registry),
            "total_tensor_memory": sum(reg.shape.memory_size for reg in self.tensor_registry.values()),
            "analysis_history_size": sum(len(analyses) for analyses in self.morphology_analyses.values()),
            "running": self.running
        }
    
    def stop(self):
        """Stop the tensor analysis system"""
        self.running = False
        if self.analysis_thread and self.analysis_thread.is_alive():
            self.analysis_thread.join(timeout=5)