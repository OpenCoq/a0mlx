"""
GGML Tensor Operations for MOSES

Provides GGML-compatible tensor operations for MOSES evolutionary program optimization,
enabling efficient tensor computations for population fitness, attention allocation,
and program evolution dynamics.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import ctypes
import struct

# Mock GGML types for compatibility (in real implementation, would import from ggml)
class GGMLTensorType(Enum):
    """GGML tensor types"""
    F32 = "f32"
    F16 = "f16" 
    Q4_0 = "q4_0"
    Q4_1 = "q4_1"
    Q8_0 = "q8_0"


@dataclass
class GGMLTensor:
    """GGML tensor representation"""
    name: str
    shape: Tuple[int, ...]
    dtype: GGMLTensorType
    data: np.ndarray
    
    def __post_init__(self):
        """Validate tensor"""
        if self.data.shape != self.shape:
            raise ValueError(f"Shape mismatch: expected {self.shape}, got {self.data.shape}")


class MOSESTensorOps:
    """GGML-compatible tensor operations for MOSES"""
    
    def __init__(self):
        """Initialize MOSES tensor operations"""
        self.tensor_cache: Dict[str, GGMLTensor] = {}
        self.operation_count = 0
        self.memory_usage = 0
        
        # MOSES-specific tensor configurations
        self.moses_tensor_configs = {
            "population_fitness": {
                "dtype": GGMLTensorType.F32,
                "typical_shape": (50,),  # population_size
                "description": "Fitness scores for population"
            },
            "population_complexity": {
                "dtype": GGMLTensorType.F32,
                "typical_shape": (50,),
                "description": "Complexity scores for population"
            },
            "attention_allocation": {
                "dtype": GGMLTensorType.F32,
                "typical_shape": (50, 5),  # population_size x attention_dimensions
                "description": "ECAN attention allocation matrix"
            },
            "program_embeddings": {
                "dtype": GGMLTensorType.F32,
                "typical_shape": (50, 20),  # population_size x max_complexity
                "description": "Program atom embeddings"
            },
            "generation_trajectory": {
                "dtype": GGMLTensorType.F32,
                "typical_shape": (100, 3),  # max_generations x metrics
                "description": "Evolution trajectory over generations"
            },
            "fitness_landscape": {
                "dtype": GGMLTensorType.F32,
                "typical_shape": (50, 50),  # fitness correlation matrix
                "description": "Program fitness landscape correlation"
            }
        }
    
    def create_tensor(self, name: str, shape: Tuple[int, ...], 
                     dtype: GGMLTensorType = GGMLTensorType.F32,
                     data: Optional[np.ndarray] = None) -> GGMLTensor:
        """Create a new GGML tensor"""
        if data is None:
            if dtype == GGMLTensorType.F32:
                data = np.zeros(shape, dtype=np.float32)
            elif dtype == GGMLTensorType.F16:
                data = np.zeros(shape, dtype=np.float16)
            else:
                data = np.zeros(shape, dtype=np.float32)
        
        tensor = GGMLTensor(name=name, shape=shape, dtype=dtype, data=data)
        self.tensor_cache[name] = tensor
        self.memory_usage += data.nbytes
        
        return tensor
    
    def moses_population_to_tensor(self, population: List[Any], 
                                 tensor_type: str = "population_fitness") -> GGMLTensor:
        """Convert MOSES population to GGML tensor"""
        if tensor_type == "population_fitness":
            data = np.array([p.fitness for p in population], dtype=np.float32)
            shape = (len(population),)
        elif tensor_type == "population_complexity":
            data = np.array([p.complexity for p in population], dtype=np.float32)
            shape = (len(population),)
        elif tensor_type == "generation_tensor":
            data = np.array([p.generation for p in population], dtype=np.float32)
            shape = (len(population),)
        elif tensor_type == "program_embeddings":
            # Create embedding matrix from program atoms
            max_atoms = max(len(p.atoms) for p in population) if population else 1
            data = np.zeros((len(population), max_atoms), dtype=np.float32)
            for i, program in enumerate(population):
                # Simple hash-based embedding (in real implementation, use proper embeddings)
                for j, atom_id in enumerate(program.atoms[:max_atoms]):
                    data[i, j] = hash(atom_id) % 1000 / 1000.0  # Normalize to [0,1]
            shape = (len(population), max_atoms)
        else:
            raise ValueError(f"Unknown tensor type: {tensor_type}")
        
        return self.create_tensor(f"moses_{tensor_type}", shape, data=data)
    
    def ecan_attention_to_tensor(self, attention_units: Dict[str, Any]) -> GGMLTensor:
        """Convert ECAN attention units to GGML tensor"""
        if not attention_units:
            return self.create_tensor("attention_empty", (0, 5))
        
        # Extract attention metrics
        attention_data = []
        for unit in attention_units.values():
            if hasattr(unit, 'moses_entity_type'):  # MOSES attention unit
                metrics = [
                    unit.attention_value,
                    unit.importance,
                    unit.confidence,
                    unit.fitness_score,
                    unit.evolution_priority
                ]
            else:  # Regular attention unit
                metrics = [
                    unit.attention_value,
                    unit.importance,
                    unit.confidence,
                    0.0,  # No fitness score
                    0.0   # No evolution priority
                ]
            attention_data.append(metrics)
        
        data = np.array(attention_data, dtype=np.float32)
        return self.create_tensor("ecan_attention", data.shape, data=data)
    
    def fitness_evolution_tensor(self, generation_history: List[Dict[str, Any]]) -> GGMLTensor:
        """Create tensor for fitness evolution over generations"""
        if not generation_history:
            return self.create_tensor("fitness_evolution_empty", (0, 3))
        
        trajectory_data = []
        for gen_data in generation_history:
            trajectory_data.append([
                gen_data.get("generation", 0),
                gen_data.get("best_fitness", 0.0),
                gen_data.get("average_fitness", 0.0)
            ])
        
        data = np.array(trajectory_data, dtype=np.float32)
        return self.create_tensor("fitness_evolution", data.shape, data=data)
    
    # GGML-compatible operations
    def tensor_add(self, a: GGMLTensor, b: GGMLTensor) -> GGMLTensor:
        """Element-wise tensor addition"""
        if a.shape != b.shape:
            raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
        
        result_data = a.data + b.data
        result = self.create_tensor(f"add_{a.name}_{b.name}", a.shape, data=result_data)
        self.operation_count += 1
        return result
    
    def tensor_mul(self, a: GGMLTensor, b: GGMLTensor) -> GGMLTensor:
        """Element-wise tensor multiplication"""
        if a.shape != b.shape:
            raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
        
        result_data = a.data * b.data
        result = self.create_tensor(f"mul_{a.name}_{b.name}", a.shape, data=result_data)
        self.operation_count += 1
        return result
    
    def tensor_matmul(self, a: GGMLTensor, b: GGMLTensor) -> GGMLTensor:
        """Matrix multiplication"""
        if len(a.shape) != 2 or len(b.shape) != 2:
            raise ValueError("Matrix multiplication requires 2D tensors")
        if a.shape[1] != b.shape[0]:
            raise ValueError(f"Matrix dimension mismatch: {a.shape} @ {b.shape}")
        
        result_data = np.matmul(a.data, b.data)
        result_shape = (a.shape[0], b.shape[1])
        result = self.create_tensor(f"matmul_{a.name}_{b.name}", result_shape, data=result_data)
        self.operation_count += 1
        return result
    
    def tensor_transpose(self, tensor: GGMLTensor) -> GGMLTensor:
        """Tensor transpose"""
        result_data = tensor.data.T
        result_shape = tuple(reversed(tensor.shape))
        result = self.create_tensor(f"transpose_{tensor.name}", result_shape, data=result_data)
        self.operation_count += 1
        return result
    
    def tensor_softmax(self, tensor: GGMLTensor, axis: int = -1) -> GGMLTensor:
        """Softmax operation"""
        # Numerically stable softmax
        shifted = tensor.data - np.max(tensor.data, axis=axis, keepdims=True)
        exp_vals = np.exp(shifted)
        result_data = exp_vals / np.sum(exp_vals, axis=axis, keepdims=True)
        
        result = self.create_tensor(f"softmax_{tensor.name}", tensor.shape, data=result_data)
        self.operation_count += 1
        return result
    
    def tensor_relu(self, tensor: GGMLTensor) -> GGMLTensor:
        """ReLU activation"""
        result_data = np.maximum(0, tensor.data)
        result = self.create_tensor(f"relu_{tensor.name}", tensor.shape, data=result_data)
        self.operation_count += 1
        return result
    
    def tensor_norm(self, tensor: GGMLTensor, ord: int = 2) -> float:
        """Tensor norm"""
        self.operation_count += 1
        return float(np.linalg.norm(tensor.data, ord=ord))
    
    def tensor_reduce_mean(self, tensor: GGMLTensor, axis: Optional[int] = None) -> GGMLTensor:
        """Reduce mean along axis"""
        result_data = np.mean(tensor.data, axis=axis)
        if axis is None:
            result_shape = ()
        else:
            result_shape = tuple(s for i, s in enumerate(tensor.shape) if i != axis)
        
        result = self.create_tensor(f"mean_{tensor.name}", result_shape, data=result_data)
        self.operation_count += 1
        return result
    
    def tensor_reduce_max(self, tensor: GGMLTensor, axis: Optional[int] = None) -> GGMLTensor:
        """Reduce max along axis"""
        result_data = np.max(tensor.data, axis=axis)
        if axis is None:
            result_shape = ()
        else:
            result_shape = tuple(s for i, s in enumerate(tensor.shape) if i != axis)
        
        result = self.create_tensor(f"max_{tensor.name}", result_shape, data=result_data)
        self.operation_count += 1
        return result
    
    # MOSES-specific operations
    def fitness_attention_correlation(self, fitness_tensor: GGMLTensor, 
                                    attention_tensor: GGMLTensor) -> float:
        """Calculate correlation between fitness and attention"""
        # Handle different attention tensor shapes
        if len(attention_tensor.shape) == 2:
            # Use first attention dimension (attention_value)
            attention_values = attention_tensor.data[:, 0]
        else:
            attention_values = attention_tensor.data
        
        # Ensure both arrays have the same length
        min_length = min(len(fitness_tensor.data), len(attention_values))
        fitness_values = fitness_tensor.data[:min_length]
        attention_values = attention_values[:min_length]
        
        if min_length < 2:
            return 0.0  # Need at least 2 points for correlation
        
        correlation = np.corrcoef(fitness_values, attention_values)[0, 1]
        self.operation_count += 1
        return float(correlation) if not np.isnan(correlation) else 0.0
    
    def population_diversity_metric(self, population_tensor: GGMLTensor) -> float:
        """Calculate population diversity using tensor operations"""
        if len(population_tensor.shape) == 1:
            # For 1D tensors (fitness), use variance
            diversity = float(np.var(population_tensor.data))
        else:
            # For 2D tensors (embeddings), use pairwise distances
            distances = []
            for i in range(population_tensor.shape[0]):
                for j in range(i + 1, population_tensor.shape[0]):
                    dist = np.linalg.norm(population_tensor.data[i] - population_tensor.data[j])
                    distances.append(dist)
            diversity = float(np.mean(distances)) if distances else 0.0
        
        self.operation_count += 1
        return diversity
    
    def evolution_momentum(self, trajectory_tensor: GGMLTensor) -> GGMLTensor:
        """Calculate evolution momentum from trajectory"""
        if trajectory_tensor.shape[0] < 2:
            return self.create_tensor("momentum_empty", (0,))
        
        # Calculate differences between consecutive generations
        fitness_series = trajectory_tensor.data[:, 1]  # Best fitness column
        momentum_data = np.diff(fitness_series)
        
        result = self.create_tensor("evolution_momentum", (len(momentum_data),), data=momentum_data)
        self.operation_count += 1
        return result
    
    def attention_reallocation_matrix(self, old_attention: GGMLTensor, 
                                   new_attention: GGMLTensor) -> GGMLTensor:
        """Create attention reallocation matrix"""
        if old_attention.shape != new_attention.shape:
            raise ValueError("Attention tensors must have same shape")
        
        # Calculate reallocation as ratio of new to old (with smoothing)
        epsilon = 1e-8
        reallocation_data = (new_attention.data + epsilon) / (old_attention.data + epsilon)
        
        result = self.create_tensor("attention_reallocation", old_attention.shape, data=reallocation_data)
        self.operation_count += 1
        return result
    
    def fitness_landscape_analysis(self, fitness_tensor: GGMLTensor, 
                                 embedding_tensor: GGMLTensor) -> Dict[str, float]:
        """Analyze fitness landscape using tensor operations"""
        if fitness_tensor.shape[0] != embedding_tensor.shape[0]:
            raise ValueError("Fitness and embedding tensors must have same number of programs")
        
        # Calculate fitness correlations with embedding dimensions
        correlations = []
        for dim in range(embedding_tensor.shape[1]):
            corr = np.corrcoef(fitness_tensor.data, embedding_tensor.data[:, dim])[0, 1]
            if not np.isnan(corr):
                correlations.append(abs(corr))
        
        # Fitness landscape metrics
        fitness_variance = float(np.var(fitness_tensor.data))
        fitness_range = float(np.max(fitness_tensor.data) - np.min(fitness_tensor.data))
        embedding_complexity = float(np.mean(np.std(embedding_tensor.data, axis=0)))
        max_correlation = float(max(correlations)) if correlations else 0.0
        
        self.operation_count += 4
        
        return {
            "fitness_variance": fitness_variance,
            "fitness_range": fitness_range,
            "embedding_complexity": embedding_complexity,
            "max_dimension_correlation": max_correlation,
            "landscape_smoothness": 1.0 - fitness_variance  # Inverse relationship
        }
    
    def validate_tensor_compatibility(self, tensor: GGMLTensor, 
                                    expected_config: str) -> Dict[str, bool]:
        """Validate tensor compatibility with MOSES configurations"""
        if expected_config not in self.moses_tensor_configs:
            return {"valid": False, "error": f"Unknown config: {expected_config}"}
        
        config = self.moses_tensor_configs[expected_config]
        validations = {}
        
        # Check dtype compatibility
        validations["dtype_compatible"] = tensor.dtype == config["dtype"]
        
        # Check shape compatibility (allow flexible dimensions)
        expected_shape = config["typical_shape"]
        if len(tensor.shape) == len(expected_shape):
            validations["shape_compatible"] = True
            # Check if dimensions are reasonable
            for i, (actual, expected) in enumerate(zip(tensor.shape, expected_shape)):
                if i == 0:  # First dimension (usually population size) can vary
                    validations[f"dim_{i}_reasonable"] = 1 <= actual <= 1000
                else:  # Other dimensions should be close to expected
                    validations[f"dim_{i}_compatible"] = actual == expected
        else:
            validations["shape_compatible"] = False
        
        # Check data validity
        validations["data_finite"] = bool(np.all(np.isfinite(tensor.data)))
        validations["data_non_negative"] = bool(np.all(tensor.data >= 0)) if "fitness" in expected_config else True
        
        # Overall validity
        validations["valid"] = all(validations.values())
        
        return validations
    
    def benchmark_operations(self, tensor_sizes: List[Tuple[int, ...]] = None) -> Dict[str, float]:
        """Benchmark GGML operations for MOSES tensors"""
        if tensor_sizes is None:
            tensor_sizes = [(50,), (50, 20), (100, 3), (50, 5)]
        
        import time
        
        benchmarks = {}
        
        for i, size in enumerate(tensor_sizes):
            # Create test tensors
            a = self.create_tensor(f"bench_a_{i}", size)
            b = self.create_tensor(f"bench_b_{i}", size)
            
            # Fill with random data
            a.data = np.random.randn(*size).astype(np.float32)
            b.data = np.random.randn(*size).astype(np.float32)
            
            # Benchmark operations
            ops_to_bench = [
                ("add", lambda: self.tensor_add(a, b)),
                ("mul", lambda: self.tensor_mul(a, b)),
                ("norm", lambda: self.tensor_norm(a)),
                ("mean", lambda: self.tensor_reduce_mean(a)),
                ("relu", lambda: self.tensor_relu(a))
            ]
            
            if len(size) == 1:  # 1D tensors
                ops_to_bench.append(("softmax", lambda: self.tensor_softmax(a)))
            
            if len(size) == 2 and size[0] == size[1]:  # Square matrices
                ops_to_bench.append(("matmul", lambda: self.tensor_matmul(a, b)))
                ops_to_bench.append(("transpose", lambda: self.tensor_transpose(a)))
            
            for op_name, op_func in ops_to_bench:
                start_time = time.time()
                try:
                    for _ in range(100):  # Run 100 times for averaging
                        result = op_func()
                    end_time = time.time()
                    
                    avg_time = (end_time - start_time) / 100 * 1000  # ms
                    benchmarks[f"{op_name}_{size}"] = avg_time
                except Exception as e:
                    benchmarks[f"{op_name}_{size}"] = float('inf')
        
        return benchmarks
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage statistics"""
        total_tensors = len(self.tensor_cache)
        total_memory = self.memory_usage
        
        tensor_breakdown = {}
        for name, tensor in self.tensor_cache.items():
            tensor_breakdown[name] = {
                "shape": tensor.shape,
                "dtype": tensor.dtype.value,
                "memory_bytes": tensor.data.nbytes,
                "memory_mb": tensor.data.nbytes / (1024 * 1024)
            }
        
        return {
            "total_tensors": total_tensors,
            "total_memory_bytes": total_memory,
            "total_memory_mb": total_memory / (1024 * 1024),
            "operation_count": self.operation_count,
            "tensor_breakdown": tensor_breakdown
        }
    
    def cleanup_tensors(self, keep_patterns: List[str] = None):
        """Clean up tensor cache"""
        if keep_patterns is None:
            keep_patterns = ["moses_", "ecan_", "fitness_evolution"]
        
        tensors_to_remove = []
        for name in self.tensor_cache:
            should_keep = any(pattern in name for pattern in keep_patterns)
            if not should_keep:
                tensors_to_remove.append(name)
        
        for name in tensors_to_remove:
            tensor = self.tensor_cache.pop(name)
            self.memory_usage -= tensor.data.nbytes
        
        return len(tensors_to_remove)


# Convenience functions for MOSES integration
def create_moses_tensor_ops() -> MOSESTensorOps:
    """Create MOSES tensor operations instance"""
    return MOSESTensorOps()


def population_to_ggml_tensors(population: List[Any], 
                             attention_units: Dict[str, Any] = None) -> Dict[str, GGMLTensor]:
    """Convert MOSES population to GGML tensors"""
    ops = create_moses_tensor_ops()
    
    tensors = {}
    
    # Population tensors
    tensors["fitness"] = ops.moses_population_to_tensor(population, "population_fitness")
    tensors["complexity"] = ops.moses_population_to_tensor(population, "population_complexity")
    tensors["generation"] = ops.moses_population_to_tensor(population, "generation_tensor")
    tensors["embeddings"] = ops.moses_population_to_tensor(population, "program_embeddings")
    
    # Attention tensors
    if attention_units:
        tensors["attention"] = ops.ecan_attention_to_tensor(attention_units)
    
    return tensors


def validate_moses_tensor_pipeline(tensors: Dict[str, GGMLTensor]) -> Dict[str, Any]:
    """Validate complete MOSES tensor pipeline"""
    ops = create_moses_tensor_ops()
    
    validation_results = {}
    
    # Validate individual tensors
    tensor_configs = {
        "fitness": "population_fitness",
        "complexity": "population_complexity", 
        "attention": "attention_allocation",
        "embeddings": "program_embeddings"
    }
    
    for tensor_name, config_name in tensor_configs.items():
        if tensor_name in tensors:
            validation = ops.validate_tensor_compatibility(tensors[tensor_name], config_name)
            validation_results[tensor_name] = validation
    
    # Cross-tensor validations
    if "fitness" in tensors and "attention" in tensors:
        try:
            correlation = ops.fitness_attention_correlation(tensors["fitness"], tensors["attention"])
            validation_results["fitness_attention_correlation"] = {
                "valid": True,
                "correlation": correlation,
                "strong_correlation": abs(correlation) > 0.3
            }
        except Exception as e:
            validation_results["fitness_attention_correlation"] = {
                "valid": False,
                "error": str(e)
            }
    
    # Population consistency
    population_sizes = []
    for tensor_name in ["fitness", "complexity", "attention", "embeddings"]:
        if tensor_name in tensors:
            population_sizes.append(tensors[tensor_name].shape[0])
    
    validation_results["population_consistency"] = {
        "valid": len(set(population_sizes)) <= 1 if population_sizes else False,
        "population_sizes": population_sizes
    }
    
    # Overall pipeline validity
    individual_validities = [result.get("valid", False) for result in validation_results.values() 
                           if isinstance(result, dict) and "valid" in result]
    validation_results["pipeline_valid"] = all(individual_validities)
    
    return validation_results