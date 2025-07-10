"""
MOSES Evolution Python API Wrapper

Provides a clean Python interface for controlling MOSES evolutionary program
optimization, with full integration to AtomSpace, PLN, and ECAN systems.
"""

import asyncio
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone
import numpy as np

from ..helpers.atomspace import AtomSpace, AtomType
from ..helpers.moses_optimizer import MOSESOptimizer, ProgramType, Program
from ..helpers.pln_reasoning import PLNInferenceEngine, TruthValue
from ..helpers.ecan_attention import ECANAttentionSystem, MOSESAttentionUnit
from .moses_evolution import MOSESEvolutionAPI


@dataclass
class EvolutionConfig:
    """Configuration for evolution session"""
    program_type: ProgramType = ProgramType.INFERENCE_RULE
    population_size: int = 50
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    selection_pressure: float = 0.3
    max_complexity: int = 20
    max_generations: int = 100
    fitness_threshold: float = 0.95
    early_stopping_patience: int = 10


@dataclass 
class EvolutionResults:
    """Results from evolution process"""
    session_id: str
    best_program: Program
    best_fitness: float
    generations_completed: int
    total_time: float
    fitness_history: List[float]
    attention_metrics: Dict[str, float]
    tensor_data: Dict[str, np.ndarray]


class MOSESPythonAPI:
    """High-level Python API for MOSES evolution system"""
    
    def __init__(self, atomspace_path: str = "/tmp/moses_python_api.db"):
        """Initialize the MOSES Python API"""
        self.api = MOSESEvolutionAPI(atomspace_path)
        self.current_session_id: Optional[str] = None
        
        # Wait for system initialization
        asyncio.create_task(self._wait_for_initialization())
    
    async def _wait_for_initialization(self):
        """Wait for all systems to initialize"""
        await self.api._initialize_systems()
    
    async def create_session(self, config: EvolutionConfig = None) -> str:
        """
        Create a new evolution session
        
        Args:
            config: Evolution configuration
            
        Returns:
            Session ID
        """
        if config is None:
            config = EvolutionConfig()
        
        session_data = {
            "program_type": config.program_type.value,
            "population_size": config.population_size,
            "mutation_rate": config.mutation_rate,
            "crossover_rate": config.crossover_rate,
            "selection_pressure": config.selection_pressure,
            "max_complexity": config.max_complexity
        }
        
        result = self.api.create_evolution_session(session_data)
        if not result["success"]:
            raise RuntimeError(f"Failed to create session: {result['error']}")
        
        self.current_session_id = result["session_id"]
        return self.current_session_id
    
    async def initialize_population(self, session_id: str = None, 
                                  seed_programs: List[Program] = None) -> List[Program]:
        """
        Initialize population for evolution
        
        Args:
            session_id: Session ID (uses current if None)
            seed_programs: Optional seed programs
            
        Returns:
            Initial population
        """
        if session_id is None:
            session_id = self.current_session_id
        
        if session_id is None:
            raise ValueError("No session ID provided and no current session")
        
        init_data = {}
        if seed_programs:
            init_data["seed_programs"] = [p.to_dict() for p in seed_programs]
        
        result = await self.api.initialize_population(session_id, init_data)
        if not result["success"]:
            raise RuntimeError(f"Failed to initialize population: {result['error']}")
        
        return [Program.from_dict(p) for p in result["population"]]
    
    async def evolve_generation(self, session_id: str = None) -> Dict[str, Any]:
        """
        Evolve one generation
        
        Args:
            session_id: Session ID (uses current if None)
            
        Returns:
            Generation statistics
        """
        if session_id is None:
            session_id = self.current_session_id
        
        if session_id is None:
            raise ValueError("No session ID provided and no current session")
        
        result = await self.api.evolve_generation(session_id, {})
        if not result["success"]:
            raise RuntimeError(f"Failed to evolve generation: {result['error']}")
        
        return result["generation_stats"]
    
    async def run_evolution(self, config: EvolutionConfig = None,
                          seed_programs: List[Program] = None,
                          progress_callback: callable = None) -> EvolutionResults:
        """
        Run complete evolution process
        
        Args:
            config: Evolution configuration
            seed_programs: Optional seed programs
            progress_callback: Optional callback for progress updates
            
        Returns:
            Evolution results
        """
        if config is None:
            config = EvolutionConfig()
        
        start_time = datetime.now(timezone.utc)
        
        # Create session
        session_id = await self.create_session(config)
        
        # Initialize population
        population = await self.initialize_population(session_id, seed_programs)
        
        # Evolution loop
        fitness_history = []
        best_fitness = 0.0
        patience_counter = 0
        generations_completed = 0
        
        for generation in range(config.max_generations):
            # Evolve generation
            gen_stats = await self.evolve_generation(session_id)
            generations_completed += 1
            
            # Track progress
            current_fitness = gen_stats["best_fitness"]
            fitness_history.append(current_fitness)
            
            # Ensure ECAN integration by updating attention for all programs
            current_population = self.api.moses_optimizer.population
            for program in current_population:
                await self.api.ecan_system.allocate_moses_attention(
                    atom_id=program.id,
                    moses_entity_type="program",
                    fitness_score=program.fitness,
                    complexity=program.complexity,
                    generation=program.generation,
                    requester_id="python_api_evolution"
                )
            
            # Progress callback
            if progress_callback:
                progress_callback({
                    "generation": generation + 1,
                    "best_fitness": current_fitness,
                    "average_fitness": gen_stats["average_fitness"],
                    "generations_completed": generations_completed
                })
            
            # Check convergence
            if current_fitness >= config.fitness_threshold:
                print(f"🎯 Fitness threshold reached: {current_fitness:.4f}")
                break
            
            # Early stopping
            if current_fitness > best_fitness:
                best_fitness = current_fitness
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= config.early_stopping_patience:
                print(f"⏹️ Early stopping after {patience_counter} generations without improvement")
                break
        
        # Get final results
        status = await self.get_session_status(session_id)
        tensors = await self.get_session_tensors(session_id)
        
        total_time = (datetime.now(timezone.utc) - start_time).total_seconds()
        
        # Create results object
        best_program = Program.from_dict(status["session"]["best_program"])
        
        # Extract tensor data
        tensor_data = {}
        if tensors["success"]:
            for name, tensor_list in tensors["tensors"].items():
                tensor_data[name] = np.array(tensor_list)
        
        # Extract attention metrics
        attention_metrics = {}
        if "attention_visualization" in status:
            viz = status["attention_visualization"]
            attention_metrics = {
                "attention_efficiency": status["performance_metrics"]["attention_efficiency"],
                "resource_utilization": status["performance_metrics"]["resource_utilization"],
                "high_fitness_attention": viz.get("entity_type_breakdown", {}).get("program", {}).get("total_attention", 0.0)
            }
        
        return EvolutionResults(
            session_id=session_id,
            best_program=best_program,
            best_fitness=best_fitness,
            generations_completed=generations_completed,
            total_time=total_time,
            fitness_history=fitness_history,
            attention_metrics=attention_metrics,
            tensor_data=tensor_data
        )
    
    async def get_session_status(self, session_id: str = None) -> Dict[str, Any]:
        """Get detailed session status"""
        if session_id is None:
            session_id = self.current_session_id
        
        if session_id is None:
            raise ValueError("No session ID provided and no current session")
        
        result = await self.api.get_session_status(session_id)
        if not result["success"]:
            raise RuntimeError(f"Failed to get session status: {result['error']}")
        
        return result
    
    async def get_session_tensors(self, session_id: str = None) -> Dict[str, Any]:
        """Get session tensor data"""
        if session_id is None:
            session_id = self.current_session_id
        
        if session_id is None:
            raise ValueError("No session ID provided and no current session")
        
        result = await self.api.get_population_tensors(session_id)
        if not result["success"]:
            raise RuntimeError(f"Failed to get session tensors: {result['error']}")
        
        return result
    
    async def optimize_for_fitness(self, target_fitness: float = 0.9,
                                 max_generations: int = 50,
                                 config: EvolutionConfig = None) -> Program:
        """
        Optimize until target fitness is reached
        
        Args:
            target_fitness: Target fitness score
            max_generations: Maximum generations to try
            config: Evolution configuration
            
        Returns:
            Best program found
        """
        if config is None:
            config = EvolutionConfig()
        
        config.fitness_threshold = target_fitness
        config.max_generations = max_generations
        
        def progress_callback(stats):
            print(f"Generation {stats['generation']}: "
                  f"best={stats['best_fitness']:.4f}, "
                  f"avg={stats['average_fitness']:.4f}")
        
        results = await self.run_evolution(config, progress_callback=progress_callback)
        
        if results.best_fitness >= target_fitness:
            print(f"🎯 Target fitness achieved: {results.best_fitness:.4f}")
        else:
            print(f"⚠️ Target fitness not reached. Best: {results.best_fitness:.4f}")
        
        return results.best_program
    
    async def analyze_evolution_dynamics(self, session_id: str = None) -> Dict[str, Any]:
        """
        Analyze evolution dynamics and attention patterns
        
        Args:
            session_id: Session ID to analyze
            
        Returns:
            Analysis results
        """
        if session_id is None:
            session_id = self.current_session_id
        
        status = await self.get_session_status(session_id)
        tensors = await self.get_session_tensors(session_id)
        
        if not (status["success"] and tensors["success"]):
            raise RuntimeError("Failed to get session data for analysis")
        
        # Fitness dynamics
        history = status["session"]["generation_history"]
        if history:
            fitness_trend = [gen["best_fitness"] for gen in history]
            avg_fitness_trend = [gen["average_fitness"] for gen in history]
            
            # Calculate improvement rates
            fitness_improvement = []
            for i in range(1, len(fitness_trend)):
                improvement = fitness_trend[i] - fitness_trend[i-1]
                fitness_improvement.append(improvement)
        else:
            fitness_trend = []
            avg_fitness_trend = []
            fitness_improvement = []
        
        # Attention dynamics
        viz = status.get("attention_visualization", {})
        attention_analysis = {}
        
        if viz.get("fitness_vs_attention"):
            points = viz["fitness_vs_attention"]
            fitness_scores = [p["fitness"] for p in points]
            attention_scores = [p["attention"] for p in points]
            
            if len(fitness_scores) > 1:
                correlation = np.corrcoef(fitness_scores, attention_scores)[0, 1]
                attention_analysis["fitness_attention_correlation"] = correlation
                attention_analysis["attention_efficiency"] = max(0.0, correlation)
            
            # Complexity vs attention
            complexity_scores = [p["complexity"] for p in points]
            if len(complexity_scores) > 1:
                complexity_attention_corr = np.corrcoef(complexity_scores, attention_scores)[0, 1]
                attention_analysis["complexity_attention_correlation"] = complexity_attention_corr
        
        # Population diversity
        tensor_data = tensors["tensors"]
        diversity_metrics = {}
        
        if tensor_data.get("fitness_tensor"):
            fitness_array = np.array(tensor_data["fitness_tensor"])
            diversity_metrics["fitness_variance"] = np.var(fitness_array)
            diversity_metrics["fitness_std"] = np.std(fitness_array)
            diversity_metrics["fitness_range"] = np.max(fitness_array) - np.min(fitness_array)
        
        if tensor_data.get("complexity_tensor"):
            complexity_array = np.array(tensor_data["complexity_tensor"])
            diversity_metrics["complexity_variance"] = np.var(complexity_array)
            diversity_metrics["avg_complexity"] = np.mean(complexity_array)
        
        # Performance metrics
        perf_metrics = status.get("performance_metrics", {})
        
        return {
            "fitness_dynamics": {
                "best_fitness_trend": fitness_trend,
                "average_fitness_trend": avg_fitness_trend,
                "fitness_improvement_per_generation": fitness_improvement,
                "convergence_rate": np.mean(fitness_improvement) if fitness_improvement else 0.0
            },
            "attention_dynamics": attention_analysis,
            "population_diversity": diversity_metrics,
            "performance_metrics": perf_metrics,
            "session_summary": {
                "generations_completed": len(history),
                "best_fitness_achieved": max(fitness_trend) if fitness_trend else 0.0,
                "total_programs_evaluated": sum(gen["population_size"] for gen in history),
                "attention_units_created": status["ecan_stats"]["moses_attention_units"]
            }
        }
    
    async def create_program_from_atoms(self, atom_ids: List[str], 
                                      program_type: ProgramType = ProgramType.INFERENCE_RULE) -> Program:
        """
        Create a program from specific atoms
        
        Args:
            atom_ids: List of atom IDs to include in program
            program_type: Type of program to create
            
        Returns:
            Created program
        """
        program = Program(
            id=f"custom_{datetime.now().timestamp()}",
            program_type=program_type,
            atoms=atom_ids,
            complexity=len(atom_ids),
            generation=0,
            parent_ids=[],
            metadata={"created_by": "python_api", "custom": True}
        )
        
        # Evaluate fitness using current MOSES optimizer
        if hasattr(self.api.moses_optimizer, '_evaluate_program'):
            program.fitness = await self.api.moses_optimizer._evaluate_program(program)
        
        return program
    
    async def get_attention_for_program(self, program_id: str) -> Optional[MOSESAttentionUnit]:
        """Get attention unit for a specific program"""
        for unit in self.api.ecan_system.moses_attention_units.values():
            if unit.atom_id == program_id:
                return unit
        return None
    
    async def boost_program_attention(self, program_id: str, boost_factor: float = 1.5) -> bool:
        """Boost attention for a specific program"""
        unit = await self.get_attention_for_program(program_id)
        if unit:
            unit.attention_value *= boost_factor
            unit.importance *= boost_factor
            return True
        return False
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all evolution sessions"""
        result = self.api.list_sessions()
        if not result["success"]:
            raise RuntimeError(f"Failed to list sessions: {result['error']}")
        
        return result["sessions"]
    
    def get_api_documentation(self) -> Dict[str, Any]:
        """Get comprehensive API documentation"""
        return self.api.get_api_documentation()


# Convenience functions for common use cases
async def quick_evolve(target_fitness: float = 0.8, 
                      max_generations: int = 20,
                      population_size: int = 30) -> EvolutionResults:
    """
    Quick evolution with reasonable defaults
    
    Args:
        target_fitness: Target fitness to achieve
        max_generations: Maximum generations
        population_size: Population size
        
    Returns:
        Evolution results
    """
    api = MOSESPythonAPI()
    await api._wait_for_initialization()
    
    config = EvolutionConfig(
        population_size=population_size,
        max_generations=max_generations,
        fitness_threshold=target_fitness
    )
    
    return await api.run_evolution(config)


async def evolve_with_seeds(seed_atom_ids: List[List[str]], 
                          target_fitness: float = 0.8) -> EvolutionResults:
    """
    Evolve with seed programs from atom IDs
    
    Args:
        seed_atom_ids: List of atom ID lists for seed programs
        target_fitness: Target fitness
        
    Returns:
        Evolution results
    """
    api = MOSESPythonAPI()
    await api._wait_for_initialization()
    
    # Create seed programs
    seed_programs = []
    for i, atom_ids in enumerate(seed_atom_ids):
        seed_program = await api.create_program_from_atoms(atom_ids)
        seed_programs.append(seed_program)
    
    config = EvolutionConfig(fitness_threshold=target_fitness)
    return await api.run_evolution(config, seed_programs)


async def analyze_best_programs(session_id: str, top_n: int = 5) -> List[Dict[str, Any]]:
    """
    Analyze the top N programs from a session
    
    Args:
        session_id: Session to analyze
        top_n: Number of top programs to analyze
        
    Returns:
        Analysis of top programs
    """
    api = MOSESPythonAPI()
    
    tensors = await api.get_session_tensors(session_id)
    if not tensors["success"]:
        raise RuntimeError("Failed to get session tensors")
    
    # Get fitness and complexity data
    fitness_data = np.array(tensors["tensors"]["fitness_tensor"])
    complexity_data = np.array(tensors["tensors"]["complexity_tensor"])
    attention_data = np.array(tensors["tensors"]["attention_tensor"])
    
    # Find top programs
    top_indices = np.argsort(fitness_data)[-top_n:][::-1]
    
    analyses = []
    for i, idx in enumerate(top_indices):
        analysis = {
            "rank": i + 1,
            "fitness": fitness_data[idx],
            "complexity": complexity_data[idx],
            "attention_metrics": {
                "attention_value": attention_data[idx][0],
                "importance": attention_data[idx][1], 
                "confidence": attention_data[idx][2],
                "evolution_priority": attention_data[idx][4]
            },
            "efficiency_ratio": fitness_data[idx] / max(1, complexity_data[idx])
        }
        analyses.append(analysis)
    
    return analyses