"""
MOSES (Meta-Optimizing Semantic Evolutionary Search) for Neural-Symbolic Reasoning

Implements evolutionary program optimization for learning and adapting
cognitive programs in the hypergraph AtomSpace.
"""

import numpy as np
from typing import Dict, List, Optional, Set, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from datetime import datetime, timezone
import json
import uuid
import random
import copy

from .atomspace import AtomSpace, Node, Link, Atom, AtomType, MOSESProgramAtom, MOSESPopulationAtom, MOSESFitnessAtom
from .pln_reasoning import PLNInferenceEngine, TruthValue


class ProgramType(Enum):
    """Types of programs that can be evolved"""
    INFERENCE_RULE = "inference_rule"
    PATTERN_MATCHER = "pattern_matcher"
    COGNITIVE_KERNEL = "cognitive_kernel"
    BEHAVIOR_TREE = "behavior_tree"


@dataclass
class Program:
    """Represents an evolvable program in the AtomSpace"""
    id: str
    program_type: ProgramType
    atoms: List[str]  # List of atom IDs that make up the program
    fitness: float = 0.0
    complexity: int = 0
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "id": self.id,
            "program_type": self.program_type.value,
            "atoms": self.atoms,
            "fitness": self.fitness,
            "complexity": self.complexity,
            "generation": self.generation,
            "parent_ids": self.parent_ids,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Program':
        """Create from dictionary"""
        return cls(
            id=data["id"],
            program_type=ProgramType(data["program_type"]),
            atoms=data["atoms"],
            fitness=data.get("fitness", 0.0),
            complexity=data.get("complexity", 0),
            generation=data.get("generation", 0),
            parent_ids=data.get("parent_ids", []),
            metadata=data.get("metadata", {}),
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.now(timezone.utc)
        )


class MOSESOptimizer:
    """
    Meta-Optimizing Semantic Evolutionary Search for program evolution
    """
    
    def __init__(self, atomspace: AtomSpace, pln_engine: PLNInferenceEngine):
        self.atomspace = atomspace
        self.pln_engine = pln_engine
        self.population: List[Program] = []
        self.population_size = 50
        self.mutation_rate = 0.1
        self.crossover_rate = 0.7
        self.selection_pressure = 0.3
        self.max_complexity = 20
        self.generation_count = 0
        self.evolution_history: List[Dict[str, Any]] = []
        
    async def initialize_population(self, program_type: ProgramType, 
                                  seed_programs: Optional[List[Program]] = None) -> List[Program]:
        """
        Initialize population of programs
        
        Args:
            program_type: Type of programs to evolve
            seed_programs: Optional seed programs to start with
            
        Returns:
            List of initialized programs
        """
        if seed_programs:
            self.population = seed_programs[:self.population_size]
        else:
            self.population = []
        
        # Fill remaining population with random programs
        while len(self.population) < self.population_size:
            program = await self._create_random_program(program_type)
            self.population.append(program)
        
        # Evaluate initial population
        await self._evaluate_population()
        
        return self.population
    
    async def _create_random_program(self, program_type: ProgramType) -> Program:
        """Create a random program of the specified type"""
        program_id = str(uuid.uuid4())
        
        # Get random atoms from the AtomSpace
        available_atoms = await self._get_available_atoms()
        
        # Create program with random atoms
        num_atoms = random.randint(1, min(10, len(available_atoms)))
        selected_atoms = random.sample(available_atoms, num_atoms)
        
        program = Program(
            id=program_id,
            program_type=program_type,
            atoms=selected_atoms,
            complexity=len(selected_atoms),
            generation=0
        )
        
        return program
    
    async def _get_available_atoms(self) -> List[str]:
        """Get available atoms from the AtomSpace"""
        # Get a sample of atoms for program construction
        node_pattern = {"atom_type": AtomType.NODE.value}
        link_pattern = {"atom_type": AtomType.LINK.value}
        
        nodes = await self.atomspace.storage.get_atoms_by_pattern(node_pattern, limit=100)
        links = await self.atomspace.storage.get_atoms_by_pattern(link_pattern, limit=100)
        
        all_atoms = nodes + links
        return [atom.id for atom in all_atoms]
    
    async def evolve_generation(self) -> Dict[str, Any]:
        """
        Evolve one generation of programs
        
        Returns:
            Dictionary with generation statistics
        """
        generation_start = datetime.now(timezone.utc)
        
        # Evaluate current population
        await self._evaluate_population()
        
        # Selection
        selected_programs = await self._selection()
        
        # Crossover and mutation
        new_generation = []
        
        # Keep best programs (elitism)
        elite_count = int(self.population_size * 0.1)
        elite_programs = sorted(self.population, key=lambda p: p.fitness, reverse=True)[:elite_count]
        new_generation.extend(elite_programs)
        
        # Generate offspring
        while len(new_generation) < self.population_size:
            if random.random() < self.crossover_rate and len(selected_programs) >= 2:
                # Crossover
                parent1, parent2 = random.sample(selected_programs, 2)
                offspring = await self._crossover(parent1, parent2)
            else:
                # Mutation only
                parent = random.choice(selected_programs)
                offspring = await self._mutate(parent)
            
            # Apply mutation
            if random.random() < self.mutation_rate:
                offspring = await self._mutate(offspring)
            
            offspring.generation = self.generation_count + 1
            new_generation.append(offspring)
        
        # Update population
        self.population = new_generation[:self.population_size]
        self.generation_count += 1
        
        # Record generation statistics
        generation_stats = {
            "generation": self.generation_count,
            "timestamp": generation_start.isoformat(),
            "duration": (datetime.now(timezone.utc) - generation_start).total_seconds(),
            "best_fitness": max(p.fitness for p in self.population),
            "average_fitness": np.mean([p.fitness for p in self.population]),
            "average_complexity": np.mean([p.complexity for p in self.population]),
            "population_size": len(self.population)
        }
        
        self.evolution_history.append(generation_stats)
        return generation_stats
    
    async def _evaluate_population(self):
        """Evaluate fitness of all programs in population"""
        for program in self.population:
            program.fitness = await self._evaluate_program(program)
    
    async def _evaluate_program(self, program: Program) -> float:
        """
        Evaluate fitness of a single program with PLN integration and uncertainty propagation
        
        Args:
            program: Program to evaluate
            
        Returns:
            Fitness score (higher is better)
        """
        if not program.atoms:
            return 0.0
        
        fitness_components = []
        uncertainty_components = []
        
        # Component 1: Semantic coherence with PLN truth values
        semantic_fitness, semantic_uncertainty = await self._evaluate_semantic_coherence_pln(program)
        fitness_components.append(semantic_fitness)
        uncertainty_components.append(semantic_uncertainty)
        
        # Component 2: Utility/performance with uncertainty
        utility_fitness, utility_uncertainty = await self._evaluate_utility_with_uncertainty(program)
        fitness_components.append(utility_fitness)
        uncertainty_components.append(utility_uncertainty)
        
        # Component 3: Complexity penalty with confidence
        complexity_penalty = max(0, 1.0 - (program.complexity / self.max_complexity))
        complexity_uncertainty = 0.05  # Low uncertainty for complexity measure
        fitness_components.append(complexity_penalty)
        uncertainty_components.append(complexity_uncertainty)
        
        # Component 4: PLN consistency check
        pln_consistency, pln_uncertainty = await self._evaluate_pln_consistency(program)
        fitness_components.append(pln_consistency)
        uncertainty_components.append(pln_uncertainty)
        
        # Combine fitness components with uncertainty propagation
        total_fitness = np.mean(fitness_components)
        total_uncertainty = np.sqrt(np.mean([u**2 for u in uncertainty_components]))
        
        # Adjust fitness based on uncertainty (penalize high uncertainty)
        uncertainty_penalty = 1.0 - (total_uncertainty * 0.2)
        final_fitness = total_fitness * uncertainty_penalty
        
        # Store detailed fitness information as MOSES fitness atom
        await self._store_detailed_fitness(program, {
            'semantic_fitness': semantic_fitness,
            'semantic_uncertainty': semantic_uncertainty,
            'utility_fitness': utility_fitness,
            'utility_uncertainty': utility_uncertainty,
            'complexity_penalty': complexity_penalty,
            'pln_consistency': pln_consistency,
            'total_uncertainty': total_uncertainty,
            'final_fitness': final_fitness
        })
        
        return final_fitness
    
    async def _evaluate_semantic_coherence_pln(self, program: Program) -> Tuple[float, float]:
        """Evaluate semantic coherence using PLN truth values with uncertainty"""
        if len(program.atoms) < 2:
            return 0.5, 0.3  # Medium fitness, high uncertainty
        
        coherence_scores = []
        confidence_scores = []
        
        # Check truth value consistency using PLN
        for atom_id in program.atoms:
            try:
                truth_value = await self.pln_engine.infer_truth_value(atom_id)
                coherence_scores.append(truth_value.strength)
                confidence_scores.append(truth_value.confidence)
            except Exception:
                coherence_scores.append(0.1)
                confidence_scores.append(0.1)
        
        if not coherence_scores:
            return 0.0, 1.0  # Zero fitness, maximum uncertainty
        
        # Calculate semantic fitness and uncertainty
        semantic_fitness = np.mean(coherence_scores)
        semantic_uncertainty = 1.0 - np.mean(confidence_scores)
        
        return semantic_fitness, semantic_uncertainty
    
    async def _evaluate_utility_with_uncertainty(self, program: Program) -> Tuple[float, float]:
        """Evaluate utility/performance with uncertainty estimation"""
        try:
            # Program type specific evaluation with uncertainty
            if program.program_type == ProgramType.INFERENCE_RULE:
                utility, uncertainty = await self._test_inference_rule_with_uncertainty(program)
            elif program.program_type == ProgramType.PATTERN_MATCHER:
                utility, uncertainty = await self._test_pattern_matcher_with_uncertainty(program)
            elif program.program_type == ProgramType.COGNITIVE_KERNEL:
                utility, uncertainty = await self._test_cognitive_kernel_with_uncertainty(program)
            else:
                # Default utility based on atom diversity with uncertainty
                diversity_score = min(1.0, len(set(program.atoms)) / 10.0)
                diversity_uncertainty = 0.2  # Medium uncertainty for diversity measure
                utility, uncertainty = diversity_score, diversity_uncertainty
            
            return utility, uncertainty
        except Exception as e:
            print(f"Error in utility evaluation: {e}")
            return 0.1, 0.9  # Low utility, high uncertainty on error
    
    async def _evaluate_pln_consistency(self, program: Program) -> Tuple[float, float]:
        """Evaluate PLN logical consistency of program atoms"""
        if not program.atoms:
            return 0.1, 0.9
        
        try:
            consistency_scores = []
            confidence_scores = []
            
            # Check logical consistency between atoms
            for i, atom_id in enumerate(program.atoms):
                for j, other_atom_id in enumerate(program.atoms[i+1:], i+1):
                    try:
                        # Use PLN to check consistency between atoms
                        consistency_result = await self.pln_engine.check_consistency(atom_id, other_atom_id)
                        consistency_scores.append(consistency_result.get('consistency', 0.5))
                        confidence_scores.append(consistency_result.get('confidence', 0.5))
                    except Exception:
                        consistency_scores.append(0.3)
                        confidence_scores.append(0.2)
            
            if not consistency_scores:
                return 0.5, 0.5
            
            pln_consistency = np.mean(consistency_scores)
            pln_uncertainty = 1.0 - np.mean(confidence_scores)
            
            return pln_consistency, pln_uncertainty
        except Exception as e:
            print(f"Error in PLN consistency evaluation: {e}")
            return 0.3, 0.8
    
    async def _test_inference_rule_with_uncertainty(self, program: Program) -> Tuple[float, float]:
        """Test inference rule program with uncertainty estimation"""
        if len(program.atoms) < 2:
            return 0.1, 0.8
        
        try:
            # Test forward chaining with uncertainty tracking
            inference_result = await self.pln_engine.forward_chaining(
                program.atoms[:3], 
                max_iterations=3
            )
            
            derived_facts = inference_result.get("derived_facts", [])
            confidence_scores = inference_result.get("confidence_scores", [])
            
            # Calculate utility and uncertainty
            utility = min(1.0, len(derived_facts) / 5.0)
            uncertainty = 1.0 - (np.mean(confidence_scores) if confidence_scores else 0.3)
            
            return utility, uncertainty
        except Exception:
            return 0.1, 0.9
    
    async def _test_pattern_matcher_with_uncertainty(self, program: Program) -> Tuple[float, float]:
        """Test pattern matcher program with uncertainty"""
        if not program.atoms:
            return 0.1, 0.8
        
        try:
            match_count = 0
            confidence_sum = 0.0
            
            for atom_id in program.atoms:
                atom = await self.atomspace.get_atom(atom_id)
                if atom:
                    match_count += 1
                    confidence_sum += atom.confidence
            
            utility = match_count / len(program.atoms) if program.atoms else 0.0
            uncertainty = 1.0 - (confidence_sum / len(program.atoms) if program.atoms else 0.0)
            
            return utility, uncertainty
        except Exception:
            return 0.1, 0.9
    
    async def _test_cognitive_kernel_with_uncertainty(self, program: Program) -> Tuple[float, float]:
        """Test cognitive kernel program with uncertainty"""
        if not program.atoms:
            return 0.1, 0.8
        
        try:
            total_truth_strength = 0.0
            total_confidence = 0.0
            
            for atom_id in program.atoms:
                tv = await self.pln_engine.infer_truth_value(atom_id)
                total_truth_strength += tv.strength
                total_confidence += tv.confidence
            
            utility = total_truth_strength / len(program.atoms)
            uncertainty = 1.0 - (total_confidence / len(program.atoms))
            
            return utility, uncertainty
        except Exception:
            return 0.1, 0.9
    
    async def _store_detailed_fitness(self, program: Program, fitness_details: Dict[str, float]):
        """Store detailed fitness evaluation as MOSES fitness atoms"""
        try:
            for fitness_type, fitness_value in fitness_details.items():
                if fitness_type.endswith('_uncertainty'):
                    continue  # Skip uncertainty values in this iteration
                
                uncertainty_key = f"{fitness_type}_uncertainty"
                uncertainty = fitness_details.get(uncertainty_key, 0.1)
                
                await self.atomspace.add_moses_fitness(
                    name=f"{fitness_type}_{program.id}",
                    fitness_type=fitness_type,
                    fitness_value=fitness_value,
                    confidence_interval=(
                        max(0.0, fitness_value - uncertainty),
                        min(1.0, fitness_value + uncertainty)
                    ),
                    evaluation_context={
                        'program_type': program.program_type.value,
                        'complexity': program.complexity,
                        'generation': program.generation,
                        'timestamp': datetime.now(timezone.utc).isoformat()
                    },
                    program_id=program.id,
                    truth_value=fitness_value,
                    confidence=1.0 - uncertainty
                )
        except Exception as e:
            print(f"Error storing detailed fitness: {e}")
    
    async def _evaluate_semantic_coherence(self, program: Program) -> float:
        """Evaluate semantic coherence of program atoms"""
        if len(program.atoms) < 2:
            return 0.5
        
        coherence_scores = []
        
        # Check truth value consistency
        for atom_id in program.atoms:
            try:
                truth_value = await self.pln_engine.infer_truth_value(atom_id)
                coherence_scores.append(truth_value.confidence)
            except Exception:
                coherence_scores.append(0.1)
        
        return np.mean(coherence_scores) if coherence_scores else 0.0
    
    async def _evaluate_utility(self, program: Program) -> float:
        """Evaluate utility/performance of program"""
        # This is a simplified utility evaluation
        # In practice, this would test the program on specific tasks
        
        utility_score = 0.0
        
        # Program type specific evaluation
        if program.program_type == ProgramType.INFERENCE_RULE:
            # Test inference capability
            utility_score = await self._test_inference_rule(program)
        elif program.program_type == ProgramType.PATTERN_MATCHER:
            # Test pattern matching capability
            utility_score = await self._test_pattern_matcher(program)
        elif program.program_type == ProgramType.COGNITIVE_KERNEL:
            # Test cognitive processing
            utility_score = await self._test_cognitive_kernel(program)
        else:
            # Default utility based on atom diversity
            utility_score = min(1.0, len(set(program.atoms)) / 10.0)
        
        return utility_score
    
    async def _test_inference_rule(self, program: Program) -> float:
        """Test inference rule program"""
        # Simple test: can it derive something meaningful?
        if len(program.atoms) < 2:
            return 0.1
        
        # Test forward chaining with program atoms as premises
        try:
            inference_result = await self.pln_engine.forward_chaining(program.atoms[:3], max_iterations=3)
            derived_facts = len(inference_result.get("derived_facts", []))
            return min(1.0, derived_facts / 5.0)
        except Exception:
            return 0.1
    
    async def _test_pattern_matcher(self, program: Program) -> float:
        """Test pattern matcher program"""
        # Test ability to find patterns in AtomSpace
        if not program.atoms:
            return 0.1
        
        try:
            # Count successful pattern matches
            match_count = 0
            for atom_id in program.atoms:
                atom = await self.atomspace.get_atom(atom_id)
                if atom:
                    match_count += 1
            
            return match_count / len(program.atoms)
        except Exception:
            return 0.1
    
    async def _test_cognitive_kernel(self, program: Program) -> float:
        """Test cognitive kernel program"""
        # Test cognitive processing capability
        if not program.atoms:
            return 0.1
        
        # Simple test: connectivity and truth value propagation
        try:
            total_truth_strength = 0.0
            for atom_id in program.atoms:
                tv = await self.pln_engine.infer_truth_value(atom_id)
                total_truth_strength += tv.strength
            
            return total_truth_strength / len(program.atoms)
        except Exception:
            return 0.1
    
    async def _selection(self) -> List[Program]:
        """Select programs for reproduction"""
        # Tournament selection
        tournament_size = max(2, int(self.population_size * self.selection_pressure))
        selected = []
        
        for _ in range(self.population_size):
            tournament = random.sample(self.population, tournament_size)
            winner = max(tournament, key=lambda p: p.fitness)
            selected.append(winner)
        
        return selected
    
    async def _crossover(self, parent1: Program, parent2: Program) -> Program:
        """Create offspring through crossover"""
        offspring_id = str(uuid.uuid4())
        
        # Combine atoms from both parents
        combined_atoms = list(set(parent1.atoms + parent2.atoms))
        
        # Select subset for offspring
        offspring_size = min(self.max_complexity, random.randint(1, len(combined_atoms)))
        offspring_atoms = random.sample(combined_atoms, offspring_size)
        
        offspring = Program(
            id=offspring_id,
            program_type=parent1.program_type,
            atoms=offspring_atoms,
            complexity=len(offspring_atoms),
            generation=max(parent1.generation, parent2.generation) + 1,
            parent_ids=[parent1.id, parent2.id]
        )
        
        return offspring
    
    async def _mutate(self, program: Program) -> Program:
        """Mutate a program"""
        mutated_program = copy.deepcopy(program)
        mutated_program.id = str(uuid.uuid4())
        mutated_program.parent_ids = [program.id]
        
        # Mutation operations
        mutation_type = random.choice(["add", "remove", "replace"])
        
        if mutation_type == "add" and len(mutated_program.atoms) < self.max_complexity:
            # Add a random atom
            available_atoms = await self._get_available_atoms()
            if available_atoms:
                new_atom = random.choice(available_atoms)
                if new_atom not in mutated_program.atoms:
                    mutated_program.atoms.append(new_atom)
        
        elif mutation_type == "remove" and len(mutated_program.atoms) > 1:
            # Remove a random atom
            mutated_program.atoms.remove(random.choice(mutated_program.atoms))
        
        elif mutation_type == "replace" and mutated_program.atoms:
            # Replace a random atom
            available_atoms = await self._get_available_atoms()
            if available_atoms:
                old_atom = random.choice(mutated_program.atoms)
                new_atom = random.choice(available_atoms)
                idx = mutated_program.atoms.index(old_atom)
                mutated_program.atoms[idx] = new_atom
        
        mutated_program.complexity = len(mutated_program.atoms)
        return mutated_program
    
    async def optimize_program(self, program_type: ProgramType, 
                             generations: int = 10,
                             seed_programs: Optional[List[Program]] = None) -> Dict[str, Any]:
        """
        Optimize programs of a specific type
        
        Args:
            program_type: Type of programs to optimize
            generations: Number of generations to evolve
            seed_programs: Optional seed programs
            
        Returns:
            Dictionary with optimization results
        """
        optimization_session = {
            "session_id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "program_type": program_type.value,
            "generations": generations,
            "results": []
        }
        
        # Initialize population
        await self.initialize_population(program_type, seed_programs)
        
        # Evolve generations
        for generation in range(generations):
            generation_stats = await self.evolve_generation()
            optimization_session["results"].append(generation_stats)
        
        # Get best program
        best_program = max(self.population, key=lambda p: p.fitness)
        optimization_session["best_program"] = best_program.to_dict()
        
        return optimization_session
    
    def get_best_programs(self, n: int = 5) -> List[Program]:
        """Get the n best programs from current population"""
        return sorted(self.population, key=lambda p: p.fitness, reverse=True)[:n]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get MOSES optimizer statistics"""
        if not self.population:
            return {
                "population_size": 0,
                "generation_count": self.generation_count,
                "best_fitness": 0.0,
                "average_fitness": 0.0
            }
        
        return {
            "population_size": len(self.population),
            "generation_count": self.generation_count,
            "best_fitness": max(p.fitness for p in self.population),
            "average_fitness": np.mean([p.fitness for p in self.population]),
            "average_complexity": np.mean([p.complexity for p in self.population]),
            "evolution_sessions": len(self.evolution_history)
        }
    
    async def save_program(self, program: Program) -> bool:
        """Save program to AtomSpace as specialized MOSES atom"""
        try:
            # Create MOSES program atom
            program_atom = await self.atomspace.add_moses_program(
                name=f"program_{program.id}",
                program_type=program.program_type.value,
                complexity=program.complexity,
                generation=program.generation,
                parent_ids=program.parent_ids,
                fitness_score=program.fitness,
                program_atoms=program.atoms,
                truth_value=program.fitness,
                confidence=min(1.0, program.fitness + 0.1),
                metadata=program.metadata
            )
            
            # Create fitness atom for detailed evaluation
            fitness_atom = await self.atomspace.add_moses_fitness(
                name=f"fitness_{program.id}",
                fitness_type="composite",
                fitness_value=program.fitness,
                confidence_interval=(max(0.0, program.fitness - 0.1), min(1.0, program.fitness + 0.1)),
                evaluation_context={
                    "program_type": program.program_type.value,
                    "complexity": program.complexity,
                    "generation": program.generation
                },
                program_id=program_atom.id,
                truth_value=program.fitness,
                confidence=min(1.0, program.fitness + 0.1)
            )
            
            return True
        except Exception as e:
            print(f"Error saving program {program.id}: {e}")
            return False
    
    async def save_population(self, population_name: str = None) -> bool:
        """Save current population to AtomSpace as MOSES population atom"""
        try:
            if not self.population:
                return False
            
            # Save all programs first
            program_atom_ids = []
            for program in self.population:
                if await self.save_program(program):
                    # Find the corresponding program atom
                    program_atoms = await self.atomspace.pattern_match({
                        'atom_type': AtomType.MOSES_PROGRAM.value,
                        'program_type': program.program_type.value
                    })
                    # Get the most recent one
                    if program_atoms:
                        program_atom_ids.append(program_atoms[-1].id)
            
            # Calculate population statistics
            fitness_scores = [p.fitness for p in self.population]
            best_fitness = max(fitness_scores) if fitness_scores else 0.0
            average_fitness = sum(fitness_scores) / len(fitness_scores) if fitness_scores else 0.0
            
            # Create population atom
            population_atom = await self.atomspace.add_moses_population(
                name=population_name or f"population_gen_{self.generation_count}",
                population_size=len(self.population),
                generation_count=self.generation_count,
                best_fitness=best_fitness,
                average_fitness=average_fitness,
                program_ids=program_atom_ids,
                truth_value=average_fitness,
                confidence=len(self.population) / max(1, self.population_size),
                metadata={
                    "mutation_rate": self.mutation_rate,
                    "crossover_rate": self.crossover_rate,
                    "selection_pressure": self.selection_pressure,
                    "max_complexity": self.max_complexity
                }
            )
            
            return True
        except Exception as e:
            print(f"Error saving population: {e}")
            return False
    
    async def load_program(self, program_id: str) -> Optional[Program]:
        """Load program from AtomSpace using MOSES program atoms"""
        try:
            # Try to find MOSES program atom first
            moses_programs = await self.atomspace.pattern_match({
                'atom_type': AtomType.MOSES_PROGRAM.value
            }, limit=1000)
            
            for program_atom in moses_programs:
                if isinstance(program_atom, MOSESProgramAtom):
                    # Check if this matches our program ID or contains it in metadata
                    if (program_atom.id == program_id or 
                        program_id in str(program_atom.metadata)):
                        
                        # Create Program object from MOSES atom
                        program = Program(
                            id=program_atom.id,
                            program_type=ProgramType(program_atom.program_type),
                            atoms=program_atom.program_atoms,
                            fitness=program_atom.fitness_score,
                            complexity=program_atom.complexity,
                            generation=program_atom.generation,
                            parent_ids=program_atom.parent_ids,
                            metadata=program_atom.metadata,
                            timestamp=program_atom.timestamp
                        )
                        return program
            
            # Fallback to old method for backward compatibility
            pattern = {"atom_type": AtomType.LINK.value, "link_type": "program"}
            links = await self.atomspace.storage.get_atoms_by_pattern(pattern, limit=1000)
            
            for link in links:
                if hasattr(link, 'metadata') and link.metadata:
                    try:
                        metadata = json.loads(link.metadata) if isinstance(link.metadata, str) else link.metadata
                        if metadata.get("id") == program_id:
                            return Program.from_dict(metadata)
                    except (json.JSONDecodeError, KeyError):
                        continue
            
            return None
        except Exception as e:
            print(f"Error loading program {program_id}: {e}")
            return None
    
    async def load_population(self, population_id: str = None) -> List[Program]:
        """Load population from AtomSpace using MOSES population atoms"""
        try:
            population_atoms = await self.atomspace.pattern_match({
                'atom_type': AtomType.MOSES_POPULATION.value
            })
            
            if not population_atoms:
                return []
            
            # Get the most recent population if no ID specified
            if population_id is None:
                population_atom = max(population_atoms, key=lambda p: p.timestamp)
            else:
                population_atom = next((p for p in population_atoms if p.id == population_id), None)
                if not population_atom:
                    return []
            
            # Load all programs in the population
            programs = []
            for program_atom_id in population_atom.program_ids:
                program_atom = await self.atomspace.get_atom(program_atom_id)
                if isinstance(program_atom, MOSESProgramAtom):
                    program = Program(
                        id=program_atom.id,
                        program_type=ProgramType(program_atom.program_type),
                        atoms=program_atom.program_atoms,
                        fitness=program_atom.fitness_score,
                        complexity=program_atom.complexity,
                        generation=program_atom.generation,
                        parent_ids=program_atom.parent_ids,
                        metadata=program_atom.metadata,
                        timestamp=program_atom.timestamp
                    )
                    programs.append(program)
            
            return programs
        except Exception as e:
            print(f"Error loading population: {e}")
            return []