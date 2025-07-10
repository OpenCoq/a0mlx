"""
MOSES Evolution REST API

Provides REST endpoints for controlling MOSES evolutionary program optimization,
including population management, evolution control, fitness evaluation, and
real-time monitoring of the evolutionary process.
"""

import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from flask import Flask, request, jsonify
import numpy as np

# Import MOSES and related systems
from ..helpers.atomspace import AtomSpace, AtomType
from ..helpers.moses_optimizer import MOSESOptimizer, ProgramType, Program
from ..helpers.pln_reasoning import PLNInferenceEngine
from ..helpers.ecan_attention import ECANAttentionSystem


class MOSESEvolutionAPI:
    """REST API for MOSES evolutionary system"""
    
    def __init__(self, atomspace_path: str = "/tmp/moses_api.db"):
        self.atomspace = AtomSpace(atomspace_path)
        self.pln_engine = PLNInferenceEngine(self.atomspace)
        self.ecan_system = ECANAttentionSystem(self.atomspace, self.pln_engine)
        self.moses_optimizer = MOSESOptimizer(self.atomspace, self.pln_engine)
        
        # Evolution session tracking
        self.evolution_sessions: Dict[str, Dict[str, Any]] = {}
        self.active_session_id: Optional[str] = None
        
        # API statistics
        self.api_stats = {
            "total_requests": 0,
            "evolution_sessions_created": 0,
            "generations_evolved": 0,
            "programs_created": 0,
            "fitness_evaluations": 0
        }
        
        # Initialize systems
        asyncio.create_task(self._initialize_systems())
    
    async def _initialize_systems(self):
        """Initialize all systems asynchronously"""
        try:
            await self.ecan_system.initialize()
            print("✅ MOSES Evolution API systems initialized")
        except Exception as e:
            print(f"❌ Error initializing MOSES API systems: {e}")
    
    def create_evolution_session(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new evolution session"""
        try:
            self.api_stats["total_requests"] += 1
            
            session_id = str(uuid.uuid4())
            program_type = ProgramType(data.get("program_type", "inference_rule"))
            population_size = data.get("population_size", 50)
            
            session = {
                "session_id": session_id,
                "program_type": program_type.value,
                "population_size": population_size,
                "generation_count": 0,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "status": "created",
                "parameters": {
                    "mutation_rate": data.get("mutation_rate", 0.1),
                    "crossover_rate": data.get("crossover_rate", 0.7),
                    "selection_pressure": data.get("selection_pressure", 0.3),
                    "max_complexity": data.get("max_complexity", 20)
                },
                "generation_history": [],
                "best_program": None,
                "tensor_shapes": {
                    "population_tensor": [population_size, data.get("max_complexity", 20)],
                    "fitness_tensor": [population_size],
                    "attention_tensor": [population_size, 5],  # 5 attention dimensions
                    "generation_tensor": [100, 3]  # 100 generations, 3 metrics
                }
            }
            
            self.evolution_sessions[session_id] = session
            self.active_session_id = session_id
            self.api_stats["evolution_sessions_created"] += 1
            
            return {
                "success": True,
                "session_id": session_id,
                "session": session
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def initialize_population(self, session_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize population for evolution session"""
        try:
            self.api_stats["total_requests"] += 1
            
            if session_id not in self.evolution_sessions:
                return {"success": False, "error": "Session not found"}
            
            session = self.evolution_sessions[session_id]
            program_type = ProgramType(session["program_type"])
            
            # Set MOSES parameters from session
            self.moses_optimizer.population_size = session["population_size"]
            self.moses_optimizer.mutation_rate = session["parameters"]["mutation_rate"]
            self.moses_optimizer.crossover_rate = session["parameters"]["crossover_rate"]
            self.moses_optimizer.selection_pressure = session["parameters"]["selection_pressure"]
            self.moses_optimizer.max_complexity = session["parameters"]["max_complexity"]
            
            # Initialize population
            seed_programs = data.get("seed_programs", None)
            if seed_programs:
                seed_programs = [Program.from_dict(p) for p in seed_programs]
            
            population = await self.moses_optimizer.initialize_population(program_type, seed_programs)
            
            # Save population to AtomSpace
            await self.moses_optimizer.save_population(f"session_{session_id}_initial")
            
            # Update session
            session["status"] = "initialized"
            session["population"] = [p.to_dict() for p in population]
            
            # Update statistics
            self.api_stats["programs_created"] += len(population)
            
            return {
                "success": True,
                "population_size": len(population),
                "population": session["population"],
                "tensor_shapes": session["tensor_shapes"]
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def evolve_generation(self, session_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Evolve one generation"""
        try:
            self.api_stats["total_requests"] += 1
            
            if session_id not in self.evolution_sessions:
                return {"success": False, "error": "Session not found"}
            
            session = self.evolution_sessions[session_id]
            
            if session["status"] != "initialized" and session["status"] != "evolving":
                return {"success": False, "error": "Session not initialized"}
            
            # Evolve generation
            generation_stats = await self.moses_optimizer.evolve_generation()
            
            # Update ECAN attention based on generation results
            await self.ecan_system.update_from_moses_generation(
                generation_stats, 
                self.moses_optimizer.population
            )
            
            # Get best program
            best_program = max(self.moses_optimizer.population, key=lambda p: p.fitness)
            
            # Create generation record
            generation_record = {
                "generation": generation_stats["generation"],
                "timestamp": generation_stats["timestamp"],
                "duration": generation_stats["duration"],
                "best_fitness": generation_stats["best_fitness"],
                "average_fitness": generation_stats["average_fitness"],
                "population_size": generation_stats["population_size"],
                "best_program": best_program.to_dict(),
                "tensor_data": {
                    "fitness_tensor": [p.fitness for p in self.moses_optimizer.population],
                    "complexity_tensor": [p.complexity for p in self.moses_optimizer.population],
                    "generation_tensor": [p.generation for p in self.moses_optimizer.population]
                }
            }
            
            # Update session
            session["generation_count"] = generation_stats["generation"]
            session["status"] = "evolving"
            session["generation_history"].append(generation_record)
            session["best_program"] = best_program.to_dict()
            
            # Save generation state
            await self.moses_optimizer.save_population(f"session_{session_id}_gen_{generation_stats['generation']}")
            
            # Update statistics
            self.api_stats["generations_evolved"] += 1
            self.api_stats["fitness_evaluations"] += generation_stats["population_size"]
            
            return {
                "success": True,
                "generation_stats": generation_record,
                "session_status": {
                    "generation_count": session["generation_count"],
                    "best_fitness": session["best_program"]["fitness"],
                    "status": session["status"]
                }
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """Get detailed session status and metrics"""
        try:
            self.api_stats["total_requests"] += 1
            
            if session_id not in self.evolution_sessions:
                return {"success": False, "error": "Session not found"}
            
            session = self.evolution_sessions[session_id]
            
            # Get ECAN attention data
            ecan_stats = self.ecan_system.get_statistics()
            moses_visualization = await self.ecan_system.get_moses_attention_visualization()
            
            # Get MOSES statistics
            moses_stats = self.moses_optimizer.get_statistics()
            
            # Calculate performance metrics
            performance_metrics = {
                "generations_per_minute": 0.0,
                "fitness_improvement_rate": 0.0,
                "attention_efficiency": 0.0,
                "resource_utilization": 0.0
            }
            
            if session["generation_history"]:
                # Calculate performance metrics
                history = session["generation_history"]
                if len(history) >= 2:
                    time_span = (datetime.fromisoformat(history[-1]["timestamp"]) - 
                               datetime.fromisoformat(history[0]["timestamp"])).total_seconds() / 60
                    if time_span > 0:
                        performance_metrics["generations_per_minute"] = len(history) / time_span
                
                # Fitness improvement rate
                first_fitness = history[0]["best_fitness"]
                last_fitness = history[-1]["best_fitness"]
                if first_fitness > 0:
                    performance_metrics["fitness_improvement_rate"] = (last_fitness - first_fitness) / first_fitness
                
                # Attention efficiency (high attention on high fitness)
                if moses_visualization.get("fitness_vs_attention"):
                    points = moses_visualization["fitness_vs_attention"]
                    fitness_attention_correlation = np.corrcoef(
                        [p["fitness"] for p in points],
                        [p["attention"] for p in points]
                    )[0, 1] if len(points) > 1 else 0.0
                    performance_metrics["attention_efficiency"] = max(0.0, fitness_attention_correlation)
            
            # Resource utilization
            if ecan_stats.get("resource_pools"):
                utilizations = [pool["utilization"] for pool in ecan_stats["resource_pools"].values()]
                performance_metrics["resource_utilization"] = np.mean(utilizations)
            
            return {
                "success": True,
                "session": session,
                "moses_stats": moses_stats,
                "ecan_stats": ecan_stats,
                "attention_visualization": moses_visualization,
                "performance_metrics": performance_metrics,
                "api_stats": self.api_stats
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def get_population_tensors(self, session_id: str) -> Dict[str, Any]:
        """Get population data as tensor arrays for analysis"""
        try:
            self.api_stats["total_requests"] += 1
            
            if session_id not in self.evolution_sessions:
                return {"success": False, "error": "Session not found"}
            
            session = self.evolution_sessions[session_id]
            
            if not self.moses_optimizer.population:
                return {"success": False, "error": "No population data available"}
            
            # Create tensor representations
            population = self.moses_optimizer.population
            
            # Fitness tensor
            fitness_tensor = np.array([p.fitness for p in population])
            
            # Complexity tensor
            complexity_tensor = np.array([p.complexity for p in population])
            
            # Generation tensor
            generation_tensor = np.array([p.generation for p in population])
            
            # Program atoms tensor (padded to max complexity)
            max_atoms = max(len(p.atoms) for p in population) if population else 1
            atom_tensor = np.zeros((len(population), max_atoms), dtype=object)
            for i, program in enumerate(population):
                for j, atom_id in enumerate(program.atoms):
                    if j < max_atoms:
                        atom_tensor[i, j] = atom_id
            
            # Attention tensor from ECAN
            attention_tensor = np.zeros((len(population), 5))  # 5 attention dimensions
            for i, program in enumerate(population):
                # Find corresponding attention unit
                for unit in self.ecan_system.moses_attention_units.values():
                    if unit.atom_id == program.id:
                        attention_tensor[i] = [
                            unit.attention_value,
                            unit.importance,
                            unit.confidence,
                            unit.fitness_score,
                            unit.evolution_priority
                        ]
                        break
            
            # Evolution trajectory tensor
            if session["generation_history"]:
                history = session["generation_history"]
                trajectory_tensor = np.array([
                    [gen["generation"], gen["best_fitness"], gen["average_fitness"]]
                    for gen in history
                ])
            else:
                trajectory_tensor = np.array([])
            
            return {
                "success": True,
                "tensor_shapes": {
                    "fitness_tensor": fitness_tensor.shape,
                    "complexity_tensor": complexity_tensor.shape,
                    "generation_tensor": generation_tensor.shape,
                    "atom_tensor": atom_tensor.shape,
                    "attention_tensor": attention_tensor.shape,
                    "trajectory_tensor": trajectory_tensor.shape
                },
                "tensors": {
                    "fitness_tensor": fitness_tensor.tolist(),
                    "complexity_tensor": complexity_tensor.tolist(),
                    "generation_tensor": generation_tensor.tolist(),
                    "atom_tensor": atom_tensor.tolist(),
                    "attention_tensor": attention_tensor.tolist(),
                    "trajectory_tensor": trajectory_tensor.tolist()
                },
                "metadata": {
                    "population_size": len(population),
                    "max_atoms_per_program": max_atoms,
                    "generation_count": session["generation_count"],
                    "attention_dimensions": ["attention_value", "importance", "confidence", "fitness_score", "evolution_priority"]
                }
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def optimize_programs(self, session_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run multi-generation optimization"""
        try:
            self.api_stats["total_requests"] += 1
            
            if session_id not in self.evolution_sessions:
                return {"success": False, "error": "Session not found"}
            
            generations = data.get("generations", 10)
            
            optimization_results = []
            
            for gen in range(generations):
                result = await self.evolve_generation(session_id, {})
                if not result["success"]:
                    break
                optimization_results.append(result["generation_stats"])
                
                # Early stopping if fitness converged
                if len(optimization_results) >= 3:
                    recent_fitness = [r["best_fitness"] for r in optimization_results[-3:]]
                    if max(recent_fitness) - min(recent_fitness) < 0.001:
                        break
            
            session = self.evolution_sessions[session_id]
            session["status"] = "completed"
            
            return {
                "success": True,
                "generations_completed": len(optimization_results),
                "final_best_fitness": optimization_results[-1]["best_fitness"] if optimization_results else 0.0,
                "optimization_results": optimization_results,
                "session_id": session_id
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def list_sessions(self) -> Dict[str, Any]:
        """List all evolution sessions"""
        try:
            self.api_stats["total_requests"] += 1
            
            sessions_summary = []
            for session_id, session in self.evolution_sessions.items():
                summary = {
                    "session_id": session_id,
                    "program_type": session["program_type"],
                    "status": session["status"],
                    "generation_count": session["generation_count"],
                    "created_at": session["created_at"],
                    "best_fitness": session["best_program"]["fitness"] if session["best_program"] else 0.0,
                    "population_size": session["population_size"]
                }
                sessions_summary.append(summary)
            
            return {
                "success": True,
                "sessions": sessions_summary,
                "total_sessions": len(sessions_summary),
                "active_session": self.active_session_id
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_api_documentation(self) -> Dict[str, Any]:
        """Get API documentation with tensor shapes and interface contracts"""
        return {
            "success": True,
            "api_version": "1.0.0",
            "description": "MOSES Evolution REST API for neural-symbolic program optimization",
            "endpoints": {
                "POST /moses/sessions": {
                    "description": "Create new evolution session",
                    "parameters": {
                        "program_type": "string (inference_rule, pattern_matcher, cognitive_kernel, behavior_tree)",
                        "population_size": "integer (default: 50)",
                        "mutation_rate": "float (default: 0.1)",
                        "crossover_rate": "float (default: 0.7)",
                        "selection_pressure": "float (default: 0.3)",
                        "max_complexity": "integer (default: 20)"
                    },
                    "returns": "session_id, session config, tensor_shapes"
                },
                "POST /moses/sessions/{session_id}/initialize": {
                    "description": "Initialize population for session",
                    "parameters": {
                        "seed_programs": "optional array of program objects"
                    },
                    "returns": "population data, tensor shapes"
                },
                "POST /moses/sessions/{session_id}/evolve": {
                    "description": "Evolve one generation",
                    "returns": "generation stats, best program, tensor data"
                },
                "GET /moses/sessions/{session_id}/status": {
                    "description": "Get session status and metrics",
                    "returns": "session data, MOSES stats, ECAN stats, attention visualization, performance metrics"
                },
                "GET /moses/sessions/{session_id}/tensors": {
                    "description": "Get population data as tensors",
                    "returns": "fitness_tensor, complexity_tensor, attention_tensor, trajectory_tensor"
                },
                "POST /moses/sessions/{session_id}/optimize": {
                    "description": "Run multi-generation optimization",
                    "parameters": {
                        "generations": "integer (default: 10)"
                    },
                    "returns": "optimization results, final fitness"
                },
                "GET /moses/sessions": {
                    "description": "List all sessions",
                    "returns": "session summaries"
                }
            },
            "tensor_shapes": {
                "fitness_tensor": "[population_size]",
                "complexity_tensor": "[population_size]", 
                "attention_tensor": "[population_size, 5]",
                "atom_tensor": "[population_size, max_atoms]",
                "trajectory_tensor": "[generations, 3]"
            },
            "data_types": {
                "Program": {
                    "id": "string",
                    "program_type": "string",
                    "atoms": "array of atom_ids",
                    "fitness": "float",
                    "complexity": "integer",
                    "generation": "integer",
                    "parent_ids": "array of strings",
                    "metadata": "object"
                },
                "GenerationStats": {
                    "generation": "integer",
                    "best_fitness": "float",
                    "average_fitness": "float",
                    "population_size": "integer",
                    "duration": "float",
                    "tensor_data": "object"
                }
            }
        }


# Flask API routes (to be integrated with existing API system)
def create_moses_routes(app: Flask, api: MOSESEvolutionAPI):
    """Create Flask routes for MOSES Evolution API"""
    
    @app.route('/moses/sessions', methods=['POST'])
    def create_session():
        data = request.get_json() or {}
        result = api.create_evolution_session(data)
        return jsonify(result)
    
    @app.route('/moses/sessions/<session_id>/initialize', methods=['POST'])
    def initialize_session(session_id):
        data = request.get_json() or {}
        result = asyncio.run(api.initialize_population(session_id, data))
        return jsonify(result)
    
    @app.route('/moses/sessions/<session_id>/evolve', methods=['POST'])
    def evolve_session(session_id):
        data = request.get_json() or {}
        result = asyncio.run(api.evolve_generation(session_id, data))
        return jsonify(result)
    
    @app.route('/moses/sessions/<session_id>/status', methods=['GET'])
    def get_session_status(session_id):
        result = asyncio.run(api.get_session_status(session_id))
        return jsonify(result)
    
    @app.route('/moses/sessions/<session_id>/tensors', methods=['GET'])
    def get_session_tensors(session_id):
        result = asyncio.run(api.get_population_tensors(session_id))
        return jsonify(result)
    
    @app.route('/moses/sessions/<session_id>/optimize', methods=['POST'])
    def optimize_session(session_id):
        data = request.get_json() or {}
        result = asyncio.run(api.optimize_programs(session_id, data))
        return jsonify(result)
    
    @app.route('/moses/sessions', methods=['GET'])
    def list_sessions():
        result = api.list_sessions()
        return jsonify(result)
    
    @app.route('/moses/docs', methods=['GET'])
    def get_docs():
        result = api.get_api_documentation()
        return jsonify(result)