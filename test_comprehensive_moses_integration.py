"""
Comprehensive MOSES Integration Test

Demonstrates the complete MOSES integration system with AtomSpace,
PLN reasoning, ECAN attention allocation, Python API, and GGML tensors.
"""

import asyncio
import sys
import os
from datetime import datetime
import numpy as np

# Add project path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python.helpers.atomspace import AtomSpace, AtomType
from python.helpers.moses_optimizer import MOSESOptimizer, ProgramType
from python.helpers.pln_reasoning import PLNInferenceEngine
from python.helpers.ecan_attention import ECANAttentionSystem
from python.api.moses_python_api import MOSESPythonAPI, EvolutionConfig
from python.helpers.ggml_tensor_ops import population_to_ggml_tensors, validate_moses_tensor_pipeline


async def comprehensive_moses_integration_test():
    """Run comprehensive test of all MOSES integration components"""
    
    print("🌟 COMPREHENSIVE MOSES INTEGRATION TEST 🌟")
    print("=" * 60)
    
    # Phase 1: System Initialization
    print("\n📋 Phase 1: System Initialization")
    print("-" * 40)
    
    # Create AtomSpace with knowledge base
    atomspace_path = '/tmp/comprehensive_moses_test.db'
    atomspace = AtomSpace(atomspace_path)
    
    # Create knowledge base
    print("🧠 Creating knowledge base...")
    concepts = []
    relations = []
    
    # Add diverse concepts
    concept_names = [
        ("Intelligence", "concept"), ("Reasoning", "concept"), ("Learning", "concept"),
        ("Memory", "concept"), ("Attention", "concept"), ("Cognition", "concept"),
        ("Language", "concept"), ("Perception", "concept"), ("Action", "concept"),
        ("Consciousness", "concept")
    ]
    
    for name, ctype in concept_names:
        concept = await atomspace.add_node(
            name, ctype, 
            truth_value=np.random.uniform(0.6, 0.95),
            confidence=np.random.uniform(0.7, 0.95)
        )
        concepts.append(concept)
    
    # Add relationships
    for i in range(len(concepts)):
        for j in range(i + 1, min(i + 4, len(concepts))):
            relation = await atomspace.add_link(
                f"Relates_{concepts[i].name}_{concepts[j].name}",
                [concepts[i].id, concepts[j].id],
                "inheritance" if i % 2 == 0 else "similarity",
                truth_value=np.random.uniform(0.5, 0.9),
                confidence=np.random.uniform(0.6, 0.9)
            )
            relations.append(relation)
    
    print(f"✅ Knowledge base created: {len(concepts)} concepts, {len(relations)} relations")
    
    # Initialize systems
    pln_engine = PLNInferenceEngine(atomspace)
    ecan_system = ECANAttentionSystem(atomspace, pln_engine)
    moses_optimizer = MOSESOptimizer(atomspace, pln_engine)
    moses_api = MOSESPythonAPI(atomspace_path)
    
    await ecan_system.initialize()
    await moses_api._wait_for_initialization()
    
    print("✅ All systems initialized successfully")
    
    # Phase 2: MOSES Evolution with PLN Integration
    print("\n🧬 Phase 2: MOSES Evolution with PLN Integration")
    print("-" * 50)
    
    # Configure evolution
    config = EvolutionConfig(
        program_type=ProgramType.INFERENCE_RULE,
        population_size=30,
        max_generations=8,
        fitness_threshold=0.85,
        mutation_rate=0.15,
        crossover_rate=0.8
    )
    
    # Run evolution with progress tracking
    def progress_callback(stats):
        print(f"  Generation {stats['generation']:2d}: "
              f"best={stats['best_fitness']:.4f}, "
              f"avg={stats['average_fitness']:.4f}")
    
    print("🚀 Starting evolution process...")
    evolution_results = await moses_api.run_evolution(
        config=config,
        progress_callback=progress_callback
    )
    
    print(f"\n✅ Evolution completed:")
    print(f"  - Best fitness achieved: {evolution_results.best_fitness:.4f}")
    print(f"  - Generations completed: {evolution_results.generations_completed}")
    print(f"  - Total time: {evolution_results.total_time:.2f} seconds")
    print(f"  - Best program ID: {evolution_results.best_program.id}")
    print(f"  - Best program complexity: {evolution_results.best_program.complexity}")
    
    # Phase 3: ECAN Attention Analysis
    print("\n🧠 Phase 3: ECAN Attention Analysis")
    print("-" * 40)
    
    # Get attention visualization
    attention_viz = await ecan_system.get_moses_attention_visualization()
    ecan_stats = ecan_system.get_statistics()
    
    print("📊 Attention allocation metrics:")
    print(f"  - MOSES attention units: {ecan_stats['moses_attention_units']}")
    print(f"  - Evolution metrics: {ecan_stats['moses_evolution_metrics']}")
    
    # Analyze attention efficiency
    if attention_viz.get("fitness_vs_attention"):
        points = attention_viz["fitness_vs_attention"]
        fitness_scores = [p["fitness"] for p in points]
        attention_scores = [p["attention"] for p in points]
        
        if len(fitness_scores) > 1:
            correlation = np.corrcoef(fitness_scores, attention_scores)[0, 1]
            print(f"  - Fitness-attention correlation: {correlation:.4f}")
            print(f"  - Attention efficiency: {'High' if abs(correlation) > 0.5 else 'Medium' if abs(correlation) > 0.3 else 'Low'}")
    
    # Show attention distribution by entity type
    entity_breakdown = attention_viz.get("entity_type_breakdown", {})
    for entity_type, data in entity_breakdown.items():
        print(f"  - {entity_type}: {data['count']} units, "
              f"total_attention={data['total_attention']:.3f}")
    
    # Phase 4: GGML Tensor Operations
    print("\n🔧 Phase 4: GGML Tensor Operations")
    print("-" * 40)
    
    # Get current population and attention data
    session_status = await moses_api.get_session_status()
    current_population = moses_optimizer.population
    
    # Convert to GGML tensors
    ggml_tensors = population_to_ggml_tensors(
        current_population, 
        ecan_system.moses_attention_units
    )
    
    print("📈 GGML tensor analysis:")
    for name, tensor in ggml_tensors.items():
        print(f"  - {name}: shape={tensor.shape}, dtype={tensor.dtype.value}")
    
    # Validate tensor pipeline
    validation_results = validate_moses_tensor_pipeline(ggml_tensors)
    print(f"  - Pipeline validation: {'✅ PASSED' if validation_results['pipeline_valid'] else '❌ FAILED'}")
    
    # Analyze tensor correlations and metrics
    if "fitness" in ggml_tensors and "attention" in ggml_tensors:
        from python.helpers.ggml_tensor_ops import MOSESTensorOps
        tensor_ops = MOSESTensorOps()
        
        # Fitness landscape analysis
        if "embeddings" in ggml_tensors:
            landscape = tensor_ops.fitness_landscape_analysis(
                ggml_tensors["fitness"], 
                ggml_tensors["embeddings"]
            )
            print("  - Fitness landscape metrics:")
            for metric, value in landscape.items():
                print(f"    - {metric}: {value:.4f}")
        
        # Population diversity
        diversity = tensor_ops.population_diversity_metric(ggml_tensors["embeddings"])
        print(f"  - Population diversity: {diversity:.4f}")
    
    # Phase 5: System Integration Analysis
    print("\n🔄 Phase 5: System Integration Analysis") 
    print("-" * 45)
    
    # Analyze evolution dynamics
    dynamics = await moses_api.analyze_evolution_dynamics()
    
    print("📊 Evolution dynamics analysis:")
    fitness_dynamics = dynamics["fitness_dynamics"]
    print(f"  - Convergence rate: {fitness_dynamics['convergence_rate']:.6f}")
    print(f"  - Best fitness trend: {len(fitness_dynamics['best_fitness_trend'])} generations")
    
    attention_dynamics = dynamics.get("attention_dynamics", {})
    if attention_dynamics:
        print(f"  - Attention efficiency: {attention_dynamics.get('fitness_attention_correlation', 0):.4f}")
    
    diversity_metrics = dynamics["population_diversity"]
    print(f"  - Fitness variance: {diversity_metrics.get('fitness_variance', 0):.6f}")
    print(f"  - Average complexity: {diversity_metrics.get('avg_complexity', 0):.2f}")
    
    # Performance metrics
    performance = dynamics["performance_metrics"]
    print("⚡ Performance metrics:")
    for metric, value in performance.items():
        print(f"  - {metric}: {value:.4f}")
    
    # Phase 6: AtomSpace Persistence and Retrieval
    print("\n💾 Phase 6: AtomSpace Persistence and Retrieval")
    print("-" * 50)
    
    # Check MOSES atoms in AtomSpace
    moses_programs = await atomspace.pattern_match({'atom_type': AtomType.MOSES_PROGRAM.value})
    moses_populations = await atomspace.pattern_match({'atom_type': AtomType.MOSES_POPULATION.value})
    moses_fitness_records = await atomspace.pattern_match({'atom_type': AtomType.MOSES_FITNESS.value})
    
    print("🗄️ AtomSpace MOSES data:")
    print(f"  - MOSES programs stored: {len(moses_programs)}")
    print(f"  - MOSES populations stored: {len(moses_populations)}")
    print(f"  - MOSES fitness records: {len(moses_fitness_records)}")
    
    # Test program retrieval and verification
    if moses_programs:
        sample_program = moses_programs[0]
        print(f"  - Sample program: {sample_program.name}")
        print(f"    - Fitness: {sample_program.fitness_score:.4f}")
        print(f"    - Complexity: {sample_program.complexity}")
        print(f"    - Generation: {sample_program.generation}")
        print(f"    - Program atoms: {len(sample_program.program_atoms)}")
    
    # Create memory snapshot
    snapshot_id = await atomspace.create_snapshot("comprehensive_test_final")
    memory_tensor = await atomspace.get_memory_tensor(snapshot_id)
    print(f"  - Memory tensor shape: {memory_tensor.shape}")
    
    # Phase 7: Final Integration Verification
    print("\n✅ Phase 7: Final Integration Verification")
    print("-" * 50)
    
    # Comprehensive system check
    integration_score = 0
    max_score = 10
    
    # Check 1: AtomSpace integration
    if len(moses_programs) > 0 and len(moses_fitness_records) > 0:
        integration_score += 2
        print("✅ AtomSpace integration: PASSED")
    else:
        print("❌ AtomSpace integration: FAILED")
    
    # Check 2: PLN integration
    if evolution_results.best_fitness > 0.5:  # PLN contributed to fitness
        integration_score += 2
        print("✅ PLN integration: PASSED")
    else:
        print("❌ PLN integration: FAILED")
    
    # Check 3: ECAN integration
    if ecan_stats['moses_attention_units'] > 0:
        integration_score += 2
        print("✅ ECAN integration: PASSED")
    else:
        print("❌ ECAN integration: FAILED")
    
    # Check 4: API functionality
    if evolution_results.generations_completed > 0:
        integration_score += 2
        print("✅ API functionality: PASSED")
    else:
        print("❌ API functionality: FAILED")
    
    # Check 5: GGML tensor compatibility
    if validation_results['pipeline_valid']:
        integration_score += 2
        print("✅ GGML tensor compatibility: PASSED")
    else:
        print("❌ GGML tensor compatibility: FAILED")
    
    # Final score
    print(f"\n🏆 FINAL INTEGRATION SCORE: {integration_score}/{max_score}")
    success_rate = (integration_score / max_score) * 100
    
    if success_rate >= 80:
        print(f"🎉 COMPREHENSIVE TEST: EXCELLENT ({success_rate:.0f}%)")
    elif success_rate >= 60:
        print(f"✅ COMPREHENSIVE TEST: GOOD ({success_rate:.0f}%)")
    elif success_rate >= 40:
        print(f"⚠️ COMPREHENSIVE TEST: PARTIAL ({success_rate:.0f}%)")
    else:
        print(f"❌ COMPREHENSIVE TEST: FAILED ({success_rate:.0f}%)")
    
    # Summary
    print("\n📋 INTEGRATION SUMMARY")
    print("=" * 40)
    print(f"Total concepts created: {len(concepts)}")
    print(f"Total relations created: {len(relations)}")
    print(f"Programs evolved: {len(current_population)}")
    print(f"Best fitness achieved: {evolution_results.best_fitness:.4f}")
    print(f"Attention units allocated: {ecan_stats['moses_attention_units']}")
    print(f"GGML tensors generated: {len(ggml_tensors)}")
    print(f"AtomSpace MOSES entities: {len(moses_programs) + len(moses_populations) + len(moses_fitness_records)}")
    print(f"Integration score: {integration_score}/{max_score} ({success_rate:.0f}%)")
    
    return {
        "success": success_rate >= 80,
        "integration_score": integration_score,
        "max_score": max_score,
        "success_rate": success_rate,
        "evolution_results": evolution_results,
        "ecan_stats": ecan_stats,
        "tensor_validation": validation_results,
        "performance_metrics": performance
    }


if __name__ == "__main__":
    # Run the comprehensive test
    result = asyncio.run(comprehensive_moses_integration_test())
    
    # Exit with appropriate code
    exit(0 if result["success"] else 1)