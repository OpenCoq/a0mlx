"""
Evolutionary Trajectory Demonstration Script

Demonstrates the evolutionary enhancements to the distributed intelligence system:
1. LLM-based goal decomposition
2. ECAN attention allocation with dynamic modulation
3. P-System membrane boundaries for cognitive encapsulation
4. Sensorimotor integration interface
5. Tensor morphology analysis system
"""

import asyncio
import sys
import os
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

# Add the project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python.helpers.atomspace import AtomSpace
from python.helpers.distributed_orchestrator import DistributedOrchestrator, TaskPriority
from python.helpers.ecan_attention import ECANAttentionSystem, AttentionType, ResourceType
from python.helpers.neural_symbolic_reasoning import NeuralSymbolicReasoningEngine
from python.helpers.p_system_membranes import PSystemMembraneNetwork, MembraneType, CognitiveObject
from python.helpers.sensorimotor_integration import (
    SensorimotorIntegrationInterface, SensorType, ActuatorType, 
    SensorConfig, ActuatorConfig, PerceptionActionLoop
)
from python.helpers.tensor_morphology_analyzer import TensorMorphologyAnalyzer, TensorType
from agent import AgentContext


class EvolutionaryTrajectoryDemo:
    """Comprehensive demonstration of evolutionary trajectory enhancements"""
    
    def __init__(self):
        self.demo_db_path = "/tmp/demo_evolutionary_trajectory.db"
        
        # Core systems
        self.atomspace = None
        self.orchestrator = None
        self.ecan_system = None
        self.reasoning_engine = None
        self.membrane_network = None
        self.sensorimotor_interface = None
        self.tensor_analyzer = None
        
        # Demo agents
        self.demo_agents = []
        
    async def setup_evolutionary_system(self):
        """Setup the complete evolutionary trajectory system"""
        print("🚀 Evolutionary Trajectory of Distributed Intelligence Systems")
        print("=" * 70)
        print("🔧 Setting up evolutionary cognitive architecture...")
        
        # 1. Initialize AtomSpace (hypergraph substrate)
        if os.path.exists(self.demo_db_path):
            os.remove(self.demo_db_path)
        
        self.atomspace = AtomSpace(self.demo_db_path)
        print("   ✅ AtomSpace hypergraph substrate initialized")
        
        # 2. Initialize Neural-Symbolic Reasoning Engine
        self.reasoning_engine = NeuralSymbolicReasoningEngine(self.atomspace)
        await self.reasoning_engine.initialize_system()
        print("   ✅ Neural-Symbolic reasoning engine activated")
        
        # 3. Initialize ECAN Attention System with dynamic modulation
        self.ecan_system = ECANAttentionSystem(self.atomspace)
        await self.ecan_system.initialize()
        print("   ✅ ECAN attention system with dynamic modulation online")
        
        # 4. Initialize Distributed Orchestrator with LLM decomposition
        self.orchestrator = DistributedOrchestrator()
        print("   ✅ Distributed orchestrator with LLM-based decomposition ready")
        
        # 5. Initialize P-System Membrane Network
        self.membrane_network = PSystemMembraneNetwork(self.atomspace)
        await self.membrane_network.initialize_cognitive_architecture()
        print("   ✅ P-System membrane boundaries for cognitive encapsulation established")
        
        # 6. Initialize Sensorimotor Integration Interface
        self.sensorimotor_interface = SensorimotorIntegrationInterface(
            self.atomspace, self.membrane_network
        )
        await self.sensorimotor_interface.initialize_embodiment_interface()
        print("   ✅ Sensorimotor integration interface for embodied agents online")
        
        # 7. Initialize Tensor Morphology Analyzer
        self.tensor_analyzer = TensorMorphologyAnalyzer(
            self.orchestrator, self.ecan_system, self.atomspace
        )
        await self.tensor_analyzer.initialize_tensor_analysis()
        print("   ✅ Tensor morphology analysis system activated")
        
        print("\n🌟 Evolutionary cognitive architecture fully operational!")
        
    async def demonstrate_llm_goal_decomposition(self):
        """Demonstrate LLM-based goal decomposition"""
        print("\n" + "="*70)
        print("🧠 1. LLM-BASED GOAL DECOMPOSITION")
        print("="*70)
        
        # Create demo agents for the orchestrator
        await self._setup_demo_agents()
        
        # Test complex goal decomposition
        complex_goals = [
            "Design and implement a distributed neural network training system with fault tolerance",
            "Develop an autonomous robotics swarm for environmental monitoring",
            "Create a real-time adaptive recommendation engine with personalization"
        ]
        
        for i, goal in enumerate(complex_goals, 1):
            print(f"\n🎯 Goal {i}: {goal}")
            print("-" * 50)
            
            # Use enhanced goal decomposition
            start_time = time.time()
            subtasks = self.orchestrator.decompose_goal(goal)
            decomposition_time = time.time() - start_time
            
            print(f"   ⏱️  Decomposition time: {decomposition_time:.2f} seconds")
            print(f"   📊 Generated {len(subtasks)} atomic subtasks:")
            
            for j, subtask in enumerate(subtasks, 1):
                print(f"      {j}. {subtask.name}")
                print(f"         📝 {subtask.description}")
                print(f"         ⏰ Duration: {subtask.estimated_duration} minutes")
                print(f"         🎯 Priority: {subtask.priority.name}")
                print(f"         🛠️  Skills: {', '.join(subtask.required_skills)}")
                if subtask.dependencies:
                    print(f"         🔗 Dependencies: {len(subtask.dependencies)}")
                print()
        
        print("✅ LLM-based goal decomposition demonstrates sophisticated cognitive parsing!")
    
    async def demonstrate_ecan_attention_modulation(self):
        """Demonstrate enhanced ECAN attention allocation with dynamic modulation"""
        print("\n" + "="*70)
        print("🎯 2. ENHANCED ECAN ATTENTION ALLOCATION")
        print("="*70)
        
        # Simulate different agents with varying performance
        agents = ["expert_agent", "learning_agent", "specialist_agent", "generalist_agent"]
        
        print("🔄 Simulating agent performance and attention modulation...")
        
        # Update agent performance scores
        performance_scores = [0.95, 0.65, 0.85, 0.75]
        for agent, score in zip(agents, performance_scores):
            self.ecan_system.update_agent_performance(agent, score)
            print(f"   📈 {agent}: performance = {score:.2f}")
        
        print("\n🧠 Testing attention allocation with modulation:")
        
        # Test attention allocation for different priorities and agents
        test_scenarios = [
            ("critical_task", 0.95, "expert_agent"),
            ("routine_task", 0.3, "learning_agent"),
            ("specialized_task", 0.8, "specialist_agent"),
            ("repeated_task", 0.6, "expert_agent"),  # Should get diminishing returns
        ]
        
        for i, (task, priority, requester) in enumerate(test_scenarios, 1):
            print(f"\n   🎯 Scenario {i}: {task} (priority: {priority:.2f}, requester: {requester})")
            
            # Allocate attention
            atom_id = f"atom_{task}_{i}"
            success = await self.ecan_system.allocate_attention(atom_id, priority, requester)
            
            reliability = self.ecan_system.get_agent_reliability(requester)
            print(f"      ✅ Allocation: {'Success' if success else 'Failed'}")
            print(f"      📊 Agent reliability: {reliability:.2f}")
            
            # Reward high performance
            if priority > 0.8:
                await self.ecan_system.reward_high_performance(requester, priority)
                print(f"      🏆 Performance reward applied")
        
        # Show attention statistics
        stats = self.ecan_system.get_statistics()
        print(f"\n📊 ECAN System Statistics:")
        print(f"   🎯 Attention allocations: {stats['attention_allocations']}")
        print(f"   📈 Average agent reliability: {stats.get('average_agent_reliability', 0):.2f}")
        print(f"   🔄 Attention redistributions: {stats.get('attention_redistributions', 0)}")
        
        print("✅ Dynamic attention modulation enhances cognitive resource management!")
    
    async def demonstrate_p_system_membranes(self):
        """Demonstrate P-System membrane boundaries"""
        print("\n" + "="*70)
        print("🧬 3. P-SYSTEM MEMBRANE BOUNDARIES")
        print("="*70)
        
        print("🏗️  Cognitive membrane architecture:")
        system_state = self.membrane_network.get_system_state()
        
        for membrane_id, membrane_info in system_state["membranes"].items():
            print(f"   🔬 {membrane_info['name']} ({membrane_info['type']})")
            print(f"      📦 Objects: {membrane_info['object_count']}/{membrane_info['capacity']}")
            print(f"      📋 Rules: {membrane_info['rule_count']}")
            print(f"      🔀 Permeability: {membrane_info['permeability']}")
            print(f"      ⚡ Energy: {membrane_info['energy_level']:.2f}")
        
        print(f"\n🎮 Injecting cognitive objects into membrane system...")
        
        # Inject various cognitive objects
        cognitive_inputs = [
            (MembraneType.PERCEPTION, "sensory_input", {"visual": "red_object_detected", "confidence": 0.9}),
            (MembraneType.PERCEPTION, "auditory_input", {"audio": "alert_sound", "frequency": 1000}),
            (MembraneType.REASONING, "inference_task", {"premise": "if A then B", "query": "A?"}),
            (MembraneType.MEMORY, "episodic_memory", {"event": "goal_completion", "timestamp": datetime.now()}),
            (MembraneType.ATTENTION, "focus_request", {"target": "urgent_task", "priority": 0.9}),
        ]
        
        for membrane_type, obj_type, content in cognitive_inputs:
            obj_id = await self.membrane_network.inject_cognitive_object(
                membrane_type, obj_type, content, priority=0.7, energy=0.8
            )
            print(f"   ➕ Injected {obj_type} into {membrane_type.value} membrane: {obj_id}")
        
        # Wait for membrane processing
        print("\n⏳ Allowing membrane system to process objects...")
        await asyncio.sleep(2)
        
        # Show updated state
        updated_state = self.membrane_network.get_system_state()
        print(f"\n📊 Membrane processing results:")
        print(f"   🔄 Execution cycles: {updated_state['global_statistics']['execution_cycles']}")
        print(f"   💬 Communications: {updated_state['global_statistics']['membrane_communications']}")
        
        for membrane_id, membrane_info in updated_state["membranes"].items():
            if membrane_info['object_count'] > 0:
                print(f"   🧠 {membrane_info['name']}: {membrane_info['object_count']} objects")
        
        print("✅ P-System membranes provide elegant cognitive encapsulation!")
    
    async def demonstrate_sensorimotor_integration(self):
        """Demonstrate sensorimotor integration interface"""
        print("\n" + "="*70)
        print("🤖 4. SENSORIMOTOR INTEGRATION INTERFACE")
        print("="*70)
        
        print("⚙️  Setting up virtual embodied agent...")
        
        # Register virtual sensors
        sensor_configs = [
            SensorConfig(
                sensor_id="camera_main",
                sensor_type=SensorType.VISUAL,
                sampling_rate=10.0,
                resolution=(320, 240),
                range_min=0.0,
                range_max=255.0,
                preprocessing_pipeline=["normalize", "feature_extract"]
            ),
            SensorConfig(
                sensor_id="proximity_front",
                sensor_type=SensorType.PROXIMITY,
                sampling_rate=20.0,
                resolution=12,
                range_min=0.0,
                range_max=3.0,
                preprocessing_pipeline=["normalize", "filter"]
            ),
            SensorConfig(
                sensor_id="imu_main",
                sensor_type=SensorType.ACCELERATION,
                sampling_rate=50.0,
                resolution=16,
                range_min=-10.0,
                range_max=10.0,
                preprocessing_pipeline=["calibrate", "denoise"]
            )
        ]
        
        # Register virtual actuators
        actuator_configs = [
            ActuatorConfig(
                actuator_id="motor_left",
                actuator_type=ActuatorType.MOTOR,
                min_value=-1.0,
                max_value=1.0,
                precision=0.01,
                response_time=0.1
            ),
            ActuatorConfig(
                actuator_id="motor_right", 
                actuator_type=ActuatorType.MOTOR,
                min_value=-1.0,
                max_value=1.0,
                precision=0.01,
                response_time=0.1
            ),
            ActuatorConfig(
                actuator_id="gripper_main",
                actuator_type=ActuatorType.GRIPPER,
                min_value=0.0,
                max_value=1.0,
                precision=0.05,
                response_time=0.5
            )
        ]
        
        # Register sensors and actuators (without callbacks for demo)
        for config in sensor_configs:
            success = await self.sensorimotor_interface.register_sensor(config)
            print(f"   📡 Registered sensor: {config.sensor_id} ({config.sensor_type.value}) - {'✅' if success else '❌'}")
        
        for config in actuator_configs:
            success = await self.sensorimotor_interface.register_actuator(config)
            print(f"   🎯 Registered actuator: {config.actuator_id} ({config.actuator_type.value}) - {'✅' if success else '❌'}")
        
        print("\n🎮 Creating perception-action loops...")
        
        # Create perception-action loops
        loops = [
            PerceptionActionLoop(
                id="obstacle_avoidance",
                name="Obstacle Avoidance Loop",
                sensor_inputs=["proximity_front"],
                processing_pipeline=["normalize", "filter"],
                decision_logic="obstacle_avoidance",
                actuator_outputs=["motor_left", "motor_right"],
                loop_frequency=10.0
            ),
            PerceptionActionLoop(
                id="visual_tracking",
                name="Visual Object Tracking",
                sensor_inputs=["camera_main"],
                processing_pipeline=["normalize", "feature_extract"],
                decision_logic="target_following",
                actuator_outputs=["motor_left", "motor_right"],
                loop_frequency=5.0
            )
        ]
        
        for loop in loops:
            success = await self.sensorimotor_interface.create_perception_action_loop(loop)
            print(f"   🔄 Created loop: {loop.name} - {'✅' if success else '❌'}")
        
        # Simulate sensor readings and actions
        print("\n🌊 Simulating sensorimotor data streams...")
        
        from python.helpers.sensorimotor_integration import SensorReading, ActuatorCommand
        
        # Simulate visual input
        visual_data = np.random.rand(320, 240, 3) * 255
        visual_reading = SensorReading(
            sensor_id="camera_main",
            sensor_type=SensorType.VISUAL,
            timestamp=datetime.now(timezone.utc),
            value=visual_data,
            confidence=0.85,
            quality=0.9
        )
        await self.sensorimotor_interface.add_sensor_reading(visual_reading)
        print("   👁️  Visual input processed")
        
        # Simulate proximity reading
        proximity_reading = SensorReading(
            sensor_id="proximity_front",
            sensor_type=SensorType.PROXIMITY,
            timestamp=datetime.now(timezone.utc),
            value=0.5,  # 0.5 meters
            confidence=0.95,
            quality=1.0
        )
        await self.sensorimotor_interface.add_sensor_reading(proximity_reading)
        print("   📏 Proximity data processed")
        
        # Send motor commands
        motor_command = ActuatorCommand(
            actuator_id="motor_left",
            actuator_type=ActuatorType.MOTOR,
            command=0.7,  # 70% speed
            priority=0.8
        )
        success = await self.sensorimotor_interface.send_actuator_command(motor_command)
        print(f"   🎯 Motor command sent - {'✅' if success else '❌'}")
        
        # Show interface state
        interface_state = self.sensorimotor_interface.get_interface_state()
        print(f"\n📊 Sensorimotor Interface Statistics:")
        print(f"   📡 Sensors: {len(interface_state['sensors'])}")
        print(f"   🎯 Actuators: {len(interface_state['actuators'])}")
        print(f"   🔄 Perception-action loops: {len(interface_state['perception_action_loops'])}")
        print(f"   📊 Sensor readings: {interface_state['global_state']['sensor_readings_count']}")
        print(f"   🎮 Actuator commands: {interface_state['global_state']['actuator_commands_count']}")
        
        print("✅ Sensorimotor integration enables embodied cognitive agents!")
    
    async def demonstrate_tensor_morphology_analysis(self):
        """Demonstrate tensor morphology analysis"""
        print("\n" + "="*70)
        print("📊 5. TENSOR MORPHOLOGY ANALYSIS")
        print("="*70)
        
        print("🔍 Analyzing tensor field morphology across cognitive architecture...")
        
        # Generate morphology report
        morphology_report = await self.tensor_analyzer.get_system_morphology_report()
        
        print(f"\n📈 System Overview:")
        overview = morphology_report["system_overview"]
        print(f"   🎯 Total tensors: {overview['total_tensors']}")
        print(f"   📊 Total elements: {overview['total_elements']:,}")
        print(f"   💾 Total memory: {overview['total_memory']:,} bytes")
        
        print(f"\n🏗️  Tensor Types:")
        for tensor_type, type_info in overview["tensor_types"].items():
            print(f"   📦 {tensor_type}:")
            print(f"      Count: {type_info['count']}")
            print(f"      Elements: {type_info['total_elements']:,}")
            print(f"      Avg Sparsity: {type_info['average_sparsity']:.3f}")
        
        print(f"\n🔬 Morphology Analysis Summary:")
        for tensor_type, analysis in morphology_report["morphology_summary"].items():
            print(f"   🧬 {tensor_type}:")
            if "average_metrics" in analysis:
                metrics = analysis["average_metrics"]
                for metric_name, value in metrics.items():
                    print(f"      {metric_name}: {value:.3f}")
        
        if morphology_report["global_patterns"]:
            print(f"\n🌐 Global Patterns Detected:")
            for pattern in morphology_report["global_patterns"]:
                print(f"   ⭐ {pattern}")
        
        if morphology_report["system_recommendations"]:
            print(f"\n💡 System Recommendations:")
            for recommendation in morphology_report["system_recommendations"]:
                print(f"   🔧 {recommendation}")
        
        # Show analysis statistics
        stats = self.tensor_analyzer.get_analysis_statistics()
        print(f"\n📊 Analysis Statistics:")
        print(f"   🎯 Total analyses: {stats['total_analyses']}")
        print(f"   🔍 Patterns discovered: {stats['patterns_discovered']}")
        print(f"   ⚠️  Anomalies detected: {stats['anomalies_detected']}")
        print(f"   💡 Optimizations suggested: {stats['optimizations_suggested']}")
        print(f"   💾 Total tensor memory: {stats['total_tensor_memory']:,} bytes")
        
        print("✅ Tensor morphology analysis reveals cognitive architecture structure!")
    
    async def demonstrate_integrated_system(self):
        """Demonstrate the integrated evolutionary system"""
        print("\n" + "="*70)
        print("🌟 6. INTEGRATED EVOLUTIONARY SYSTEM")
        print("="*70)
        
        print("🔗 Demonstrating system integration and emergent behavior...")
        
        # Complex multi-system scenario
        complex_goal = "Develop an adaptive swarm robotics system for disaster response"
        
        print(f"\n🎯 Complex Goal: {complex_goal}")
        print("-" * 50)
        
        # 1. LLM-based decomposition
        print("🧠 1. LLM-based goal decomposition...")
        subtasks = self.orchestrator.decompose_goal(complex_goal)
        print(f"   ✅ Generated {len(subtasks)} cognitive subtasks")
        
        # 2. Inject into membrane system
        print("🧬 2. Injecting tasks into P-System membranes...")
        task_objects = []
        for subtask in subtasks[:3]:  # Limit for demo
            obj_id = await self.membrane_network.inject_cognitive_object(
                MembraneType.REASONING,
                "cognitive_task",
                {
                    "task_name": subtask.name,
                    "description": subtask.description,
                    "priority": subtask.priority.value,
                    "skills": subtask.required_skills
                },
                priority=0.8,
                energy=0.9
            )
            task_objects.append(obj_id)
        print(f"   ✅ Injected {len(task_objects)} task objects")
        
        # 3. ECAN attention allocation
        print("🎯 3. ECAN attention allocation...")
        for i, obj_id in enumerate(task_objects):
            priority = 0.9 - (i * 0.1)  # Decreasing priority
            agent_id = f"swarm_agent_{i}"
            success = await self.ecan_system.allocate_attention(obj_id, priority, agent_id)
            print(f"   ✅ Attention allocated: {obj_id} -> {agent_id} (priority: {priority:.1f})")
        
        # 4. Sensorimotor integration
        print("🤖 4. Sensorimotor embodiment simulation...")
        # Simulate swarm robot sensors
        swarm_sensor = SensorReading(
            sensor_id="swarm_communication",
            sensor_type=SensorType.PROXIMITY,
            timestamp=datetime.now(timezone.utc),
            value={"neighbor_count": 5, "signal_strength": 0.8},
            confidence=0.95
        )
        await self.sensorimotor_interface.add_sensor_reading(swarm_sensor)
        print("   ✅ Swarm communication sensors active")
        
        # 5. Tensor morphology tracking
        print("📊 5. Tensor morphology evolution tracking...")
        # Register a new tensor representing swarm state
        swarm_tensor = np.random.rand(10, 10, 5)  # 10 robots, 10 tasks, 5 states
        tensor_id = await self.tensor_analyzer.register_tensor(
            "swarm_coordination",
            TensorType.COGNITIVE_STATE,
            swarm_tensor,
            {"scenario": "disaster_response_swarm"}
        )
        analysis = await self.tensor_analyzer.get_tensor_analysis(tensor_id)
        print(f"   ✅ Swarm tensor registered: {analysis.tensor_id}")
        print(f"      Sparsity: {analysis.metrics.get('sparsity', 0):.3f}")
        print(f"      Complexity: {analysis.metrics.get('complexity', 0):.3f}")
        
        # Allow system to process
        print("\n⏳ Allowing integrated system to evolve...")
        await asyncio.sleep(3)
        
        # Show emergent statistics
        print("\n🌟 Emergent System State:")
        
        # Orchestration state
        print(f"   🎯 Active subtasks: {len(self.orchestrator._subtasks)}")
        
        # Membrane state
        membrane_stats = self.membrane_network.get_system_state()["global_statistics"]
        print(f"   🧬 Membrane communications: {membrane_stats['membrane_communications']}")
        
        # ECAN state
        ecan_stats = self.ecan_system.get_statistics()
        print(f"   🎯 Attention allocations: {ecan_stats['attention_allocations']}")
        
        # Sensorimotor state
        sensorimotor_stats = self.sensorimotor_interface.get_interface_state()["global_state"]
        print(f"   🤖 Sensor readings: {sensorimotor_stats['sensor_readings_count']}")
        
        # Tensor analysis state
        tensor_stats = self.tensor_analyzer.get_analysis_statistics()
        print(f"   📊 Tensor analyses: {tensor_stats['total_analyses']}")
        
        print("\n✨ Integrated evolutionary system demonstrates emergent cognitive behavior!")
    
    async def _setup_demo_agents(self):
        """Setup demo agents for orchestration"""
        agent_configs = [
            ("data_analyst", ["data_collection", "data_analysis", "statistics", "research"]),
            ("ml_engineer", ["machine_learning", "model_training", "ai", "implementation"]),
            ("systems_architect", ["design", "architecture", "planning", "systems"]),
            ("qa_specialist", ["testing", "qa", "validation", "quality_assurance"]),
            ("project_manager", ["planning", "coordination", "general", "management"])
        ]
        
        for agent_name, skills in agent_configs:
            # Create mock agent context
            agent_context = AgentContext(
                id=agent_name,
                name=agent_name.replace("_", " ").title(),
                agent=None  # Mock for demo
            )
            
            # Register with orchestrator
            agent_id = self.orchestrator.register_agent(agent_context, skills)
            self.demo_agents.append((agent_id, agent_context))
        
        print(f"   🤖 Registered {len(self.demo_agents)} specialized agents")
    
    async def cleanup_demo_systems(self):
        """Cleanup demo systems"""
        print("\n🧹 Cleaning up evolutionary systems...")
        
        # Stop all systems
        if self.tensor_analyzer:
            self.tensor_analyzer.stop()
        
        if self.sensorimotor_interface:
            self.sensorimotor_interface.stop()
        
        if self.membrane_network:
            self.membrane_network.stop()
        
        if self.ecan_system:
            self.ecan_system.stop()
        
        print("   ✅ All systems stopped gracefully")


async def main():
    """Main demonstration of evolutionary trajectory"""
    
    demo = EvolutionaryTrajectoryDemo()
    
    try:
        # Setup evolutionary system
        await demo.setup_evolutionary_system()
        
        # Run demonstrations
        await demo.demonstrate_llm_goal_decomposition()
        await demo.demonstrate_ecan_attention_modulation()
        await demo.demonstrate_p_system_membranes()
        await demo.demonstrate_sensorimotor_integration()
        await demo.demonstrate_tensor_morphology_analysis()
        await demo.demonstrate_integrated_system()
        
        # Final summary
        print("\n" + "="*70)
        print("🎉 EVOLUTIONARY TRAJECTORY DEMONSTRATION COMPLETE")
        print("="*70)
        print("🌟 The cognitive chrysalis has emerged as a self-organizing,")
        print("   recursive cognitive membrane with:")
        print("   • LLM-driven goal decomposition")
        print("   • Dynamic ECAN attention allocation")
        print("   • P-System cognitive encapsulation")
        print("   • Embodied sensorimotor integration")
        print("   • Tensor morphology optimization")
        print("   • Hypergraph neural-symbolic reasoning")
        print("\n💫 Ready for AGI emergence via distributed architecture!")
        
    except KeyboardInterrupt:
        print("\n⚡ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await demo.cleanup_demo_systems()


if __name__ == "__main__":
    asyncio.run(main())