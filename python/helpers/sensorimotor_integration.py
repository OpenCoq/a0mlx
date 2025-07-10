"""
Sensorimotor Integration Interface for Embodied Agents

Provides interfaces for integrating sensory data streams and actuator controls
with the distributed orchestration system, enabling embodied AI agents.
"""

import numpy as np
from typing import Dict, List, Optional, Set, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from datetime import datetime, timezone
import json
import uuid
import threading
from collections import defaultdict, deque
import math

from .atomspace import AtomSpace
from .neural_symbolic_reasoning import NeuralSymbolicReasoningEngine
from .p_system_membranes import PSystemMembraneNetwork, MembraneType, CognitiveObject


class SensorType(Enum):
    """Types of sensors for embodied agents"""
    VISUAL = "visual"
    AUDITORY = "auditory"
    TACTILE = "tactile"
    PROPRIOCEPTIVE = "proprioceptive"  # Body position/movement
    TEMPERATURE = "temperature"
    PROXIMITY = "proximity"
    ACCELERATION = "acceleration"
    GYROSCOPIC = "gyroscopic"
    MAGNETIC = "magnetic"
    CHEMICAL = "chemical"


class ActuatorType(Enum):
    """Types of actuators for embodied agents"""
    MOTOR = "motor"              # Wheel/track motors
    SERVO = "servo"              # Servo motors for joints
    GRIPPER = "gripper"          # Gripper/manipulator
    SPEAKER = "speaker"          # Audio output
    DISPLAY = "display"          # Visual output
    LED = "led"                  # Light indicators
    PUMP = "pump"                # Fluid/air pumps
    ELECTROMAGNET = "electromagnet"
    TEMPERATURE_CONTROL = "temperature_control"


class SensorModalityFusion(Enum):
    """Sensor fusion strategies"""
    RAW = "raw"                  # No fusion, raw sensor data
    EARLY_FUSION = "early_fusion"  # Combine raw sensor data
    LATE_FUSION = "late_fusion"    # Combine processed features
    HYBRID_FUSION = "hybrid_fusion"  # Multi-level fusion


@dataclass
class SensorReading:
    """Individual sensor reading"""
    sensor_id: str
    sensor_type: SensorType
    timestamp: datetime
    value: Union[float, np.ndarray, Dict[str, Any]]
    confidence: float = 1.0
    quality: float = 1.0  # Signal quality (0-1)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActuatorCommand:
    """Command to an actuator"""
    actuator_id: str
    actuator_type: ActuatorType
    command: Union[float, np.ndarray, Dict[str, Any]]
    priority: float = 0.5
    duration: Optional[float] = None  # Duration in seconds
    feedback_required: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SensorConfig:
    """Configuration for a sensor"""
    sensor_id: str
    sensor_type: SensorType
    sampling_rate: float  # Hz
    resolution: Union[int, Tuple[int, ...]]
    range_min: float
    range_max: float
    calibration_data: Dict[str, Any] = field(default_factory=dict)
    preprocessing_pipeline: List[str] = field(default_factory=list)


@dataclass
class ActuatorConfig:
    """Configuration for an actuator"""
    actuator_id: str
    actuator_type: ActuatorType
    max_value: float
    min_value: float
    precision: float
    response_time: float  # seconds
    safety_limits: Dict[str, float] = field(default_factory=dict)


@dataclass
class PerceptionActionLoop:
    """A complete perception-action loop"""
    id: str
    name: str
    sensor_inputs: List[str]  # Sensor IDs
    processing_pipeline: List[str]  # Processing steps
    decision_logic: str  # Decision algorithm
    actuator_outputs: List[str]  # Actuator IDs
    active: bool = True
    loop_frequency: float = 10.0  # Hz
    statistics: Dict[str, Any] = field(default_factory=dict)


class SensorimotorIntegrationInterface:
    """
    Interface for integrating sensory and actuator systems with cognitive architecture
    """
    
    def __init__(self, atomspace: AtomSpace, 
                 membrane_network: Optional[PSystemMembraneNetwork] = None):
        self.atomspace = atomspace
        self.membrane_network = membrane_network
        
        # Sensor management
        self.sensors: Dict[str, SensorConfig] = {}
        self.sensor_readings: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.sensor_threads: Dict[str, threading.Thread] = {}
        self.sensor_callbacks: Dict[str, Callable] = {}
        
        # Actuator management
        self.actuators: Dict[str, ActuatorConfig] = {}
        self.actuator_commands: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.actuator_threads: Dict[str, threading.Thread] = {}
        self.actuator_callbacks: Dict[str, Callable] = {}
        
        # Perception-action loops
        self.perception_action_loops: Dict[str, PerceptionActionLoop] = {}
        self.loop_threads: Dict[str, threading.Thread] = {}
        
        # State management
        self.running = False
        self.global_state = {
            "total_sensors": 0,
            "total_actuators": 0,
            "active_loops": 0,
            "sensor_readings_count": 0,
            "actuator_commands_count": 0,
            "fusion_operations": 0,
            "planning_cycles": 0
        }
        
        # Sensor fusion capabilities
        self.fusion_strategies: Dict[str, Callable] = {
            "early_fusion": self._early_fusion,
            "late_fusion": self._late_fusion,
            "hybrid_fusion": self._hybrid_fusion
        }
        
        # Preprocessing pipelines
        self.preprocessing_functions: Dict[str, Callable] = {
            "normalize": self._normalize_sensor_data,
            "filter": self._filter_sensor_data,
            "calibrate": self._calibrate_sensor_data,
            "denoise": self._denoise_sensor_data,
            "feature_extract": self._extract_features
        }
    
    async def initialize_embodiment_interface(self):
        """Initialize the sensorimotor interface"""
        self.running = True
        
        # Set up default sensor and actuator configurations
        await self._setup_default_configurations()
        
        # Initialize perception processing pipeline
        await self._initialize_perception_pipeline()
        
        print("🤖 Sensorimotor Integration Interface initialized")
    
    async def register_sensor(self, config: SensorConfig, 
                            data_callback: Optional[Callable] = None) -> bool:
        """Register a new sensor"""
        try:
            self.sensors[config.sensor_id] = config
            
            if data_callback:
                self.sensor_callbacks[config.sensor_id] = data_callback
            
            # Initialize sensor reading buffer
            self.sensor_readings[config.sensor_id] = deque(maxlen=1000)
            
            # Start sensor thread if callback is provided
            if data_callback:
                thread = threading.Thread(
                    target=self._sensor_loop,
                    args=(config.sensor_id,),
                    daemon=True
                )
                self.sensor_threads[config.sensor_id] = thread
                thread.start()
            
            self.global_state["total_sensors"] += 1
            
            # Add sensor to AtomSpace
            await self._add_sensor_to_atomspace(config)
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to register sensor {config.sensor_id}: {e}")
            return False
    
    async def register_actuator(self, config: ActuatorConfig,
                              command_callback: Optional[Callable] = None) -> bool:
        """Register a new actuator"""
        try:
            self.actuators[config.actuator_id] = config
            
            if command_callback:
                self.actuator_callbacks[config.actuator_id] = command_callback
            
            # Initialize command buffer
            self.actuator_commands[config.actuator_id] = deque(maxlen=100)
            
            # Start actuator thread if callback is provided
            if command_callback:
                thread = threading.Thread(
                    target=self._actuator_loop,
                    args=(config.actuator_id,),
                    daemon=True
                )
                self.actuator_threads[config.actuator_id] = thread
                thread.start()
            
            self.global_state["total_actuators"] += 1
            
            # Add actuator to AtomSpace
            await self._add_actuator_to_atomspace(config)
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to register actuator {config.actuator_id}: {e}")
            return False
    
    async def add_sensor_reading(self, reading: SensorReading):
        """Add a sensor reading to the system"""
        # Store reading
        self.sensor_readings[reading.sensor_id].append(reading)
        self.global_state["sensor_readings_count"] += 1
        
        # Process reading through pipeline
        processed_reading = await self._process_sensor_reading(reading)
        
        # Inject into membrane network if available
        if self.membrane_network:
            await self.membrane_network.inject_cognitive_object(
                MembraneType.PERCEPTION,
                "sensory_input",
                {
                    "sensor_id": reading.sensor_id,
                    "sensor_type": reading.sensor_type.value,
                    "processed_value": processed_reading,
                    "confidence": reading.confidence,
                    "timestamp": reading.timestamp.isoformat()
                },
                priority=reading.confidence,
                energy=reading.quality
            )
        
        # Add to AtomSpace as perception nodes
        await self._add_reading_to_atomspace(processed_reading)
    
    async def send_actuator_command(self, command: ActuatorCommand) -> bool:
        """Send command to an actuator"""
        if command.actuator_id not in self.actuators:
            return False
        
        # Validate command against actuator limits
        if not self._validate_actuator_command(command):
            return False
        
        # Store command
        self.actuator_commands[command.actuator_id].append(command)
        self.global_state["actuator_commands_count"] += 1
        
        # Execute command if callback is available
        if command.actuator_id in self.actuator_callbacks:
            try:
                callback = self.actuator_callbacks[command.actuator_id]
                result = callback(command)
                
                # Add command execution to AtomSpace
                await self._add_command_to_atomspace(command, result)
                
                return True
            except Exception as e:
                print(f"❌ Actuator command failed: {e}")
                return False
        
        return True
    
    async def create_perception_action_loop(self, loop_config: PerceptionActionLoop) -> bool:
        """Create a new perception-action loop"""
        try:
            self.perception_action_loops[loop_config.id] = loop_config
            
            # Start loop thread
            thread = threading.Thread(
                target=self._perception_action_loop,
                args=(loop_config.id,),
                daemon=True
            )
            self.loop_threads[loop_config.id] = thread
            thread.start()
            
            self.global_state["active_loops"] += 1
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to create perception-action loop {loop_config.id}: {e}")
            return False
    
    def _sensor_loop(self, sensor_id: str):
        """Sensor data collection loop"""
        config = self.sensors[sensor_id]
        callback = self.sensor_callbacks[sensor_id]
        
        # Calculate sleep time based on sampling rate
        sleep_time = 1.0 / config.sampling_rate
        
        while self.running:
            try:
                # Get sensor data from callback
                raw_data = callback()
                
                # Create sensor reading
                reading = SensorReading(
                    sensor_id=sensor_id,
                    sensor_type=config.sensor_type,
                    timestamp=datetime.now(timezone.utc),
                    value=raw_data,
                    confidence=1.0,  # Could be computed from signal quality
                    quality=1.0
                )
                
                # Add to system asynchronously
                asyncio.run(self.add_sensor_reading(reading))
                
                # Sleep for appropriate interval
                import time
                time.sleep(sleep_time)
                
            except Exception as e:
                print(f"❌ Sensor loop error for {sensor_id}: {e}")
                import time
                time.sleep(1.0)
    
    def _actuator_loop(self, actuator_id: str):
        """Actuator command execution loop"""
        while self.running:
            try:
                # Check for pending commands
                if self.actuator_commands[actuator_id]:
                    command = self.actuator_commands[actuator_id].popleft()
                    
                    # Execute command
                    callback = self.actuator_callbacks[actuator_id]
                    result = callback(command)
                    
                    # Log result
                    print(f"🎯 Actuator {actuator_id} executed command: {result}")
                
                # Sleep briefly
                import time
                time.sleep(0.01)
                
            except Exception as e:
                print(f"❌ Actuator loop error for {actuator_id}: {e}")
                import time
                time.sleep(1.0)
    
    def _perception_action_loop(self, loop_id: str):
        """Execute a perception-action loop"""
        loop_config = self.perception_action_loops[loop_id]
        sleep_time = 1.0 / loop_config.loop_frequency
        
        while self.running and loop_config.active:
            try:
                # Gather sensor inputs
                sensor_data = {}
                for sensor_id in loop_config.sensor_inputs:
                    if (sensor_id in self.sensor_readings and 
                        self.sensor_readings[sensor_id]):
                        latest_reading = self.sensor_readings[sensor_id][-1]
                        sensor_data[sensor_id] = latest_reading
                
                # Process through pipeline
                processed_data = await asyncio.run(
                    self._process_perception_pipeline(sensor_data, loop_config)
                )
                
                # Make decision
                decision = await asyncio.run(
                    self._execute_decision_logic(processed_data, loop_config)
                )
                
                # Generate actuator commands
                if decision:
                    await asyncio.run(
                        self._generate_actuator_commands(decision, loop_config)
                    )
                
                self.global_state["planning_cycles"] += 1
                
                # Sleep for appropriate interval
                import time
                time.sleep(sleep_time)
                
            except Exception as e:
                print(f"❌ Perception-action loop error for {loop_id}: {e}")
                import time
                time.sleep(1.0)
    
    async def _process_sensor_reading(self, reading: SensorReading) -> Any:
        """Process a sensor reading through preprocessing pipeline"""
        if reading.sensor_id not in self.sensors:
            return reading.value
        
        config = self.sensors[reading.sensor_id]
        processed_value = reading.value
        
        # Apply preprocessing pipeline
        for step in config.preprocessing_pipeline:
            if step in self.preprocessing_functions:
                func = self.preprocessing_functions[step]
                processed_value = func(processed_value, config)
        
        return processed_value
    
    async def _process_perception_pipeline(self, sensor_data: Dict[str, SensorReading],
                                         loop_config: PerceptionActionLoop) -> Dict[str, Any]:
        """Process sensor data through perception pipeline"""
        processed_data = {}
        
        # Extract values from sensor readings
        for sensor_id, reading in sensor_data.items():
            processed_data[sensor_id] = reading.value
        
        # Apply fusion if multiple sensors
        if len(sensor_data) > 1:
            fused_data = await self._fuse_sensor_data(sensor_data)
            processed_data["fused"] = fused_data
            self.global_state["fusion_operations"] += 1
        
        return processed_data
    
    async def _execute_decision_logic(self, processed_data: Dict[str, Any],
                                    loop_config: PerceptionActionLoop) -> Optional[Dict[str, Any]]:
        """Execute decision logic for perception-action loop"""
        
        # Simple example decision logic
        decision_logic = loop_config.decision_logic
        
        if decision_logic == "obstacle_avoidance":
            return await self._obstacle_avoidance_decision(processed_data)
        elif decision_logic == "target_following":
            return await self._target_following_decision(processed_data)
        elif decision_logic == "exploration":
            return await self._exploration_decision(processed_data)
        else:
            # Generic decision based on processed data
            return {"action": "continue", "intensity": 0.5}
    
    async def _generate_actuator_commands(self, decision: Dict[str, Any],
                                        loop_config: PerceptionActionLoop):
        """Generate actuator commands based on decision"""
        
        for actuator_id in loop_config.actuator_outputs:
            if actuator_id in self.actuators:
                # Generate command based on decision
                command_value = self._decision_to_command(decision, actuator_id)
                
                command = ActuatorCommand(
                    actuator_id=actuator_id,
                    actuator_type=self.actuators[actuator_id].actuator_type,
                    command=command_value,
                    priority=decision.get("priority", 0.5)
                )
                
                await self.send_actuator_command(command)
    
    # Fusion strategies
    async def _fuse_sensor_data(self, sensor_data: Dict[str, SensorReading]) -> Dict[str, Any]:
        """Fuse multiple sensor readings"""
        # Default to early fusion
        return await self._early_fusion(sensor_data)
    
    async def _early_fusion(self, sensor_data: Dict[str, SensorReading]) -> Dict[str, Any]:
        """Early fusion: combine raw sensor data"""
        fused = {}
        
        # Combine numerical values
        numerical_values = []
        for reading in sensor_data.values():
            if isinstance(reading.value, (int, float)):
                numerical_values.append(reading.value)
        
        if numerical_values:
            fused["mean"] = np.mean(numerical_values)
            fused["std"] = np.std(numerical_values)
            fused["max"] = np.max(numerical_values)
            fused["min"] = np.min(numerical_values)
        
        return fused
    
    async def _late_fusion(self, sensor_data: Dict[str, SensorReading]) -> Dict[str, Any]:
        """Late fusion: combine processed features"""
        # Extract features from each sensor first
        features = {}
        for sensor_id, reading in sensor_data.items():
            features[sensor_id] = await self._extract_features(reading.value, None)
        
        # Combine features
        fused_features = {}
        for sensor_id, sensor_features in features.items():
            for feature_name, feature_value in sensor_features.items():
                if feature_name not in fused_features:
                    fused_features[feature_name] = []
                fused_features[feature_name].append(feature_value)
        
        # Aggregate features
        result = {}
        for feature_name, values in fused_features.items():
            result[feature_name] = np.mean(values)
        
        return result
    
    async def _hybrid_fusion(self, sensor_data: Dict[str, SensorReading]) -> Dict[str, Any]:
        """Hybrid fusion: combine early and late fusion"""
        early = await self._early_fusion(sensor_data)
        late = await self._late_fusion(sensor_data)
        
        return {"early_fusion": early, "late_fusion": late}
    
    # Preprocessing functions
    def _normalize_sensor_data(self, data: Any, config: SensorConfig) -> Any:
        """Normalize sensor data to [0, 1] range"""
        if isinstance(data, (int, float)):
            range_span = config.range_max - config.range_min
            if range_span > 0:
                return (data - config.range_min) / range_span
        return data
    
    def _filter_sensor_data(self, data: Any, config: SensorConfig) -> Any:
        """Apply filtering to sensor data"""
        # Simple moving average filter for numerical data
        if isinstance(data, (int, float)):
            # This would use historical data in a real implementation
            return data  # Placeholder
        return data
    
    def _calibrate_sensor_data(self, data: Any, config: SensorConfig) -> Any:
        """Apply calibration to sensor data"""
        if isinstance(data, (int, float)) and "offset" in config.calibration_data:
            offset = config.calibration_data["offset"]
            scale = config.calibration_data.get("scale", 1.0)
            return (data - offset) * scale
        return data
    
    def _denoise_sensor_data(self, data: Any, config: SensorConfig) -> Any:
        """Remove noise from sensor data"""
        # Placeholder for denoising algorithms
        return data
    
    def _extract_features(self, data: Any, config: Optional[SensorConfig]) -> Dict[str, float]:
        """Extract features from sensor data"""
        features = {}
        
        if isinstance(data, (int, float)):
            features["magnitude"] = abs(data)
            features["sign"] = 1.0 if data >= 0 else -1.0
        elif isinstance(data, np.ndarray):
            features["mean"] = np.mean(data)
            features["std"] = np.std(data)
            features["max"] = np.max(data)
            features["min"] = np.min(data)
            features["energy"] = np.sum(data ** 2)
        
        return features
    
    # Decision logic implementations
    async def _obstacle_avoidance_decision(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Obstacle avoidance decision logic"""
        # Simple proximity-based avoidance
        proximity_threshold = 0.3
        
        # Check for proximity sensors
        for key, value in data.items():
            if isinstance(value, (int, float)) and value < proximity_threshold:
                return {
                    "action": "avoid",
                    "direction": "left",  # or could be computed
                    "intensity": 1.0 - value,
                    "priority": 1.0
                }
        
        return {"action": "continue", "intensity": 0.5}
    
    async def _target_following_decision(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Target following decision logic"""
        # Placeholder for target following logic
        return {"action": "approach", "intensity": 0.7}
    
    async def _exploration_decision(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Exploration decision logic"""
        # Simple random exploration
        import random
        directions = ["forward", "left", "right", "backward"]
        
        return {
            "action": "explore",
            "direction": random.choice(directions),
            "intensity": random.uniform(0.3, 0.8)
        }
    
    def _decision_to_command(self, decision: Dict[str, Any], actuator_id: str) -> Any:
        """Convert decision to actuator command"""
        action = decision.get("action", "continue")
        intensity = decision.get("intensity", 0.5)
        direction = decision.get("direction", "forward")
        
        # Simple mapping for motor commands
        if actuator_id.startswith("motor"):
            if action == "avoid" and direction == "left":
                return -intensity  # Turn left
            elif action == "avoid" and direction == "right":
                return intensity   # Turn right
            elif action == "continue":
                return intensity   # Move forward
            else:
                return 0.0
        
        # Default command
        return intensity
    
    def _validate_actuator_command(self, command: ActuatorCommand) -> bool:
        """Validate actuator command against safety limits"""
        config = self.actuators[command.actuator_id]
        
        if isinstance(command.command, (int, float)):
            return config.min_value <= command.command <= config.max_value
        
        return True  # Assume valid for complex commands
    
    # AtomSpace integration
    async def _add_sensor_to_atomspace(self, config: SensorConfig):
        """Add sensor configuration to AtomSpace"""
        await self.atomspace.add_node(
            f"sensor_{config.sensor_id}",
            "sensor_config",
            1.0,  # truth value
            0.9   # confidence
        )
    
    async def _add_actuator_to_atomspace(self, config: ActuatorConfig):
        """Add actuator configuration to AtomSpace"""
        await self.atomspace.add_node(
            f"actuator_{config.actuator_id}",
            "actuator_config", 
            1.0,  # truth value
            0.9   # confidence
        )
    
    async def _add_reading_to_atomspace(self, processed_reading: Any):
        """Add sensor reading to AtomSpace"""
        # Create perception node
        await self.atomspace.add_node(
            f"perception_{uuid.uuid4().hex[:8]}",
            "perception",
            0.8,  # truth value
            0.7   # confidence
        )
    
    async def _add_command_to_atomspace(self, command: ActuatorCommand, result: Any):
        """Add actuator command and result to AtomSpace"""
        # Create action node
        await self.atomspace.add_node(
            f"action_{uuid.uuid4().hex[:8]}",
            "action",
            0.8,  # truth value
            0.7   # confidence
        )
    
    async def _setup_default_configurations(self):
        """Set up default sensor and actuator configurations"""
        # Example sensor configurations
        default_sensors = [
            SensorConfig(
                sensor_id="proximity_front",
                sensor_type=SensorType.PROXIMITY,
                sampling_rate=20.0,
                resolution=12,
                range_min=0.0,
                range_max=5.0,
                preprocessing_pipeline=["normalize", "filter"]
            ),
            SensorConfig(
                sensor_id="camera_main",
                sensor_type=SensorType.VISUAL,
                sampling_rate=30.0,
                resolution=(640, 480),
                range_min=0.0,
                range_max=255.0,
                preprocessing_pipeline=["normalize", "feature_extract"]
            )
        ]
        
        # Example actuator configurations
        default_actuators = [
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
            )
        ]
        
        # Store configurations (but don't register without callbacks)
        for sensor_config in default_sensors:
            self.sensors[sensor_config.sensor_id] = sensor_config
        
        for actuator_config in default_actuators:
            self.actuators[actuator_config.actuator_id] = actuator_config
    
    async def _initialize_perception_pipeline(self):
        """Initialize perception processing pipeline"""
        # Example perception-action loop
        example_loop = PerceptionActionLoop(
            id="obstacle_avoidance_loop",
            name="Basic Obstacle Avoidance",
            sensor_inputs=["proximity_front"],
            processing_pipeline=["normalize", "filter"],
            decision_logic="obstacle_avoidance",
            actuator_outputs=["motor_left", "motor_right"],
            loop_frequency=10.0
        )
        
        self.perception_action_loops[example_loop.id] = example_loop
    
    def get_interface_state(self) -> Dict[str, Any]:
        """Get current interface state"""
        return {
            "global_state": self.global_state,
            "sensors": {sid: {
                "type": config.sensor_type.value,
                "sampling_rate": config.sampling_rate,
                "readings_count": len(self.sensor_readings[sid])
            } for sid, config in self.sensors.items()},
            "actuators": {aid: {
                "type": config.actuator_type.value,
                "commands_count": len(self.actuator_commands[aid])
            } for aid, config in self.actuators.items()},
            "perception_action_loops": {lid: {
                "name": loop.name,
                "active": loop.active,
                "frequency": loop.loop_frequency
            } for lid, loop in self.perception_action_loops.items()},
            "running": self.running
        }
    
    def stop(self):
        """Stop the sensorimotor interface"""
        self.running = False
        
        # Stop all threads
        for thread in self.sensor_threads.values():
            if thread.is_alive():
                thread.join(timeout=2)
        
        for thread in self.actuator_threads.values():
            if thread.is_alive():
                thread.join(timeout=2)
        
        for thread in self.loop_threads.values():
            if thread.is_alive():
                thread.join(timeout=2)