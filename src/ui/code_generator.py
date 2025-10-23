"""
Code Generation System for Visual Programming Interface
Converts visual behavior trees to executable Python code with debugging support.
"""

from typing import Dict, List, Any, Optional, Tuple
import json
import ast
import textwrap
from dataclasses import dataclass
from enum import Enum

class ValidationResult:
    """Result of behavior tree validation"""
    
    def __init__(self, is_valid: bool, errors: List[str] = None, warnings: List[str] = None):
        self.is_valid = is_valid
        self.errors = errors or []
        self.warnings = warnings or []
    
    def add_error(self, error: str):
        """Add an error to the validation result"""
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: str):
        """Add a warning to the validation result"""
        self.warnings.append(warning)
    
    def __str__(self):
        result = f"Valid: {self.is_valid}\n"
        if self.errors:
            result += f"Errors: {', '.join(self.errors)}\n"
        if self.warnings:
            result += f"Warnings: {', '.join(self.warnings)}\n"
        return result


class ExecutionState(Enum):
    """Execution states for debugging"""
    READY = "ready"
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"
    PAUSED = "paused"


@dataclass
class DebugState:
    """Debug state information for a behavior node"""
    node_id: str
    node_type: str
    state: ExecutionState
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    execution_time: float
    error_message: Optional[str] = None


class CodeGenerator:
    """Converts visual behavior trees to executable Python code"""
    
    def __init__(self):
        self.code_templates = self._load_code_templates()
        self.validation_rules = self._load_validation_rules()
        self.debug_enabled = False
        self.debug_states = {}
    
    def _load_code_templates(self) -> Dict[str, str]:
        """Load code templates for different block types"""
        return {
            # Sensor blocks
            'camera_sensor': '''
def camera_sensor_{node_id}(vehicle_state, environment):
    """Camera sensor implementation"""
    try:
        # Get camera parameters
        resolution = {resolution}
        fps = {fps}
        
        # Simulate camera capture
        camera_data = environment.get_camera_view(
            vehicle_state.position,
            vehicle_state.orientation,
            resolution=resolution
        )
        
        return {{
            'success': True,
            'data': camera_data,
            'timestamp': environment.current_time
        }}
    except Exception as e:
        return {{
            'success': False,
            'error': str(e),
            'data': None
        }}
''',
            
            'lidar_sensor': '''
def lidar_sensor_{node_id}(vehicle_state, environment):
    """LIDAR sensor implementation"""
    try:
        # Get LIDAR parameters
        range_limit = {range}
        resolution = {resolution}
        
        # Simulate LIDAR scan
        lidar_data = environment.get_lidar_scan(
            vehicle_state.position,
            vehicle_state.orientation,
            range_limit=range_limit,
            resolution=resolution
        )
        
        return {{
            'success': True,
            'data': lidar_data,
            'timestamp': environment.current_time
        }}
    except Exception as e:
        return {{
            'success': False,
            'error': str(e),
            'data': None
        }}
''',
            
            'gps_sensor': '''
def gps_sensor_{node_id}(vehicle_state, environment):
    """GPS sensor implementation"""
    try:
        # Get GPS parameters
        accuracy = {accuracy}
        
        # Simulate GPS reading with noise
        gps_data = environment.get_gps_reading(
            vehicle_state.position,
            accuracy=accuracy
        )
        
        return {{
            'success': True,
            'data': gps_data,
            'timestamp': environment.current_time
        }}
    except Exception as e:
        return {{
            'success': False,
            'error': str(e),
            'data': None
        }}
''',
            
            # Condition blocks
            'obstacle_detected': '''
def obstacle_detected_{node_id}(sensor_data, vehicle_state):
    """Obstacle detection condition"""
    try:
        distance_threshold = {distance_threshold}
        angle_range = {angle_range}
        
        if not sensor_data or not sensor_data.get('success'):
            return {{'success': False, 'detected': False, 'reason': 'No sensor data'}}
        
        # Check for obstacles in sensor data
        obstacles = sensor_data['data'].get('obstacles', [])
        
        for obstacle in obstacles:
            distance = obstacle.get('distance', float('inf'))
            angle = obstacle.get('angle', 0)
            
            if distance <= distance_threshold and abs(angle) <= angle_range:
                return {{
                    'success': True,
                    'detected': True,
                    'obstacle': obstacle,
                    'distance': distance,
                    'angle': angle
                }}
        
        return {{'success': True, 'detected': False}}
        
    except Exception as e:
        return {{
            'success': False,
            'detected': False,
            'error': str(e)
        }}
''',
            
            'speed_check': '''
def speed_check_{node_id}(vehicle_state):
    """Speed check condition"""
    try:
        min_speed = {min_speed}
        max_speed = {max_speed}
        
        current_speed = vehicle_state.velocity.magnitude()
        
        speed_ok = min_speed <= current_speed <= max_speed
        
        return {{
            'success': True,
            'speed_ok': speed_ok,
            'current_speed': current_speed,
            'min_speed': min_speed,
            'max_speed': max_speed
        }}
        
    except Exception as e:
        return {{
            'success': False,
            'speed_ok': False,
            'error': str(e)
        }}
''',
            
            'lane_detection': '''
def lane_detection_{node_id}(camera_data):
    """Lane detection condition"""
    try:
        confidence_threshold = {confidence_threshold}
        
        if not camera_data or not camera_data.get('success'):
            return {{'success': False, 'lanes_detected': False, 'reason': 'No camera data'}}
        
        # Simulate lane detection algorithm
        lanes = camera_data['data'].get('lanes', [])
        
        valid_lanes = [lane for lane in lanes 
                      if lane.get('confidence', 0) >= confidence_threshold]
        
        return {{
            'success': True,
            'lanes_detected': len(valid_lanes) > 0,
            'lanes': valid_lanes,
            'confidence_threshold': confidence_threshold
        }}
        
    except Exception as e:
        return {{
            'success': False,
            'lanes_detected': False,
            'error': str(e)
        }}
''',
            
            # Action blocks
            'accelerate': '''
def accelerate_{node_id}(vehicle_controller, vehicle_state):
    """Acceleration action"""
    try:
        acceleration = {acceleration}
        max_speed = {max_speed}
        
        current_speed = vehicle_state.velocity.magnitude()
        
        if current_speed >= max_speed:
            return {{
                'success': True,
                'action_taken': False,
                'reason': 'Already at max speed'
            }}
        
        # Apply acceleration
        vehicle_controller.set_throttle(acceleration)
        
        return {{
            'success': True,
            'action_taken': True,
            'acceleration': acceleration,
            'max_speed': max_speed
        }}
        
    except Exception as e:
        return {{
            'success': False,
            'action_taken': False,
            'error': str(e)
        }}
''',
            
            'brake': '''
def brake_{node_id}(vehicle_controller, vehicle_state):
    """Braking action"""
    try:
        brake_force = {brake_force}
        
        current_speed = vehicle_state.velocity.magnitude()
        
        if current_speed <= 0.1:  # Already stopped
            return {{
                'success': True,
                'action_taken': False,
                'reason': 'Already stopped'
            }}
        
        # Apply brakes
        vehicle_controller.set_brake(brake_force)
        
        return {{
            'success': True,
            'action_taken': True,
            'brake_force': brake_force
        }}
        
    except Exception as e:
        return {{
            'success': False,
            'action_taken': False,
            'error': str(e)
        }}
''',
            
            'steer': '''
def steer_{node_id}(vehicle_controller, vehicle_state):
    """Steering action"""
    try:
        steering_angle = {steering_angle}
        duration = {duration}
        
        # Apply steering
        vehicle_controller.set_steering(steering_angle, duration)
        
        return {{
            'success': True,
            'action_taken': True,
            'steering_angle': steering_angle,
            'duration': duration
        }}
        
    except Exception as e:
        return {{
            'success': False,
            'action_taken': False,
            'error': str(e)
        }}
''',
            
            'follow_path': '''
def follow_path_{node_id}(vehicle_controller, vehicle_state, path_manager):
    """Path following action"""
    try:
        path_id = "{path_id}"
        speed = {speed}
        
        # Get path from path manager
        path = path_manager.get_path(path_id)
        if not path:
            return {{
                'success': False,
                'action_taken': False,
                'error': f'Path {{path_id}} not found'
            }}
        
        # Calculate steering and throttle for path following
        steering, throttle = path_manager.calculate_control(
            vehicle_state.position,
            vehicle_state.orientation,
            path,
            target_speed=speed
        )
        
        vehicle_controller.set_steering(steering)
        vehicle_controller.set_throttle(throttle)
        
        return {{
            'success': True,
            'action_taken': True,
            'path_id': path_id,
            'speed': speed,
            'steering': steering,
            'throttle': throttle
        }}
        
    except Exception as e:
        return {{
            'success': False,
            'action_taken': False,
            'error': str(e)
        }}
''',
            
            # Composite blocks
            'sequence': '''
def sequence_{node_id}(children_results):
    """Sequence composite - all children must succeed"""
    try:
        for i, result in enumerate(children_results):
            if not result.get('success', False):
                return {{
                    'success': False,
                    'failed_child': i,
                    'reason': f'Child {{i}} failed: {{result.get("error", "Unknown error")}}'
                }}
        
        return {{
            'success': True,
            'children_executed': len(children_results)
        }}
        
    except Exception as e:
        return {{
            'success': False,
            'error': str(e)
        }}
''',
            
            'selector': '''
def selector_{node_id}(children_results):
    """Selector composite - first successful child wins"""
    try:
        for i, result in enumerate(children_results):
            if result.get('success', False):
                return {{
                    'success': True,
                    'successful_child': i,
                    'result': result
                }}
        
        return {{
            'success': False,
            'reason': 'All children failed'
        }}
        
    except Exception as e:
        return {{
            'success': False,
            'error': str(e)
        }}
''',
            
            'parallel': '''
def parallel_{node_id}(children_results):
    """Parallel composite - specified number of children must succeed"""
    try:
        success_count = {success_count}
        
        successful_children = sum(1 for result in children_results 
                                if result.get('success', False))
        
        success = successful_children >= success_count
        
        return {{
            'success': success,
            'successful_children': successful_children,
            'required_successes': success_count,
            'total_children': len(children_results)
        }}
        
    except Exception as e:
        return {{
            'success': False,
            'error': str(e)
        }}
'''
        }
    
    def _load_validation_rules(self) -> Dict[str, List[str]]:
        """Load validation rules for different block types"""
        return {
            'camera_sensor': [
                'resolution must be a string',
                'fps must be a positive integer'
            ],
            'lidar_sensor': [
                'range must be a positive float',
                'resolution must be a positive float'
            ],
            'gps_sensor': [
                'accuracy must be a positive float'
            ],
            'obstacle_detected': [
                'distance_threshold must be a positive float',
                'angle_range must be a positive float'
            ],
            'speed_check': [
                'min_speed must be a non-negative float',
                'max_speed must be greater than min_speed'
            ],
            'lane_detection': [
                'confidence_threshold must be between 0 and 1'
            ],
            'accelerate': [
                'acceleration must be a positive float',
                'max_speed must be a positive float'
            ],
            'brake': [
                'brake_force must be a positive float'
            ],
            'steer': [
                'steering_angle must be a float between -1 and 1',
                'duration must be a positive float'
            ],
            'follow_path': [
                'path_id must be a non-empty string',
                'speed must be a positive float'
            ],
            'sequence': [],
            'selector': [],
            'parallel': [
                'success_count must be a positive integer'
            ]
        }
    
    def validate_behavior_tree(self, tree_data: Dict[str, Any]) -> ValidationResult:
        """Validate a behavior tree for correctness"""
        result = ValidationResult(True)
        
        if not tree_data:
            result.add_error("Empty behavior tree")
            return result
        
        blocks = tree_data.get('blocks', [])
        if not blocks:
            result.add_error("No blocks in behavior tree")
            return result
        
        # Check for root node (node with no input connections)
        root_nodes = []
        for block in blocks:
            has_input = any(block['id'] in other_block.get('connections', []) 
                          for other_block in blocks)
            if not has_input:
                root_nodes.append(block)
        
        if len(root_nodes) == 0:
            result.add_error("No root node found")
        elif len(root_nodes) > 1:
            result.add_warning("Multiple root nodes found")
        
        # Validate individual blocks
        for block in blocks:
            block_result = self._validate_block(block)
            result.errors.extend(block_result.errors)
            result.warnings.extend(block_result.warnings)
            if not block_result.is_valid:
                result.is_valid = False
        
        # Check for unreachable nodes
        if root_nodes:
            reachable_nodes = self._find_reachable_nodes(root_nodes[0], blocks)
            all_node_ids = {block['id'] for block in blocks}
            unreachable = all_node_ids - reachable_nodes
            
            if unreachable:
                result.add_warning(f"Unreachable nodes: {list(unreachable)}")
        
        return result
    
    def _validate_block(self, block: Dict[str, Any]) -> ValidationResult:
        """Validate an individual block"""
        result = ValidationResult(True)
        
        # Check required fields
        required_fields = ['id', 'type', 'data']
        for field in required_fields:
            if field not in block:
                result.add_error(f"Block missing required field: {field}")
        
        if not result.is_valid:
            return result
        
        block_type = block['type']
        block_data = block['data']
        
        # Check if block type is supported
        if block_type not in self.code_templates:
            result.add_error(f"Unsupported block type: {block_type}")
            return result
        
        # Validate parameters
        parameters = block_data.get('parameters', {})
        rules = self.validation_rules.get(block_type, [])
        
        for rule in rules:
            if not self._check_validation_rule(parameters, rule):
                result.add_error(f"Block {block['id']} ({block_type}): {rule}")
        
        return result
    
    def _check_validation_rule(self, parameters: Dict[str, Any], rule: str) -> bool:
        """Check a specific validation rule"""
        # This is a simplified rule checker
        # In a real implementation, this would be more sophisticated
        
        if 'must be a positive float' in rule:
            param_name = rule.split(' ')[0]
            value = parameters.get(param_name)
            return isinstance(value, (int, float)) and value > 0
        
        elif 'must be a positive integer' in rule:
            param_name = rule.split(' ')[0]
            value = parameters.get(param_name)
            return isinstance(value, int) and value > 0
        
        elif 'must be a non-negative float' in rule:
            param_name = rule.split(' ')[0]
            value = parameters.get(param_name)
            return isinstance(value, (int, float)) and value >= 0
        
        elif 'must be between 0 and 1' in rule:
            param_name = rule.split(' ')[0]
            value = parameters.get(param_name)
            return isinstance(value, (int, float)) and 0 <= value <= 1
        
        elif 'must be between -1 and 1' in rule:
            param_name = rule.split(' ')[0]
            value = parameters.get(param_name)
            return isinstance(value, (int, float)) and -1 <= value <= 1
        
        elif 'must be a non-empty string' in rule:
            param_name = rule.split(' ')[0]
            value = parameters.get(param_name)
            return isinstance(value, str) and len(value) > 0
        
        elif 'must be greater than' in rule:
            parts = rule.split(' ')
            param1 = parts[0]
            param2 = parts[-1]
            value1 = parameters.get(param1)
            value2 = parameters.get(param2)
            return (isinstance(value1, (int, float)) and 
                   isinstance(value2, (int, float)) and 
                   value1 > value2)
        
        return True  # Default to pass for unknown rules
    
    def _find_reachable_nodes(self, root_node: Dict[str, Any], 
                            all_blocks: List[Dict[str, Any]]) -> set:
        """Find all nodes reachable from the root node"""
        reachable = set()
        to_visit = [root_node['id']]
        
        # Create connection map
        connections = {}
        for block in all_blocks:
            connections[block['id']] = block.get('connections', [])
        
        while to_visit:
            current_id = to_visit.pop()
            if current_id in reachable:
                continue
            
            reachable.add(current_id)
            
            # Add connected nodes to visit list
            for connected_id in connections.get(current_id, []):
                if connected_id not in reachable:
                    to_visit.append(connected_id)
        
        return reachable
    
    def generate_code(self, tree_data: Dict[str, Any]) -> str:
        """Generate executable Python code from behavior tree"""
        # First validate the tree
        validation_result = self.validate_behavior_tree(tree_data)
        if not validation_result.is_valid:
            raise ValueError(f"Invalid behavior tree: {validation_result.errors}")
        
        blocks = tree_data.get('blocks', [])
        
        # Generate imports and setup
        code_parts = [
            "# Generated behavior tree code",
            "# This code was automatically generated from a visual behavior tree",
            "",
            "import math",
            "import time",
            "from typing import Dict, Any, List",
            "",
            "class BehaviorTreeExecutor:",
            "    def __init__(self, vehicle_controller, environment, path_manager=None):",
            "        self.vehicle_controller = vehicle_controller",
            "        self.environment = environment", 
            "        self.path_manager = path_manager",
            "        self.debug_enabled = False",
            "        self.debug_states = {}",
            "",
        ]
        
        # Generate individual block functions
        for block in blocks:
            block_code = self._generate_block_code(block)
            code_parts.append(textwrap.indent(block_code, "    "))
            code_parts.append("")
        
        # Generate main execution function
        execution_code = self._generate_execution_code(blocks)
        code_parts.append(textwrap.indent(execution_code, "    "))
        
        return "\n".join(code_parts)
    
    def _generate_block_code(self, block: Dict[str, Any]) -> str:
        """Generate code for a single block"""
        block_type = block['type']
        block_id = block['id'].replace('-', '_')  # Make valid Python identifier
        block_data = block['data']
        
        if block_type not in self.code_templates:
            return f"# Unsupported block type: {block_type}"
        
        template = self.code_templates[block_type]
        
        # Extract parameters
        parameters = block_data.get('parameters', {})
        
        # Format template with parameters
        try:
            # Process parameters to ensure proper formatting
            formatted_params = {}
            for key, value in parameters.items():
                if isinstance(value, str):
                    formatted_params[key] = f'"{value}"'
                else:
                    formatted_params[key] = value
            
            formatted_code = template.format(
                node_id=block_id,
                **formatted_params
            )
        except KeyError as e:
            return f"# Error formatting template for {block_type}: missing parameter {e}"
        
        return formatted_code
    
    def _generate_execution_code(self, blocks: List[Dict[str, Any]]) -> str:
        """Generate the main execution function"""
        # Find root node
        root_nodes = []
        for block in blocks:
            has_input = any(block['id'] in other_block.get('connections', []) 
                          for other_block in blocks)
            if not has_input:
                root_nodes.append(block)
        
        if not root_nodes:
            return "# Error: No root node found"
        
        root_node = root_nodes[0]
        
        execution_code = f'''
def execute_behavior_tree(self, vehicle_state):
    """Execute the behavior tree"""
    try:
        # Start execution from root node
        result = self._execute_node("{root_node['id']}", vehicle_state)
        return result
    except Exception as e:
        return {{
            'success': False,
            'error': str(e)
        }}

def _execute_node(self, node_id, vehicle_state):
    """Execute a specific node"""
    # This would contain the execution logic for traversing the tree
    # For now, return a placeholder
    return {{
        'success': True,
        'node_id': node_id
    }}
'''
        
        return execution_code
    
    def enable_debugging(self):
        """Enable debugging mode"""
        self.debug_enabled = True
        self.debug_states = {}
    
    def disable_debugging(self):
        """Disable debugging mode"""
        self.debug_enabled = False
        self.debug_states = {}
    
    def get_debug_state(self, node_id: str) -> Optional[DebugState]:
        """Get debug state for a specific node"""
        return self.debug_states.get(node_id)
    
    def set_debug_state(self, node_id: str, state: DebugState):
        """Set debug state for a specific node"""
        self.debug_states[node_id] = state
    
    def step_through_execution(self, tree_data: Dict[str, Any], 
                             vehicle_state: Any) -> List[DebugState]:
        """Step through behavior tree execution for debugging"""
        if not self.debug_enabled:
            self.enable_debugging()
        
        debug_steps = []
        blocks = tree_data.get('blocks', [])
        
        # Find root node
        root_nodes = [block for block in blocks 
                     if not any(block['id'] in other_block.get('connections', []) 
                              for other_block in blocks)]
        
        if not root_nodes:
            return debug_steps
        
        # Simulate step-by-step execution in original order
        for block in blocks:
            debug_state = DebugState(
                node_id=block['id'],
                node_type=block['type'],
                state=ExecutionState.READY,
                input_data={},
                output_data={},
                execution_time=0.0
            )
            debug_steps.append(debug_state)
        
        return debug_steps


class BehaviorTreeDebugger:
    """Debugging tools for behavior tree execution"""
    
    def __init__(self, code_generator: CodeGenerator):
        self.code_generator = code_generator
        self.breakpoints = set()
        self.current_step = 0
        self.execution_history = []
    
    def add_breakpoint(self, node_id: str):
        """Add a breakpoint at a specific node"""
        self.breakpoints.add(node_id)
    
    def remove_breakpoint(self, node_id: str):
        """Remove a breakpoint from a specific node"""
        self.breakpoints.discard(node_id)
    
    def clear_breakpoints(self):
        """Clear all breakpoints"""
        self.breakpoints.clear()
    
    def start_debug_session(self, tree_data: Dict[str, Any], vehicle_state: Any):
        """Start a debugging session"""
        self.current_step = 0
        self.execution_history = []
        self.code_generator.enable_debugging()
        
        # Get step-by-step execution
        debug_steps = self.code_generator.step_through_execution(tree_data, vehicle_state)
        self.execution_history = debug_steps
        
        return debug_steps
    
    def step_forward(self) -> Optional[DebugState]:
        """Step forward one execution step"""
        if self.current_step < len(self.execution_history):
            current_state = self.execution_history[self.current_step]
            self.current_step += 1
            return current_state
        return None
    
    def step_backward(self) -> Optional[DebugState]:
        """Step backward one execution step"""
        if self.current_step > 0:
            self.current_step -= 1
            # Return the state we just stepped back to
            if self.current_step > 0:
                return self.execution_history[self.current_step - 1]
            else:
                return self.execution_history[0] if self.execution_history else None
        return None
    
    def get_current_state(self) -> Optional[DebugState]:
        """Get the current debug state"""
        if 0 <= self.current_step < len(self.execution_history):
            return self.execution_history[self.current_step]
        return None
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get a summary of the execution"""
        if not self.execution_history:
            return {'total_steps': 0, 'current_step': 0}
        
        successful_steps = sum(1 for state in self.execution_history 
                             if state.state == ExecutionState.SUCCESS)
        failed_steps = sum(1 for state in self.execution_history 
                         if state.state == ExecutionState.FAILURE)
        
        return {
            'total_steps': len(self.execution_history),
            'current_step': self.current_step,
            'successful_steps': successful_steps,
            'failed_steps': failed_steps,
            'completion_rate': successful_steps / len(self.execution_history) if self.execution_history else 0
        }


class RealTimeBehaviorValidator:
    """Real-time validation and error checking for behavior trees"""
    
    def __init__(self):
        self.validation_cache = {}
        self.error_callbacks = []
    
    def add_error_callback(self, callback):
        """Add a callback for validation errors"""
        self.error_callbacks.append(callback)
    
    def validate_real_time(self, tree_data: Dict[str, Any]) -> ValidationResult:
        """Perform real-time validation of behavior tree"""
        # Create cache key
        cache_key = self._create_cache_key(tree_data)
        
        # Check cache first
        if cache_key in self.validation_cache:
            return self.validation_cache[cache_key]
        
        # Perform validation
        code_generator = CodeGenerator()
        result = code_generator.validate_behavior_tree(tree_data)
        
        # Cache result
        self.validation_cache[cache_key] = result
        
        # Notify error callbacks
        if not result.is_valid:
            for callback in self.error_callbacks:
                callback(result)
        
        return result
    
    def clear_cache(self):
        """Clear the validation cache"""
        self.validation_cache = {}
    
    def _create_cache_key(self, tree_data: Dict[str, Any]) -> str:
        """Create a cache key for the tree data"""
        # Create a deterministic string representation
        return json.dumps(tree_data, sort_keys=True)
    
    def check_syntax_errors(self, code: str) -> List[str]:
        """Check for syntax errors in generated code"""
        errors = []
        
        try:
            ast.parse(code)
        except SyntaxError as e:
            errors.append(f"Syntax error at line {e.lineno}: {e.msg}")
        except Exception as e:
            errors.append(f"Code parsing error: {str(e)}")
        
        return errors
    
    def check_runtime_errors(self, tree_data: Dict[str, Any]) -> List[str]:
        """Check for potential runtime errors in behavior tree"""
        errors = []
        blocks = tree_data.get('blocks', [])
        
        # Check for blocks with missing input connections
        for block in blocks:
            block_data = block.get('data', {})
            
            # If block requires input but has no input connections
            if block_data.get('has_input', False):
                has_input_connection = any(
                    block['id'] in other_block.get('connections', [])
                    for other_block in blocks
                )
                
                if not has_input_connection:
                    errors.append(f"Block '{block['id']}' requires input but has no input connection")
            
            # Check parameter types
            parameters = block_data.get('parameters', {})
            for param_name, param_value in parameters.items():
                if 'force' in param_name or 'speed' in param_name or 'threshold' in param_name:
                    if not isinstance(param_value, (int, float)):
                        errors.append(f"Parameter '{param_name}' in block '{block['id']}' should be numeric")
        
        return errors