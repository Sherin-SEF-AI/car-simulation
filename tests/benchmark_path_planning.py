"""
Performance benchmarks for path planning algorithms
"""

import time
import statistics
from typing import List, Tuple
import matplotlib.pyplot as plt

from src.core.path_planning import (
    Point2D, Obstacle, AStarPlanner, RRTPlanner, 
    DynamicPathPlanner, create_test_environment
)

class PathPlanningBenchmark:
    """Benchmark suite for path planning algorithms"""
    
    def __init__(self):
        self.results = {}
    
    def benchmark_astar(self, scenarios: List[Tuple], iterations: int = 10) -> dict:
        """Benchmark A* algorithm performance"""
        print("Benchmarking A* Algorithm...")
        
        results = {
            'algorithm': 'A*',
            'scenarios': [],
            'avg_times': [],
            'success_rates': [],
            'path_lengths': []
        }
        
        for i, (obstacles, start, goal, description) in enumerate(scenarios):
            print(f"  Scenario {i+1}: {description}")
            
            planner = AStarPlanner(grid_resolution=1.0)
            planner.set_bounds(Point2D(-50, -50), Point2D(50, 50))
            planner.set_obstacles(obstacles)
            
            times = []
            successes = 0
            path_lengths = []
            
            for _ in range(iterations):
                start_time = time.time()
                path = planner.plan_path(start, goal)
                end_time = time.time()
                
                times.append(end_time - start_time)
                
                if path is not None:
                    successes += 1
                    path_lengths.append(path.total_length)
            
            avg_time = statistics.mean(times)
            success_rate = successes / iterations
            avg_path_length = statistics.mean(path_lengths) if path_lengths else 0
            
            results['scenarios'].append(description)
            results['avg_times'].append(avg_time)
            results['success_rates'].append(success_rate)
            results['path_lengths'].append(avg_path_length)
            
            print(f"    Avg Time: {avg_time:.4f}s, Success Rate: {success_rate:.2%}, Avg Path Length: {avg_path_length:.2f}")
        
        return results
    
    def benchmark_rrt(self, scenarios: List[Tuple], iterations: int = 10) -> dict:
        """Benchmark RRT algorithm performance"""
        print("Benchmarking RRT Algorithm...")
        
        results = {
            'algorithm': 'RRT',
            'scenarios': [],
            'avg_times': [],
            'success_rates': [],
            'path_lengths': []
        }
        
        for i, (obstacles, start, goal, description) in enumerate(scenarios):
            print(f"  Scenario {i+1}: {description}")
            
            planner = RRTPlanner(step_size=2.0, max_iterations=2000, goal_bias=0.1)
            planner.set_bounds(Point2D(-50, -50), Point2D(50, 50))
            planner.set_obstacles(obstacles)
            
            times = []
            successes = 0
            path_lengths = []
            
            for _ in range(iterations):
                start_time = time.time()
                path = planner.plan_path(start, goal)
                end_time = time.time()
                
                times.append(end_time - start_time)
                
                if path is not None:
                    successes += 1
                    path_lengths.append(path.total_length)
            
            avg_time = statistics.mean(times)
            success_rate = successes / iterations
            avg_path_length = statistics.mean(path_lengths) if path_lengths else 0
            
            results['scenarios'].append(description)
            results['avg_times'].append(avg_time)
            results['success_rates'].append(success_rate)
            results['path_lengths'].append(avg_path_length)
            
            print(f"    Avg Time: {avg_time:.4f}s, Success Rate: {success_rate:.2%}, Avg Path Length: {avg_path_length:.2f}")
        
        return results
    
    def benchmark_dynamic_planning(self, base_scenarios: List[Tuple], iterations: int = 5) -> dict:
        """Benchmark dynamic path planning performance"""
        print("Benchmarking Dynamic Path Planning...")
        
        results = {
            'algorithm': 'Dynamic',
            'scenarios': [],
            'avg_times': [],
            'replan_counts': [],
            'success_rates': []
        }
        
        for i, (obstacles, start, goal, description) in enumerate(base_scenarios):
            print(f"  Scenario {i+1}: {description}")
            
            base_planner = AStarPlanner(grid_resolution=1.0)
            base_planner.set_bounds(Point2D(-50, -50), Point2D(50, 50))
            
            dynamic_planner = DynamicPathPlanner(base_planner, replan_threshold=1.0)
            
            times = []
            replan_counts = []
            successes = 0
            
            for _ in range(iterations):
                start_time = time.time()
                
                # Initial planning
                initial_path = dynamic_planner.plan_initial_path(start, goal)
                
                if initial_path is not None:
                    successes += 1
                    
                    # Simulate dynamic obstacle updates
                    replan_count = 0
                    for step in range(5):  # 5 simulation steps
                        # Add dynamic obstacles
                        dynamic_obstacles = obstacles + [
                            Obstacle(Point2D(10 + step * 2, 10 + step), 1.5)
                        ]
                        
                        dynamic_planner.update_obstacles(dynamic_obstacles)
                        updated_path = dynamic_planner.update_path(step * 1.5)
                        
                        if updated_path != dynamic_planner.current_path:
                            replan_count += 1
                    
                    replan_counts.append(replan_count)
                
                end_time = time.time()
                times.append(end_time - start_time)
            
            avg_time = statistics.mean(times)
            success_rate = successes / iterations
            avg_replan_count = statistics.mean(replan_counts) if replan_counts else 0
            
            results['scenarios'].append(description)
            results['avg_times'].append(avg_time)
            results['replan_counts'].append(avg_replan_count)
            results['success_rates'].append(success_rate)
            
            print(f"    Avg Time: {avg_time:.4f}s, Success Rate: {success_rate:.2%}, Avg Replans: {avg_replan_count:.1f}")
        
        return results
    
    def create_benchmark_scenarios(self) -> List[Tuple]:
        """Create various benchmark scenarios"""
        scenarios = []
        
        # Scenario 1: No obstacles (baseline)
        scenarios.append((
            [],
            Point2D(0, 0),
            Point2D(30, 30),
            "No obstacles"
        ))
        
        # Scenario 2: Few obstacles
        obstacles_few = [
            Obstacle(Point2D(10, 10), 3.0),
            Obstacle(Point2D(20, 20), 2.5)
        ]
        scenarios.append((
            obstacles_few,
            Point2D(0, 0),
            Point2D(30, 30),
            "Few obstacles"
        ))
        
        # Scenario 3: Many obstacles
        obstacles_many = []
        for i in range(10):
            x = 5 + (i % 5) * 6
            y = 5 + (i // 5) * 6
            obstacles_many.append(Obstacle(Point2D(x, y), 2.0))
        
        scenarios.append((
            obstacles_many,
            Point2D(0, 0),
            Point2D(30, 30),
            "Many obstacles"
        ))
        
        # Scenario 4: Narrow passage
        obstacles_narrow = [
            Obstacle(Point2D(15, 5), 4.0),
            Obstacle(Point2D(15, 25), 4.0),
            Obstacle(Point2D(10, 15), 2.0),
            Obstacle(Point2D(20, 15), 2.0)
        ]
        scenarios.append((
            obstacles_narrow,
            Point2D(0, 15),
            Point2D(30, 15),
            "Narrow passage"
        ))
        
        # Scenario 5: Complex maze-like
        obstacles_maze = []
        # Create maze-like structure
        for x in range(5, 26, 5):
            for y in range(5, 26, 5):
                if (x + y) % 10 == 0:
                    obstacles_maze.append(Obstacle(Point2D(x, y), 2.0))
        
        scenarios.append((
            obstacles_maze,
            Point2D(0, 0),
            Point2D(30, 30),
            "Maze-like"
        ))
        
        return scenarios
    
    def run_full_benchmark(self, iterations: int = 10):
        """Run complete benchmark suite"""
        print("=== Path Planning Algorithm Benchmark ===\n")
        
        scenarios = self.create_benchmark_scenarios()
        
        # Benchmark A*
        astar_results = self.benchmark_astar(scenarios, iterations)
        self.results['astar'] = astar_results
        
        print()
        
        # Benchmark RRT
        rrt_results = self.benchmark_rrt(scenarios, iterations)
        self.results['rrt'] = rrt_results
        
        print()
        
        # Benchmark Dynamic Planning
        dynamic_results = self.benchmark_dynamic_planning(scenarios, max(1, iterations // 2))
        self.results['dynamic'] = dynamic_results
        
        print("\n=== Benchmark Complete ===")
        
        return self.results
    
    def generate_report(self):
        """Generate benchmark report"""
        if not self.results:
            print("No benchmark results available. Run benchmark first.")
            return
        
        print("\n=== BENCHMARK REPORT ===\n")
        
        # Performance comparison table
        print("Performance Comparison (Average Times):")
        print("-" * 60)
        print(f"{'Scenario':<20} {'A*':<10} {'RRT':<10} {'Dynamic':<10}")
        print("-" * 60)
        
        for i, scenario in enumerate(self.results['astar']['scenarios']):
            astar_time = self.results['astar']['avg_times'][i]
            rrt_time = self.results['rrt']['avg_times'][i]
            dynamic_time = self.results['dynamic']['avg_times'][i]
            
            print(f"{scenario:<20} {astar_time:<10.4f} {rrt_time:<10.4f} {dynamic_time:<10.4f}")
        
        print()
        
        # Success rate comparison
        print("Success Rate Comparison:")
        print("-" * 60)
        print(f"{'Scenario':<20} {'A*':<10} {'RRT':<10} {'Dynamic':<10}")
        print("-" * 60)
        
        for i, scenario in enumerate(self.results['astar']['scenarios']):
            astar_success = self.results['astar']['success_rates'][i]
            rrt_success = self.results['rrt']['success_rates'][i]
            dynamic_success = self.results['dynamic']['success_rates'][i]
            
            print(f"{scenario:<20} {astar_success:<10.2%} {rrt_success:<10.2%} {dynamic_success:<10.2%}")
        
        print()
        
        # Path quality comparison
        print("Path Length Comparison:")
        print("-" * 50)
        print(f"{'Scenario':<20} {'A*':<15} {'RRT':<15}")
        print("-" * 50)
        
        for i, scenario in enumerate(self.results['astar']['scenarios']):
            astar_length = self.results['astar']['path_lengths'][i]
            rrt_length = self.results['rrt']['path_lengths'][i]
            
            print(f"{scenario:<20} {astar_length:<15.2f} {rrt_length:<15.2f}")
        
        print()
        
        # Summary statistics
        print("Summary Statistics:")
        print("-" * 40)
        
        astar_avg_time = statistics.mean(self.results['astar']['avg_times'])
        rrt_avg_time = statistics.mean(self.results['rrt']['avg_times'])
        dynamic_avg_time = statistics.mean(self.results['dynamic']['avg_times'])
        
        astar_avg_success = statistics.mean(self.results['astar']['success_rates'])
        rrt_avg_success = statistics.mean(self.results['rrt']['success_rates'])
        dynamic_avg_success = statistics.mean(self.results['dynamic']['success_rates'])
        
        print(f"A* - Avg Time: {astar_avg_time:.4f}s, Avg Success: {astar_avg_success:.2%}")
        print(f"RRT - Avg Time: {rrt_avg_time:.4f}s, Avg Success: {rrt_avg_success:.2%}")
        print(f"Dynamic - Avg Time: {dynamic_avg_time:.4f}s, Avg Success: {dynamic_avg_success:.2%}")

def main():
    """Run path planning benchmarks"""
    benchmark = PathPlanningBenchmark()
    
    # Run benchmarks with fewer iterations for faster execution
    results = benchmark.run_full_benchmark(iterations=5)
    
    # Generate report
    benchmark.generate_report()

if __name__ == '__main__':
    main()