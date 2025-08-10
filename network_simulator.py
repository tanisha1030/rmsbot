"""
Network Traffic Simulator using SimPy for Botnet Detection
Simulates realistic robotic network behavior and botnet activities
"""

import simpy
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Tuple
import random
from enum import Enum
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrafficType(Enum):
    NORMAL = "normal"
    BOTNET = "botnet"
    COMMAND_CONTROL = "c2"

@dataclass
class NetworkPacket:
    """Represents a network packet with timing and size information"""
    timestamp: float
    source_id: str
    destination_id: str
    packet_size: int
    traffic_type: TrafficType
    interval_from_last: float = 0.0

class RobotAgent:
    """Simulates a robotic agent in the network"""
    
    def __init__(self, env: simpy.Environment, robot_id: str, is_compromised: bool = False):
        self.env = env
        self.robot_id = robot_id
        self.is_compromised = is_compromised
        self.packets_sent = []
        self.last_packet_time = 0.0
        self.behavioral_params = self._initialize_behavior()
        
    def _initialize_behavior(self) -> Dict:
        """Initialize behavioral parameters for the robot"""
        if self.is_compromised:
            return {
                'packet_size_mean': random.uniform(800, 1200),
                'packet_size_std': random.uniform(50, 150),
                'send_interval_mean': random.uniform(0.1, 0.8),
                'send_interval_std': random.uniform(0.05, 0.2),
                'burst_probability': 0.3,
                'burst_size': random.randint(5, 15)
            }
        else:
            return {
                'packet_size_mean': random.uniform(300, 700),
                'packet_size_std': random.uniform(100, 200),
                'send_interval_mean': random.uniform(1.0, 3.0),
                'send_interval_std': random.uniform(0.5, 1.0),
                'burst_probability': 0.05,
                'burst_size': random.randint(2, 5)
            }
    
    def generate_packet(self, destination: str = "server") -> NetworkPacket:
        """Generate a network packet based on robot behavior"""
        current_time = self.env.now
        interval = current_time - self.last_packet_time
        
        # Determine packet size based on behavior
        packet_size = max(64, int(np.random.normal(
            self.behavioral_params['packet_size_mean'],
            self.behavioral_params['packet_size_std']
        )))
        
        # Determine traffic type
        if self.is_compromised:
            # Compromised robots might send different types of traffic
            traffic_type = random.choices(
                [TrafficType.BOTNET, TrafficType.COMMAND_CONTROL],
                weights=[0.8, 0.2]
            )[0]
        else:
            traffic_type = TrafficType.NORMAL
        
        packet = NetworkPacket(
            timestamp=current_time,
            source_id=self.robot_id,
            destination_id=destination,
            packet_size=packet_size,
            traffic_type=traffic_type,
            interval_from_last=interval
        )
        
        self.packets_sent.append(packet)
        self.last_packet_time = current_time
        
        return packet

    def normal_activity(self):
        """Simulate normal robot activities"""
        while True:
            # Regular communication interval
            interval = max(0.1, np.random.normal(
                self.behavioral_params['send_interval_mean'],
                self.behavioral_params['send_interval_std']
            ))
            
            yield self.env.timeout(interval)
            
            # Generate and send packet
            packet = self.generate_packet()
            logger.debug(f"Robot {self.robot_id} sent packet: {packet.packet_size} bytes")
            
            # Occasional burst activity
            if random.random() < self.behavioral_params['burst_probability']:
                burst_size = self.behavioral_params['burst_size']
                for _ in range(burst_size):
                    yield self.env.timeout(random.uniform(0.01, 0.1))
                    burst_packet = self.generate_packet()
                    logger.debug(f"Robot {self.robot_id} sent burst packet: {burst_packet.packet_size} bytes")

class BotnetController:
    """Simulates botnet command and control activities"""
    
    def __init__(self, env: simpy.Environment, compromised_robots: List[RobotAgent]):
        self.env = env
        self.compromised_robots = compromised_robots
        self.commands_sent = 0
        
    def send_commands(self):
        """Periodically send commands to compromised robots"""
        while True:
            # Wait for command interval (irregular timing to avoid detection)
            command_interval = random.uniform(5.0, 30.0)
            yield self.env.timeout(command_interval)
            
            # Send commands to subset of compromised robots
            target_robots = random.sample(
                self.compromised_robots, 
                k=random.randint(1, len(self.compromised_robots))
            )
            
            for robot in target_robots:
                # Simulate command packet
                command_packet = NetworkPacket(
                    timestamp=self.env.now,
                    source_id="botnet_server",
                    destination_id=robot.robot_id,
                    packet_size=random.randint(100, 500),
                    traffic_type=TrafficType.COMMAND_CONTROL,
                    interval_from_last=0.0
                )
                robot.packets_sent.append(command_packet)
            
            self.commands_sent += 1
            logger.info(f"Botnet controller sent command #{self.commands_sent} to {len(target_robots)} robots")

class NetworkSimulator:
    """Main network simulation environment"""
    
    def __init__(self, num_normal_robots: int = 50, num_compromised_robots: int = 10, 
                 simulation_time: float = 300.0):
        self.env = simpy.Environment()
        self.num_normal_robots = num_normal_robots
        self.num_compromised_robots = num_compromised_robots
        self.simulation_time = simulation_time
        self.robots = []
        self.all_packets = []
        
    def setup_network(self):
        """Initialize the network with robots and botnet controller"""
        logger.info(f"Setting up network with {self.num_normal_robots} normal and {self.num_compromised_robots} compromised robots")
        
        # Create normal robots
        for i in range(self.num_normal_robots):
            robot = RobotAgent(self.env, f"robot_normal_{i}", is_compromised=False)
            self.robots.append(robot)
            self.env.process(robot.normal_activity())
        
        # Create compromised robots
        compromised_robots = []
        for i in range(self.num_compromised_robots):
            robot = RobotAgent(self.env, f"robot_compromised_{i}", is_compromised=True)
            self.robots.append(robot)
            compromised_robots.append(robot)
            self.env.process(robot.normal_activity())
        
        # Create botnet controller
        if compromised_robots:
            botnet_controller = BotnetController(self.env, compromised_robots)
            self.env.process(botnet_controller.send_commands())
    
    def run_simulation(self) -> pd.DataFrame:
        """Run the network simulation and return collected data"""
        logger.info(f"Starting network simulation for {self.simulation_time} seconds")
        
        self.setup_network()
        self.env.run(until=self.simulation_time)
        
        # Collect all packets from all robots
        for robot in self.robots:
            self.all_packets.extend(robot.packets_sent)
        
        logger.info(f"Simulation completed. Collected {len(self.all_packets)} packets")
        
        # Convert to DataFrame for analysis
        return self._packets_to_dataframe()
    
    def _packets_to_dataframe(self) -> pd.DataFrame:
        """Convert collected packets to a pandas DataFrame"""
        if not self.all_packets:
            return pd.DataFrame()
        
        data = []
        for packet in self.all_packets:
            data.append({
                'timestamp': packet.timestamp,
                'source_id': packet.source_id,
                'destination_id': packet.destination_id,
                'packet_size': packet.packet_size,
                'interval': packet.interval_from_last,
                'traffic_type': packet.traffic_type.value,
                'is_botnet': 1 if packet.traffic_type in [TrafficType.BOTNET, TrafficType.COMMAND_CONTROL] else 0
            })
        
        df = pd.DataFrame(data)
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Recalculate intervals for better accuracy
        df['interval'] = df.groupby('source_id')['timestamp'].diff().fillna(0)
        
        return df
    
    def get_network_statistics(self) -> Dict:
        """Get comprehensive network statistics"""
        df = self._packets_to_dataframe()
        
        if df.empty:
            return {}
        
        stats = {
            'total_packets': len(df),
            'normal_packets': len(df[df['is_botnet'] == 0]),
            'botnet_packets': len(df[df['is_botnet'] == 1]),
            'unique_sources': df['source_id'].nunique(),
            'simulation_duration': self.simulation_time,
            'avg_packet_size': df['packet_size'].mean(),
            'avg_interval': df['interval'].mean(),
            'packets_per_second': len(df) / self.simulation_time,
            'botnet_ratio': df['is_botnet'].mean()
        }
        
        return stats

def run_enhanced_simulation(output_file: str = "simulated_robot_logs.csv", 
                          num_normal: int = 50, num_compromised: int = 10,
                          sim_time: float = 300.0) -> Tuple[pd.DataFrame, Dict]:
    """
    Run an enhanced network simulation and save results
    
    Args:
        output_file: Path to save the generated dataset
        num_normal: Number of normal robots
        num_compromised: Number of compromised robots  
        sim_time: Simulation time in seconds
    
    Returns:
        Tuple of (DataFrame with network data, statistics dictionary)
    """
    # Create and run simulation
    simulator = NetworkSimulator(
        num_normal_robots=num_normal,
        num_compromised_robots=num_compromised,
        simulation_time=sim_time
    )
    
    # Run simulation
    network_data = simulator.run_simulation()
    
    # Get statistics
    stats = simulator.get_network_statistics()
    
    # Save to file
    if not network_data.empty:
        network_data.to_csv(output_file, index=False)
        logger.info(f"Network data saved to {output_file}")
    
    # Print summary
    print(f"\n=== Simulation Summary ===")
    print(f"Total packets generated: {stats.get('total_packets', 0):,}")
    print(f"Normal traffic: {stats.get('normal_packets', 0):,}")
    print(f"Botnet traffic: {stats.get('botnet_packets', 0):,}")
    print(f"Botnet ratio: {stats.get('botnet_ratio', 0):.2%}")
    print(f"Average packet size: {stats.get('avg_packet_size', 0):.1f} bytes")
    print(f"Average interval: {stats.get('avg_interval', 0):.2f} seconds")
    print(f"Packets per second: {stats.get('packets_per_second', 0):.1f}")
    
    return network_data, stats

if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    random.seed(42)
    
    # Run simulation with default parameters
    data, statistics = run_enhanced_simulation(
        output_file="simulated_robot_logs.csv",
        num_normal=30,
        num_compromised=8,
        sim_time=180.0  # 3 minutes
    )
    
    print(f"\nGenerated dataset shape: {data.shape}")
    print(f"Sample data:")
    print(data.head())
