import os
import traci
import numpy as np
from pathlib import Path

class HighwayEnvironment:
    def __init__(self, 
                 num_lanes=3,
                 highway_length=1000,
                 ramp_length=200,
                 gui=False):
        self.num_lanes = num_lanes
        self.highway_length = highway_length
        self.ramp_length = ramp_length
        self.gui = gui
        
        # Create necessary paths
        self.base_dir = Path("sumo_env")
        self.base_dir.mkdir(exist_ok=True)
        
        # Generate necessary SUMO files
        self._generate_network()
        self.generate_route_file()
        self.generate_config_file()

    def _generate_network(self):
        """Generate SUMO network with highway and ramp"""
        with open(self.base_dir / "highway.netcfg", "w") as netfile:
            netfile.write("""<?xml version="1.0" encoding="UTF-8"?>
<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
    <input>
        <node-files value="nodes.nod.xml"/>
        <edge-files value="edges.edg.xml"/>
        <connection-files value="connections.con.xml"/>
    </input>
    <output>
        <output-file value="highway.net.xml"/>
    </output>
</configuration>""")
            
        with open(self.base_dir / "nodes.nod.xml", "w") as nodesfile:
            nodesfile.write("""<?xml version="1.0" encoding="UTF-8"?>
<nodes>
    <node id="start" x="0" y="0"/>
    <node id="ramp_start" x="400" y="-100"/>
    <node id="ramp_end" x="500" y="0" type="traffic_light"/>
    <node id="end" x="1000" y="0"/>
</nodes>""")
            
        with open(self.base_dir / "edges.edg.xml", "w") as edgesfile:
            edgesfile.write(f"""<?xml version="1.0" encoding="UTF-8"?>
<edges>
    <edge id="highway1" from="start" to="ramp_end" numLanes="{self.num_lanes}" speed="27.78" />
    <edge id="highway2" from="ramp_end" to="end" numLanes="{self.num_lanes}" speed="27.78" />
    <edge id="ramp" from="ramp_start" to="ramp_end" numLanes="1" speed="16.67" />
</edges>""")
            
        with open(self.base_dir / "connections.con.xml", "w") as confile:
            confile.write("""<?xml version="1.0" encoding="UTF-8"?>
<connections>
    <connection from="highway1" to="highway2"/>
    <connection from="ramp" to="highway2" via="ramp_end"/>
    <tlLogic id="ramp_end" type="static" programID="1" offset="0">
        <phase duration="10" state="GGGG"/>
        <phase duration="3" state="yGGG"/>
        <phase duration="10" state="rGGG"/>
    </tlLogic>
</connections>
""")
        
        os.system(f"netconvert -c {self.base_dir}/highway.netcfg")
        
    def generate_route_file(self):
        """Generate route file with traffic flows"""
        with open(self.base_dir / "routes.rou.xml", "w") as routefile:
            routefile.write(f"""<?xml version="1.0" encoding="UTF-8"?>
<routes>
    <vType id="car" accel="2.6" decel="4.5" color="red" laneChangeModel="LC2013" sigma="0.5" length="5" minGap="2.5" maxSpeed="30" lcStrategic="10.0" lcKeepRight="0.1"/>
    <vType id="taxi_sfra" accel="2.6" decel="4.5" color="yellow" laneChangeModel="LC2013" sigma="0.5" length="5" minGap="2.5" maxSpeed="30" lcStrategic="10.0" lcKeepRight="0.1"/>
    <vType id="truck" length="12.00" maxSpeed="25.00" color="blue" laneChangeModel="LC2013" lcStrategic="1.0" lcSpeedGain="0.8" lcKeepRight="0.4" accel="1.0" decel="3.0" sigma="0.6"/>
    <vType id="bus" length="12.00" maxSpeed="20.00" color="yellow" laneChangeModel="LC2013" lcStrategic="1.0" lcSpeedGain="0.7" lcKeepRight="0.5" accel="1.2" decel="2.5" sigma="0.5"/>    
    <vType id="motorcycle" length="2.50" maxSpeed="40.00" color="green" laneChangeModel="LC2013" lcStrategic="1.2" lcSpeedGain="1.2" lcKeepRight="0.1" accel="3.0" decel="4.5" sigma="0.3"/>

    <flow id="highway_flow_taxi" type="taxi_sfra" begin="0" end="3600" period="2" departLane="random">
        <route edges="highway1 highway2"/>
    </flow>
    <flow id="ramp_flow_taxi" type="taxi_sfra" begin="0" end="3600" period="30">
        <route edges="ramp highway2"/>    
    </flow> 
    <flow id="highway_flow_truck" type="truck" begin="0" end="3600" period="50" departLane="random">
        <route edges="highway1 highway2"/>
    </flow>
    <flow id="highway_flow_bus" type="bus" begin="0" end="3600" period="40" departLane="random">
        <route edges="highway1 highway2"/>
    </flow> 
    <flow id="highway_flow_moto" type="motorcycle" begin="0" end="3600" period="10" departLane="random">
        <route edges="highway1 highway2"/>
    </flow>
    <flow id="ramp_flow_moto" type="motorcycle" begin="0" end="3600" period="30">
        <route edges="ramp highway2"/>    
    </flow>  
""")
            # Define multiple flows
            flows = [
                {"start": 0, "end": 600, "highway_flow": 4000, "ramp_flow": 600},  # Heavy flow at the start
                {"start": 600, "end": 1600, "highway_flow": 1000, "ramp_flow": 150},  # Lighter flow
                {"start": 1600, "end": 2800, "highway_flow": 6000, "ramp_flow": 800},  # Very heavy flow
                {"start": 2800, "end": 3600, "highway_flow": 1000, "ramp_flow": 600},  # Lighter flow
            ]

            for flow in flows:
                highway_period = 3600 / flow["highway_flow"]
                ramp_period = 3600 / flow["ramp_flow"]
                
                routefile.write(f"""
    <flow id="highway_flow_{flow['start']}" type="car" begin="{flow['start']}" end="{flow['end']}" period="{highway_period}" departLane="random">
        <route edges="highway1 highway2"/>
    </flow>
    <flow id="ramp_flow_{flow['start']}" type="car" begin="{flow['start']}" end="{flow['end']}" period="{ramp_period}">
        <route edges="ramp highway2"/>    
    </flow>    
""")
            routefile.write("</routes>")

    def generate_config_file(self):
        """Generate SUMO configuration file"""
        with open(self.base_dir / "highway.sumocfg", "w") as configfile:
            configfile.write(f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <input>
        <net-file value="highway.net.xml"/>
        <route-files value="routes.rou.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="7200"/>
    </time>
</configuration>""")

    def reset(self):
        """Reset the environment for a new episode"""
        config_path = str(self.base_dir / "highway.sumocfg")
        print(f"Loading SUMO configuration from: {config_path}")
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        sumo_binary = "sumo-gui" if self.gui else "sumo"
        traci.start([sumo_binary, "-c", config_path])
        
        return self.get_state()

    def get_state(self):
        """Get the current state of the environment"""
        vehicle_ids = traci.vehicle.getIDList()
        if len(vehicle_ids) == 0:
            avg_speed = 0
        else:
            avg_speed = np.mean([traci.vehicle.getSpeed(veh) for veh in vehicle_ids])
        queue_length = len(vehicle_ids)  # Simplified queue length for ramp
        
        return np.array([avg_speed, queue_length])

    def step(self, action):
        """Step through the environment based on the action taken"""
        green_light_duration = action * 5  # Scale action to seconds
        
        traci.trafficlight.setPhase("ramp_end", 1)  
        
        traci.simulationStep()

        waiting_time = sum(traci.vehicle.getWaitingTime(veh) for veh in traci.vehicle.getIDList())
        
        reward = -waiting_time
        
        next_state = self.get_state()
        
        return next_state, reward

    def start_simulation(self):
        """Start SUMO simulation"""
        if 'SUMO_HOME' not in os.environ:
            raise ValueError("Please set SUMO_HOME environment variable")
            
        sumo_binary = "sumo-gui" if self.gui else "sumo"
        
        sumo_cmd = [sumo_binary,
                   "-c", str(self.base_dir / "highway.sumocfg"),
                   "--time-to-teleport", "-1"]
        
        traci.start(sumo_cmd)
        
    def end_simulation(self):
        """End SUMO simulation"""
        traci.close()