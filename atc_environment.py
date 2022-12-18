# -*- coding: utf-8 -*-
import time
from atc_action_space import ATCActionSpace, FCFS_ActionSpace
from atc_state_space import ATCStateSpace
from vertiport import Vertiport
from evtol import eVTOL

import airsim
import gym
from gym import spaces
import numpy as np
from sympy.geometry import Segment2D
from stable_baselines3.common.utils import set_random_seed


class ATC_Environment(gym.Env):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.vertiport_config = config["Vertiport"]
        self.evtol_config = config["eVTOL"]
        self.test = config["test"]
        self.noise = config["noise"]
        self.noise_percent = config["noise_percent"]
        self.evtol_count = config["number_of_eVTOLs"]
        self.clock_speed = config["clock_speed"]
        self.type = config["agent_type"]
        self.evtol_offsets = config["eVTOL_offsets"]

        self.current_evtol = None
        self.state_manager = None
        self.sleep_time = 0.5 / self.clock_speed # Half a second in simulation
        cont_bound = np.finfo(np.float32).max
        self.action_space = spaces.Discrete(11) 

        if config["agent_type"] == "baseline":
            self.n_features = self.evtol_count * 5 + 9 * 4 + 8  # number of evtols times their features plus number of ports times their features + next evtol features
            self.observation_space = spaces.Dict(
                                        dict(
                                            next_evtol_embedding = spaces.Box(low=-cont_bound, high=cont_bound, shape=(self.n_features,), dtype=np.float32),
                                            mask = spaces.Box(low=-cont_bound, high=cont_bound, shape=(11,), dtype=np.float32)))
        elif config["agent_type"] == "graph_learning":
            self.observation_space = spaces.Dict(
            dict(
                vertiport_features = spaces.Box(low=-cont_bound, high=cont_bound, shape=(9, 4), dtype=np.float32),
                vertiport_edge = spaces.Box(low=0.0, high=9.0, shape=(2, 72), dtype=np.float32),
                evtol_features = spaces.Box(low=-cont_bound, high=cont_bound, shape=(4, 5), dtype=np.float32),
                evtol_edge = spaces.Box(low=0.0, high=3.0, shape=(2, 12), dtype=np.float32),
                next_evtol_embedding = spaces.Box(low=-cont_bound, high=cont_bound, shape=(8,), dtype=np.float32),
                mask = spaces.Box(low=0, high=1, shape=(11,), dtype=np.float32)
            ))
        self.client = airsim.MultirotorClient()

    """
    Gym Functions
    """

    def step(self, action):
        """One of 11 actions is decoded and applied to the simulation environment."""
        start_time = time.time()

        self.select_next_evtol()

        if self.test == "FCFS":
            decoded_action, self.test_takeoff_queue, self.test_land_queue = self.action_manager.decode_action_fcfs(self.current_evtol, self.test_takeoff_queue, self.test_land_queue)
        else:
            decoded_action = self.action_manager.decode_action_atc(self.current_evtol, action)
        
        #Ideally, all movements should take place here, and ports that were previously unavailable should free up
        if decoded_action["action"] == "land in normal port":
            new_position = decoded_action["position"]
            self.complete_landing(self.current_evtol.evtol_name, new_position)
            self.current_evtol.set_status("in-action", "in-port")
            self.current_evtol.job_status["initial_loc"] = self.current_evtol.current_location
            self.current_evtol.job_status["final_dest"] = new_position

            
        elif decoded_action["action"] == "land in battery port":
            # print(["coded action", coded_action, ", current_evtol:", self.current_evtol.evtol_name])
            new_position = decoded_action["position"]
            self.complete_landing(self.current_evtol.evtol_name, new_position)
            self.current_evtol.set_status("in-action", "battery-port")
            self.current_evtol.job_status["initial_loc"] = self.current_evtol.current_location
            self.current_evtol.job_status["final_dest"] = new_position
            
        elif decoded_action["action"] == "takeoff":
            if self.noise and np.random.uniform() < self.noise_percent:
                pass
            else:
                new_position = decoded_action["position"]
                self.complete_takeoff(self.current_evtol.evtol_name, new_position)
                self.current_evtol.set_status("in-action", "in-destination")
                self.current_evtol.job_status["initial_loc"] = self.current_evtol.current_location
                self.current_evtol.job_status["final_dest"] = new_position

        elif decoded_action["action"] == "hover":
            new_position = decoded_action["position"]
            self.move_position(self.current_evtol.evtol_name, new_position,join=0)
            self.current_evtol.status = self.current_evtol.all_states['in-air']
            self.current_evtol.job_status["initial_loc"] = self.current_evtol.current_location
            self.current_evtol.job_status["final_dest"] = new_position
        
        elif decoded_action['action'] == 'takeoff to a hovering spot':
            if self.noise and np.random.uniform() < self.noise_percent:
                pass
            else:
                new_position = decoded_action["position"]
                self.complete_takeoff(self.current_evtol.evtol_name,new_position, hover=True)
                self.current_evtol.status = self.current_evtol.all_states['in-air']
                self.current_evtol.job_status["initial_loc"] = self.current_evtol.current_location
                self.current_evtol.job_status["final_dest"] = new_position


        elif decoded_action['action'] == 'move to battery port':
            if self.noise and np.random.uniform() < self.noise_percent:
                pass
            else:
                new_position = decoded_action["position"]
                self.change_port(self.current_evtol.evtol_name, new_position)
                self.current_evtol.job_status['final_dest'] = new_position
                self.current_evtol.status_to_set = self.current_evtol.all_states['battery-port']
                self.current_evtol.job_status["initial_loc"] = self.current_evtol.current_location
                self.current_evtol.job_status["final_dest"] = new_position

        elif decoded_action["action"] == "stay":
            if self.current_evtol.status == self.current_evtol.all_states["in-air"]:
                self.client.moveToZAsync(-4, self.current_evtol.evtol_name)
                time.sleep(self.sleep_time)
                self.client.hoverAsync(self.current_evtol.evtol_name).join()

        elif decoded_action["action"] == "continue" or "avoid collision":
            pass

        else:
            new_position = decoded_action["position"]
            self.change_port(self.current_evtol.evtol_name, new_position)
            old_position = self.current_evtol.current_location
            reduce = self.current_evtol.calculate_reduction(old_position,new_position) #Ditto
            self.current_evtol.update_battery(reduce) #Ditto
            self.current_evtol.job_status["initial_loc"] = self.current_evtol.current_location
            self.current_evtol.job_status["final_dest"] = new_position
            self.current_evtol.set_status('in-action','in-port')

        self.update_atc_environment()
        reward = self.calculate_reward(decoded_action['action'])
        self.test_reward += reward
        # print(f"reward: {reward}")
        
        observation = self.get_observation()
        self.environment_time_seconds = (time.time() - self.start_time_seconds)  * self.clock_speed
        self.total_timesteps += 1
        self.test_battery += self.average_battery
        done = self.done
        if self.total_timesteps == (432000 / self.clock_speed) and self.test:  # 43200 seconds in 12 hours, which is a day of operation.
            self.total_delay = 0
            self.test_step_time /= self.total_timesteps
            self.done = done = True
            for i in self.all_evtols:
                i.assign_schedule(self.port) # To get the final delay. 
                self.total_delay += i.upcoming_schedule["total-delay"]
            info = {"cumulative reward":self.test_reward,
                    "number of collisions":self.collisions,
                    "total delay":self.total_delay,
                    "good takeoffs":self.good_takeoffs,
                    "good landings":self.good_landings,
                    "average battery":self.test_battery / self.total_timesteps,
                    "average time per step":self.test_step_time}
        elif self.total_timesteps == (432000 / self.clock_speed):
            self.done = done = True
            info = {"action":action}
        self.step_time = time.time() - start_time
        self.test_step_time += self.step_time

        return observation, reward, done, info
    

    def get_observation(self):
        return self.state_manager.get_obs(self.current_evtol, self.type, self.graph_prop)
    
    def calculate_reward(self, action):
        """
        The reward function is split into five components:
        - Tau: Good takeoff component. A good takeoff is when the eVTOL takesoff within 5 minutes of it's scheduled takeoff time, 
               and the battery state is sufficiently charged or greater.
        - gamma: Good landing component. A good landing is when the eVTOL landgs within 5 minutes of it's scheduled landing time or less,
                 and the battery state is sufficiently charged or greater.
        - lambda_: Battery component. The battery component is positive when the battery is over 30%, and proportional to the battery levels after that.
                   It can be negative when the battery levels are under 30%.
        - beta_: Delay component. The delay is measured as the amount of time past the scheduled takeoff/landing time that the eVTOL has not yet taken off/landed (with a 5 minute buffer).
        
        -safety: Safety component. The algorithm will test for potential collisions at each step, meaning if two eVTOLs are set to collide in a 2D abstraction.
                 If that is the case, and the agent chooses not to avoid a collision, the safety component will be negative. 

        There are five weights, one for each component, which decide the importance of each. They all start out at 1/5, reducing the range of each component to [-1, 1]. 
        From there they can be increased or decreased to account for certain metrics the user will want. 
        """
        evtol = self.current_evtol
        evtol_states = evtol.all_states
        evtol_battery_states = evtol.all_battery_states
        Tau = 0
        gamma = 0
        w1 = w2 = w3 = w4 = w5 = 1 / 5

        if action == "takeoff": # Takeoff
            if evtol.schedule_status == 0 and evtol.battery_state in [evtol_battery_states["sufficient"], evtol_battery_states["full"]]:
                self.good_takeoffs += 1
                Tau = 5
            else:
                Tau = -5
        
        if "land" in action: # Landing
            if evtol.schedule_status in [0, 1] and evtol.battery_state in [evtol_battery_states["sufficient"], evtol_battery_states["full"]]:
                self.good_landings += 1
                gamma = 5
            else:
                gamma = -5

        # #Penalized for bad takeoffs 
        # Tao_bad = -5 + 5 * np.exp(-(evtol.tasks_completed - evtol.good_takeoffs))
        # if Tao_bad > 0: # More good takeoffs than bad
        #     Tao_bad = 5

        #Penalized for bad landings - useful for delayed reward if we implement that
        # gamma_bad = -5 + 5 * np.exp(-(evtol.tasks_completed - evtol.good_takeoffs))

        # Depends on the battery
        if evtol.battery_remaining == evtol_battery_states["critical"]:
            lambda_ = -5
        else:
            lambda_ = (evtol.battery_remaining / 100) * 5
        
        beta = evtol.upcoming_schedule["delay"] # Depends on the delay
        if not beta: # If no delay or if evtol is early.
            beta = 0
        beta_ = -5 + 10 * np.exp(-beta/60)

        safety = self.calculate_safety(action)

        w1 += 1 / 10 # 3rd most important
        w2 += 1 / 10 # 3rd most important
        w3 += 1.5 / 10 # 2nd most important
        w4 -= 1 / 10  # 4th most important
        w5 += 1.5 / 10  # Most important

        formula = w1*gamma + w2*Tau + w3*lambda_ + w4*beta_ + w5*safety

        return formula

    def reset(self):
        print('resetting')
        self.client.reset()
        time.sleep(self.sleep_time)
        self.initialize()
        self.current_evtol = self.all_evtols[0]
        self.done = False

        return self.state_manager.get_obs(self.current_evtol, self.type, self.graph_prop)
    
    def calculate_safety(self, action):
        """
        eVTOL paths are traced as 2D lines. If two paths intersect and both eVTOLs are currently moving, the euclidean distance will be calculated 
        as a function of time, to determine at which time there will be a minimum distance between the two eVTOLs. This time is plugged back into the equation, 
        and minimum distance is compared with a safety threshold (currently 3 meters).
        (TODO) This function seems to assume collisions if both eVTOLs are returning to base, which causes the agent to prefer landing late. A better collision check
        would be to use the client locations at each step and make sure no two eVTOLs are too close to each other. This minimum separation can be different depending on which area 
        the eVTOLs are in.
        """
        if self.current_evtol.status != self.current_evtol.all_states["in-action"]:
            return 0

        threshold_meters = 3  # minimum safe separation distance in meters
        curr_evtol = self.current_evtol
        curr_pos = curr_evtol.current_location
        final_pos  = curr_evtol.job_status["final_dest"]
        curr_segment = Segment2D(tuple(curr_pos[:-1]), tuple(final_pos[:-1]))

        other_evtols = self.all_evtols.copy()
        other_evtols.pop(other_evtols.index(curr_evtol))
        for other_evtol in other_evtols:
            if other_evtol.status == curr_evtol.all_states["in-action"]:
                other_loc = other_evtol.current_location
                other_final_pos = other_evtol.job_status["final_dest"]
                other_segment = Segment2D(tuple(other_loc[:-1]), tuple(other_final_pos[:-1]))
                intersection = curr_segment.intersect(other_segment)
                if intersection:  # Check the distance between each evtol and the intersection point, and calc the times.

                    # intersection = np.array(intersection.args[0].coordinates).astype(np.float32)
                    # curr_segment = np.array(curr_segment.args[0].coordinates).astype(np.float32)
                    # other_segment = np.array(other_segment.args[0].coordinates).astype(np.float32)

                    # inter_norm = np.linalg.norm(intersection)
                    # curr_norm, curr_vnorm = np.linalg.norm(curr_segment), np.linalg.norm(np.array([curr_evtol_v.x_val, curr_evtol_v.y_val]).astype(np.float32))
                    # other_norm, other_vnorm = np.linalg.norm(other_segment), np.linalg.norm(np.array([other_evtol_v.x_val, other_evtol_v.y_val]).astype(np.float32))

                    # # Attempting to solve for time of intersection for both evtols and checking if the times are too close.
                    # # Basically, P2 = P1 + V * T_intersect, and vice versa for the other evtol.

                    # ti_curr, ti_other = (inter_norm - curr_norm) / curr_vnorm, (inter_norm - other_norm) / other_vnorm

                    # if abs(ti_curr - ti_other) <= threshold:
                    #     print(f"evtols will collide, intersection 1: {ti_curr}s, intersection 2: {ti_other}s.")
                    #     return -5
                    # else:
                    #     return 5

                    curr_evtol_v = self.client.getMultirotorState(vehicle_name=curr_evtol.evtol_name).kinematics_estimated.linear_velocity
                    other_evtol_v = self.client.getMultirotorState(vehicle_name=other_evtol.evtol_name).kinematics_estimated.linear_velocity

                    numerator = 2*(curr_evtol_v.x_val - other_evtol_v.x_val)*(curr_pos[0] - other_loc[0]) + \
                                2*(curr_evtol_v.y_val - other_evtol_v.y_val)*(curr_pos[1] - other_loc[1])
                    denominator = 2*(curr_evtol_v.x_val - other_evtol_v.x_val)**2 + 2*(curr_evtol_v.y_val - other_evtol_v.y_val)**2
                    if denominator == 0:
                        return 0
                    t_min_sep = - numerator / denominator

                    x_ = (curr_pos[0] - other_loc[0] + t_min_sep*(curr_evtol_v.x_val - other_evtol_v.x_val))**2
                    y_ = (curr_pos[1] - other_loc[1] + t_min_sep*(curr_evtol_v.y_val - other_evtol_v.y_val))**2
                    min_sep = np.sqrt(x_ + y_)  # Euclid distance
                    # print(f"minimum separation: {min_sep}m")

                    if 0 < min_sep < threshold_meters:
                        if action == "avoid collision":
                            # print(f"evtols will not collide, agent chose correctly: minimum separation is {min_sep}m")
                            self.avoided_collisions += 1
                            return 5
                        else:
                            # print(f"evtols will collide, agent chose incorrectly.")
                            self.collisions += 1
                            return -5
        return 0

    """
    Environment Helper Functions
    """

    def initialize(self):
        """
        Each eVTOL in use is initialized and added to a list for easy management. 
        Ideally any variables that need to reset with the environment are reset here.
        """
        self.start_time_seconds = time.time()                  
        self.environment_time_seconds = (time.time() - self.start_time_seconds) *self.clock_speed
        self.total_timesteps = 0
        self.tasks_completed = 0    # A task is considered as a evtol going to a destination and returning.
        self.good_takeoffs = 0
        self.test_reward = 0
        self.test_step_time = 0
        self.test_land_queue = []
        self.test_takeoff_queue = []
        self.test_battery = 0
        self.good_landings = 0
        self.total_delay = 0
        self.collisions = 0
        self.avoided_collisions = 0
        self.average_battery = 0
        self.all_evtols = list()
        self.port = Vertiport(self.vertiport_config)

        self.graph_prop = {'vertiport_features':{},'vertiport_edge':{},
                            'evtol_features':{},'evtol_edge':{},
                            'next_evtol_embedding':{}, 'mask':{}}
        self.graph_prop['vertiport_edge'] = self.create_edge_connect(num_nodes=self.port.total_port_count)
        self.graph_prop['evtol_edge'] = self.create_edge_connect(num_nodes=self.evtol_count)

        self.evtol_feature_mat = np.zeros((self.evtol_count, 5)) #five features per evtol

        for i in range(self.evtol_count):
            evtol_name = "evtol"+str(i)
            offset = self.evtol_offsets[i]
            self.all_evtols.append(eVTOL(evtol_name, offset, self.evtol_config))

        self.toggle_client_protocols(False)
        time.sleep(self.sleep_time)
        self.toggle_client_protocols(True)

        for i in self.all_evtols:   
            client_position = self.client.getMultirotorState(i.evtol_name).kinematics_estimated.position
            evtol_location = [client_position.x_val, client_position.y_val, client_position.z_val]
            i.evtol_locs[self.all_evtols.index(i)] = evtol_location
            i.current_location = i.previous_location = evtol_location
            self.takeoff(i.evtol_name)

        self.state_manager = ATCStateSpace(self.port)
        if self.test == "FCFS":
            self.action_manager = FCFS_ActionSpace(self.port)
        else:
            self.action_manager = ATCActionSpace(self.port)
        
        self.assign_initial_schedules()

    def assign_initial_schedules(self): 
        """Each eVTOL is assigned to an initial destination, and receives a flight plan consisting of a landing and takeoff time."""
        for evtol in self.all_evtols:
            # choice = random.randint(0,1) #Controls whether a destination or hover port is picked
            choice = 0 #All UAMs start by going to destinations
            evtol.assign_schedule(port=self.port, choice=choice)
            if evtol.job_status['final_dest'] in self.vertiport_config["destinations"]:
                self.test_land_queue.append(evtol)
            else:
                self.test_takeoff_queue.append(evtol) #Added for first-come first-serve
            initial_destination = evtol.job_status['final_dest']
            if initial_destination in self.port.hovering_spots:
                evtol.set_status("in-action", "in-air")
                evtol.port_identification = {'type':'hover','port_no':self.port.hover_spots.index(initial_destination)}
            else:
                evtol.set_status("in-action", "in-destination")
            evtol.job_status['initial_loc'] = evtol.current_location

            self.move_position(evtol.evtol_name, initial_destination, join=0)
        self.update_atc_environment()
            
    """
    eVTOL functions
    """
    def update_atc_environment(self):
        """
        Each eVTOL and port is updated with respect to the simulation environment. Their locations are refreshed and batteries are discharged according to their current status.
        This is where the graph features get updated for both the ports and eVTOLs.

        """
        self.tasks_completed = 0
        self.total_delay = 0
        self.average_battery = 0

        for i in self.all_evtols:    
            client_position = self.client.getMultirotorState(i.evtol_name).kinematics_estimated.position
            evtol_location = [x, y, z] = [client_position.x_val, client_position.y_val, client_position.z_val]
            i.current_location = evtol_location
            i.evtol_locs[self.all_evtols.index(i)] = evtol_location
            self.environment_time_seconds = (time.time() - self.start_time_seconds) * self.clock_speed
            i.update(evtol_location,self.client,self.port,self.environment_time_seconds, self.noise)
            i.get_state_status()
            self.tasks_completed += i.tasks_completed
            self.average_battery += i.battery_remaining
            self.evtol_feature_mat[self.all_evtols.index(i)] = [i.battery_state, i.status, i.schedule_status, x, y] 
            self.total_delay += i.upcoming_schedule["total-delay"]
        self.average_battery /= self.evtol_count
        self.port.update_all()
        self.graph_prop['vertiport_features'] = self.port.feature_mat
        self.graph_prop['evtol_features'] = self.evtol_feature_mat
    
    
    def select_next_evtol(self):
        failed_attempts = 0
        if not self.current_evtol:
            self.current_evtol = self.all_evtols[0]
            new_evtol_index = 0
        else:
            old_evtol_index = self.all_evtols.index(self.current_evtol)
            new_evtol_index = old_evtol_index + 1
            
        if new_evtol_index < self.evtol_count:
            evtol = self.current_evtol = self.all_evtols[new_evtol_index]
            self.update_atc_environment()
            initial_location = evtol.current_location
            collision = self.client.simGetCollisionInfo(self.current_evtol.evtol_name)
            while evtol.status == evtol.all_states["in-destination"] or \
            ((collision.has_collided == True and collision.object_name[:-1] == 'evtol') and (self.calculate_distance(initial_location, evtol.current_location) < 3)):   
                time.sleep(self.sleep_time)
                self.update_atc_environment()
                collision = self.client.simGetCollisionInfo(self.current_evtol.evtol_name)

                failed_attempts += 1
                if failed_attempts > 30: #Failsafe
                    self.client.reset()
                    time.sleep(self.sleep_time)
                    self.toggle_client_protocols(False)
                    time.sleep(self.sleep_time)
                    self.toggle_client_protocols(True)
                    self.port.reset_ports()
                    for i in range(self.evtol_count):
                        self.takeoff(self.all_evtols[i].evtol_name, join=1)
                    self.assign_initial_schedules()
                    return
        else:
            self.current_evtol = self.all_evtols[0]
        self.update_atc_environment()
    
    def get_final_position(self, position, offset):
        return [position[0] + offset[0] , position[1] + offset[1], position[2]]
    
    def calculate_distance(self, point1, point2):
        dist = np.linalg.norm(np.array(point1)-np.array(point2))
        return dist

    """
    Airsim client functions
    """

    def complete_takeoff(self, evtol_name, fly_port, hover=False):
        """Takeoff to a destination using the airsim client."""
        self.client.takeoffAsync(vehicle_name=evtol_name).join()
        self.move_position(evtol_name, fly_port, join=0) if not hover else self.move_position(evtol_name, fly_port, join=1)
        
    def move_position(self, evtol_name, position, join=0):
        """Move to a position using the airsim client."""
        if self.noise and np.random.uniform() < self.noise_percent:
            velocity = 1 - np.random.uniform()
        else:
            velocity = 1

        if join == 0:
            self.client.moveToPositionAsync(position[0],position[1],position[2], velocity=velocity,timeout_sec=15, vehicle_name=evtol_name)
        else:
            self.client.moveToPositionAsync(position[0],position[1],position[2], velocity=velocity,timeout_sec=15, vehicle_name=evtol_name).join()

    def toggle_client_protocols(self, control = False):
        """Enables/disables client api control and arm/disarm capabilities for each eVTOL. Necessary when resetting the environment."""
        for i in range(self.evtol_count):
            self.client.enableApiControl(control, self.all_evtols[i].evtol_name)
        for i in range(self.evtol_count):
             self.client.armDisarm(control, self.all_evtols[i].evtol_name)

    def change_port(self, evtol_name, new_port):
        """Change from one port to another with the airsim client."""
        self.client.takeoffAsync(vehicle_name=evtol_name).join()
        self.client.moveToPositionAsync(new_port[0], new_port[1], new_port[2], velocity=1, vehicle_name=evtol_name).join()
        self.client.landAsync(vehicle_name=evtol_name).join()
    
    def takeoff(self, evtol_name, join=0):
        """Asynchronous or synchronous takeoff with the airsim client."""
        if join == 0:
            self.client.takeoffAsync(vehicle_name=evtol_name)
        else:
            self.client.takeoffAsync(vehicle_name=evtol_name).join()
    
    def complete_landing(self, evtol_name, location):     
        """Land the eVTOL using the airsim client."""   
        self.move_position(evtol_name, location,join=1)        
        self.client.landAsync(vehicle_name=evtol_name).join()


    """
    Graph Learning Function(s)
    """
    def create_edge_connect(self, num_nodes=5, adj_mat=None, direct_connect=0):
        """Returns the undirected edge connectivity matrix (ECM). Additionally, the adjacency matrix can be supplied for a directly connected ECM"""
        if direct_connect == 0: #undirected graph, every node is connected and shares info with each other
            k = num_nodes - 1
            num_edges = k*num_nodes
            blank_edge = np.ones( (2, num_edges) )
            top_index = 0
            bot_index = np.arange(num_nodes)
            index = 1
            for i in range(num_nodes):
                blank_edge[0][k*i:k*index] = top_index
                blank_edge[1][k*i:k*index] = np.delete(bot_index, top_index)
                index+=1
                top_index+=1
        elif direct_connect == 1: #directed graph, in which case we need the adjacency matrix to create the edge list tensor
            blank_edge = np.array([]).reshape(2, 0)
            for i in range(adj_mat.shape[0]):
                for j in range(adj_mat.shape[1]):
                    if adj_mat[i,j] == 1:
                        blank_edge = np.concatenate((blank_edge, np.array([i, j]).reshape(2, 1)), axis=1, dtype=np.float32)
        # return tensor(blank_edge,dtype=long) #gym.spaces don't want tensors, they tensorfy the output anyway
        return blank_edge

    """
    Miscellaneous
    """

    def _seed(self, seed=None, cuda=False):
        """Seeds the environment using stable baselines set_random_seed method."""
        if isinstance(seed, int) and seed >= 0:
            set_random_seed(seed=seed, using_cuda=cuda)
        else:
            raise TypeError(f"You must use a positive integer value for the seed, not {seed}")
