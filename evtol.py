# -*- coding: utf-8 -*-
import time
import random 

import numpy as np


class eVTOL:
    def __init__(self, evtol_name, offset, config):
        self.noise_percent = config["noise_percent"]
        self.clock_speed = config["clock_speed"]
        self.evtol_locs = config["eVTOL_offsets"]


        self.evtol_name = evtol_name
        self.evtol_no = evtol_name.split("l")[1]
        self.velocity_mps = 1
        self.all_battery_states = {'critical':0,'sufficient':1,'full':2}
        self.battery_state = self.all_battery_states["full"]
        self.battery_remaining = 100
        self.distance_traveled = 0
        self.all_states = {"in-air":0, "in-port":1, "battery-port":2, "in-action":3, "in-destination":4}
        self.job_status = {"initial_loc":None, "final_dest":None, "current_pos": None}
        self.status = 1
        self.status_to_set = 1
        self.offset = offset
        self.current_location = []
        self.previous_location = []
        self.in_portzone = False
        self.port_center_loc =[0,0,-4] #Filler
        self.dist_threshold_meters = 10
        self.current_location = None
        self.in_battery_port = 0
        self.port_identification = None
        self.upcoming_schedule = {"landing-time": 0, "takeoff-time":0, 'delay':None, 'total-delay':0, 'time':0, 'end-port':None}
        self.env_time_seconds = 0
        self.schedule_status = 0 
        self.tasks_completed = 0
        self.good_takeoffs = 0
        self.sleep_time = 0.5 / self.clock_speed

    def get_status(self):
        if self.status == self.all_states['in-air']:
            status = 0
        else:
            status = 1
        return status
    
    def set_status(self,status, final_status):
        self.status = self.all_states[status]
        self.status_to_set = self.all_states[final_status]

    def get_battery_state(self):
        return self.battery_state 
    
    def calculate_reduction(self, old_position, new_position): 
        time_travelled = np.linalg.norm(np.array(old_position)-np.array(new_position)) / self.velocity_mps
        discharge_rate = 0.50

        if time_travelled < 1 and self.status == self.all_states["in-air"]:
            return 2
        if time_travelled < 1 and self.status in [self.all_states["in-port"], self.all_states["battery-port"]]:
            return 4

        return discharge_rate * time_travelled

    def update_battery(self, reduce):
        self.battery_remaining -= reduce
        if self.battery_remaining < 0:
            self.battery_remaining = 0
        if self.battery_remaining == 100:
            self.battery_state = self.all_battery_states['full']

        elif 30 <= self.battery_remaining <= 100: 
            self.battery_state = self.all_battery_states['sufficient'] 

        elif 0 <= self.battery_remaining <= 30:
            self.battery_state = self.all_battery_states['critical'] 
    
    def check_zone(self):
        dist = self.calculate_distance(self.current_location)
        if dist < self.dist_threshold_meters:
            self.in_portzone = True
        else:
            self.in_portzone = False
    
    def update_evtol(self, current_loc, client, port, env_time, noise=False):
        if noise and np.random.uniform() < self.noise_percent:
            velocity = 1 - np.random.uniform()
        else:
            velocity = 1
        self.upcoming_schedule["time"] = env_time
        self.env_time_seconds = env_time
        # print(self.env_time)
        reduction = self.calculate_reduction(self.previous_location, self.current_location)
        self.update_battery(reduction)
        self.previous_location = self.current_location

        if self.status == self.all_states['in-action']: 
            if self.status_to_set == self.all_states['in-destination']: 
                distance_meters = self.calculate_distance(current_loc[:-1],self.job_status['final_dest'][:-1])
                if distance_meters < 0.75: #evtol reached destination and is ready for the next task
                    self.set_status('in-destination','in-action')
                    self.tasks_completed += 1
                else:
                    final_pos = self.job_status['final_dest']
                    client.moveToPositionAsync(final_pos[0],final_pos[1],final_pos[2], velocity=velocity, vehicle_name=self.evtol_name)

            elif self.status_to_set == self.all_states['battery-port']:
                distance_meters = self.calculate_distance(current_loc[:-1],self.job_status['final_dest'][:-1])
                if distance_meters < 0.75: #evtol reached the battery port and is ready to charge
                    client.landAsync(vehicle_name = self.evtol_name)
                    self.in_battery_port = 1
                    self.set_status('battery-port','in-action')
                    if noise and np.random.uniform() < self.noise_percent:
                        self.battery_remaining += 0
                    else:
                        self.battery_remaining += 10
                else:
                    final_pos = self.job_status['final_dest']
                    client.moveToPositionAsync(final_pos[0],final_pos[1],final_pos[2], velocity=velocity, vehicle_name=self.evtol_name)
                    time.sleep(self.sleep_time)

            elif self.status_to_set == self.all_states['in-air']:
                distance_meters = self.calculate_distance(current_loc[:-1],self.job_status['final_dest'][:-1])
                if distance_meters < 0.75: #evtol reached the hover spot and is ready for the next task
                    client.hoverAsync(vehicle_name = self.evtol_name)
                    self.set_status('in-air','in-action')
                else:
                    final_pos = self.job_status['final_dest']
                    client.moveToPositionAsync(final_pos[0],final_pos[1],final_pos[2], velocity=velocity, vehicle_name=self.evtol_name)
                    time.sleep(self.sleep_time)
            elif self.status_to_set == self.all_states['in-port']:
                distance_meters = self.calculate_distance(current_loc[:-1],self.job_status['final_dest'][:-1])
                if distance_meters < 0.75: #evtol reached destination and is ready for the next task
                    self.set_status('in-port','in-action')
                else:
                    final_pos = self.job_status['final_dest']
                    client.moveToPositionAsync(final_pos[0],final_pos[1],final_pos[2], velocity=velocity, vehicle_name=self.evtol_name)
                    time.sleep(self.sleep_time)

        elif self.status == self.all_states['battery-port']:
            if self.battery_remaining >= 100:
                self.battery_remaining = 100
                self.set_status('battery-port','in-action')
            else:
                if noise and np.random.uniform() < 0.2:
                    self.battery_remaining += 0
                else:
                    self.battery_remaining += 10

        elif self.status == self.all_states['in-port']:

            self.set_status('in-port','in-action')

        elif self.status == self.all_states['in-air']:
            pass

        elif self.status == self.all_states['in-destination']:
            self.assign_schedule(port, choice=1) #Assigning a hover port
            self.port_identification = {'type':'hover','port_no':port.hovering_spots.index(self.job_status['final_dest'])}
            des = self.job_status['final_dest']
            final_pos = self.get_offset_position(des, self.offset)
            client.moveToPositionAsync(final_pos[0],final_pos[1],final_pos[2], velocity=velocity, vehicle_name=self.evtol_name)
            time.sleep(self.sleep_time)
            self.set_status('in-action','in-air')
        
    def calculate_distance(self,cur_location, dest):
        return np.linalg.norm(np.array(dest)-np.array(cur_location))      
        
    def get_offset_position(self, port, offset):
        return [port[0] - offset[0] , port[1] - offset[1], port[2]]
    
    def assign_schedule(self, port, choice = 0):
        """
        The eVTOL is given a destination from the list of destination ports. The eVTOL is also given a random landing and random takeoff time 
        depending on which state it's currently in. 
        """
        self.job_status['final_dest'] = port.get_destination(choice)
        random_landing_seconds = random.randint(1,3) * self.clock_speed 
        random_takeoff_seconds = random.randint(2,4) * self.clock_speed
        self.upcoming_schedule["landing-time"] = random_landing_seconds + self.env_time_seconds
        self.upcoming_schedule["takeoff-time"] = random_takeoff_seconds + self.env_time_seconds

        if self.upcoming_schedule["delay"]:
            self.upcoming_schedule["total-delay"] += self.upcoming_schedule["delay"]
        self.upcoming_schedule["delay"] = None
        self.upcoming_schedule["end-port"] = self.job_status['final_dest']

    def get_schedule_status(self):
        """
        If the eVTOL is in the air or moving to a new location, the landing time will be taken into account for the schedule status. Vice versa for the takeoff time.
        If the eVTOL hasn't taken off/landed and is within 5 minutes of it's predetermined takeoff/landing time, then it is on time (schedule status=1). 
        Any earlier than that is considered early (schedule status=0) and any later is considered late (schedule status=1)

        """
        threshold_seconds = 300 
        if self.status == self.all_states['in-air'] or self.status == self.all_states['in-action']:
            if (self.upcoming_schedule["landing-time"] - threshold_seconds) <= self.upcoming_schedule['time'] <= (self.upcoming_schedule["landing-time"] + threshold_seconds):
                self.schedule_status = 0
                return 0
            elif (self.upcoming_schedule['time']) <= (self.upcoming_schedule["landing-time"] - threshold_seconds):
                self.schedule_status = 1
                return 1
            elif (self.upcoming_schedule["landing-time"] + threshold_seconds) <= (self.upcoming_schedule['time']): # Late
                self.schedule_status = 2
                self.upcoming_schedule["delay"] = self.upcoming_schedule["time"] - self.upcoming_schedule["landing-time"] 
                return 2

        elif self.status == self.all_states['in-port'] or self.status == self.all_states['battery-port']:
            if (self.upcoming_schedule["takeoff-time"] - threshold_seconds <= self.upcoming_schedule['time'] <= self.upcoming_schedule["takeoff-time"] + threshold_seconds):
                self.schedule_status = 0
                return 0
            elif (self.upcoming_schedule['time'] <= self.upcoming_schedule["takeoff-time"] - threshold_seconds):
                self.schedule_status = 1
                return 1
            elif (self.upcoming_schedule["takeoff-time"] + threshold_seconds <= self.upcoming_schedule['time']):
                self.schedule_status = 2
                self.upcoming_schedule["delay"] = self.upcoming_schedule["time"] - self.upcoming_schedule["takeoff-time"]
                return 2
        else:
            self.schedule_status = 0
            return 0
