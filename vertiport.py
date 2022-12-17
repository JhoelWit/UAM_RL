# -*- coding: utf-8 -*-
import time
import random 

import numpy as np


class Vertiport():
    def __init__(self, config):
        self.normal_ports = config["normal_ports"]
        self.battery_ports = config["battery_ports"]
        self.destinations = config["destinations"]
        self.hovering_spots = config["hovering_spots"]

        self.num_ports = len(self.normal_ports)
        self.num_battery_ports = len(self.battery_ports)
        self.num_hovering_spots = len(self.hovering_spots)
        self.port_status = {}
        self.port_center_loc =[0,0,-4] #Filler
        self.distance_threshold_meters = 10
        self.reset_ports()

    def reset_ports(self):
        for i in range(self.num_ports):
            self.port_status[i] = {"port_no":i, "position":self.normal_ports[i], "occupied":False, "type":0}
            
        self.battery_port_status = {}
        for i in range(self.num_battery_ports):
            self.battery_port_status[i] = {"port_no":i, "position":self.battery_ports[i], "occupied":False, "type":1}
        
        self.hover_spot_status = {}
        for i in range(self.num_hovering_spots):
            self.hover_spot_status[i] = {"port_no":i,"position":self.hovering_spots[i], "occupied":False, "type":2}

        self.total_port_count = self.num_ports + self.num_battery_ports + self.num_hovering_spots
        self.feature_mat = np.zeros((self.total_port_count, 4)) #four features per port

    def update_port(self, port):
        """Updates the port/spot status as either available or unavailable."""
        if port:
            if port['type'] == 'normal':
                self.change_status_normal_port(port['port_no'], False)
            elif port['type'] == 'battery':
                self.change_status_battery_port(port['port_no'], False)
            elif port['type'] == 'hover':
                self.change_hover_spot_status(port['port_no'], False)

    def update_all(self):
        """This function will iterate through all ports, battery ports, and hover spots and 
        update the vertiport feature matrix accordingly
        """
        for i in range(self.num_ports):
            if self.port_status[i]['occupied'] == True:
                availability = 0
            else:
                availability = 1
            node_type = 0
            self.feature_mat[i] = [availability, node_type, self.port_status[i]["position"][0], self.port_status[i]["position"][1]]
        for i in range(self.num_battery_ports):
            if self.battery_port_status[i]['occupied'] == True:
                availability = 0
            else:
                availability = 1
            node_type = 1
            self.feature_mat[i+self.num_ports] = [availability, node_type, self.battery_port_status[i]["position"][0], self.battery_port_status[i]["position"][1]]
        for i in range(self.num_hovering_spots):
            if self.hover_spot_status[i]['occupied'] == True:
                availability = 0
            else:
                availability = 1
            node_type = 2
            self.feature_mat[i+self.num_ports+self.num_battery_ports] = [availability, node_type, self.hover_spot_status[i]["position"][0], self.hover_spot_status[i]["position"][1]]
                       
    def get_empty_port(self):
        for i in range(self.num_ports):
            if self.port_status[i]["occupied"] == False:
                self.change_status_normal_port(self.port_status[i]['port_no'],True)
                return self.port_status[i]
        return None
    
    def get_empty_battery_port(self):
        for i in range(self.num_battery_ports):
            if self.battery_port_status[i]["occupied"] == False:
                self.change_status_battery_port(self.battery_port_status[i]['port_no'],True)
                return self.battery_port_status[i]
        return None
    
    def get_empty_hover_status(self):
        for i in range(self.num_hovering_spots):
            if self.hover_spot_status[i]["occupied"] == False:
                self.change_hover_spot_status(self.hover_spot_status[i]['port_no'],True)
                return self.hover_spot_status[i]
        return None

    def get_destination(self, choice = 0, number = None):
        if number:
            return self.destinations[number]
        if choice == 0:
            return random.choice(self.destinations)
        else:
            empty_port = self.get_empty_hover_status()
            return empty_port['position']

    def change_status_normal_port(self, port_no, occupied):
        self.port_status[port_no]["occupied"] = occupied
        
    def change_status_battery_port(self, port_no, occupied):
        self.battery_port_status[port_no]["occupied"] = occupied
    
    def change_hover_spot_status(self, port_no, occupied):
        self.hover_spot_status[port_no]["occupied"] = occupied
            
    def get_count_empty_port(self):
        cnt = 0
        for i in range(self.num_ports):
            if self.port_status[i]["occupied"] == False:
                cnt+=1
        return cnt
    
    def get_count_empty_battery_port(self):
        cnt = 0
        for i in range(self.num_battery_ports):
            if self.port_status[i]["occupied"] == False:
                cnt+=1
        return cnt
    
    def get_count_empty_hover_Spot(self):
        cnt = 0
        for i in range(self.num_hovering_spots):
            if self.hover_spot_status[i]["occupied"] == False:
                cnt+=1

        return cnt   
    
    def get_availability_ports(self,drone_locs):
        empty_ports = self.get_count_empty_port()
        uams_inside = self.count_uavs_inside(drone_locs)
        percent = empty_ports/uams_inside
        if uams_inside > 0:
            percent = empty_ports/uams_inside
            if percent > 0.8:
                return 2
            elif percent> 0.5:
                return 1
            else:
                return 0
        else:
            return 2


    def get_availability_battery_ports(self,drone_locs):
        empty_ports = self.get_count_empty_battery_port()
        uams_inside = self.count_uavs_inside(drone_locs)
        if uams_inside > 0:
            percent = empty_ports/uams_inside
            if percent > 0.8:
                return 2
            elif percent> 0.5:
                return 1
            else:
                return 0
        else:
            return 2
        
    def get_availability_hover_spots(self,drone_locs):
        empty_ports = self.get_count_empty_hover_Spot()
        uams_inside = self.count_uavs_inside(drone_locs)
        percent = empty_ports/uams_inside
        if uams_inside > 0:
            percent = empty_ports/uams_inside
            if percent > 0.8:
                return 2
            elif percent> 0.5:
                return 1
            else:
                return 0
        else:
            return 2
    
    def get_port_status(self, port, port_type): #Changed from port_status to avoid key errors
        if port_type == 'normal':
                 return self.port_status[port]["occupied"]
        elif port_type == 'battery':
                return self.battery_port_status[port]["occupied"]
        elif port_type == 'hover':
                return self.hover_spot_status[port]["occupied"]
    
    def count_uavs_inside(self,drone_locs):
        UAVs_inside = 0
        for i in range(len(drone_locs)):
            dist= self._calculate_distance(drone_locs[i])
            if dist<self.distance_threshold_meters: #Switched from > to <
                UAVs_inside +=1
        return UAVs_inside
    
    def _calculate_distance(self,cur_location):

        return np.linalg.norm(np.array(self.port_center_loc)-np.array(cur_location)) #math.dist starts at python3.8, I'm using 3.7 lol
    
    def get_all_port_statuses(self):
        
        return [self.port_status , self.battery_port_status, self.hover_spot_status]
