# -*- coding: utf-8 -*-


class ATCActionSpace:
    """
    This class should decode the actions and send the coordinate to the Main environment
    """
    def __init__(self, port):
        self.port = port
        self.actions = {0 : "stay still",
                        1 : "takeoff",
                        2 : "normal-1",
                        3 : "normal-2",
                        4 : "battery-1",
                        5 : "hover-1",
                        6 : "hover-2",
                        7 : "hover-3",
                        8 : "hover-4",
                        9 : "continue",
                        10 : "avoid collision"}

    
    def action_decode(self, drone, action):
        """
        action count:

            staystill - 0
            takeoff - 1
            move to normal port 1 - 2
            move to normal port 2 - 3
            move to battery port 1 - 4
            move to hover spot 1 - 5
            move to hover spot 2 - 6
            move to hover spot 3 - 7
            move to hover spot 4 - 8
            continue moving - 9
            avoid collision - 10
            

            
    
        First check the status then assign the port/hoveringspot/takeoff/land .... etc
        Update: We will use masking so no need to check the status
        Updateupdate: if using random walk might need to add extra conditionals to the if-elif statements

        Parameters
        ----------
        drone : drone object
            DESCRIPTION.
        action : just a scalar value [0-4]
            DESCRIPTION.


        Returns
        -------
        None.

        """
        action = self.actions[action]   # Converting to a string for readability. 
        status = drone.status
        # print(f"action {action}")

        if action == "continue":
            return {'action':'continue'}

        if action == "stay still":
            return {"action" : "stay", "position" : None}
        
        elif action == "takeoff":
            # dest_num = int(action[-1]) - 1
            # dest = self.port.get_destination(choice=0)
            dest = drone.upcoming_schedule["end-port"]
            final_pos = self.get_final_pos(dest, drone.offset)
            self.port.update_port(drone.port_identification)

            return {"position" : final_pos, "action": "takeoff"}

        elif action in ["normal-1", "normal-2"]:
            norm_num = int(action[-1]) - 1
            if self.port.get_port_status(norm_num, 'normal') is False:                    
                final_pos = self.get_final_pos(self.port.port_status[norm_num]["position"], drone.offset)

                drone.in_battery_port = 0
                self.port.update_port(drone.port_identification)
                drone.port_identification = {'type':'normal','port_no':norm_num}

                #self.port.update_port(drone.port_identification)
                self.port.change_status_normal_port(norm_num, True)

                return {"position" : final_pos, "action": "land in normal port"}

        elif action == "battery-1":
            if self.port.get_port_status(0, 'battery') is False:
                final_pos = self.get_final_pos(self.port.battery_port_status[0]["position"], drone.offset)
                drone.in_battery_port = 1
                self.port.update_port(drone.port_identification)
                drone.port_identification = {'type':'battery', 'port_no':0}

                self.port.change_status_battery_port(0, True)
                #self.port.update_port(drone.port_identification)

                return {"position" : final_pos, "action": "land in battery port"}        
        
        elif action in ["hover-1", "hover-2", "hover-3", "hover-4"]:
            hover_num = int(action[-1]) - 1
            if not self.port.get_port_status(hover_num, "hover"):
                final_pos = self.get_final_pos(self.port.hover_spot_status[hover_num]["position"], drone.offset)
                drone.in_battery_port = 0
                self.port.update_port(drone.port_identification)
                drone.port_identification = {'type':'hover','port_no':hover_num}
                self.port.change_hover_spot_status(hover_num, True)
                if status in [drone.all_states["in-air"], drone.all_states["in-action"]]:
                    return {"position" : final_pos, "action": "hover"}
                else:
                    return {"position" : final_pos, "action": "takeoff to a hovering spot"}
        
        elif action == "avoid collision":
            return {"action" : "avoid collision", "position" : drone.job_status["final_dest"]}
    
        return {"action" : "continue"}  # Mainly for the random walk, since that can't use masking. 

    def get_final_pos(self,port, offset):
        return [port[0] + offset[0] , port[1] + offset[1], port[2]]
        
    
    
"""
All actions for reference
LAnding
1.1 stay still - 0
1.2 land in empty port - 1
1.3 land in battery port - 2
1.4 move to empty hovering spot - 3

Takeoff
2.1 stay still - 0
2.2 takeoff - 1
2.3 move to battery port - 2
2.4 move from battery port - 3


"""

class FCFS_ActionSpace:
    """
    This class should decode the actions and send the coordinate to the Main environment
    """
    def __init__(self, port):
        self.port = port
        self.actions = {0 : "stay still",
                        1 : "takeoff",
                        2 : "normal-1",
                        3 : "normal-2",
                        4 : "battery-1",
                        5 : "hover-1",
                        6 : "hover-2",
                        7 : "hover-3",
                        8 : "hover-4",
                        9 : "continue",
                        10 : "avoid collision"}

    
    def action_decode(self,drone, takeoff_queue, land_queue):
        """
        action count:

            staystill - 0
            takeoff - 1
            move to normal port 1 - 2
            move to normal port 2 - 3
            move to battery port 1 - 4
            move to hover spot 1 - 5
            move to hover spot 2 - 6
            move to hover spot 3 - 7
            move to hover spot 4 - 8
            continue moving - 9
            avoid collision - 10
            

            
    
        First check the status then assign the port/hoveringspot/takeoff/land .... etc
        Update: We will use masking so no need to check the status
        Updateupdate: if using random walk might need to add extra conditionals to the if-elif statements

        Parameters
        ----------
        drone : drone object
            DESCRIPTION.
        action : just a scalar value [0-4]
            DESCRIPTION.


        Returns
        -------
        None.

        """
        action = None
        status = drone.status
        if drone in takeoff_queue:
            if drone == takeoff_queue[0] and drone.battery_remaining >= 60:
                action = "takeoff"
                takeoff_queue.pop(0)
                if drone not in land_queue:
                    land_queue.append(drone)
                dest = drone.upcoming_schedule["end-port"]
                final_pos = self.get_final_pos(dest, drone.offset)
                self.port.update_port(drone.port_identification)
                return {"position" : final_pos, "action": "takeoff"}, takeoff_queue, land_queue
            elif drone == takeoff_queue[0] and drone.battery_remaining <= 60:
                drone.status = 2
                # drone.battery_remaining += 10

        if drone in land_queue:
            if drone == land_queue[0] and drone.status in [0, 1]:
                if self.port.get_port_status(0, 'battery') is False:
                    final_pos = self.get_final_pos(self.port.battery_port_status[0]["position"], drone.offset)
                    drone.in_battery_port = 1
                    self.port.change_status_battery_port(0, True)
                    self.port.update_port(drone.port_identification)
                    drone.port_identification = {'type':'battery', 'port_no':0}
                    land_queue.pop(0)
                    if drone not in takeoff_queue:
                        takeoff_queue.append(drone)
                    if drone.status == 0:
                        return {"position" : final_pos, "action": "land in battery port"}, takeoff_queue, land_queue
                    else:
                        return {"position" : final_pos, "action": "move to battery port"}, takeoff_queue, land_queue
                elif drone.status == 0:
                    empty_port = self.port.get_empty_port()   
                    if empty_port:        
                        final_pos = self.get_final_pos(empty_port["position"], drone.offset)
                        drone.in_battery_port = 0
                        self.port.update_port(drone.port_identification)
                        drone.port_identification = {'type':'normal','port_no':empty_port["port_no"]}
                        #self.port.update_port(drone.port_identification)
                        self.port.change_status_normal_port(empty_port["port_no"], True)
                        if drone not in takeoff_queue:
                            takeoff_queue.append(drone)
                        return {"position" : final_pos, "action": "land in normal port"}, takeoff_queue, land_queue

        if not action:
            return {"action" : "continue"}, takeoff_queue, land_queue
        
    def get_final_pos(self,port, offset):
        return [port[0] + offset[0] , port[1] + offset[1], port[2]]