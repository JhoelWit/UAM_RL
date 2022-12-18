# -*- coding: utf-8 -*-

def get_offset_position(port, offset):
    """Each eVTOL has an offset due to their simulated initial position, which is applied to each waypoint they travel to."""
    return [port[0] + offset[0] , port[1] + offset[1], port[2]]


class ATCActionSpace:
    def __init__(self, port):
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
        """
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

    
    def decode_action_atc(self, evtol, action):
        """
        The ATC action decoder will first make any changes to port availability and eVTOL status, and then return the decoded action to the ATC environment.
        """
        action = self.actions[action]   # Converting to a string for readability. 
        status = evtol.status
        # print(f"action {action}")

        if action == "continue":
            return {'action':'continue'}

        if action == "stay still":
            return {"action" : "stay", "position" : None}
        
        elif action == "takeoff":
            # dest_num = int(action[-1]) - 1
            # dest = self.port.get_destination(choice=0)
            dest = evtol.upcoming_schedule["end-port"]
            final_pos = get_offset_position(dest, evtol.offset)
            self.port.update_port(evtol.port_identification)

            return {"position" : final_pos, "action": "takeoff"}

        elif action in ["normal-1", "normal-2"]:
            port_index = int(action[-1]) - 1
            if self.port.get_port_status(port_index, 'normal') is False:                    
                final_pos = get_offset_position(self.port.port_status[port_index]["position"], evtol.offset)

                evtol.in_battery_port = 0
                self.port.update_port(evtol.port_identification)
                evtol.port_identification = {'type':'normal','port_no':port_index}

                #self.port.update_port(evtol.port_identification)
                self.port.change_status_normal_port(port_index, True)

                return {"position" : final_pos, "action": "land in normal port"}

        elif action == "battery-1":
            if self.port.get_port_status(0, 'battery') is False:
                final_pos = get_offset_position(self.port.battery_port_status[0]["position"], evtol.offset)
                evtol.in_battery_port = 1
                self.port.update_port(evtol.port_identification)
                evtol.port_identification = {'type':'battery', 'port_no':0}

                self.port.change_status_battery_port(0, True)
                #self.port.update_port(evtol.port_identification)

                return {"position" : final_pos, "action": "land in battery port"}        
        
        elif action in ["hover-1", "hover-2", "hover-3", "hover-4"]:
            hover_num = int(action[-1]) - 1
            if not self.port.get_port_status(hover_num, "hover"):
                final_pos = get_offset_position(self.port.hover_spot_status[hover_num]["position"], evtol.offset)
                evtol.in_battery_port = 0
                self.port.update_port(evtol.port_identification)
                evtol.port_identification = {'type':'hover','port_no':hover_num}
                self.port.change_hover_spot_status(hover_num, True)
                if status in [evtol.all_states["in-air"], evtol.all_states["in-action"]]:
                    return {"position" : final_pos, "action": "hover"}
                else:
                    return {"position" : final_pos, "action": "takeoff to a hovering spot"}
        
        elif action == "avoid collision":
            return {"action" : "avoid collision", "position" : evtol.job_status["final_dest"]}
    
        return {"action" : "continue"}  # Mainly for the random walk, since that can't use masking. 

    def decode_action_fcfs(self,evtol, takeoff_queue, land_queue):
        """
        The first come first serve (FCFS) action decoder will carry out actions in order of que number. eVTOLs at the front of the que will land/takeoff first, and once landed
        each eVTOL will charge to 60% before taking off again. eVTOLs in the back of the que will be skipped until it is their turn to land/takeoff. This decoder simulates 
        real life FCFS methodologies for urban air mobility.
        """
        action = None
        status = evtol.status
        if evtol in takeoff_queue:
            if evtol == takeoff_queue[0] and evtol.battery_remaining >= 60:
                action = "takeoff"
                takeoff_queue.pop(0)
                if evtol not in land_queue:
                    land_queue.append(evtol)
                dest = evtol.upcoming_schedule["end-port"]
                final_pos = get_offset_position(dest, evtol.offset)
                self.port.update_port(evtol.port_identification)
                return {"position" : final_pos, "action": "takeoff"}, takeoff_queue, land_queue
            elif evtol == takeoff_queue[0] and evtol.battery_remaining <= 60:
                evtol.status = 2
                # evtol.battery_remaining += 10

        if evtol in land_queue:
            if evtol == land_queue[0] and evtol.status in [0, 1]:
                if self.port.get_port_status(0, 'battery') is False:
                    final_pos = get_offset_position(self.port.battery_port_status[0]["position"], evtol.offset)
                    evtol.in_battery_port = 1
                    self.port.change_status_battery_port(0, True)
                    self.port.update_port(evtol.port_identification)
                    evtol.port_identification = {'type':'battery', 'port_no':0}
                    land_queue.pop(0)
                    if evtol not in takeoff_queue:
                        takeoff_queue.append(evtol)
                    if evtol.status == 0:
                        return {"position" : final_pos, "action": "land in battery port"}, takeoff_queue, land_queue
                    else:
                        return {"position" : final_pos, "action": "move to battery port"}, takeoff_queue, land_queue
                elif evtol.status == 0:
                    empty_port = self.port.get_empty_port()   
                    if empty_port:        
                        final_pos = get_offset_position(empty_port["position"], evtol.offset)
                        evtol.in_battery_port = 0
                        self.port.update_port(evtol.port_identification)
                        evtol.port_identification = {'type':'normal','port_no':empty_port["port_no"]}
                        #self.port.update_port(evtol.port_identification)
                        self.port.change_status_normal_port(empty_port["port_no"], True)
                        if evtol not in takeoff_queue:
                            takeoff_queue.append(evtol)
                        return {"position" : final_pos, "action": "land in normal port"}, takeoff_queue, land_queue

        if not action:
            return {"action" : "continue"}, takeoff_queue, land_queue