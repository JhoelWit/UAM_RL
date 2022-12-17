# -*- coding: utf-8 -*-
import numpy as np

class ATCStateSpace:
    def __init__(self, ports):
        self.ports = ports
        
    
    def get_obs(self, evtol, type, graph_prop = None):
        """
        
        
        Battery capacity of current vehicle(0,1)
        Empty ports (0,1,2)
            3.1 Not available – 0
            3.2 availability limited – 1
            3.3 Sufficient ports available - 2
        empty hovering spots(0,1,2)
        (same choices as empty ports)
        Battery ports(0,1,2)
        (same choices as empty ports)
        Status of the vehicle(0,1,2)
          6.1 in-air– 0
          6.2 in-port– 1
          6.3 in-battery port- 2
        On-time to takeoff/land (0,1)
       # not implemented -  Collision possibility (0,1,2) => 0 – safe; 2 – highly unsafe


        Returns
        -------
        None.

        """
        def generate_mask():
            """Returns a mask to use for the GRL and RL policy models.
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
            avoid collision - 10"""

            if evtol.status == evtol.all_states["in-action"]:
                mask = [1] * 11
                mask[-1] = mask[-2] = 0
                return np.array(mask)

            mask = [0] * 11
            for row_idx, row in enumerate(self.ports.feature_mat):  # Looks at availability of each port and masks if the port is not available
                mask[row_idx + 2] = 1 if row[0] == 0 else 0
            mask[-1] = mask[-2] = 1
            # print("mask", mask)
            return np.array(mask)
            
        evtol_locs = evtol.evtol_locs 
        battery_capacity = evtol.get_battery_state()
        empty_port = self.ports.get_availability_ports(evtol_locs)                       
        empty_hovering_spots = self.ports.get_availability_hover_spots(evtol_locs)        
        empty_battery_ports = self.ports.get_availability_battery_ports(evtol_locs)       
        status = evtol.get_status()
        schedule = evtol.get_state_status() 
        next_embedding = np.array([battery_capacity,
                            empty_port,
                            empty_hovering_spots,
                            empty_battery_ports,
                            status,
                            schedule, 
                            evtol.current_location[0], 
                            evtol.current_location[1]])
        states = np.hstack((graph_prop["vertiport_features"].flatten(), graph_prop["evtol_features"].flatten(), next_embedding))
        if type == "regular":
            return dict(next_evtol_embedding = states, mask = generate_mask())
            
        elif type == "graph_learning":
            graph_prop['next_evtol_embedding'] = next_embedding
            graph_prop["mask"] = generate_mask()
            return graph_prop