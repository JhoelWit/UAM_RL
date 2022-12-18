# -*- coding: utf-8 -*-
import numpy as np

class ATCStateSpace:
    def __init__(self, ports):
        self.ports = ports
        
    
    def get_obs(self, evtol, type, graph_prop = None):
        """
        The observation space exists for two types of agents. One is a baseline agent with no graph neural networks. Another is the graph learning agent.
        
        The graph learning observation space consists of:
        - Vertiport graph features (availability, port type, port 2D location)
        - eVTOL graph features (battery status, schedule status, 2D location, eVTOL status)
        - Next eVTOL embedding (availability of each port, and next drone features)

        The baseline observation is the same as the graph learning, however all graph features are flattened.
        """
        def generate_mask():
            """
            Returns a mask to use for the GRL and RL policy models. Unavailable actions to take (such as avoiding collisions while grounded) will be masked at each step.
            """

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
        empty_port = self.ports.get_availability_of_ports(evtol_locs)                       
        empty_hovering_spots = self.ports.get_availability_hover_spots(evtol_locs)        
        empty_battery_ports = self.ports.get_availability_battery_ports(evtol_locs)       
        status = evtol.get_status()
        schedule = evtol.get_schedule_status() 
        next_embedding = np.array([battery_capacity,
                            empty_port,
                            empty_hovering_spots,
                            empty_battery_ports,
                            status,
                            schedule, 
                            evtol.current_location[0], 
                            evtol.current_location[1]])
        states = np.hstack((graph_prop["vertiport_features"].flatten(), graph_prop["evtol_features"].flatten(), next_embedding))
        if type == "baseline":
            return dict(next_evtol_embedding=states, mask=generate_mask())
            
        elif type == "graph_learning":
            graph_prop['next_evtol_embedding'] = next_embedding
            graph_prop["mask"] = generate_mask()
            return graph_prop