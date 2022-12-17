# -*- coding: utf-8 -*-
from typing import Any, Dict, Optional, Tuple, Type

import numpy as np
from cmath import inf
import gym
import torch
from torch import long, nn, Tensor, tensor, bool
from stable_baselines3.common.type_aliases import Schedule
from torch_geometric.nn import GCNConv, BatchNorm
from stable_baselines3.common.policies import MultiInputActorCriticPolicy, BasePolicy,BaseFeaturesExtractor
from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    make_proba_distribution,
)


class CustomGLPolicy(BasePolicy):
    def __init__(self, 
                observation_space: gym.spaces.Dict,
                action_space: gym.spaces.Discrete,
                lr_schedule: Schedule = Schedule,
                log_std_init: float = 0.0,
                use_sde: bool = False,
                squash_output: bool = False,
                ortho_init: bool = True,
                optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
                optimizer_kwargs: Optional[Dict[str, Any]] = None,
                policy_kwargs: Optional[Dict[str, Any]] = None,
                ):
        super(CustomGLPolicy,self).__init__(observation_space,
                                            action_space,
                                            optimizer_class = optimizer_class,
                                            optimizer_kwargs = optimizer_kwargs,   
                                            squash_output = squash_output                                         
                                            )

        self.features_extractor = GNNFeatureExtractor(observation_space, policy_kwargs)
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
        self.action_dist = make_proba_distribution(action_space, use_sde=use_sde)

    def _predict(self, observation: Tensor, deterministic: bool = True) -> Tensor:
            actions, _, _ = self.forward(observation, deterministic=deterministic)
            return tensor([actions])

    def _build(self):
        pass

    def evaluate_actions(self, obs: Tensor, actions: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        distribution, values = self.get_distribution(obs)
        log_prob = distribution.log_prob(actions)

        return values, log_prob, distribution.entropy()

    def forward(self, obs, deterministic = False):

        distribution,values = self.get_distribution(obs)
        actions = distribution.get_actions(deterministic=True)
        log_prob = distribution.log_prob(actions)

        return actions, values, log_prob

    def predict_values(self,obs):
        _, values = self.get_distribution(obs)
        return values

    def get_distribution(self, observation):

        latent_sde, values, mean_actions = self.extract_features(observation)

        if isinstance(self.action_dist, DiagGaussianDistribution):
            distribution =  self.action_dist.proba_distribution(mean_actions, self.log_std)
        elif isinstance(self.action_dist, CategoricalDistribution):
            # Here mean_actions are the logits before the softmax
            distribution =  self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            # Here mean_actions are the flattened logits
            distribution =  self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, BernoulliDistribution):
            # Here mean_actions are the logits (before rounding to get the binary actions)
            distribution =  self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            distribution =  self.action_dist.proba_distribution(mean_actions, self.log_std, latent_sde)
        else:
            raise ValueError("Invalid action distribution")

        return distribution, values

class GNNFeatureExtractor(nn.Module):
    def __init__(self, observation_space, policy_kwargs): #This custom GNN receives the obs dict for the action log-probabilities
        super(GNNFeatureExtractor, self).__init__()
        
        if not policy_kwargs: #For Optuna trials
            import yaml
            with open('policy_config.yaml', 'r') as file:
                policy_kwargs = yaml.safe_load(file)
                policy_kwargs = policy_kwargs["policy_kwargs"]

        activations = {"relu":nn.ReLU(), "rrelu":nn.RReLU(0.1, 0.3), "elu":nn.ELU(), "leaky":nn.LeakyReLU(0.1), "tanh":nn.Tanh()}

        print(policy_kwargs)
        #Input features
        
        vertiport_input_channels = policy_kwargs["policy"]["vertiport"]["input_channels"]
        vertiport_hidden_channels = policy_kwargs["policy"]["vertiport"]["hidden_channels"]
        vertiport_output_channels = policy_kwargs["policy"]["vertiport"]["output_channels"]
        
        evtol_input_channels = policy_kwargs["policy"]["evtol"]["input_channels"]
        evtol_hidden_channels = policy_kwargs["policy"]["evtol"]["hidden_channels"]
        evtol_output_channels = policy_kwargs["policy"]["evtol"]["output_channels"]
        
        mlp_hidden1 = policy_kwargs["policy"]["mlp"]["hidden1"]
        mlp_hidden2 = policy_kwargs["policy"]["mlp"]["hidden2"]

        value_hidden1 = mlp_hidden1 = policy_kwargs["value"]["hidden1"]
        value_hidden2 = mlp_hidden1 = policy_kwargs["value"]["hidden2"]
        value_activ = activations[policy_kwargs["value"]["activ"]]
        value_bias = False



        next_evtol_feature_size = np.prod(observation_space["next_evtol_embedding"].shape)
        input_dim = vertiport_output_channels + evtol_output_channels + next_evtol_feature_size #length of feature vector for MLP
        latent_pi_dim = 11 # Output action dimensions
        latent_vf_dim = 1 # Output value dimensions
        


        self.vertiport = GCN(vertiport_input_channels, vertiport_hidden_channels, vertiport_output_channels) #input channels, hidden channels, output channels
        self.evtols = GCN(evtol_input_channels, evtol_hidden_channels, evtol_output_channels) #Input channels, hidden channels, output channels
        self.output_space = GRLMLP(input_dim, mlp_hidden1, mlp_hidden2, latent_pi_dim) #Input dimension, output dimension

        self.value_network = nn.Sequential(nn.Linear(input_dim, value_hidden1, bias=value_bias), 
                                            value_activ, 
                                            nn.Linear(value_hidden1, value_hidden2, bias=value_bias),
                                            value_activ,
                                            nn.Linear(value_hidden2, latent_vf_dim),
                                            )

    def forward(self,data):
        # print('data from PPO',data)
        verti_features = data['vertiport_features'].float()
        verti_edge = data['vertiport_edge'][0].long() #edge connectivity matrix doesn't change, so only need the first instance
        ev_features = data['evtol_features'].float()
        ev_edge = data['evtol_edge'][0].long()
        next_drone = data['next_evtol_embedding']
        mask = data['mask'].bool()

        verti_embed = self.vertiport(verti_features, verti_edge)
        ev_embed = self.evtols(ev_features, ev_edge)
        final_features = torch.cat((verti_embed, ev_embed, next_drone), dim=1)
        value = self.value_network(final_features)
        output = self.output_space(final_features, mask) #Testing how the custom feature extractor works

        return final_features, value, output


class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels) -> None:
        super().__init__()
        self.gcn_layer1_prev = None
        self.gcn_layer2_prev = None
        self.gcn_layer3_prev = None
        self.gcn_layer4_prev = None
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, out_channels)
        self.Leaky_ReLU = nn.LeakyReLU(negative_slope=0.1)
        self.RRelu = nn.RReLU(lower=0.1, upper=0.3)
        self.BatchNorm = BatchNorm(in_channels, track_running_stats=True) #Supposedly works the same as BatchNorm1d

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.gcn_layer1_prev = self.conv1(x, edge_index) 
        x = self.RRelu(x)
        x = self.gcn_layer2_prev = self.conv2(x, edge_index) + self.gcn_layer1_prev 
        x = self.RRelu(x)
        x = self.gcn_layer3_prev = self.conv3(x, edge_index) + self.gcn_layer2_prev
        x = self.RRelu(x)
        x = self.conv4(x, edge_index) + self.gcn_layer3_prev[:,:,:self.out_channels]

        # x = self.conv1(x, edge_index) 
        # x = self.Leaky_ReLU(x)
        # x = self.conv2(x, edge_index)
        # x = self.Leaky_ReLU(x)
        # x = self.conv3(x, edge_index)
        # x = self.Leaky_ReLU(x)
        # x = self.conv4(x, edge_index)
        # print('x without pooling',x.shape)

        return x.mean(dim=1)

class GRLMLP(nn.Module):
    def __init__(self,input_dim, hidden1, hidden2, output_dim) -> None:
        super().__init__()

        self.input = nn.Linear(input_dim, hidden1)
        self.hidden_layer = nn.Linear(hidden1, hidden1)
        self.hidden_layer2 = nn.Linear(hidden1, hidden2)
        self.output = nn.Linear(hidden2, output_dim)
        self.Leaky_ReLU = nn.LeakyReLU(negative_slope=0.1)
        self.RRelu = nn.RReLU(lower=0.1, upper=0.3)
        self.tanh = nn.Tanh()


    def forward(self, x, mask=None):

        #x should be (input_dim,-1)
        # batch_size = x.shape[0]
        # x = x.view(batch_size,-1)
 
        #Forward propogation
        h1 = self.input(x)
        # l1 = self.tanh(h1)
        l1 = self.RRelu(h1)

        h2 = self.hidden_layer(l1)
        l2 = self.RRelu(h2)
        # l2 = self.tanh(h2)

        h3 = self.hidden_layer2(l2)
        # l3 = self.tanh(h3)
        l3 = self.RRelu(h3)

        # print('mask',mask)
        ypred = self.output(l3)
        ypred[mask] = -inf
        return torch.log_softmax(ypred, dim = -1)
