import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Policy(nn.Module):
    def __init__(self, joint_log_softmax_temperature, foot_log_softmax_temperature, softmax_temperature_min, stability_epsilon, policy_mean_abs_clip, policy_std_min_clip, policy_std_max_clip):
        super(Policy, self).__init__()
        self.joint_log_softmax_temperature = joint_log_softmax_temperature
        self.foot_log_softmax_temperature = foot_log_softmax_temperature
        self.softmax_temperature_min = softmax_temperature_min
        self.stability_epsilon = stability_epsilon
        self.policy_mean_abs_clip = policy_mean_abs_clip
        self.policy_std_min_clip = policy_std_min_clip
        self.policy_std_max_clip = policy_std_max_clip

        self.dynamic_joint_state_mask1 = nn.Linear(23, 64)
        self.latent_dynamic_joint_state = nn.Linear(3, 4)
        self.dynamic_foot_state_mask1 = nn.Linear(10, 32)
        self.latent_dynamic_foot_state = nn.Linear(2, 4)
        self.action_latent1 = nn.Linear(400, 512)
        self.action_layer_norm = nn.LayerNorm(512, eps=1e-6)
        self.action_latent2 = nn.Linear(512, 256)
        self.action_latent3 = nn.Linear(256, 128)
        self.action_description_latent1 = nn.Linear(23, 128)
        self.action_description_latent2 = nn.Linear(128, 128)
        self.policy_mean_layer1 = nn.Linear(260, 128)
        self.policy_mean_layer_norm = nn.LayerNorm(128, eps=1e-6)
        self.policy_mean_layer2 = nn.Linear(128, 1)
        self.policy_logstd_layer = nn.Linear(128, 1)

    def forward(self, dynamic_joint_description, dynamic_joint_state, dynamic_foot_description, dynamic_foot_state, general_state):
        dynamic_joint_state_mask = torch.tanh(self.dynamic_joint_state_mask1(dynamic_joint_description))
        dynamic_joint_state_mask = torch.clamp(dynamic_joint_state_mask, -1.0 + self.stability_epsilon, 1.0 - self.stability_epsilon)

        latent_dynamic_joint_state = F.elu(self.latent_dynamic_joint_state(dynamic_joint_state))

        joint_e_x = torch.exp(dynamic_joint_state_mask / (torch.exp(self.joint_log_softmax_temperature) + self.softmax_temperature_min))
        dynamic_joint_state_mask = joint_e_x / (joint_e_x.sum(dim=-1, keepdim=True) + self.stability_epsilon)
        dynamic_joint_state_mask = dynamic_joint_state_mask.unsqueeze(-1).repeat(1, 1, 1, latent_dynamic_joint_state.size(-1))
        masked_dynamic_joint_state = dynamic_joint_state_mask * latent_dynamic_joint_state.unsqueeze(-2)
        masked_dynamic_joint_state = masked_dynamic_joint_state.view(masked_dynamic_joint_state.shape[:-2] + (masked_dynamic_joint_state.shape[-2] * masked_dynamic_joint_state.shape[-1],))
        dynamic_joint_latent = masked_dynamic_joint_state.sum(dim=-2)

        dynamic_foot_state_mask = torch.tanh(self.dynamic_foot_state_mask1(dynamic_foot_description))
        dynamic_foot_state_mask = torch.clamp(dynamic_foot_state_mask, -1.0 + self.stability_epsilon, 1.0 - self.stability_epsilon)

        latent_dynamic_foot_state = F.elu(self.latent_dynamic_foot_state(dynamic_foot_state))

        foot_e_x = torch.exp(dynamic_foot_state_mask / (torch.exp(self.foot_log_softmax_temperature) + self.softmax_temperature_min))
        dynamic_foot_state_mask = foot_e_x / (foot_e_x.sum(dim=-1, keepdim=True) + self.stability_epsilon)
        dynamic_foot_state_mask = dynamic_foot_state_mask.unsqueeze(-1).repeat(1, 1, 1, latent_dynamic_foot_state.size(-1))
        masked_dynamic_foot_state = dynamic_foot_state_mask * latent_dynamic_foot_state.unsqueeze(-2)
        masked_dynamic_foot_state = masked_dynamic_foot_state.view(masked_dynamic_foot_state.shape[:-2] + (masked_dynamic_foot_state.shape[-2] * masked_dynamic_foot_state.shape[-1],))
        dynamic_foot_latent = masked_dynamic_foot_state.sum(dim=-2)

        combined_input = torch.cat([dynamic_joint_latent, dynamic_foot_latent, general_state], dim=-1)

        action_latent = self.action_latent1(combined_input)
        action_latent = F.elu(self.action_layer_norm(action_latent))
        action_latent = F.elu(self.action_latent2(action_latent))
        action_latent = self.action_latent3(action_latent)

        action_description_latent = F.elu(self.action_description_latent1(dynamic_joint_description))
        action_description_latent = self.action_description_latent2(action_description_latent)

        action_latent = action_latent.unsqueeze(-2).repeat(1, action_description_latent.size(-2), 1)
        combined_action_latent = torch.cat([action_latent, latent_dynamic_joint_state.detach(), action_description_latent], dim=-1)
        
        policy_mean = self.policy_mean_layer1(combined_action_latent)
        policy_mean = F.elu(self.policy_mean_layer_norm(policy_mean))
        policy_mean = self.policy_mean_layer2(policy_mean)
        policy_mean = torch.clamp(policy_mean, -self.policy_mean_abs_clip, self.policy_mean_abs_clip)

        return policy_mean.squeeze(-1)


def get_policy():
    joint_log_softmax_temperature = torch.tensor(np.load("jax_nn_weights/joint_log_softmax_temperature.npy"))
    foot_log_softmax_temperature = torch.tensor(np.load("jax_nn_weights/foot_log_softmax_temperature.npy"))
    softmax_temperature_min = 0.015
    stability_epsilon = 0.00000001
    policy_mean_abs_clip = 10.0
    policy_std_min_clip = 0.00000001
    policy_std_max_clip = 2.0

    policy = Policy(joint_log_softmax_temperature, foot_log_softmax_temperature, softmax_temperature_min, stability_epsilon, policy_mean_abs_clip, policy_std_min_clip, policy_std_max_clip)
    policy = torch.jit.script(policy)

    # Load weights
    policy.dynamic_joint_state_mask1.weight.data = torch.tensor(np.load("jax_nn_weights/Dense_1_kernel.npy")).T
    policy.dynamic_joint_state_mask1.bias.data = torch.tensor(np.load("jax_nn_weights/Dense_1_bias.npy"))
    policy.latent_dynamic_joint_state.weight.data = torch.tensor(np.load("jax_nn_weights/Dense_2_kernel.npy")).T
    policy.latent_dynamic_joint_state.bias.data = torch.tensor(np.load("jax_nn_weights/Dense_2_bias.npy"))
    policy.dynamic_foot_state_mask1.weight.data = torch.tensor(np.load("jax_nn_weights/Dense_4_kernel.npy")).T
    policy.dynamic_foot_state_mask1.bias.data = torch.tensor(np.load("jax_nn_weights/Dense_4_bias.npy"))
    policy.latent_dynamic_foot_state.weight.data = torch.tensor(np.load("jax_nn_weights/Dense_5_kernel.npy")).T
    policy.latent_dynamic_foot_state.bias.data = torch.tensor(np.load("jax_nn_weights/Dense_5_bias.npy"))
    policy.action_latent1.weight.data = torch.tensor(np.load("jax_nn_weights/Dense_6_kernel.npy")).T
    policy.action_latent1.bias.data = torch.tensor(np.load("jax_nn_weights/Dense_6_bias.npy"))
    policy.action_layer_norm.weight.data = torch.tensor(np.load("jax_nn_weights/LayerNorm_0_scale.npy"))
    policy.action_layer_norm.bias.data = torch.tensor(np.load("jax_nn_weights/LayerNorm_0_bias.npy"))
    policy.action_latent2.weight.data = torch.tensor(np.load("jax_nn_weights/Dense_7_kernel.npy")).T
    policy.action_latent2.bias.data = torch.tensor(np.load("jax_nn_weights/Dense_7_bias.npy"))
    policy.action_latent3.weight.data = torch.tensor(np.load("jax_nn_weights/Dense_8_kernel.npy")).T
    policy.action_latent3.bias.data = torch.tensor(np.load("jax_nn_weights/Dense_8_bias.npy"))
    policy.action_description_latent1.weight.data = torch.tensor(np.load("jax_nn_weights/Dense_9_kernel.npy")).T
    policy.action_description_latent1.bias.data = torch.tensor(np.load("jax_nn_weights/Dense_9_bias.npy"))
    policy.action_description_latent2.weight.data = torch.tensor(np.load("jax_nn_weights/Dense_10_kernel.npy")).T
    policy.action_description_latent2.bias.data = torch.tensor(np.load("jax_nn_weights/Dense_10_bias.npy"))
    policy.policy_mean_layer1.weight.data = torch.tensor(np.load("jax_nn_weights/Dense_11_kernel.npy")).T
    policy.policy_mean_layer1.bias.data = torch.tensor(np.load("jax_nn_weights/Dense_11_bias.npy"))
    policy.policy_mean_layer_norm.weight.data = torch.tensor(np.load("jax_nn_weights/LayerNorm_1_scale.npy"))
    policy.policy_mean_layer_norm.bias.data = torch.tensor(np.load("jax_nn_weights/LayerNorm_1_bias.npy"))
    policy.policy_mean_layer2.weight.data = torch.tensor(np.load("jax_nn_weights/Dense_12_kernel.npy")).T
    policy.policy_mean_layer2.bias.data = torch.tensor(np.load("jax_nn_weights/Dense_12_bias.npy"))
    policy.policy_logstd_layer.weight.data = torch.tensor(np.load("jax_nn_weights/Dense_13_kernel.npy")).T
    policy.policy_logstd_layer.bias.data = torch.tensor(np.load("jax_nn_weights/Dense_13_bias.npy"))

    return policy


if __name__ == "__main__":
    policy = get_policy()
    
    dummy_dynamic_joint_description = np.zeros((1, 13, 23))
    dummy_dynamic_joint_state = np.zeros((1, 13, 3))
    dummy_dynamic_foot_description = np.zeros((1, 4, 10))
    dummy_dynamic_foot_state = np.zeros((1, 4, 2))
    dummy_general_policy_state = np.zeros((1, 16))

    dummy_dynamic_joint_description = torch.tensor(dummy_dynamic_joint_description, dtype=torch.float32)
    dummy_dynamic_joint_state = torch.tensor(dummy_dynamic_joint_state, dtype=torch.float32)
    dummy_dynamic_foot_description = torch.tensor(dummy_dynamic_foot_description, dtype=torch.float32)
    dummy_dynamic_foot_state = torch.tensor(dummy_dynamic_foot_state, dtype=torch.float32)
    dummy_general_policy_state = torch.tensor(dummy_general_policy_state, dtype=torch.float32)

    import time

    nr_evals = 1_000
    start = time.time()
    for i in range(nr_evals):
        with torch.no_grad():
            action = policy(dummy_dynamic_joint_description, dummy_dynamic_joint_state, dummy_dynamic_foot_description, dummy_dynamic_foot_state, dummy_general_policy_state)
    end = time.time()
    print("Average time per evaluation: ", (end - start) / nr_evals)
