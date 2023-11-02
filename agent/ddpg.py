import torch
from torch.optim import Adam
from gym import spaces
from model.actor import DeterministicPolicy
from model.critic import QNetwork
from model.utils import *
from module.measure import *

class DDPG(object):
    def __init__(self, in_channels, num_actions, args, device, coord):

        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha

        self.target_update_term = args.target_update_term
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.device = device
        self.coord = coord
        self.max_step = args.num_steps
        self.width = args.width
        self.height = args.height
        self.model_based = args.model_based

        action_space = spaces.Box(low=0, high=1, shape=(num_actions, ))
        
        self.critic = QNetwork(in_channels, action_space.shape[0], args.hidden_size, args.width, args.height, self.model_based).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)

        self.critic_target = QNetwork(in_channels, action_space.shape[0], args.hidden_size, args.width, args.height, self.model_based).to(self.device)
        hard_update(self.critic_target, self.critic)

        self.num_updates = 0

        self.policy = DeterministicPolicy(in_channels, action_space.shape[0], args.hidden_size, action_space, width=args.width, height=args.height).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)
        
        self.policy_target = DeterministicPolicy(in_channels, action_space.shape[0], args.hidden_size, action_space, width=args.width, height=args.height).to(self.device)
        hard_update(self.policy_target, self.policy)

    def select_action(self, state, evaluate=False):
        self.policy.eval()
        action = self.policy(state)
        if evaluate is False:
            pass
        self.policy.train()
        return action

    def update_parameters(self, env, memory, batch_size, recalculate_reward=True):
        with torch.autograd.set_detect_anomaly(True):
            # Sample a batch from memory
            canvas_batch, source_batch, goal_batch, step_batch, action_batch, shape_batch, reward_batch, next_canvas_batch, next_source_batch, mask_batch = memory.sample(batch_size=batch_size)

            coord_batch = self.coord.repeat(batch_size, 1, 1, 1)
            step_batch = torch.FloatTensor(step_batch).to(self.device).unsqueeze(2).unsqueeze(3).repeat(1, 1, self.height, self.width)
            t_channel_batch = (self.max_step - step_batch)/self.max_step
            next_t_channel_batch = (self.max_step - (step_batch + 1))/self.max_step
            canvas_batch = torch.FloatTensor(canvas_batch).to(self.device)
            source_batch = torch.FloatTensor(source_batch).to(self.device)
            goal_batch = torch.FloatTensor(goal_batch).to(self.device)
            shape_batch = torch.FloatTensor(shape_batch).unsqueeze(1).to(self.device)
            next_canvas_batch = torch.FloatTensor(next_canvas_batch).to(self.device)
            next_source_batch = torch.FloatTensor(next_source_batch).to(self.device)
            
            state_batch = torch.cat([canvas_batch, goal_batch, source_batch, t_channel_batch, coord_batch], dim=1)
            next_state_batch = torch.cat([next_canvas_batch, goal_batch, next_source_batch, next_t_channel_batch, coord_batch], dim=1)
            action_batch = torch.FloatTensor(action_batch).to(self.device)
            reward_batch = torch.FloatTensor(reward_batch).to(self.device)
            mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

            if recalculate_reward and not self.model_based:
                reward = env.get_reward(state_batch[:, :3, :, :], next_state_batch[:, :3, :, :], state_batch[:, 3:6, :, :], use_tensor=True, shape=shape_batch)
                reward_batch = reward

            with torch.no_grad():
                next_action = self.policy_target(next_state_batch)
                if self.model_based:
                    next_next_state_batch, next_info = env.independent_step(next_state_batch, next_action)
                    next_reward_batch = env.get_reward(next_state_batch[:, :3, :, :], next_next_state_batch[:, :3, :, :], next_state_batch[:, 3:6, :, :], use_tensor=True, shape=next_info['shape'])
                    next_next_v = self.critic_target(next_next_state_batch, None)
                    target_q = mask_batch * self.gamma * (next_reward_batch + next_next_v)
                else: 
                    next_q = self.critic_target(next_state_batch, next_action)
                    target_q = reward_batch + mask_batch * self.gamma * next_q

            if self.model_based:
                next_state_batch, info = env.independent_step(state_batch, action_batch)
                reward_batch = env.get_reward(state_batch[:, :3, :, :], next_state_batch[:, :3, :, :], state_batch[:, 3:6, :, :], use_tensor=True, shape=shape_batch, grad=True)
                next_v = self.critic(next_state_batch, None)
                cur_q = reward_batch + next_v
                target_q += reward_batch.detach()
            else: 
                cur_q = self.critic(state_batch, action_batch)
            critic_loss = mse(cur_q, target_q)

            self.critic_optim.zero_grad()
            if self.model_based:
                critic_loss.backward(retain_graph=True)
            else:
                critic_loss.backward()
            self.critic_optim.step()

            action = self.policy(state_batch)

            if self.model_based:
                next_pi_state_batch, info_pi = env.independent_step(state_batch.detach(), action)
                reward_pi_batch = env.get_reward(state_batch[:, :3, :, :].detach(), next_pi_state_batch[:, :3, :, :], state_batch[:, 3:6, :, :].detach(), use_tensor=True, shape=info_pi['shape'], grad=True)
                next_v_pi = self.critic(next_pi_state_batch, None)

                pre_q = reward_pi_batch + next_v_pi

            else:
                pre_q = self.critic(state_batch, action)

            policy_loss = -pre_q.mean()

            self.policy_optim.zero_grad()
            if self.model_based:
                policy_loss.backward(retain_graph=True)
            else:
                policy_loss.backward()
            self.policy_optim.step()

            if self.num_updates % self.target_update_term == 0:
                soft_update(self.policy_target, self.policy, self.tau)
                soft_update(self.critic_target, self.critic, self.tau)

            self.num_updates += 1
        return critic_loss.item(), policy_loss.item()

    def get_critic_lr(self):
        critic_lr = self.critic_optim.param_groups[0]['lr']
        return critic_lr
    
    def get_policy_lr(self):
        policy_lr = self.policy_optim.param_groups[0]['lr']
        return policy_lr

    def get_action_value(self, obs, action, use_tensor=False):
        if not use_tensor:
            action = torch.FloatTensor(action).to(self.device)
        with torch.no_grad():
            q = self.critic(obs, action)
        return q.detach().cpu().numpy()

    # only valid when model_based
    def get_value(self, obs):
        with torch.no_grad():
            v = self.critic(obs, None)
        return v.detach().cpu().numpy()