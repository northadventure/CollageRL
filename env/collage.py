import cv2
import copy
import sys
import torch
from torchvision.transforms import Resize as resize
from env.utils import *
from module.measure import *
from model.gan import GAN


class CollageEnv:
    def __init__(self, args, drawer, device, dataloader=None, dis=None, coord=None, one_batch=False):
        self.target_width = args.target_width
        self.target_height = args.target_height
        self.width = args.width
        self.height = args.height
        self.drawer = drawer
        
        self.wmin = args.wmin
        self.wmax = args.wmax
        self.hmin = args.hmin
        self.hmax = args.hmax
        self.max_step = args.num_steps
        self.initial_canvas = args.initial_canvas
        self.device = device
        self.dataloader = dataloader
        if dis is None:
            self.dis = GAN(args.dis, args.width, args.height, device)
        else:
            self.dis = dis
        self.mse_reward = args.mse_reward
        
        if one_batch:
            self.num_multi_env = 1
        else:
            self.num_multi_env = args.num_multi_env
        self.noop = args.noop

        # CoordConv
        if coord is None:
            coord = torch.zeros([1, 2, self.width, self.height])
            for i in range(self.width):
                for j in range(self.height):
                    coord[0, 0, i, j] = i / float(self.width-1)
                    coord[0, 1, i, j] = j / float(self.height-1)
            self.coord = coord.to(device)
        else:
            self.coord = coord

    def reset(self, canvas=None, goal=None, goal_label=None, goal_idx=None, source=None, mode='train'):
        # Initialize canvas
        if canvas is None:
            self.canvas = self.initialize_canvas(mode=mode)

        # Sample goal
        if goal is None:
            self.goal = self.dataloader.get_random_goals(self.num_multi_env, mode)
        else:
            self.goal, self.goal_label, self.goal_idx = goal, goal_label, goal_idx

        # Sample material
        if source is None:
            self.source = self.dataloader.get_random_sources(self.num_multi_env, mode)
        else:
            self.source = source
        
        self.steps = np.zeros([self.num_multi_env])
        
        self.shape = None
            
        obs = self.make_obs()

        self.done = np.zeros([self.num_multi_env])
        self.real_steps = 0

        self.episode_reward_dis = 0
        self.episode_reward_mse = 0
        self.episode_reward_total = 0
        
        return obs

    def step(self, action, source=None, mode='train', calculate_rewards=False, paper_like=False):
        # draw
        next_canvas, shape = self.drawer.draw_nondiff(tensor(self.canvas), tensor(self.source), action, self.target_width, self.target_height, self.wmin, self.wmax, self.hmin, self.hmax, self.device, rounding=True, paper_like=paper_like, shape=self.shape)
        # (Next state = current state) for already done canvas
        done_tensor = torch.FloatTensor(self.done).to(self.device)
        done_tensor_grid = done_tensor.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 3, self.height, self.width)
        next_canvas = (1 - done_tensor_grid)*next_canvas + done_tensor_grid * torch.FloatTensor(self.canvas).permute(0, 3, 1, 2).to(device)
        next_canvas = next_canvas.permute(0, 2, 3, 1).detach().cpu().numpy()

        if calculate_rewards:
            # When in evaluation & inference, calculate rewards here for further information
            reward = self.calculate_rewards(next_canvas, action, shape.unsqueeze(1))
        else:
            # When in training, we don't need to calculate the reward here (the reward will be calculated during the agent updates).
            reward = 0

        self.canvas = next_canvas

        # Sample next material
        if source is None:
            self.source = self.dataloader.get_random_sources(self.num_multi_env, mode)
        else:
            self.source = source

        # Count for pasted scraps
        if self.noop:
            self.steps += (1 - self.done) * (action[:, 0] > 0.5).detach().cpu().numpy()
        else:
            self.steps += (1 - self.done)
        self.real_steps += 1

        # Make next obs
        next_obs = self.make_obs()

        info = {
            'steps': self.steps,
            'episode_reward_dis': self.episode_reward_dis,
            'episode_reward_mse': self.episode_reward_mse,
            'episode_reward_total': self.episode_reward_total,
            'mse': mse(self.canvas, self.goal),
            'shape': shape,
            'already_done': copy.deepcopy(self.done)
        }

        self.done = self.is_terminal()
        
        return next_obs, reward, copy.deepcopy(self.done), info

    def calculate_rewards(self, next_canvas, action, shape):
        reward_dis = self.get_reward_dis(self.canvas[0], next_canvas[0], self.goal[0], shape=shape[0:1])
        reward_mse = self.get_reward_mse(self.canvas[0], next_canvas[0], self.goal[0])
        self.episode_reward_dis += reward_dis
        self.episode_reward_mse += reward_mse
        reward = reward_dis
        self.episode_reward_total += reward
        return {'dis':reward_dis, 'mse':reward_mse}

    def independent_step(self, obs_batch, action_batch, nondiff=False, maintain_source=False):
        canvas_batch = obs_batch[:, :3]#.clone()
        goal_batch = obs_batch[:, 3:6]#.clone()
        source_batch = obs_batch[:, 6:9]#.clone()
        t_channel_batch = obs_batch[:, 9:10].clone()
        coord_batch = obs_batch[:, 10:12]#.clone()

        if nondiff:
            next_canvas_batch, shape = self.drawer.draw_nondiff(canvas_batch, source_batch, action_batch, self.width, self.height, self.wmin, self.wmax, self.hmin, self.hmax, self.device)
        else:
            next_canvas_batch, shape = self.drawer.draw_diff(canvas_batch, source_batch, action_batch, self.width, self.height, self.wmin, self.wmax, self.hmin, self.hmax, self.device)

        # remaining steps
        if self.noop:
            t_channel_batch -= 1/self.max_step * (action_batch[:, 0:1] > 0.5).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.height, self.width)
        else:
            t_channel_batch -= 1/self.max_step

        if maintain_source:
            new_source_batch = source_batch
        else:
            new_sources = self.dataloader.get_random_sources(canvas_batch.size(0), 'train')
            new_source_batch = torch.FloatTensor(new_sources).permute(0, 3, 1, 2).to(device)
        
        next_obs = torch.cat([
            next_canvas_batch,
            goal_batch,
            new_source_batch,
            t_channel_batch,
            coord_batch,
        ], dim=1)

        info = {
            'shape': shape
        }

        return next_obs, info

    def initialize_canvas(self, mode):
        width, height = self.target_width, self.target_height
        if self.initial_canvas == 'random':
            if mode == 'train':
                canvas = self.dataloader.get_random_goals(self.num_multi_env, mode)
            elif mode == 'eval':
                canvas = np.ones([self.num_multi_env, height, width, 3])
            else:
                print(f'There is no mode {mode} for initializing canvas. mode should be \'train\' or \'eval\'')
                sys.exit(0)
        elif self.initial_canvas == 'white':
            canvas = np.ones([self.num_multi_env, height, width, 3])
        else:
            print(f'There is no initial_canvas option {self.initial_canvas} for initializing canvas')
            sys.exit(0)

        return canvas

    def make_obs(self):
        t_channel = torch.ones(self.num_multi_env, 1, self.height, self.width).to(self.device) * (self.max_step - torch.FloatTensor(self.steps).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(self.device))/self.max_step
        obs = torch.cat([
            resize((self.width, self.height))(tensor(self.canvas)),
            resize((self.width, self.height))(tensor(self.goal)),
            resize((self.width, self.height))(tensor(self.source)),
            t_channel,
            self.coord.repeat(self.num_multi_env, 1, 1, 1),
        ], dim=1)
        return obs

    def get_reward_dis(self, canvas, next_canvas, goal, use_tensor=False, shape=None, grad=False, complexity_bonus=True):
        # GAN reward
        if not use_tensor:
            canvas = cv2.resize(canvas, (self.width, self.height))
            next_canvas = cv2.resize(next_canvas, (self.width, self.height))
            goal = cv2.resize(goal, (self.width, self.height))
            current_set = torch.cat([tensor(canvas), tensor(goal)], dim=1)
            next_set = torch.cat([tensor(next_canvas), tensor(goal)], dim=1)
        else:
            current_set = torch.cat([canvas, goal], dim=1)
            next_set = torch.cat([next_canvas, goal], dim=1)
        sims_input = torch.cat([current_set, next_set], dim=0)
        sims = self.dis.similarity_tensor(sims_input, grad=grad)
        if not use_tensor:
            reward_dis = (sims[1] - sims[0]).item()  # C(t+1) sim - C(t) sim
        else:
            half = int(sims_input.size(0)/2)
            reward_dis = (sims[half:] - sims[:half])  # C(t+1) sim - C(t) sim

        return reward_dis

    def get_reward_mse(self, canvas, next_canvas, goal, use_tensor=False):
        # MSE reward
        reward_mse = mse(canvas, goal, use_tensor=use_tensor) - mse(next_canvas, goal, use_tensor=use_tensor)
        return reward_mse

    def get_reward(self, canvas, next_canvas, goal, use_tensor=False, shape=None, grad=False, step_penalty=True, complexity_bonus=True):
        if self.mse_reward:
            reward = self.get_reward_mse(canvas, next_canvas, goal, use_tensor=use_tensor).view(-1, 1)
        else:
            reward = self.get_reward_dis(canvas, next_canvas, goal, use_tensor=use_tensor, shape=shape, grad=grad, complexity_bonus=complexity_bonus)
        if step_penalty:
            reward -= 1
        return reward

    def is_terminal(self):
        if self.real_steps > 100:  # set maximum steps (to terminate when too many NOOPs occur)
            return self.steps >= 0
        return self.steps >= self.max_step

    def func(self):
        print('ouch!')