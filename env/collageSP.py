import copy
import sys
from torchvision.transforms import Resize as resize
from env.collage import CollageEnv
from env.utils import *
from module.measure import *

# Inherit from collage.py
class CollageSPEnv(CollageEnv):
    def __init__(self, args, drawer, device, dataloader=None, dis=None, coord=None, one_batch=False):
        super().__init__(args, drawer, device, dataloader, dis, coord, one_batch)
        self.source_sample_size = args.source_sample_size

    def reset(self, canvas=None, goal=None, goal_label=None, goal_idx=None, source_pool=None, mode='train'):
        # Initialize canvas
        if canvas is None:
            self.canvas = self.initialize_canvas(mode=mode)
        else:
            self.canvas = canvas

        # Sample goal
        if goal is None:
            self.goal = self.dataloader.get_random_goals(self.num_multi_env, mode)
        else:
            self.goal, self.goal_label, self.goal_idx = goal, goal_label, goal_idx

        # Sample material pool
        if source_pool is None:
            if self.source_sample_size is None:
                print('Num of sources must be defined if source pool is not customly given.')
                sys.exit(0)
            self.source_pool, _ = self.dataloader.get_random_source_pool(self.source_sample_size, mode=mode)
        else:
            self.source_pool = source_pool

        self.steps = np.zeros([self.num_multi_env])
        
        self.shape = None

        obs = self.make_obs()

        self.done = np.zeros([self.num_multi_env])
        self.real_steps = 0

        self.episode_reward_dis = 0
        self.episode_reward_mse = 0
        self.episode_reward_total = 0
        
        return obs

    def step(self, action, source_idx, mode='train', calculate_rewards=False, paper_like=False, disallow_duplicate=False):
        # Get selected source
        source = self.source_pool[source_idx]

        # Draw
        next_canvas, shape = self.drawer.draw_nondiff(tensor(self.canvas), tensor(source), action, self.target_width, self.target_height, self.wmin, self.wmax, self.hmin, self.hmax, self.device, rounding=True, paper_like=paper_like, shape=self.shape)
        next_canvas = next_canvas.permute(0, 2, 3, 1).detach().cpu().numpy()
        
        if disallow_duplicate:
            # Delete used source
            self.source_pool = np.delete(self.source_pool, source_idx, axis=0)

        if calculate_rewards:
            reward = self.calculate_rewards(next_canvas, action, shape)
        else:
            # If in training, reward doesn't need to be calculated here (reward will be calculated while in the agent update).
            reward = 0

        self.canvas = next_canvas

        # Count for pasted scraps
        if self.noop:
            self.steps += (1 - self.done) * (action[:, 0] > 0.5).detach().cpu().numpy()
        else:
            self.steps += (1 - self.done)
        self.real_steps += 1

        # Make next obs
        next_obs = self.make_obs()

        self.done = self.is_terminal()

        info = {
            'steps': self.steps,
            'episode_reward_dis': self.episode_reward_dis,
            'episode_reward_mse': self.episode_reward_mse,
            'episode_reward_total': self.episode_reward_total,
            'mse': mse(self.canvas, self.goal),
            'shape': shape,
        }
        
        return next_obs, reward, copy.deepcopy(self.done), info

    def make_obs(self):
        # (n_batch, n_sources, n_channels, height, width)
        ns = len(self.source_pool)
        bs = self.num_multi_env
        t_channel = torch.ones(self.num_multi_env, ns, 1, self.height, self.width).to(self.device) * (self.max_step - torch.FloatTensor(self.steps).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(self.device))/self.max_step
        obs = torch.cat([
            resize((self.width, self.height))(tensor(self.canvas)).unsqueeze(1).repeat(1, ns, 1, 1, 1),
            resize((self.width, self.height))(tensor(self.goal)).unsqueeze(1).repeat(1, ns, 1, 1, 1),
            resize((self.width, self.height))(tensor(self.source_pool)).unsqueeze(0).repeat(bs, 1, 1, 1, 1),
            t_channel,
            self.coord.unsqueeze(0).repeat(bs, ns, 1, 1, 1),
        ], dim=2)
        return obs