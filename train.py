import argparse
import os
import sys
import cv2
import random
import torch
import numpy as np
from torchvision.transforms import Resize as resize
from datetime import datetime
from env.draw import Drawer
from env.collage import CollageEnv
from env.collageSP import CollageSPEnv
from env.utils import tensor
from agent.sac import SAC
from agent.ddpg import DDPG
from model.gan import GAN
from module.gpu import *  # Set device here
from module.rpm import ReplayMemory
from module.evaluate import Evaluator
from module.logger import Logger
from module.dataloader import DataLoader


parser = argparse.ArgumentParser(description='Collage Training Arguments')
parser.add_argument('--algo', type=str, default='sac')  # sac | ddpg
parser.add_argument('--eval', type=bool, default=True)
parser.add_argument('--gamma', type=float, default=0.95)
parser.add_argument('--tau', type=float, default=0.005)
parser.add_argument('--lr', type=float, default=0.0003)
parser.add_argument('--alpha', type=float, default=0.2)
parser.add_argument('--automatic_entropy_tuning', action='store_true')
parser.add_argument('--seed', type=int, default=123456)
parser.add_argument('--update_steps', type=int, default=5)  # number of training steps per one update
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--hidden_size', type=int, default=128)
parser.add_argument('--start_episodes', type=int, default=10)  # pre-episodes to fill the replay memory before training
parser.add_argument('--replay_size', type=int, default=1e5)
parser.add_argument('--model_based', action='store_true')
parser.add_argument('--num_multi_env', type=int, default=16)
parser.add_argument('--noop', action='store_true')  # Enable NOOP action
parser.add_argument('--mse_reward', action='store_true')  # Use MSE reward (default: discriminator reward)
parser.add_argument('--dis', type=str, default='wgan')  # Type of discriminator. wgan | sngan | gngan
parser.add_argument('--scale', type=str, default='small')  # Network input resolution scale. tiny: 8x8 | mini: 16x16 | little: 32x32 | small: 64x64 | medium: 128x128 | big: 256x256 | large: 512x512 | huge: 1024x1024
parser.add_argument('--target_width', type=int, default=224)
parser.add_argument('--target_height', type=int, default=224)
parser.add_argument('--width', type=int, default=224)
parser.add_argument('--height', type=int, default=224)
parser.add_argument('--wmin', type=float, default=0.05)
parser.add_argument('--wmax', type=float, default=0.3)
parser.add_argument('--hmin', type=float, default=0.05)
parser.add_argument('--hmax', type=float, default=0.3)
parser.add_argument('--data_path', type=str, default='~/Datasets')  # A folder where the datasets are living in
parser.add_argument('--goal', type=str, default='imagenet')  # imagenet | mnist | scene
parser.add_argument('--source', type=str, default='dtd')  # imagenet | dtd
parser.add_argument('--initial_canvas', type=str, default='random')  # random | white
parser.add_argument('--num_steps', type=int, default=100)  # Episode length
parser.add_argument('--num_episodes', type=int, default=1e6)  # Training episodes
parser.add_argument('--learning_term_agent', type=int, default=1)  # Update episode term for agent
parser.add_argument('--learning_term_dis', type=int, default=1)  # Update episode term for discriminator)
parser.add_argument('--target_update_term', type=int, default=1)  # Update episode term for target Q network
parser.add_argument('--eval_episodes', type=int, default=5)  # How many episodes should be sampled for one evaluation
parser.add_argument('--eval_term', type=int, default=1e2)  # Episode term for evaluation
parser.add_argument('--source_sample_size', type=int, default=30)  # Subset size for material candidates (only for CollageSP env)
parser.add_argument('--save_term', type=int, default=1e3)  # Episode term for saving network
parser.add_argument('--save_history_term', type=int, default=1e4)  # Episode term for make network history
parser.add_argument('--logging_term', type=int, default=1)  # Episode term for logging
parser.add_argument('--printing_term', type=int, default=1)  # Episode term for status printing
parser.add_argument('--no_log', action='store_true')

args = parser.parse_args()

if args.scale == 'tiny':
    args.width, args.height = 8, 8
elif args.scale == 'mini':
    args.width, args.height = 16, 16
elif args.scale == 'little':
    args.width, args.height = 32, 32
elif args.scale == 'small':
    args.width, args.height = 64, 64
elif args.scale == 'medium':
    args.width, args.height = 128, 128
elif args.scale == 'big':
    args.width, args.height = 256, 256
elif args.scale == 'large':
    args.width, args.height = 512, 512
elif args.scale == 'huge':
    args.width, args.height = 1024, 1024
else:
    print(f'There is no scale option {args.scale}')
    sys.exit(0)

# Seed control
seed = 142
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Setup some values
print('Preparing train')
dataloader = DataLoader(data_path=args.data_path, goal=args.goal, source=args.source, width=args.target_width, height=args.target_height, debug=args.no_log)
dis = GAN(args.dis, args.width, args.height, device)
coord = torch.zeros([1, 2, args.width, args.height])
for i in range(args.width):
    for j in range(args.height):
        coord[0, 0, i, j] = i / float(args.width-1)
        coord[0, 1, i, j] = j / float(args.height-1)
coord = coord.to(device)

# Build env
print('Building Env')
drawer = Drawer(args, device)
env = CollageEnv(args, drawer, device, dataloader, dis, coord)
envsp = CollageSPEnv(args, drawer, device, dataloader, dis, coord, one_batch=True)
in_channels = 12
num_actions = 11
if args.noop:
    num_actions += 1
if args.algo == 'sac':
    agent = SAC(in_channels=in_channels, num_actions=num_actions, args=args, device=device, coord=coord)
elif args.algo == 'ddpg':
    agent = DDPG(in_channels=in_channels, num_actions=num_actions, args=args, device=device, coord=coord)
memory = ReplayMemory(args.replay_size)
logger = Logger('Collage', args, args.no_log)
evaluator = Evaluator()

# Make folders for outputs
if not args.no_log:
    datetime_str = datetime.today().strftime("%Y-%m-%d-%H:%M:%S")
    output_path = f'outputs/{datetime_str}/'
    weight_path = f'{output_path}/weights/'
    weight_best_path = f'{output_path}/weights_best/'
    weight_history_path = f'{output_path}/weights_history/'
    os.mkdir(output_path)
    os.mkdir(weight_path)
    os.mkdir(weight_best_path)
    os.mkdir(weight_history_path)
else:
    weight_best_path = None

# Start training
print('Stacking pre-interactions')
for episode in range(int(args.num_episodes)):
    obs = env.reset()
    
    canvas = resize((args.width, args.height))(tensor(env.canvas, cpu=True))
    goal = resize((args.width, args.height))(tensor(env.goal, cpu=True))

    while True:
        with torch.no_grad():
            action = agent.select_action(obs)
        source = resize((args.width, args.height))(tensor(env.source, cpu=True))
        next_obs, reward, done, info = env.step(action)
        next_canvas = resize((args.width, args.height))(tensor(env.canvas, cpu=True))
        next_source = resize((args.width, args.height))(tensor(env.source, cpu=True))
        for ad, c, s, g, st, a, sh, nc, ns, d in zip(info['already_done'], canvas, source, goal, env.steps, action.detach().cpu(), info['shape'].cpu(), next_canvas, next_source, torch.tensor(done)):
            if not ad:
                memory.push(c.unsqueeze(0), s.unsqueeze(0), g.unsqueeze(0), torch.tensor([st]), a.unsqueeze(0), sh.unsqueeze(0), torch.FloatTensor(np.array([reward])), nc.unsqueeze(0), ns.unsqueeze(0), not d)
        obs = next_obs
        canvas = next_canvas

        if done.all():
            break

    if episode < args.start_episodes:
        continue
    elif episode == args.start_episodes:
        print('Learning started')

    # Train agent
    if episode % args.learning_term_agent == 0:
        for update_step in range(args.update_steps):
            if args.algo == 'sac':
                qf1_loss, qf2_loss, policy_loss, alpha_loss, alpha_tlogs = agent.update_parameters(env, memory, args.batch_size)
            elif args.algo == 'ddpg':
                critic_loss, policy_loss = agent.update_parameters(env, memory, args.batch_size)

    # Train discriminator
    if not args.mse_reward:
        if episode % args.learning_term_dis == 0:
            d_fake, d_real, cost = env.dis.update(env, memory, args.batch_size)
    else:
        d_fake, d_real, cost = 0, 0, 0

    # Save networks
    if episode % args.save_term == 0:
        if not args.no_log:
            torch.save(agent.policy.state_dict(), f'{weight_path}/actor.pkl')
            torch.save(agent.critic.state_dict(), f'{weight_path}/critic.pkl')
            torch.save(env.dis.netD.state_dict(), f'{weight_path}/dis.pkl')

    # Save networks for history
    if episode % args.save_history_term == 0:
        if not args.no_log:
            torch.save(agent.policy.state_dict(), f'{weight_history_path}/actor{episode}.pkl')
            torch.save(agent.critic.state_dict(), f'{weight_history_path}/critic{episode}.pkl')
            torch.save(env.dis.netD.state_dict(), f'{weight_history_path}/dis{episode}.pkl')

    # Log
    if episode % args.logging_term == 0:
        logger.log('Train/mse', info['mse'], episode + 1)
        if args.algo == 'sac':
            logger.log('Loss/critic_loss', (qf1_loss + qf2_loss)/2, episode + 1)
            logger.log('Loss/alpha_loss', alpha_loss, episode + 1)
            logger.log('Loss/alpha', alpha_tlogs, episode + 1)
        else:
            logger.log('Loss/critic_loss', critic_loss, episode + 1)
        logger.log('Loss/actor_loss', policy_loss, episode + 1)
        logger.log('Loss/dis_fake', d_fake, episode + 1)
        logger.log('Loss/dis_real', d_real, episode + 1)
        logger.log('Loss/dis_cost', cost, episode + 1)
        logger.log('Loss/critic_lr', agent.get_critic_lr(), episode + 1)
        logger.log('Loss/policy_lr', agent.get_policy_lr(), episode + 1)
            
    # Print status
    if episode % args.printing_term == 0:
        mse = round(info['mse'], 4)
        print(f'Train | Ep {episode + 1} | MSE {mse} | Total Eps {episode * args.num_multi_env}')
    
    # Evaluate (save for the best)
    if (episode + 1) % args.eval_term == 0:
        evaluator.evaluate(env, agent, args, logger, episode + 1, weight_best_path)
        evaluator.evaluate_sp(envsp, agent, args, logger, episode + 1)
        print(f'current replay buffer size: {len(memory)}')