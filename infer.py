import argparse
import os
import sys
import cv2
import copy
import io
import base64
import time
import scipy.stats as st
from tqdm import tqdm
from env.draw import Drawer
from env.collageSP import CollageSPEnv
from env.utils import sliding_window, num_sliding_windows, shuffled_sliding_window, tensor
from agent.sac import SAC
from agent.ddpg import DDPG
from model.gan import GAN
from module.gpu import *  # Set device here
from module.evaluate import Evaluator
from module.logger import Logger
from module.measure import get_complexity_heatmap, calculate_complexity


parser = argparse.ArgumentParser(description='Collage Training Arguments')
parser.add_argument('--algo', type=str, default='sac')  # sac | ddpg
parser.add_argument('--eval', type=bool, default=True)
parser.add_argument('--gamma', type=float, default=0.95)
parser.add_argument('--tau', type=float, default=0.005)
parser.add_argument('--lr', type=float, default=0.0003)
parser.add_argument('--alpha', type=float, default=0.2)
parser.add_argument('--automatic_entropy_tuning', action='store_true')
parser.add_argument('--seed', type=int, default=123456)
parser.add_argument('--update_steps', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--hidden_size', type=int, default=128)
parser.add_argument('--model_based', action='store_true')
parser.add_argument('--num_multi_env', type=int, default=16)
parser.add_argument('--noop', action='store_true')
parser.add_argument('--mse_reward', action='store_true')
parser.add_argument('--dis', type=str, default='wgan')  # wgan | sngan | gngan
parser.add_argument('--scale', type=str, default='small')
parser.add_argument('--target_width', type=int, default=224)
parser.add_argument('--target_height', type=int, default=224)
parser.add_argument('--width', type=int, default=224)
parser.add_argument('--height', type=int, default=224)
parser.add_argument('--wmin', type=float, default=0.05)
parser.add_argument('--wmax', type=float, default=0.3)
parser.add_argument('--hmin', type=float, default=0.05)
parser.add_argument('--hmax', type=float, default=0.3)
parser.add_argument('--learned_max_steps', type=int, default=10)
parser.add_argument('--initial_canvas', type=str, default='random')
parser.add_argument('--num_cycles', type=int)
parser.add_argument('--num_steps', type=int, default=1)
parser.add_argument('--num_episodes', type=int, default=1e6)
parser.add_argument('--learning_term_agent', type=int, default=1)
parser.add_argument('--learning_term_dis', type=int, default=1)
parser.add_argument('--target_update_term', type=int, default=1)
parser.add_argument('--eval_episodes', type=int, default=5)
parser.add_argument('--eval_term', type=int, default=1e2)
parser.add_argument('--save_term', type=int, default=1e3)
parser.add_argument('--logging_term', type=int, default=1)
parser.add_argument('--printing_term', type=int, default=1)
parser.add_argument('--no_log', action='store_true')
parser.add_argument('--goal', type=str)
parser.add_argument('--source_dir', type=str)  # Folder path for material images
parser.add_argument('--source_load_limit', type=int)  # Maximum number of original material images to load
parser.add_argument('--source_sample_size', type=int, default=30)  # Subset size for material candidates
parser.add_argument('--min_source_complexity', type=int)  # Discard materials with complexity over this threshold
parser.add_argument('--weight_path', type=str, default='.')
parser.add_argument('--window_ratio', type=float, default=0.5)
parser.add_argument('--goal_resolution', type=int)  # Goal image size
parser.add_argument('--goal_resolution_fit', type=str, default='horizontal')  # Set how to adjust the goal image size ratio. (horizontal | vertical | square)
parser.add_argument('--source_resolution_ratio', type=float, default=1.0)  # Set material image size resolution ratio (<= 1.0) (the bigger, the more close-up for materials)
parser.add_argument('--skip_negative_reward', action='store_true')  # Activate skipping if MSE reward < 0
parser.add_argument('--paper_like', action='store_true')  # Activate teared paper effect
parser.add_argument('--complexity_aware', action='store_true')  # Activate complexity-aware collage
parser.add_argument('--disallow_duplicate', action='store_true')  # Activate non-duplicate material usage
parser.add_argument('--scale_order', nargs='+')
parser.add_argument('--min_scrap_size', type=float)  # Set minimum scrap size to limit scraps that are too small
parser.add_argument('--sensitivity', type=float)  # The level of low-complexity part abstraction (the bigger, the more skipping low-complexity parts)
parser.add_argument('--fixed_t', type=float, default=9)  # Fixed t for custom t_channel
parser.add_argument('--video_fps', type=int, default=30)  # FPS for sequence video

args = parser.parse_args()
args.white_start = True  # Start from white canvas
args.no_log = True  # No log for testing

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

# Prepare the test
print('Preparing Test')
dis = GAN(args.dis, args.width, args.height, device)
dis.load(args.weight_path)
dis.netD.eval()
coord = torch.zeros([1, 2, args.width, args.height])
for i in range(args.width):
    for j in range(args.height):
        coord[0, 0, i, j] = i / float(args.width-1)
        coord[0, 1, i, j] = j / float(args.height-1)
coord = coord.to(device)
in_channels = 12
num_actions = 11
if args.noop:
    num_actions += 1
if args.algo == 'sac':
    agent = SAC(in_channels=in_channels, num_actions=num_actions, args=args, device=device, coord=coord)
elif args.algo == 'ddpg':
    agent = DDPG(in_channels=in_channels, num_actions=num_actions, args=args, device=device, coord=coord)
agent.policy.load_state_dict(torch.load(f'{args.weight_path}/actor.pkl'))
agent.policy.eval()
agent.critic.load_state_dict(torch.load(f'{args.weight_path}/critic.pkl'))
agent.critic.eval()
logger = Logger('Collage', args, args.no_log)
evaluator = Evaluator()

# Build env
print('Building Env')
drawer = Drawer(args, device)
env = CollageSPEnv(args, drawer, device, dis=dis, coord=coord)

# Configurations
scale_order = list(map(int, args.scale_order))
goal_path = args.goal
goal_name = goal_path.split('/')[-1].split('.')[0]
sensitivity = args.sensitivity
source_name = args.source_dir.split('/')[-1]

# Make results saving paths
result_dir = f'{goal_name}{args.goal_resolution}{args.goal_resolution_fit}_'
for scale in args.scale_order:
    result_dir += f'{scale}'
    if int(scale) != len(args.scale_order)-1:
        result_dir += '__'
result_dir += f'_{source_name}x{args.source_resolution_ratio}_sensitivity{sensitivity}_cycle{args.num_cycles}_fixt{args.fixed_t}_mcskip{args.min_source_complexity}'
if args.disallow_duplicate:
    result_dir += '_nodup'
if not args.paper_like:
    result_dir += '_noline'
result_path = f'samples/results/{result_dir}'
sequence_path = result_path + '/sequence'
os.makedirs(sequence_path, exist_ok=True)

# Load target image
full_goal = cv2.imread(goal_path)
goal_resolution = args.goal_resolution
if args.goal_resolution_fit == 'square':
    goal_resolution = (goal_resolution, goal_resolution)
elif args.goal_resolution_fit == 'horizontal':
    goal_resolution = (goal_resolution, int(goal_resolution / full_goal.shape[1] * full_goal.shape[0]))
elif args.goal_resolution_fit == 'vertical':
    goal_resolution = (int(goal_resolution / full_goal.shape[0] * full_goal.shape[1]), goal_resolution)
full_goal = cv2.cvtColor(cv2.resize(full_goal, goal_resolution, cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB)/255

# Make initial canvas
full_canvas = np.ones_like(full_goal)

# Counters & other info gatherers
total_steps = 0  # count all steps including skipped steps
total_valid_steps = 0  # count steps without skipped steps
scale_final_steps = []  # count valid steps per scale
skip_counter = {  # count skipped steps with reasons
    'low_complexity': 0,
    'negative_reward': 0,
    'too_small_scrap_size': 0,
    'noop': 0,
}
scale_step = -1
start_time = time.time()
end_times = []

# Now begin the collage
print('Collage started')
# Perform multi-scale collage following assigned scale order
for ss, (target_width, target_height) in enumerate(zip(scale_order, scale_order)):
    # Skip the scale if the actual scale of the goal image is smaller than current assigned scale
    if target_width > full_goal.shape[1] or target_height > full_goal.shape[0]:
        continue
    else:
        scale_step += 1

    # Set env configuration as current assigned scale
    env.target_width, env.target_height = target_width, target_height

    # Set sliding window
    step_size = int(target_width*args.window_ratio)
    window_size = (target_height, target_width)

    # Load materials
    sources = []
    print(f'Loading materials for scale {ss+1}/{len(scale_order)}↓')
    names = os.listdir(args.source_dir)
    image_names = [name for name in names if name.split('.')[-1] in ['jpg', 'png', 'JPG', 'PNG']]
    np.random.shuffle(image_names)
    for img_cnt, full_source_path in enumerate(tqdm(image_names)):
        if img_cnt > args.source_load_limit:
            break
        # Divide materials into the network input size
        full_source = cv2.imread(f'{args.source_dir}/{full_source_path}')
        source_target_resolution = (int(full_source.shape[1]*args.source_resolution_ratio), int(full_source.shape[0]*args.source_resolution_ratio))
        full_source = cv2.cvtColor(cv2.resize(full_source, source_target_resolution, cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB)/255
        for _, _, source_part in sliding_window(full_source, window_size[0], window_size, no_dup=True):
            # Discard low complexity materials
            if calculate_complexity(source_part, use_tensor=False) < args.min_source_complexity:
                continue
            sources.append(source_part)
    sources = np.array(sources)
    num_divided_goals = num_sliding_windows(full_goal, step_size, window_size)
    if args.disallow_duplicate:
        used_source_ids = []

    # Print scale information
    print(f'+-Information (scale {ss+1}/{len(scale_order)})-------------------------------+')
    print(f'| %-30s%-24s|' % ('Goal image:', goal_path.split('/')[-1]))
    print(f'| %-30s%-24s|' % ('Goal size:', f'{full_goal.shape[0]} x {full_goal.shape[1]}'))
    print(f'| %-30s%-24s|' % ('Divided size -> Input size:', f'{target_width} -> {args.width}'))
    print(f'| %-30s%-24s|' % ('Divided goals:', f'{num_divided_goals}'))
    print(f'| %-30s%-24s|' % ('Available sources:', f'{len(image_names)}'))
    print(f'| %-30s%-24s|' % ('Divided sources:', f'{len(sources)}'))
    print(f'| %-30s%-24s|' % ('Steps per section:', f'{args.source_sample_size}'))
    print(f'| %-30s%-24s|' % ('Maximum total steps:', f'{args.source_sample_size * num_divided_goals}'))
    print(f'| %-30s%-24s|' % ('Skip negative reward:', f'{args.skip_negative_reward}'))
    print('+-------------------------------------------------------+')

    # Calculate goal image complexities at current scale
    if args.complexity_aware:
        complexity_heatmap = get_complexity_heatmap(full_goal, step_size, window_size, standard=True)

        # Assign the number of cycles for each sub-canvas (higher the sub-goal image complexity is, higher the number of cycles assigned to corresponding sub-canvas)
        cycle_limit_map = []
        for complexity in complexity_heatmap.reshape(-1):
            if scale_step >= 1:  # apply after second scale order
                cycle_limit_map.append(int(round(args.num_cycles * st.norm.cdf(complexity) ** sensitivity)))
            else:
                cycle_limit_map.append(copy.deepcopy(args.num_cycles))

        cycle_count = [0 for i in range(len(complexity_heatmap.reshape(-1)))]

    # Start cylcles at current scale
    for cycle in range(args.num_cycles):
        print(f'Scale {ss+1}/{len(scale_order)}, Cycle {cycle+1}/{args.num_cycles}↓')
        
        # Start work on (sub-canvas, sub-goal) pairs following random shuffled sliding window sequence
        for part_pair in tqdm(shuffled_sliding_window(full_canvas, full_goal, step_size, window_size), total=num_divided_goals): # int(full_canvas.shape[0]*full_canvas.shape[1]/step_size/step_size):
            
            total_steps += 1
            
            part_id, x, y, canvas_part, goal_part = part_pair
                
            if y + window_size[0] < full_canvas.shape[0]:
                if x + window_size[1] < full_canvas.shape[1]:
                    canvas_part = full_canvas[y:y + window_size[0], x:x + window_size[1]]
                else:
                    canvas_part = full_canvas[y:y + window_size[0], full_canvas.shape[1] - window_size[1]:]
            else:
                if x + window_size[1] < full_canvas.shape[1]:
                    canvas_part = full_canvas[full_canvas.shape[0] - window_size[0]:, x:x + window_size[1]]
                else:
                    canvas_part = full_canvas[full_canvas.shape[0] - window_size[0]:, full_canvas.shape[1] - window_size[1]:]
            
            # Skip if current cycle already reached the assigned number of cycles on current (sub-canvas, sub-goal)
            if args.complexity_aware:
                if cycle_count[part_id] == cycle_limit_map[part_id]:
                    skip_counter['low_complexity'] += 1
                    continue
                else:
                    cycle_count[part_id] += 1
                
            canvas_part = np.expand_dims(canvas_part, 0)
            goal_part = np.expand_dims(goal_part, 0)

            # Make material pool
            source_ids = np.array([i for i in range(len(sources))])
            sampled_ids = np.random.choice(source_ids, min(len(sources), args.source_sample_size), replace=False)
            if args.disallow_duplicate:
                for used_source_id in used_source_ids:
                    if len(source_pool) > 1:
                        sampled_ids = sampled_ids[sampled_ids != used_source_id]
            source_pool = sources[sampled_ids]
            
            # Get obs
            obs = env.reset(canvas=canvas_part, goal=goal_part, source_pool=source_pool, mode='eval')
            obs = obs.squeeze(0)
            
            # Material & action selection
            for step in range(1):
                # Adjust t_channel
                obs[:, 9:10] = torch.ones(len(source_pool), 1, env.height, env.width).to(device) * (args.learned_max_steps - args.fixed_t)/args.learned_max_steps

                with torch.no_grad():
                    action = agent.select_action(obs, evaluate=True)

                # Select material based on Q-value
                if args.model_based:
                    next_obs, info = env.independent_step(obs, action, nondiff=True, maintain_source=True)
                    reward = env.get_reward_dis(obs[:, :3, :, :], next_obs[:, :3, :, :], obs[:, 3:6, :, :], use_tensor=True).detach().cpu().numpy()
                    next_v = agent.get_value(next_obs)
                    q = reward + args.gamma * next_v
                    q = np.reshape(q, -1)
                    selected_source_idx = (-q).argsort()[0]  # select material with highest Q-value
                else:
                    q = agent.get_action_value(obs, action, use_tensor=True)
                    selected_source_idx = np.argmax(q)  # select material with highest Q-value

                if args.disallow_duplicate:
                    used_source_ids.append(sampled_ids[selected_source_idx])

                past_canvas = env.canvas
                
                selected_action = action[selected_source_idx:selected_source_idx+1]

                # Skip when the scrap size is too small
                _, shape = drawer.draw_nondiff(tensor(env.canvas), tensor(env.source_pool[selected_source_idx]), selected_action, target_width, target_height, args.wmin, args.wmax, args.hmin, args.hmax, device, rounding=True, shape=env.shape)
                scrap_size = shape.sum().item() / (target_width*target_height)
                if scrap_size < args.min_scrap_size:
                    skip_counter['too_small_scrap_size'] += 1
                    continue

                # Do the collage on current (sub-canvas, sub-goal) using selected material and action
                next_obs, reward, done, info = env.step(selected_action, selected_source_idx, calculate_rewards=True, paper_like=args.paper_like, disallow_duplicate=args.disallow_duplicate)

                # Skip when the MSE reward is 0 (if so, there was a noop action from the agent)
                if reward['mse'] == 0:
                    skip_counter['noop'] += 1
                    break
                
                # Skip when the MSE reward is negative
                if args.skip_negative_reward:
                    if reward['mse'] < 0:
                        skip_counter['negative_reward'] += 1
                        break
                
                total_valid_steps += 1

                # Update full_canvas
                obs = next_obs
                if y + window_size[0] < full_canvas.shape[0]:
                    if x + window_size[1] < full_canvas.shape[1]:
                        full_canvas[y:y+window_size[0], x:x+window_size[1]] = env.canvas
                    else:
                        full_canvas[y:y+window_size[0], full_canvas.shape[1]-window_size[1]:] = env.canvas
                else:
                    if x + window_size[1] < full_canvas.shape[1]:
                        full_canvas[full_canvas.shape[0]-window_size[0]:, x:x+window_size[1]] = env.canvas
                    else:
                        full_canvas[full_canvas.shape[0]-window_size[0]:, full_canvas.shape[1]-window_size[1]:] = env.canvas

                # Save collage sequence
                sequence_number = str.zfill(str(total_valid_steps), 4)
                sequence_fname = f'{sequence_number}.jpg'
                full_canvas = np.clip(full_canvas, 0, 1)
                # if total_valid_steps % 50 == 0:  # 간격
                cv2.imwrite(f'{sequence_path}/{sequence_fname}', cv2.cvtColor(np.uint8(full_canvas*255), cv2.COLOR_BGR2RGB))

        # One cycle ends
    # One scale ends
    scale_final_steps.append(total_valid_steps)

    end_times.append(time.time() - start_time)

    # Save scale progress
    goal_fname = goal_path.split('/')[-1].split('.')[0]
    result_path_scale = f'{result_path}/{goal_fname}_{source_name}_{scale_order[scale_step]}.jpg'
    full_canvas = np.clip(full_canvas, 0, 1)
    cv2.imwrite(result_path_scale, cv2.cvtColor(np.uint8(full_canvas*255), cv2.COLOR_BGR2RGB))
    
# Collage ends

# Make sequence video
frames = []
fps = 30
for fname in tqdm(sorted(os.listdir(sequence_path))):
    if fname[-3:] == 'jpg':
        frames.append(cv2.imread(f'{sequence_path}/{fname}'))
VIDEO_WIDTH = frames[0].shape[1]
VIDEO_HEIGHT = frames[0].shape[0]
vname = 'video'
video_name = f'{result_path}/video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(f'{result_path}/{vname}.mp4', fourcc, fps, (VIDEO_WIDTH, VIDEO_HEIGHT), 1)
for frame in frames:
    video.write(np.uint8(frame))
video.release()

# Print work info
print('\nCollage process done.')
print(f'Total steps: {total_steps}')
print(f'Total valid steps (actual pasted scraps): {total_valid_steps}')
print(f'Result saved at {result_path}')
print(f'Actual pasted scraps at each scale:')
print(scale_final_steps)
print('Skip reasons and counts↓')
for key, val in skip_counter.items():
    print(f'\t{key}: {val}')
print(f'Valid step rate: {round(100*(total_valid_steps/total_steps), 2)}%')
print(f'Cumulative times for each scale (sec):')
print(end_times)