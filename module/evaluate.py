import math
import numpy as np
import torch
import cv2
import os
from module.measure import calculate_ssim, calculate_psnr

class Evaluator:
    def __init__(self):
        self.best_mse = math.inf

    # Evaluation, no choice for materials (materials are randomly sampled at every timestep)
    def evaluate(self, env, agent, args, logger, episode, weight_best_path):
        episode_rewards_dis = []
        episode_rewards_mse = []
        episode_rewards_total = []
        mses = []
        ssims = []
        psnrs = []

        for eval_episode in range(args.eval_episodes):
            obs = env.reset(mode='eval')
            while True:
                action = agent.select_action(obs, evaluate=True)
                next_obs, reward, done, info = env.step(action, mode='eval', calculate_rewards=True)
                obs = next_obs
                
                if done.all():
                    break
                
            episode_rewards_dis.append(info['episode_reward_dis'])
            episode_rewards_mse.append(info['episode_reward_mse'])
            episode_rewards_total.append(info['episode_reward_total'])
            mses.append(info['mse'])
            ssims.append(calculate_ssim(obs[:, :3, :, :], obs[:, 3:6, :, :]).mean().item())
            psnrs.append(calculate_psnr(obs[:, :3, :, :], obs[:, 3:6, :, :]).item())

        mean_episode_reward_dis = np.mean(episode_rewards_dis)
        mean_episode_reward_mse = np.mean(episode_rewards_mse)
        mean_episode_reward_total = np.mean(episode_rewards_total)
        mean_mse = np.mean(mses)
        mean_ssim = np.mean(ssims)
        mean_psnr = np.mean(psnrs)

        logger.log('Eval/episode_reward(dis)', mean_episode_reward_dis, episode)
        logger.log('Eval/episode_reward(mse)', mean_episode_reward_mse, episode)
        logger.log('Eval/episode_reward(total)', mean_episode_reward_total, episode)

        logger.log('Eval/mse', mean_mse, episode)
        logger.log('Eval/ssim', mean_ssim, episode)
        logger.log('Eval/psnr', mean_psnr, episode)

        # Only log an image of the final episode of this evaluation
        logger.log('Eval/canvas', np.concatenate([env.canvas[0], np.ones([args.target_height, 10, 3]), env.goal[0]], axis=1), episode, type='image')

        Rd = round(mean_episode_reward_dis, 2)
        Rm = round(mean_episode_reward_mse, 2)
        M = round(mean_mse, 2)
        Rt = round(mean_episode_reward_total, 2)
        
        print(f'Eval | Ep {episode} | R(d) {Rd} | R(m) {Rm} | R(t) {Rt} | M {M}')

        if mean_mse < self.best_mse:
            self.best_mse = mean_mse
            if weight_best_path is not None:
                torch.save(agent.policy.state_dict(), f'{weight_best_path}/actor.pkl')
                torch.save(agent.critic.state_dict(), f'{weight_best_path}/critic.pkl')
                torch.save(env.dis.netD.state_dict(), f'{weight_best_path}/dis.pkl')


    # Evaluation with source pool
    def evaluate_sp(self, env, agent, args, logger, episode, save_result=None):
        in_channels = 12
        num_actions = 11
            
        episode_rewards_dis = []
        episode_rewards_mse = []
        episode_rewards_total = []
        mses = []
        ssims = []
        psnrs = []

        for eval_episode in range(args.eval_episodes):
            obs = env.reset(mode='eval')
            while True:
                obs = obs.squeeze()
                if args.model_based:
                    with torch.no_grad():
                        action = agent.select_action(obs, evaluate=True)  # 각 obs에 대한 행동을 구함
                    next_obs, info = env.independent_step(obs, action, nondiff=True, maintain_source=True)
                    reward = env.get_reward(obs[:, :3, :, :], next_obs[:, :3, :, :], obs[:, 3:6, :, :], use_tensor=True).detach().cpu().numpy()
                    next_v = agent.get_value(next_obs)
                    q = reward + args.gamma * next_v
                    selected_source_idx = np.argmax(q)
                    selected_action = action[selected_source_idx:selected_source_idx+1]
                    next_obs, reward, done, info = env.step(selected_action, selected_source_idx, mode='eval', calculate_rewards=True)
                else:
                    with torch.no_grad():
                        action = agent.select_action(obs, evaluate=True)
                    q = agent.get_action_value(obs.view(-1, in_channels, args.height, args.width), action, use_tensor=True)
                    selected_source_idx = np.argmax(q)
                    next_obs, reward, done, info = env.step(action[selected_source_idx:selected_source_idx+1], selected_source_idx, mode='eval', calculate_rewards=True)
            
                obs = next_obs.squeeze()
                
                if done.all():
                    break
                
            episode_rewards_dis.append(info['episode_reward_dis'])
            episode_rewards_mse.append(info['episode_reward_mse'])
            episode_rewards_total.append(info['episode_reward_total'])
            mses.append(info['mse'])
            ssims.append(calculate_ssim(obs[:, :3, :, :], obs[:, 3:6, :, :]).mean().item())
            psnrs.append(calculate_psnr(obs[:, :3, :, :], obs[:, 3:6, :, :]).item())

            if save_result is not None:
                for j in range(len(env.canvas)):
                    for i in range(10000):
                        result_path = save_result + f'/result_{str(i).zfill(4)}.jpg'
                        goal_path = save_result + f'/goal_{str(i).zfill(4)}.jpg'
                        compare_path = save_result + f'/compare_{str(i).zfill(4)}.jpg'
                        if not os.path.isfile(result_path):
                            break
                    cv2.imwrite(result_path, cv2.cvtColor(np.uint8(env.canvas[j]*255), cv2.COLOR_BGR2RGB))
                    cv2.imwrite(goal_path, cv2.cvtColor(np.uint8(env.goal[j]*255), cv2.COLOR_BGR2RGB))
                    cv2.imwrite(compare_path, cv2.cvtColor(np.uint8(np.concatenate([env.canvas[j], np.ones([args.target_height, 10, 3]), env.goal[j]], axis=1)*255), cv2.COLOR_BGR2RGB))

        mean_episode_reward_dis = np.mean(episode_rewards_dis)
        mean_episode_reward_mse = np.mean(episode_rewards_mse)
        mean_episode_reward_total = np.mean(episode_rewards_total)
        mean_mse = np.mean(mses)
        mean_ssim = np.mean(ssims)
        mean_psnr = np.mean(psnrs)

        logger.log('EvalSP/episode_reward(dis)', mean_episode_reward_dis, episode)
        logger.log('EvalSP/episode_reward(mse)', mean_episode_reward_mse, episode)
        logger.log('EvalSP/episode_reward(total)', mean_episode_reward_total, episode)

        logger.log('EvalSP/mse', mean_mse, episode)
        logger.log('EvalSP/ssim', mean_ssim, episode)
        logger.log('EvalSP/psnr', mean_psnr, episode)

        logger.log('EvalSP/canvas', np.concatenate([env.canvas[0], np.ones([args.target_height, 10, 3]), env.goal[0]], axis=1), episode, type='image')  # 마지막 eval episode의 결과만 로깅 (이미지 통합, 중간에 10px 흰색 구분선 포함)
        # logger.log('EvalSP/canvas', env.canvas, episode, type='image')  # 마지막 eval episode의 결과만 로깅
        # logger.log('EvalSP/goal', env.goal, episode, type='image')  # 마지막 eval episode의 결과만 로깅

        Rd = round(mean_episode_reward_dis, 2)
        Rm = round(mean_episode_reward_mse, 2)
        M = round(mean_mse, 4)
        Rt = round(mean_episode_reward_total, 2)
        
        print(f'EvalSP | Ep {episode} | R(d) {Rd} | R(m) {Rm} | R(t) {Rt} | M {M}')