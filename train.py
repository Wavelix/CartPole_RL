import numpy as np
import os
import torch
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import gymnasium as gym

# 导入自定义模块
sys.path.append("d:/Projects/PythonProjects/CartPole_RL")
from config import *
from envs.cartpole_env import CartPoleEnvWrapper
from models.dqn import DQNAgent
from utils.replay_buffer import ReplayBuffer

def train():
    """训练DQN智能体"""
    
    # 创建保存模型的目录
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    
    # 创建环境
    env = CartPoleEnvWrapper(ENV_NAME)
    
    # 获取状态和动作空间大小
    state_size = env.observation_space.shape[0]
    action_size = env.n_actions
    
    # 创建DQN智能体
    agent = DQNAgent(state_size, action_size)
    
    # 创建经验回放缓冲区
    memory = ReplayBuffer(BUFFER_SIZE, DEVICE)
    
    # 训练统计
    rewards_history = []
    losses = []
    epsilons = []
    
    # 填充经验回放缓冲区
    state, _ = env.reset()
    for _ in range(MIN_REPLAY_SIZE):
        action = np.random.randint(action_size)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        memory.add(state, action, reward, next_state, done)
        
        if done:
            state, _ = env.reset()
        else:
            state = next_state
    
    # 主训练循环
    state, _ = env.reset()
    episode_reward = 0
    episode_count = 0
    
    progress_bar = tqdm(range(NUM_EPISODES * MAX_STEPS), desc="Training")
    
    for step in range(NUM_EPISODES * MAX_STEPS):
        # 计算epsilon（探索率）
        epsilon = max(EPSILON_END, EPSILON_START - (step / EPSILON_DECAY) * (EPSILON_START - EPSILON_END))
        epsilons.append(epsilon)
        
        # 选择动作
        action = agent.act(state, epsilon)
        
        # 执行动作
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # 存储经验
        memory.add(state, action, reward, next_state, done)
        
        # 更新状态和奖励
        state = next_state
        episode_reward += reward
        
        # 从经验回放缓冲区中学习
        if len(memory) >= BATCH_SIZE:
            experiences = memory.sample(BATCH_SIZE)
            loss = agent.learn(experiences, TARGET_UPDATE_FREQ)
            losses.append(loss)
        
        # 如果episode结束
        if done:
            rewards_history.append(episode_reward)
            
            # 打印训练统计
            progress_bar.set_postfix({
                "episode": episode_count, 
                "reward": episode_reward, 
                "epsilon": epsilon
            })
            
            # 保存最佳模型
            if len(rewards_history) > 100 and np.mean(rewards_history[-100:]) > REWARD_THRESHOLD:
                print(f"\nEnvironment solved in {episode_count} episodes!")
                agent.save(MODEL_PATH)
                break
                
            if episode_count % EVAL_FREQ == 0:
                agent.save(MODEL_PATH)
            
            # 重置环境
            state, _ = env.reset()
            episode_reward = 0
            episode_count += 1
        
        progress_bar.update(1)
    
    # 关闭环境
    env.close()
    
    # 绘制训练结果
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(rewards_history)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    plt.subplot(3, 1, 2)
    plt.plot(losses)
    plt.title('Loss')
    plt.xlabel('Training step')
    plt.ylabel('Loss')
    
    plt.subplot(3, 1, 3)
    plt.plot(epsilons)
    plt.title('Epsilon')
    plt.xlabel('Training step')
    plt.ylabel('Epsilon')
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()

if __name__ == "__main__":
    train()