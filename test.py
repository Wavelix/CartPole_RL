import numpy as np
import os
import torch
import time
import sys
import gymnasium as gym

# 导入自定义模块
sys.path.append("d:/Projects/PythonProjects/CartPole_RL")
from config import MODEL_PATH, ENV_NAME
from envs.cartpole_env import CartPoleEnvWrapper
from models.dqn import DQNAgent

def test(render=True, episodes=10):
    """测试训练好的DQN智能体"""
    
    # 创建环境
    if render:
        # 使用自定义包装环境并添加render_mode参数
        env = CartPoleEnvWrapper(ENV_NAME, pos_penalty_weight=0.3)
        # 获取原始环境并设置渲染模式
        env.env = gym.make("InvertedPendulum-v4", render_mode="human")
    else:
        env = CartPoleEnvWrapper(ENV_NAME, pos_penalty_weight=0.3)
    
    # 获取状态和动作空间大小
    state_size = env.observation_space.shape[0]
    action_size = env.n_actions  # 现在这个属性存在了
    
    # 创建DQN智能体
    agent = DQNAgent(state_size, action_size)
    
    # 加载训练好的模型
    if os.path.exists(MODEL_PATH):
        agent.load(MODEL_PATH)
        print(f"加载模型: {MODEL_PATH}")
    else:
        print(f"错误: 找不到模型: {MODEL_PATH}")
        return
    
    # 测试智能体
    total_rewards = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            if render:
                env.render()
                time.sleep(0.01)  # 减缓渲染速度以便观察
            
            # 选择动作 (测试时不需要探索)
            action = agent.act(state, epsilon=0.0)
            
            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 更新状态和奖励
            state = next_state
            episode_reward += reward
        
        total_rewards.append(episode_reward)
        print(f"Episode {episode+1}/{episodes}, Reward: {episode_reward:.2f}")
    
    # 打印测试结果
    print(f"Average Reward over {episodes} episodes: {np.mean(total_rewards):.2f}")
    
    # 关闭环境
    env.close()

if __name__ == "__main__":
    test(render=True, episodes=5)