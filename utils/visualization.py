import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import animation
import gymnasium as gym
import sys

sys.path.append("d:/Projects/PythonProjects/CartPole_RL")
from models.dqn import DQNAgent
from envs.cartpole_env import CartPoleEnvWrapper
from config import MODEL_PATH

def save_frames_as_gif(frames, filename='cartpole_animation.gif'):
    """将帧保存为GIF动画"""
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)
    patch = plt.imshow(frames[0])
    plt.axis('off')
    
    def animate(i):
        patch.set_data(frames[i])
    
    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    anim.save(filename, writer='pillow', fps=30)
    plt.close()

def visualize_agent_behavior(env_name="InvertedPendulum-v4", model_path=MODEL_PATH, num_episodes=1):
    """可视化智能体在环境中的行为"""
    # 创建环境
    env = gym.make(env_name, render_mode="rgb_array")
    env_wrapped = CartPoleEnvWrapper(env_name)
    
    # 获取状态和动作空间大小
    state_size = env_wrapped.observation_space.shape[0]
    action_size = env_wrapped.n_actions
    
    # 创建DQN智能体
    agent = DQNAgent(state_size, action_size)
    agent.load(model_path)
    
    frames = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        
        while not done:
            # 渲染环境
            frames.append(env.render())
            
            # 选择动作
            action = agent.act(state, epsilon=0.0)
            
            # 将离散动作转换为连续动作
            continuous_action = np.array([env_wrapped.discrete_actions[action]])
            
            # 执行动作
            state, reward, terminated, truncated, _ = env.step(continuous_action)
            done = terminated or truncated
    
    env.close()
    
    # 保存为GIF
    save_frames_as_gif(frames)
    print(f"Animation saved as cartpole_animation.gif")

if __name__ == "__main__":
    visualize_agent_behavior()