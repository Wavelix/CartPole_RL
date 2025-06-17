import numpy as np
import gymnasium as gym

class CartPoleEnvWrapper:
    """
    倒立摆环境的包装器，用于添加小车位置约束的奖励
    """
    
    def __init__(self, env_name="InvertedPendulum-v4", pos_penalty_weight=0.3, render_mode=None):
        self.env = gym.make(env_name, render_mode=render_mode)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.pos_penalty_weight = pos_penalty_weight  # 小车位置惩罚权重
        
        # 判断是否是离散动作空间
        if isinstance(self.action_space, gym.spaces.Box):
            # 将连续动作空间离散化
            self.is_discrete = False
            self.n_actions = 9  # 定义9个离散动作：-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1
            self.discrete_actions = np.linspace(-1, 1, self.n_actions)
        else:
            self.is_discrete = True
            self.n_actions = self.action_space.n
    
    def reset(self):
        """重置环境"""
        obs, info = self.env.reset()
        return obs, info
    
    def step(self, action):
        """执行动作并返回结果"""
        if not self.is_discrete:
            # 将离散动作索引转换为连续动作值
            action = np.array([self.discrete_actions[action]])
        
        # 执行动作
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 添加对小车位置的惩罚(假设小车位置是观测值的第一个元素)
        cart_pos = obs[0]
        pos_penalty = self.pos_penalty_weight * np.abs(cart_pos)
        
        # 修改奖励以包含位置惩罚
        reward = reward - pos_penalty
        
        # 如果小车位置超出阈值，则终止episode
        if np.abs(cart_pos) > 2.4:  # 通常倒立摆的位置阈值
            terminated = True
            reward -= 10  # 额外惩罚
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        """渲染环境"""
        return self.env.render()
    
    def close(self):
        """关闭环境"""
        self.env.close()