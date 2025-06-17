import numpy as np
import random
from collections import deque
import torch

class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, buffer_size, device):
        self.buffer = deque(maxlen=buffer_size)
        self.device = device
        
    def add(self, state, action, reward, next_state, done):
        """添加经验到缓冲区"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """从缓冲区中随机采样批量经验"""
        experiences = random.sample(self.buffer, batch_size)
        
        # 转换为PyTorch张量
        states = torch.tensor(np.array([e[0] for e in experiences]), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.array([e[1] for e in experiences]), dtype=torch.int64, device=self.device)
        rewards = torch.tensor(np.array([e[2] for e in experiences]), dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.array([e[3] for e in experiences]), dtype=torch.float32, device=self.device)
        dones = torch.tensor(np.array([e[4] for e in experiences]), dtype=torch.float32, device=self.device)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """返回缓冲区中经验的数量"""
        return len(self.buffer)