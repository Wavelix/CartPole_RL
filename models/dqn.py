import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math
import sys
sys.path.append("d:/Projects/PythonProjects/CartPole_RL")
from config import DEVICE, GAMMA, LEARNING_RATE, CLIP_GRAD

class QNetwork(nn.Module):
    """Q网络：用于近似Q值函数"""
    
    def __init__(self, state_size, action_size, hidden_size=64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
    
    def forward(self, state):
        """前向传播"""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    """DQN智能体"""
    
    def __init__(self, state_size, action_size, hidden_size=64):
        self.state_size = state_size
        self.action_size = action_size
        
        # 创建Q网络和目标网络
        self.q_network = QNetwork(state_size, action_size, hidden_size).to(DEVICE)
        self.target_network = QNetwork(state_size, action_size, hidden_size).to(DEVICE)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # 目标网络不需要梯度
        
        # 优化器
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=LEARNING_RATE)
        
        # 训练步数计数器
        self.t_step = 0
        
    def act(self, state, epsilon=0.):
        """根据当前状态选择动作"""
        # 探索：随机选择动作
        if np.random.random() < epsilon:
            return np.random.randint(self.action_size)
        
        # 利用：选择Q值最大的动作
        state = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        self.q_network.eval()
        with torch.no_grad():
            action_values = self.q_network(state)
        self.q_network.train()
        return torch.argmax(action_values).item()
    
    def learn(self, experiences, target_update_freq):
        """从经验中学习更新Q网络"""
        states, actions, rewards, next_states, dones = experiences
        
        # 获取目标Q值
        self.target_network.eval()
        with torch.no_grad():
            target_q_values = self.target_network(next_states)
            max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]
            targets = rewards.unsqueeze(1) + GAMMA * max_target_q_values * (1 - dones.unsqueeze(1))
        
        # 获取当前Q值
        q_values = self.q_network(states)
        actions_q_values = q_values.gather(dim=1, index=actions.unsqueeze(1))
        
        # 计算损失并更新网络
        loss = F.mse_loss(actions_q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), CLIP_GRAD)
        
        self.optimizer.step()
        
        # 更新目标网络
        self.t_step += 1
        if self.t_step % target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            
        return loss.item()
    
    def save(self, path):
        """保存模型"""
        torch.save(self.q_network.state_dict(), path)
    
    def load(self, path):
        """加载模型"""
        self.q_network.load_state_dict(torch.load(path, map_location=DEVICE))
        self.target_network.load_state_dict(self.q_network.state_dict())