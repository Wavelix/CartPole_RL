import torch

# 环境参数
ENV_NAME = "InvertedPendulum-v4"  # MuJoCo环境名称
MAX_STEPS = 1000                 # 每个episode的最大步数
REWARD_THRESHOLD = 950           # 认为训练成功的奖励阈值

# DQN参数
GAMMA = 0.99                     # 折扣因子
BATCH_SIZE = 64                  # 批量大小
BUFFER_SIZE = 100000             # 经验回放缓冲区大小
MIN_REPLAY_SIZE = 1000           # 开始学习前的最小经验数量
EPSILON_START = 1.0              # 起始探索率
EPSILON_END = 0.05               # 最终探索率
EPSILON_DECAY = 10000            # 探索率衰减步数
TARGET_UPDATE_FREQ = 1000        # 目标网络更新频率

# 训练参数
NUM_EPISODES = 1000              # 训练的episode数量
LEARNING_RATE = 1e-4             # 学习率
CLIP_GRAD = 1.0                  # 梯度裁剪阈值
EVAL_FREQ = 100                  # 评估频率

# 设备配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型保存路径
MODEL_PATH = "saved_models/dqn_cartpole.pt"