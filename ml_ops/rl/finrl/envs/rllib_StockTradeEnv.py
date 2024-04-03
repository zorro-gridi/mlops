from __future__ import annotations
import logging
from pathlib import Path

from typing import List

import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gymnasium import spaces
from gymnasium.utils import seeding
from stable_baselines3.common.vec_env import DummyVecEnv
from ray.rllib.env import EnvContext

import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
dir_list = [dname for dname in current_dir.as_posix().split('/')]
home_dir = '/'.join([dirname for dirname in dir_list[:dir_list.index('zorro')+1]])
env_path = '/'.join([dirname for dirname in dir_list[:dir_list.index('pycharm')+1]])
sys.path.append(env_path)

from mlops.ml_ops.rl.finrl.envs.rllib_BaseTradeEnv import BaseTradeEnv


matplotlib.use("Agg")


"""
@Author: Zorro
@Date: 2024-01-01
@Desc:
    本代码定义了基于 rllib 强化学习训练框架的 gym 环境模型
"""


class StockTradeEnv(BaseTradeEnv):
    """
    Desc:
        A stock trading environment for OpenAI gym
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, config: EnvContext):
        super().__init__(config)
        # 最终方案 -> action_space: MultiDiscrete, 多维离散空间
        self.action_space = spaces.MultiDiscrete([11] * self.stock_dim)


    def step(self, actions):
        # MultiDiscrete start 参数在实际运行中不起作用，需要手动调节 actions
        actions = actions - 5
        return super().step(actions)