import numpy as np
import time
from strategy import Strategy
from unit_manager import UnitManager
from task_manager import TaskManager
from tactics import Tactics
from environment import EnvironmentModel
from utils import init_debug, debug

class Agent:  

  def __init__(self, player: str, env_cfg):
    init_debug()
    self.team_id = 0 if player == "player_0" else 1
    self.opp_team_id = 1 if self.team_id == 0 else 0
    self.env = EnvironmentModel(self, env_cfg)
    self.unitman = UnitManager(self, env_cfg)
    self.taskman = TaskManager(self, env_cfg)
    self.tactics = Tactics(self, env_cfg)
    self.strategy = Strategy(self, env_cfg)
    self.tot_time = 0

  def update(self, obs, step):
    self.env.update(obs, step)
    self.unitman.update(obs)
    self.taskman.update(obs)
    self.tactics.update(obs)
    self.strategy.update(obs)

  def actions(self):
    live_units = self.unitman.live_units()
    self.taskman.assign_tasks(live_units)
    actions = dict()
    for uid in self.unitman.live_units():
      actions[uid] = self.unitman.my_unit(uid).action()
    self.tactics.last_actions = actions
    return actions

  def act(self, step: int, obs, remainingOverageTime: int = 60):
    debug(f"Step {step}. Time {int(self.tot_time)} / {remainingOverageTime}\n")
    self.time = remainingOverageTime
    tstart = time.time()
    self.update(obs, step)
    actions = np.zeros((self.env.max_units, 3), dtype=int)
    if not self.env.is_last_step(0): # skip last step
      unit_actions = self.actions()
      for uid in unit_actions:
        actions[uid] = unit_actions[uid]
    tend = time.time()
    debug(f"time={tend - tstart}\n")
    self.tot_time += (tend - tstart)
    return actions
