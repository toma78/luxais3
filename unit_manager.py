import numpy as np
from random import choice
from task_manager import Task
from utils import debug, dir2str, pos_to_plus, pos_to_3x3, manhattan, max_distance

class Unit:

  def __init__(self, agent, id, pos, energy):
    self.agent = agent
    self.id = id
    self.pos = pos
    self.energy = energy
    self.task = None
    self.path = []
    self.last_pos = None
    self.last_energy = None
    self.old_task = None
    self.task_step = 0

  def __repr__(self):
    return f"<U{self.id} {self.pos} {self.energy}>"

  def set_task(self, task, path=[]):
    self.old_task = self.task
    self.task = task
    self.path = path
    if self.task is not None and (self.old_task is None or self.old_task != self.task):
      self.task_step = self.agent.env.step  

  def update_position(self, pos):
    self.last_pos = self.pos
    self.pos = pos

  def update_energy(self, energy):
    self.last_energy = self.energy
    self.energy = energy

  def action(self):
    # no task
    if self.task is None:
      return [ 0, 0, 0 ]
    # sap task
    if self.task.type == Task.SAP:
      return [ 5, self.task.pos[0], self.task.pos[1] ]
    # at destination
    if self.pos == self.task.pos:
      return [ 0, 0, 0 ]
    # go towards task destination
    if len(self.path) > 0:
      dirto = self.path.pop(0)
    else:
      dirto = 0
    return [ dirto, 0, 0 ]

class UnitManager:

  def __init__(self, agent, env_cfg):
    self.agent = agent
    self.units = (dict(), dict()) # unit ids for both players id -> Unit
    self.dead_units = set()
    self.vis_units_history = [] # visible units history 
    self.positions = (dict(), dict())

  def update(self, obs):
    if self.agent.env.is_reset_step():
      self.units = (dict(), dict())
      self.vis_units_history = []
      debug("unitman match reset\n")
    # update units
    unit_mask = np.array(obs["units_mask"]) # shape (2, max_units, )
    unit_positions = np.array(obs["units"]["position"]) # shape (2, max_units, 2)
    unit_energys = np.array(obs["units"]["energy"]) # shape (2, max_units, 1)
    available_unit_ids = [ np.where(unit_mask[0])[0], np.where(unit_mask[1])[0] ]
    visible_units = (set(), set())
    self.positions = (dict(), dict())
    for team_id in [ 0, 1 ]:
      for uid in available_unit_ids[team_id]:
        uid = int(uid)
        # track unit existance
        visible_units[team_id].add(uid)
        pos = int(unit_positions[team_id][uid][0]), int(unit_positions[team_id][uid][1]) 
        energy = int(unit_energys[team_id][uid]) 
        if uid not in self.units[team_id]:
          self.units[team_id][uid] = Unit(self.agent, uid, pos, energy)
        else:
          self.units[team_id][uid].update_position(pos)
          self.units[team_id][uid].update_energy(energy)
        self.positions[team_id][pos] = self.units[team_id][uid]
        # estimate nebula energy reduction (TODO: don't assume no void/sap loss)
        if not self.agent.env.nebula_enred_known:
          self._infer_ner(team_id, uid)
    # append to history
    self.vis_units_history.append(visible_units)
    # record casualties and remove them from units
    self._casualties()

  def _infer_ner(self, team_id, uid):
    unit = self.units[team_id][uid]
    if unit.last_pos is not None and unit.last_energy > 0 and unit.last_energy < 400 and unit.energy > 0 and unit.energy < 400:
      if unit.pos != unit.last_pos and (not self.agent.env.drift_detected or self.agent.env.drift_speed_known) and self.agent.env.is_last_nebula_tile(unit.pos):
        cost = self.agent.env.unit_move_cost - self.agent.env.tile_energy(unit.pos)          
        ner = (unit.last_energy - unit.energy) - cost
        self.agent.env.set_nebula_energy_reduction(ner)
      if unit.pos == unit.last_pos and (not self.agent.env.drift_detected or self.agent.env.drift_speed_known) and self.agent.env.is_last_nebula_tile(unit.pos):
        cost = -self.agent.env.tile_energy(unit.pos)
        ner = (unit.last_energy - unit.energy) - cost
        self.agent.env.set_nebula_energy_reduction(ner)

  def all_visible_units(self, team_id):
    return [ unit for unit in self.units[team_id].values() if unit.id in self.vis_units_history[-1][team_id] ]

  def visible_units_on(self, pos, team_id):
    return [ unit for unit in self.all_visible_units(team_id) if unit.pos == pos ]

  def visible_units_on_plus(self, pos, team_id):
    return [ unit for unit in self.all_visible_units(team_id) if unit.pos in pos_to_plus[pos] ]

  def visible_units_on_3x3(self, pos, team_id):
    return [ unit for unit in self.all_visible_units(team_id) if unit.pos in pos_to_3x3[pos] ]

  def my_unit(self, uid):
    return self.units[self.agent.team_id][uid]

  def opp_visible_units(self):
    return [ unit for unit in self.units[self.agent.opp_team_id].values() if unit.id in self.vis_units_history[-1][self.agent.opp_team_id] ]

  def opp_unit(self, uid):
    return self.units[self.agent.opp_team_id][uid]
  
  def live_units(self):
    return { unit.id for unit in self.all_visible_units(self.agent.team_id) if unit.energy >= 0 }

  def _casualties(self):
    # TODO: ignoring units respawning at the next turn
    self.dead_units = set()
    if len(self.vis_units_history) < 2:
      return     
    for uid in (self.vis_units_history[-2][self.agent.team_id] - self.vis_units_history[-1][self.agent.team_id]):
      u = self.units[self.agent.team_id].pop(uid)
      self.dead_units.add(u.pos)
      debug(f"{uid} dead\n")
    