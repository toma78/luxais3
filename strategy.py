import numpy as np
from task_manager import Task
from utils import debug, spawn_points, manhattan, center, pos_to_plus, clip, euclid, pos_to_5x5, mirror, pos_to_belt, pos_to_7x7, pos_to_9x9

class Strategy:


  def __init__(self, agent, env_cfg):
    self.agent = agent
    self.team_points = (0, 0)
    self.min_hg_energy = 8

  def update(self, obs):
    self.reward = obs["team_points"][self.agent.team_id] - self.team_points[self.agent.team_id]
    self.oppreward = obs["team_points"][self.agent.opp_team_id] - self.team_points[self.agent.opp_team_id]
    self.team_points = int(obs["team_points"][0]), int(obs["team_points"][1])
    self.team_wins = int(obs["team_wins"][0]), int(obs["team_wins"][1])
    debug(f"rew={self.reward} pts={self.team_points} win={self.team_wins}\n")
    # fragments confirmation
    self.abandon_fragments = False
    if self.reward > 0:
      self.abandon_fragments = self.agent.env.confirm_fragments(self.reward)
    elif self.reward == 0:
      self.agent.env.absent_fragments()
    if self.agent.env.is_reset_step():
      self.agent.tactics.oppmodel.match_reset()
    else:
      self.agent.tactics.oppmodel.update(self.oppreward)
    # divide fragments
    pfragments = self.agent.env.all_fragments()
    cfragments = self.agent.env.confirmed_fragments()
    self.all_confirmed = len(pfragments) == len(cfragments)
    self.myfrags = set()
    self.oppfrags = set()
    for f in cfragments:
      if manhattan(f, spawn_points[self.agent.team_id]) <= manhattan(f, spawn_points[self.agent.opp_team_id]):
        self.myfrags.add(f)
      else:
        self.oppfrags.add(f)    
    # determine match phase tasks
    self.phase_tasks = { Task.SAP, Task.IMPROVE, Task.RECHARGE }
    if self.abandon_fragments:
      self.phase_tasks.add(Task.LEAVE)
    if len(self.myfrags) < 10 and (len(self.agent.env.relics) == 0 or self.keep_exploring()):
      self.phase_tasks.add(Task.EXPLORE)
    if len(pfragments) > 0:
      self.phase_tasks.add(Task.COLLECT)
      self.phase_tasks.add(Task.BACKUP)

    self.high_ground = dict() # pos -> value
    fnotcovered = set()
    for f in self.oppfrags:
      if not any(self.agent.unitman.my_unit(uid).pos == f for uid in self.agent.unitman.live_units()):
        fnotcovered.add(f)
    self.high_ground = dict() # pos -> value
    for x in range(24):
      for y in range(24):
        if self.agent.env.tot_tile_energy((x,y), 0) >= self.min_hg_energy:
          fisr = len([ f for f in fnotcovered if self.agent.env.in_sap_range((x,y), f) ])
          if fisr > 0:
            fisr = min(5, fisr)
            self.high_ground[x,y] = fisr

  def fstatus(self):
    s = "High ground:\n"
    for y in range(24):
      for x in range(24):
        hg = self.high_ground[x,y] if (x,y) in self.high_ground else 0
        s += str(hg).zfill(2) + " "
      s += "\n"
    return s

  def max_explorers(self):
    return 10 - self.agent.env.unit_sensor_range

  def max_collectors(self):
    return 16

  def keep_exploring(self):
    if len(self.agent.env.relic_steps) == 1 and self.agent.env.match() > 1:
      return False # relic not found in 2. match -> don't explore anymore
    return len(self.agent.env.relic_steps) < 3 and len(self.agent.env.relic_steps) <= self.agent.env.match()

  def needs_exploring(self, p):
    if p not in self.agent.env.discovered:
      return True
    discstep = self.agent.env.discovered[p].step
    if self.agent.env.match_step(discstep) > 50 or self.agent.env.match(discstep) > 2:
      return False
    if self.agent.env.step - discstep > (5+self.agent.env.unit_sensor_range):
      return True
    return False

  def path_crowding(self, pos, dstep):
    danger = 30 if self.agent.tactics.danger(pos, dstep) else 1 # it is a dangerous world
    crowding = self.agent.taskman.crowding(pos, dstep) 
    return int(danger * crowding)

  def path_cost(self, cost, steps):
    return cost + 10*steps

  def priority(self, unit, task):
    steps_factor = 10 #10 
    crowd_factor = 30 #30
    if task.type == Task.LEAVE:
      prior = 2000
      prior -= task.cost
      prior -= steps_factor * task.steps
    elif task.type == Task.SAP:
      prior = 500 
      sappot = task.cost # actually sap value
      prior += sappot + (unit.energy // 5)
    elif task.type == Task.EXPLORE:
      prior = 500
      prior -= task.cost
      prior -= steps_factor * task.steps
      # small or no effect
      #age = self.agent.env.step - self.agent.env.last_seen(task.pos) # tile last seen
      #age = min(20, age)
      #prior += 2*age
    elif task.type == Task.RECHARGE:
      prior = 500 - 20
      prior -= task.cost
      prior -= steps_factor * task.steps
      prior -= crowd_factor * self.agent.taskman.crowding(task.pos, task.steps)
      prior += 3 * (50 - self.agent.env.match_step())
      fisr = self.high_ground[task.pos]
      prior += 20 * fisr # number of fragments in sap range
      if not self.agent.env.is_confirmed_fragment(unit.pos) and task.steps <= 7 and self.agent.env.in_sap_range(task.pos, unit.pos):
        ounits = [ u for u in self.agent.unitman.opp_visible_units() if u.pos == task.pos and u.energy >= 0 ]
        if len(ounits) > 0 and sum(u.energy for u in ounits) < unit.energy - task.cost - 20: 
          prior += 150 # ram opp units
      prior += (200 - unit.energy)
    elif task.type == Task.COLLECT or task.type == Task.BACKUP:
      prior = 500
      prior -= task.cost
      prior -= steps_factor * task.steps
      prior -= crowd_factor * self.agent.taskman.crowding(task.pos, task.steps)
      prior += 100 if task.pos in self.myfrags else 0
      if task.steps <= 7 and self.agent.env.in_sap_range(task.pos, unit.pos):
        ounits = [ u for u in self.agent.unitman.opp_visible_units() if u.pos == task.pos and u.energy >= 0 ]
        if len(ounits) > 0 and sum(u.energy for u in ounits) < unit.energy - task.cost - 20: 
          prior += 150# ram opp units
      uenergy = (10 - unit.energy // 40) if unit.pos == task.pos else 0 # keep unit with lowest energy on fragment
      prior += uenergy
      if self.agent.env.is_confirmed_fragment(task.pos) and self.agent.env.tot_tile_energy(task.pos, task.steps) > 0:
        tenergy = self.agent.env.tot_tile_energy(task.pos, task.steps) # tile energy
        prior += 5 * tenergy
      if task.type == Task.BACKUP and task.steps == 1: # assure backup gets picked before collect
        prior += crowd_factor * self.agent.taskman.crowding(task.pos, task.steps)
        prior += task.cost + steps_factor * task.steps + 11
    elif task.type == Task.IMPROVE:
      prior = 0
      prior -= task.cost
      prior -= steps_factor * task.steps
      prior -= crowd_factor * self.agent.taskman.crowding(task.pos, task.steps)
      ediff = self.agent.env.tot_tile_energy(task.pos, task.steps) - self.agent.env.tot_tile_energy(unit.pos, 0)
      nsteps = 10 - min(10, task.steps)
      prior += nsteps * ediff
      pos = (1 + abs(task.pos[0] - center[0])) * (1 + abs(task.pos[1] - center[1]))
      prior -= 5 * pos

    if (task.type == Task.SAP or unit.pos == task.pos) and not self.agent.tactics.is_safe(unit.pos, unit.energy):
      prior -= 1000 # about to get rammed

    #if self.agent.taskman.task_kept(unit, task):
    #  prior += 5 # slightly prefer continuing old task
    return int(prior)
