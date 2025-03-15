import numpy as np
from utils import debug, mirror

class Tile:

  EMPTY = 0
  NEBULA = 1
  ASTEROID = 2

  def __init__(self, pos, energy, status, step):
    self.pos = pos
    self.energy = int(energy)
    self.status = int(status)
    self.step = step
  
  def empty(self):
    return self.status == Tile.EMPTY
  
  def passable(self):
    return self.status != Tile.ASTEROID

class EnvironmentModel:
  # constants
  FR_ABSENT = -1
  FR_UNKNOWN = 0
  FR_CONFIRMED = 1
  # fixed parameters
  W = 24
  H = 24
  spawn_rate = 3
  max_units = 16
  max_energy_nodes = 6
  max_energy_per_tile = 20
  min_energy_per_tile = -20
  max_relic_nodes = 6
  relic_config_size = 5
  init_unit_energy = 100
  max_unit_energy = 400
  max_steps_in_match = 100
  max_matches = 5
  # infered parameter ranges
  nebula_energy_reduction_options = [ 0, 1, 2, 3, 5, 25 ]
  nebula_drift_speed_options = [-0.15, -0.1, -0.05, -0.025, 0.025, 0.05, 0.1, 0.15]
  unit_sap_dropoff_options = [ 0.25, 0.5, 1 ]
  unit_sensor_range_options = [1, 2, 3, 4]
  enode_drift_speed_options = [ 0.01, 0.02, 0.03, 0.04, 0.05 ]
  # parameter guess
  nebula_enred_known = False
  nebula_enred = 5
  unit_sdoff_known = False
  unit_sap_dropoff_factor = 0.5
  unit_sr_known = False
  unit_sensor_range = 1
  drift_speed_known = False
  drift_speed = (0.0, "upright")
  endrift_speed_known = False
  endrift_speed = 0.03

  unit_energy_void_factor = 0.2

  def __init__(self, agent, env_cfg):
    self.agent = agent
    self.step = 0
    self.unit_move_cost = env_cfg["unit_move_cost"]
    self.unit_sap_cost = env_cfg["unit_sap_cost"] 
    self.unit_sap_range = env_cfg["unit_sap_range"]
    self.max_steps_in_match = env_cfg["max_steps_in_match"]
    self.max_matches = env_cfg["match_count_per_episode"]
    debug(f"env unit params: mc={self.unit_move_cost} sc={self.unit_sap_cost} sr={self.unit_sap_range}\n")
    self.observations = [] # sets of visible tiles at each step
    self.relic_steps = [] # steps of relic discovery
    self.relics = set() # pos of relics
    self.fragments = dict() # pos in relic area -> known status
    self.features = dict() # reconstruction of asteroids at step 0 
    self.nebula = dict() # reconstruction of nebula at step 0
    self.last_drift_step = 1 # step of last drift detected 
    self.discovered = dict() # pos -> tile as last seen
    self.drift_detected = False 

  def update(self, obs, step):
    self.step = step
    tiles = self._visible_tiles(obs)
    self.observations.append(tiles)
    for tpos, tile in tiles.items():
      self.discovered[tpos] = tile
      self.discovered[mirror(tpos)] = tile
    # relics
    relic_nodes = np.array(obs["relic_nodes"]) # shape (max_relic_nodes, 2)
    observed_relic_nodes_mask = np.array(obs["relic_nodes_mask"]) # shape (max_relic_nodes, )
    visible_relic_node_ids = set(np.where(observed_relic_nodes_mask)[0])
    for id in visible_relic_node_ids:
      rpos = int(relic_nodes[id][0]), int(relic_nodes[id][1])
      if rpos not in self.relics:
        self.relics.add(rpos)
        self.relics.add(mirror(rpos))
        self.relic_steps.append(self.step)
        self._update_fragments(rpos)
    # infer drift speed
    self.drift_detected = False 
    if not self.drift_speed_known:
      self._infere_drift()
    # update features mask
    self._update_features()
    # infer energy node drift speed
    if not self.endrift_speed_known:
      self._infere_endrift()

  def _visible_tiles(self, obs):
    tiles = dict() # pos -> tile
    visible = np.array(obs["sensor_mask"])
    mapfeats = obs["map_features"]["tile_type"]
    mapenergy = obs["map_features"]["energy"]
    for x in range(self.W):
      for y in range(self.H):
        if visible[x][y]:
          tiles[x, y] = Tile((x, y), mapenergy[x][y], mapfeats[x][y], self.step)
    return tiles

  def _infere_drift(self):
    if len(self.observations) < 2:
      return
    previous = self.observations[-2]
    current = self.observations[-1]
    changed = set()
    for tpos in previous:
      if tpos in current and current[tpos].status != previous[tpos].status:
          changed.add(tpos)
    if len(changed) == 0:
      return
    drift_diff = self.step - self.last_drift_step
    self.last_drift_step = self.step
    drift_speed = 1.0 / drift_diff
    drift_speed = min(self.nebula_drift_speed_options, key=lambda o: abs(o-drift_speed))
    debug(f"drift speed {drift_speed}\n")
    upright, downleft = 0, 0
    for tpos in changed:
      x, y, = tpos
      urpos = (x+1) % self.W, (y+self.H-1) % self.H
      if urpos in changed:
        if previous[urpos].status == Tile.EMPTY and current[urpos].status != Tile.EMPTY and current[urpos].status == previous[tpos].status:
          upright += 1 # asteroid or nebula moved on empty up-right tile 
        if previous[urpos].status != Tile.EMPTY and current[urpos].status == Tile.EMPTY and current[tpos].status == previous[urpos].status:
          downleft += 1 # asteroid or nebula moved on empty up-right tile 
      dlpos = (x+self.W-1) % self.W, (y+1) % self.H
      if dlpos in changed:
        if previous[dlpos].status == Tile.EMPTY and current[dlpos].status != Tile.EMPTY and current[dlpos].status == previous[tpos].status:
          downleft += 1 # asteroid or nebula moved on empty down-left tile 
        if previous[dlpos].status != Tile.EMPTY and current[dlpos].status == Tile.EMPTY and current[tpos].status == previous[dlpos].status:
          upright += 1 # asteroid or nebula moved on empty up-right tile 
    debug(f"drift dir {upright} {downleft}\n")
    if upright > downleft:
      self.drift_speed, self.drift_speed_known = (drift_speed, "upright"), True
      debug(f"drift {self.drift_speed}\n")
      self._reconstruct_features()
    if upright < downleft:
      self.drift_speed, self.drift_speed_known = (drift_speed, "downleft"), True
      debug(f"drift {self.drift_speed}\n")
      self._reconstruct_features()
    self.features = dict() # reset features for detected unknown drift
    self.drift_detected = True # skip ner estimation

  def _infere_endrift(self):
    if self.step <= 10 or self.step > 100 or len(self.observations) < 2:
      return
    previous = self.observations[-2]
    current = self.observations[-1]
    for tpos in previous:
      if tpos in current and current[tpos].energy != previous[tpos].energy:
        self.endrift_speed = round(1.0 / (self.step - 2), 2)
        debug(f"enode speed {self.endrift_speed}\n")
        self.endrift_speed_known = True
        return

  def _update_features(self):
    for tpos, tile in self.observations[-1].items():
      tpos = self._rewind_drift(tpos, self.step + 1)
      self.features[tpos] = tile.status
      self.features[mirror(tpos)] = tile.status

  def _reconstruct_features(self):
    for step, tiles in enumerate(self.observations):
      for tpos, tile in tiles.items():
        tpos = self._rewind_drift(tpos, step + 1)
        self.features[tpos] = tile.status
        self.features[mirror(tpos)] = tile.status

  def _rewind_drift(self, tpos, step, show=False):
    if not self.drift_speed_known:
      return tpos
    dspeed, ddir = self.drift_speed
    ndrifts = int((step-2) * dspeed)
    if show:
      debug(f"{step} {ndrifts}\n")
    x, y, = tpos
    if ddir == "upright":
      return (x+10*self.W-ndrifts) % self.W, (y+ndrifts) % self.H
    else:
      return (x+ndrifts) % self.W, (y+10*self.H-ndrifts) % self.H

  def _update_fragments(self, rpos):
    rx, ry = rpos
    debug(f"new relic at {(rx, ry)}\n")
    for dx in range(-(self.relic_config_size // 2), self.relic_config_size // 2 + 1):
      for dy in range(-(self.relic_config_size // 2), self.relic_config_size // 2 + 1):
        fpos = rx+dx, ry+dy
        if self.on_map(fpos) and not self.is_confirmed_fragment(fpos):
          self.fragments[fpos] = self.FR_UNKNOWN
          self.fragments[mirror(fpos)] = self.FR_UNKNOWN

  def __repr__(self):
    s = "Features map 0:\n"
    for y in range(self.H):
      for x in range(self.W):
        if (x,y) in self.features:
          s += str(self.features[x,y]) + " "
        else:
          s += ". "
      s += "\n"
    s += "Features map current:\n"
    for y in range(self.H):
      for x in range(self.W):
        if self.is_nebula((x,y), 0):
          s += "1 "
        elif not self.passable((x,y), 0):
          s += "2 "
        else:
          s += ". "
      s += "\n"
    return s

  def fstatus(self):
    s = "Fragments map:\n"
    for y in range(self.H):
      for x in range(self.W):
        if (x,y) not in self.fragments:
          s += ". "
        elif self.fragments[x,y] == self.FR_ABSENT:
          s += "A "
        elif self.fragments[x,y] == self.FR_CONFIRMED:
          s += "C "
        elif self.fragments[x,y] == self.FR_UNKNOWN:
          s += "? "
      s += "\n"
    return s

  def on_map(self, pos):
    return 0 <= pos[0] < self.W and 0 <= pos[1] < self.H

  def in_sap_range(self, posa, posb):
    return abs(posa[0]-posb[0]) <= self.unit_sap_range and abs(posa[1]-posb[1]) <= self.unit_sap_range

  def is_my_half(self, pos):
    if self.agent.team_id == 0:
      return pos[0] + pos[1] < 24
    else:
      return pos[0] + pos[1] >= 24

  def match_step(self, step=None):
    if step is None:
      step = self.step
    return step % 101

  def match(self, step=None):
    if step is None:
      step = self.step
    return step // 101

  def is_fragment(self, p):
    return p in self.fragments and self.fragments[p] != self.FR_ABSENT

  def is_confirmed_fragment(self, p):
    return p in self.fragments and self.fragments[p] == self.FR_CONFIRMED

  def is_unknown_fragment(self, p):
    return p in self.fragments and self.fragments[p] == self.FR_UNKNOWN

  def nfragments(self):
    return sum(1 for p in self.fragments if self.is_fragment(p))

  def all_fragments(self):
    return set(p for p in self.fragments if self.fragments[p] != self.FR_ABSENT)

  def confirmed_fragments(self):
    return { p for p in self.fragments if self.is_confirmed_fragment(p) }

  def tile_energy(self, pos):
    return self.discovered[pos].energy if pos in self.discovered else 0
  
  def passable(self, pos, dstep):
    tpos = self._rewind_drift(pos, self.step + dstep)
    return tpos not in self.features or self.features[tpos] != Tile.ASTEROID

  def is_nebula(self, pos, dstep):
    tpos = self._rewind_drift(pos, self.step + 1 + dstep)
    return tpos in self.features and self.features[tpos] == Tile.NEBULA

  def tot_tile_energy(self, pos, dstep):
    if self.endrift_speed_known and self.is_enode_change(pos, dstep):
      return 0 - (self.nebula_enred if self.is_nebula(pos, dstep) else 0)
    else:
      return self.tile_energy(pos) - (self.nebula_enred if self.is_nebula(pos, dstep) else 0)

  def is_last_nebula_tile(self, pos):
    tpos = self._rewind_drift(pos, self.step)
    return tpos in self.features and self.features[tpos] == Tile.NEBULA

  def is_visible(self, pos):
    return pos in self.observations[-1]

  def last_seen(self, pos):
    return 0 if pos not in self.discovered else self.discovered[pos].step    

  def is_last_step(self, dstep):
    return (self.step + dstep) in { 101, 202, 303, 404, 505 }

  def is_reset_step(self):
    return self.step in { 102, 203, 304, 405 }

  def is_enode_change(self, pos, dstep):
    if pos not in self.discovered:
      return False
    step1, step2 = self.discovered[pos].step, self.step + dstep
    return (step2-step1) > int(1.0 / self.endrift_speed)

  def set_sensor_range(self, sr):
    if sr in self.unit_sensor_range_options and sr > self.unit_sensor_range:
      self.unit_sensor_range, self.unit_sr_known = sr, self.step > 100
      debug(f"vis estimated: {sr}\n")  

  def set_nebula_energy_reduction(self, ner):
    if ner in self.nebula_energy_reduction_options:
      self.nebula_enred, self.nebula_enred_known = ner, True
      debug(f"ner estimated: {ner}\n")  

  def set_sap_dropoff(self, denergy, nsaps):
    for o in self.unit_sap_dropoff_options:
      if -denergy == int(self.unit_sap_cost*o*nsaps):
        self.unit_sap_dropoff_factor, self.unit_sdoff_known = o, True
        debug(f"sdo estimated: {o}\n")  

  def absent_fragments(self):
    covered = { unit.pos for unit in self.agent.unitman.all_visible_units(self.agent.team_id) if unit.energy >= 0 } # dying units don't get points
    for pos in covered:
      self.fragments[pos] = self.FR_ABSENT
      self.fragments[mirror(pos)] = self.FR_ABSENT

  def confirm_fragments(self, reward):
    covered_fragments = { unit.pos for unit in self.agent.unitman.all_visible_units(self.agent.team_id) 
                          if self.agent.env.is_fragment(unit.pos) and unit.energy >= 0 } # dying units don't get points
    unknown = { pos for pos in covered_fragments if self.fragments[pos] == self.FR_UNKNOWN }
    confirmed = { pos for pos in covered_fragments if self.fragments[pos] == self.FR_CONFIRMED }
    unkreward = reward - len(confirmed)
    if len(unknown) > 0:
      if unkreward == len(unknown):
        for pos in unknown:
          self.fragments[pos] = self.FR_CONFIRMED
          self.fragments[mirror(pos)] = self.FR_CONFIRMED
          debug(f"frg confirmed {pos} \n")
      elif unkreward == 0:
        for pos in unknown:
          self.fragments[pos] = self.FR_ABSENT
          self.fragments[mirror(pos)] = self.FR_ABSENT
          debug(f"frg absent {pos} \n")
      else:
        debug(f"frg abandon\n")        
        return True
    return False