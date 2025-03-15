import numpy as np
from unit_manager import Unit
from utils import debug, dir2delta, path2str, pos_to_3x3, manhattan, spawn_points, pos_to_plus

class OpponentModel:

  def __init__(self, agent):
    self.agent = agent
    self.oid = self.agent.opp_team_id
    self.match_reset()
  
  def match_reset(self):
    self.opp_tracking = { uid: Unit(None, uid, spawn_points[self.oid], 100) for uid in range(16) } # ouid -> Unit
    self.sap_danger = { (x,y): 0 for x in range(24) for y in range(24) }
    self.hidden = set()

  def dead(self, uid):
    self.opp_tracking[uid] = Unit(None, uid, spawn_points[self.oid], 100)
  
  def seen(self, uid, pos, energy):
    self.opp_tracking[uid].update_position(pos)
    self.opp_tracking[uid].update_energy(energy)
    self._update_danger(uid)
    self._update_potential(uid)
  
  def _update_danger(self, uid):
    ounit = self.opp_tracking[uid]
    for p in pos_to_plus[ounit.pos]:
      self.opp_threat[p] += ounit.energy
    if ounit.energy >= self.agent.env.unit_sap_cost:
      for dx in range(-1-self.agent.env.unit_sap_range, self.agent.env.unit_sap_range+2): # sap_range + 1 for side saps
        for dy in range(-1-self.agent.env.unit_sap_range, self.agent.env.unit_sap_range+2):
            p = ounit.pos[0] + dx, ounit.pos[1] + dy
            if self.agent.env.on_map(p):
              self.sap_danger[p] += 1
              self.sap_danger[p] = min(10, self.sap_danger[p])

  def _update_potential(self, uid):
    ounit = self.opp_tracking[uid]
    for sp in pos_to_3x3[ounit.pos]:
      if sp == ounit.pos:
        self.sap_potential[sp] += 100
        if self.agent.env.is_confirmed_fragment(sp):
          self.sap_potential[sp] += 50
      else:
        self.sap_potential[sp] += int(90*self.agent.env.unit_sap_dropoff_factor) # prefer direct

  def update(self, oppreward):
    self.sap_potential = { (x,y): 0 for x in range(24) for y in range(24) }
    self.opp_threat = { (x,y): 0 for x in range(24) for y in range(24) }
    for x in range(24):
      for y in range(24):
        self.sap_danger[x,y] -= 1 # reduce sap danger each turn
        self.sap_danger[x,y] = max(0, self.sap_danger[x,y])
    # update visible units
    opp_visible = self.agent.unitman.vis_units_history[-1][self.agent.opp_team_id]
    for uid in range(16):
      if uid in opp_visible:
        ounit = self.agent.unitman.opp_unit(uid)
        if ounit.energy >= 0:
          self.seen(uid, ounit.pos, ounit.energy)
        else:
          self.dead(uid)
      elif self.agent.env.is_confirmed_fragment(self.opp_tracking[uid].pos) and not self.agent.env.is_visible(self.opp_tracking[uid].pos):
        self._update_danger(uid) # from memory
    # infer invisible opps on fragments
    self.hidden = set()
    if oppreward > 0:
      # visible opp on fragments
      for uid in opp_visible:
        ounit = self.agent.unitman.opp_unit(uid)
        if ounit.energy >= 0 and self.agent.env.is_confirmed_fragment(ounit.pos):
          oppreward -= 1
      # invisible opp on fragments
      fpos = [ fp for fp in self.agent.env.confirmed_fragments() ]
      fpos = sorted(fpos, key=lambda p: manhattan(p, spawn_points[self.oid]))
      while oppreward > 0 and len(fpos) > 0:
        fp = fpos.pop(0)
        if not self.agent.env.is_visible(fp): # invisible fragments only
          self.sap_potential[fp] += 100
          self.hidden.add(fp)
          oppreward -= 1

  def __repr__(self):
    s = "Danger map:\n"
    for y in range(24):
      for x in range(24):
        s += str(self.sap_danger[x,y]).zfill(2) + " "
      s += "\n"
    s += "Potential map:\n"
    for y in range(24):
      for x in range(24):
        s += str(self.sap_potential[x,y]).zfill(4) + " "
      s += "\n"
    s += "Threat map:\n"
    for y in range(24):
      for x in range(24):
        s += str(self.opp_threat[x,y]).zfill(4) + " "
      s += "\n"
    return s

class Tactics:

  def __init__(self, agent, env_cfg):
    self.agent = agent
    self.last_actions = None
    self.oppmodel = OpponentModel(self.agent) 

  def update(self, obs):
    # sap dropoff estimation
    if not self.agent.env.unit_sdoff_known:
      self._infere_sdoff()
    # infere vision
    if not self.agent.env.unit_sr_known:
      self._infere_vision()

  def _infere_vision(self):
    if len(self.agent.unitman.all_visible_units(self.agent.team_id)) == 0:
      return
    maxd = 0
    for tpos in self.agent.env.observations[-1]:
      mindx = min(abs(u.pos[0] - tpos[0]) for u in self.agent.unitman.all_visible_units(self.agent.team_id))
      mindy = min(abs(u.pos[1] - tpos[1]) for u in self.agent.unitman.all_visible_units(self.agent.team_id))
      if mindx > maxd:
        maxd = mindx
      if mindy > maxd:
        maxd = mindy
    self.agent.env.set_sensor_range(maxd)

  def _infere_sdoff(self):
    if not self.agent.env.nebula_enred_known:
      return
    if len(self.agent.unitman.vis_units_history) < 2:
      return
    # sapping units that didn't die (i.e. changed position, in case of dying and respawning right away)
    sapunits = [ u for u in self.agent.unitman.all_visible_units(self.agent.team_id) if u.pos == u.last_pos ]
    sapunits = [ u for u in sapunits if u.id in self.agent.tactics.last_actions and self.agent.tactics.last_actions[u.id][0] == 5]
    sapunits = { u.id:self.agent.tactics.last_actions[u.id][1:3] for u in sapunits }
    if len(sapunits) == 0:
      return
    # count number of saps on each position
    sappos = { (self.agent.unitman.my_unit(uid).pos[0]+dx, self.agent.unitman.my_unit(uid).pos[1]+dy):0 for uid, (dx, dy) in sapunits.items() }
    for uid, (dx, dy) in sapunits.items():
      x, y = self.agent.unitman.my_unit(uid).pos
      sappos[x+dx, y+dy] += 1
    # opponent units visible on both this and previous round
    visible_opp_units = set()
    for uid in self.agent.unitman.vis_units_history[-1][self.agent.opp_team_id]:
      if uid in self.agent.unitman.vis_units_history[-2][self.agent.opp_team_id]:
        visible_opp_units.add(uid)
    debug(f"sdoff: {sapunits} {sappos} {visible_opp_units}\n")
    # check for energy difference for units next to sap positions
    for ouid in visible_opp_units:
      ounit = self.agent.unitman.opp_unit(ouid)
      if len(self.agent.unitman.visible_units_on_plus(ounit.pos, self.agent.team_id)) > 0:
        continue # skip units affected by void  
      nsaps = sum(sappos[spos] for spos in sappos if ounit.pos != spos and ounit.pos in pos_to_3x3[spos])
      if nsaps > 0:
        dene = ounit.energy - ounit.last_energy
        if ounit.energy >= 0: # only live unit gets energy from tile
          dene -= self.agent.env.tot_tile_energy(ounit.pos, 0)
        dene += self.agent.env.unit_move_cost if ounit.pos != ounit.last_pos else 0 # add movement 
        debug(f"dene: {ounit} {dene} / {nsaps}\n")
        self.agent.env.set_sap_dropoff(dene, nsaps)

  def is_safe(self, pos, energy):
    return energy >= self.oppmodel.opp_threat[pos]

  def can_sap(self, unit):
    return unit.energy >= self.agent.env.unit_sap_cost
  
  def sap_potential(self, pos):
    return self.oppmodel.sap_potential[pos]

  def _apply_sap(self, unit, dpos, opp_units_ene):
    x, y = unit.pos[0] + dpos[0], unit.pos[1] + dpos[1]
    for u in self.agent.unitman.visible_units_on((x,y), self.agent.opp_team_id):
      opp_units_ene[u.id] -= self.agent.env.unit_sap_cost
    for u in self.agent.unitman.visible_units_on_3x3((x,y), self.agent.opp_team_id):
      if u.pos != (x,y):
        opp_units_ene[u.id] -= self.agent.env.unit_sap_cost * self.agent.env.unit_sap_dropoff_factor

  def _evaluate_sap(self, unit, dpos, opp_units_ene, opp_units_ram):
    spos = unit.pos[0] + dpos[0], unit.pos[1] + dpos[1]
    sappot = 0 
    if self.agent.env.is_confirmed_fragment(unit.pos):
      sappot += 30
    # damaging visible units
    for u in self.agent.unitman.visible_units_on_3x3(spos, self.agent.opp_team_id):
      if u.id in opp_units_ram or opp_units_ene[u.id] < 0: # skip units to be rammed or dying
        continue
      if self.agent.env.is_confirmed_fragment(u.pos):
        if spos == u.pos: # direct
          sappot += 100
        else: # side * 0.9 so direct is preferred even with sdoff = 1
          sappot += int(90*self.agent.env.unit_sap_dropoff_factor)
      else:
       if spos == u.pos: # direct
         sappot += 70
       else: # side * 0.9 so direct is preferred even with sdoff = 1
         sappot += int(60*self.agent.env.unit_sap_dropoff_factor)
    # damaging invisible units on fragemnts
    if self.agent.strategy.all_confirmed:
      for sp in pos_to_3x3[spos]:
        if sp in self.oppmodel.hidden:
          if spos == sp: # direct
            sappot += 100
          else: # side
            sappot += int(90 * self.agent.env.unit_sap_dropoff_factor)
    return sappot
    
  def danger(self, pos, steps):
    return self.oppmodel.sap_danger[pos] > 0
