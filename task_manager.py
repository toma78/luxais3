import heapq
from utils import debug, DELTAS, delta2dir, pos_to_3x3, pos_to_srange, dir2str, dir2delta, pos_to_plus, manhattan, spawn_points

class Task:

  LEAVE = 'L'
  SAP = 'S'
  COLLECT = 'C'
  EXPLORE = 'E'
  IMPROVE = 'I'
  RECHARGE = 'R'
  BACKUP = 'B'

  def __init__(self, type, pos, cost, steps):
    self.type = type
    self.pos = pos
    self.cost = cost
    self.steps = steps
    self.priority = None

  def __eq__(self, other):
    return self.type == other.type and self.pos == other.pos

  def __ne__(self, other):
    return not self.__eq__(other)
  
  def __repr__(self):
    return f"<{self.type}{self.pos} {self.cost} {self.steps} {self.priority}>"

class TaskManager:

  def __init__(self, agent, env_cfg):
    self.agent = agent
    self.maxdelsteps = 2
    self.track_len = 30
    def sort_deltas(x, y):
      if x < 12 and y < 12:
        if x < y:
          return [ (1, 0), (0, 1), (0, -1), (-1, 0) ]
        else:
          return [ (0, 1), (1, 0), (-1, 0), (0, -1) ]
      if x >= 12 and y >= 12:
        if x >= y:
          return [ (-1, 0), (0, -1), (0, 1), (1, 0)  ]
        else:
          return [ (0, -1), (-1, 0), (1, 0), (0, 1) ]
      if x < 12 and y >= 12:
        if (12-x) < (y-12):
          return [ (1, 0), (0, -1), (0, 1), (-1, 0) ]
        else:
          return [ (0, -1), (1, 0), (-1, 0), (0, 1) ]
      if x >= 12 and y < 12:
        if (12-y) < (x-12):
          return [ (0, 1), (-1, 0), (1, 0), (0, -1) ]
        else:
          return [ (-1, 0), (0, 1), (0, -1), (1, 0) ]

    self.ord_deltas = { (x,y): sort_deltas(x, y) for x in range(24) for y in range(24) }

  def update(self, obs):
    pass

  def estimate_cost(self, pos, dstep):
    return self.agent.env.unit_move_cost - self.agent.env.tot_tile_energy(pos, dstep)

  def _sap_tasks(self, unit):
    sap_positions = []
    for dsap in self.unit_saps[unit.id]:
      sp = self.agent.tactics._evaluate_sap(unit, dsap, self.opp_units_ene, self.opp_units_ram)        
      sap_positions.append((dsap, sp))
    sap_positions = [ s for s in sap_positions if s[1] >= 100 ]
    self.unit_saps[unit.id] = [ dsap for dsap, _ in sap_positions ]
    return sap_positions

  def _valid_tasks(self, unit, pos, cost, steps):
    tasks = set()
    if self.agent.env.is_unknown_fragment(unit.pos) and not self.agent.env.is_unknown_fragment(pos):
      tasks.add(Task.LEAVE)
    if self.agent.env.is_fragment(pos) and pos not in self.collecting:
      tasks.add(Task.COLLECT)
    if self.agent.env.is_confirmed_fragment(pos) and pos not in self.backuping and \
      pos in self.agent.unitman.positions[self.agent.team_id] and \
      self.agent.unitman.positions[self.agent.team_id][pos].energy > 3 * (unit.energy - cost) / 2 and \
      manhattan(unit.pos, spawn_points[self.agent.team_id]) < manhattan(pos, spawn_points[self.agent.team_id]):
      tasks.add(Task.BACKUP)
    if self.agent.strategy.needs_exploring(pos) and self.agent.env.is_my_half(pos) and pos not in self.exploring:
      tasks.add(Task.EXPLORE)
    if pos in self.agent.strategy.high_ground and pos not in self.recharging:
      tasks.add(Task.RECHARGE)
    elif self.agent.env.tot_tile_energy(pos, steps) >= 0 or self.agent.env.tot_tile_energy(pos, steps) >= self.agent.env.tot_tile_energy(unit.pos, steps):
      tasks.add(Task.IMPROVE)
    return tasks

  def _valid_tile(self, unit, pos, cost, steps):
    if self.agent.env.is_last_step(steps) or cost > unit.energy:
      return False
    if steps < 5 and not self.agent.tactics.is_safe(pos, unit.energy-cost):
      return False
    if steps in self.probing and pos != self.probing[steps] and self.agent.env.is_unknown_fragment(pos):
      return False
    return True

  def heap_fix_up(self, pq, i):
    while i > 0:
      p = (i-1) // 2
      if pq[i] < pq[p]:
        pq[i], pq[p] = pq[p], pq[i]
        i = p
      else:
        break

  def _search_tasks(self, unit):
    found = [] # (dest, cost, steps, tasks)
    hpqcnt = 0
    # current pos cost
    ac = self.estimate_cost(unit.pos, 0) - self.agent.env.unit_move_cost # doesn't move
    pcost = self.agent.strategy.path_cost(ac, 0)
    pcost += self.agent.strategy.path_crowding(unit.pos, 0)
    pq = [ (pcost, 0, unit.pos, 0, 0, 0) ] # (path_cost, heap_cnt, pos, energy_cost, steps, delays)
    visited = { (unit.pos, 0): None } # (pos, dt) -> parent
    tasks_pos = set() # to avoid refinding same task with different delays
    while len(pq) > 0:
      npcost, _, pos, cost, steps, delays = heapq.heappop(pq)
      if pos not in tasks_pos:
        tasks_pos.add(pos)        
        vtasks = self._valid_tasks(unit, pos, cost, steps)
        found.append((pos, cost, steps, vtasks, npcost))
      if unit.energy - cost < self.agent.env.unit_move_cost: # can't move further
        continue
      # insert delays
      if delays < 1 and steps < self.maxdelsteps:
        ac = self.estimate_cost(pos, steps+1) - self.agent.env.unit_move_cost # doesn't move
        if self._valid_tile(unit, pos, cost+ac, steps+1):
          pcost = self.agent.strategy.path_cost(cost+ac, steps+1)
          pcost += self.agent.strategy.path_crowding(pos, steps+1)
          hpqcnt += 1
          node = (pcost, hpqcnt, pos, cost+ac, steps+1, delays+1)
          if (pos, steps+1) not in visited:
            heapq.heappush(pq, node)
            visited[pos, steps+1] = (pos, steps)
          else: # update neighbours
            for i, (pc, hpc, p, _, ont, dly) in enumerate(pq):
              if pos == p and dly == delays and ont >= steps:
                pcost = self.agent.strategy.path_cost(cost+ac, steps+1)
                pcost += self.agent.strategy.path_crowding(pos, steps+1)            
                if pcost < pc:
                  pq[i] = (pcost, hpc, pos, cost+ac, steps+1, delays)
                  #heapq.heapify(pq)
                  self.heap_fix_up(pq, i)
                  visited[pos, min(steps+1, self.maxdelsteps)] = (pos, min(steps, self.maxdelsteps))
                break
      # expand neighbours  
      x, y = pos
      for dx, dy in self.ord_deltas[pos]:
        nx, ny, nt = x+dx, y+dy, steps+1
        if self.agent.env.on_map((nx, ny)) and self.agent.env.passable((nx, ny), nt):
          ac = self.estimate_cost((nx, ny), nt)
          if not self._valid_tile(unit, (nx, ny), cost+ac, nt):
            continue
          if ((nx, ny), min(nt, self.maxdelsteps)) not in visited:
            pcost = self.agent.strategy.path_cost(cost+ac, nt)
            pcost += self.agent.strategy.path_crowding((nx, ny), nt)
            hpqcnt += 1
            node = (pcost, hpqcnt, (nx, ny), cost+ac, nt, delays)
            heapq.heappush(pq, node)
            visited[(nx, ny), min(nt, self.maxdelsteps)] = ((x, y), min(steps, self.maxdelsteps))
          else: # update neighbours
            for i, (pc, hpc, p, _, ont, dly) in enumerate(pq):
              if (nx, ny) == p and dly == delays and ont >= steps:
                pcost = self.agent.strategy.path_cost(cost+ac, nt)
                pcost += self.agent.strategy.path_crowding((nx, ny), nt)            
                if pcost < pc:
                  pq[i] = (pcost, hpc, (nx, ny), cost+ac, nt, delays)
                  heapq.heapify(pq)
                  visited[(nx, ny), min(nt, self.maxdelsteps)] = ((x, y), min(steps, self.maxdelsteps))
                break
    return found, visited

  def _recover_path(self, node, visited):
    path = []
    while visited[node] is not None:
      (x, y), _ = node
      (xp, yp), _ = visited[node]
      path = [ delta2dir[x-xp, y-yp] ] + path
      node = visited[node]
    return path

  def _reset_coordination(self):
    self.exploring = set()
    self.recharging = set()
    self.collecting = set()
    self.backuping = set()
    self.probing = dict() # dstep -> pos
    self.nexplorers = 0
    self.ncollectors = 0
    self.nfragments = self.agent.env.nfragments()
    self.opp_units_ram = set()
    self.opp_units_ene = { unit.id: unit.energy for unit in self.agent.unitman.all_visible_units(self.agent.opp_team_id) }
    self.track_units = [ dict() for _ in range(self.track_len) ] # dstep -> { pos -> nunits }   
    self.track_side_units = [ dict() for _ in range(self.track_len) ] # dstep -> { pos -> nunits in 3x3 }   

  def crowding(self, pos, dstep):
    if dstep >= self.track_len:
      return 0
    c = self.track_units[dstep][pos] if pos in self.track_units[dstep] else 0
    s = self.track_side_units[dstep][pos] if pos in self.track_side_units[dstep] else 0
    return c + s * self.agent.env.unit_sap_dropoff_factor 

  def _track_no_task(self, unit):
    if self.agent.env.is_unknown_fragment(unit.pos):
      self.probing[0] = unit.pos
    for p in pos_to_3x3[unit.pos]:
      if p == unit.pos:
        if p not in self.track_units[0]:
          self.track_units[0][p] = 0
        self.track_units[0][p] += 1
      else:
        if p not in self.track_side_units[0]:
          self.track_side_units[0][p] = 0
        self.track_side_units[0][p] += 1

  def _track_path(self, pos, path):
    # coordinate probing for fragments 
    upos = pos
    for dstep, ddir in enumerate(path):
      delta = dir2delta[ddir]
      upos = upos[0] + delta[0], upos[1] + delta[1]
      if self.agent.env.is_unknown_fragment(upos):
        self.probing[dstep+1] = upos
    # crowding
    path = [ 0 ] + path
    path = path[:self.track_len]
    if len(path) < self.track_len:
      path = path + [ 0 ] * (self.track_len - len(path))
    upos = pos
    for dstep, ddir in enumerate(path):
      delta = dir2delta[ddir]
      upos = upos[0] + delta[0], upos[1] + delta[1]
      for p in pos_to_3x3[upos]:
        if p == upos:
          if p not in self.track_units[dstep]:
            self.track_units[dstep][p] = 0
          self.track_units[dstep][p] += 1
        else:
          if p not in self.track_side_units[dstep]:
            self.track_side_units[dstep][p] = 0
          self.track_side_units[dstep][p] += 1
  
  def _precalculate(self, free_uids):
    self.unit_saps = dict()
    for uid in free_uids:
      unit = self.agent.unitman.my_unit(uid)
      sap_positions = []
      for dx in range(-self.agent.env.unit_sap_range, self.agent.env.unit_sap_range+1):
        for dy in range(-self.agent.env.unit_sap_range, self.agent.env.unit_sap_range+1):
          x, y = unit.pos[0] + dx, unit.pos[1] + dy
          if self.agent.env.on_map((x, y)) and self.agent.tactics.sap_potential((x,y)) > 0:
            sap_positions.append((dx, dy))
      self.unit_saps[uid] = sap_positions

  def _start_task(self, unit, task, path):
    unit.set_task(task, path)
    self._track_path(unit.pos, path)
    if task.type == Task.SAP:
      if self.agent.env.is_fragment(unit.pos): # sapping from fragment
        self.collecting.add(unit.pos)
      self.agent.tactics._apply_sap(unit, task.pos, self.opp_units_ene)
    elif task.type == Task.EXPLORE:
      for p in pos_to_srange[self.agent.env.unit_sensor_range][task.pos]:
        self.exploring.add(p)
      self.nexplorers += 1
      if self.nexplorers >= self.agent.strategy.max_explorers():
        self.allowed_tasks.remove(Task.EXPLORE)
    elif task.type == Task.COLLECT:
      self.collecting.add(task.pos)
      self.ncollectors += 1
      if self.ncollectors >= self.agent.strategy.max_collectors():
        self.allowed_tasks.remove(Task.COLLECT)
    elif task.type == Task.BACKUP:
      self.backuping.add(task.pos)
      self.collecting.add(task.pos)
      self.ncollectors += 1
      if self.ncollectors >= self.agent.strategy.max_collectors():
        self.allowed_tasks.remove(Task.COLLECT)
    elif task.type == Task.RECHARGE:
      self.recharging.add(task.pos)
    # track rammed units
    if task.steps <= 7:
      for u in self.agent.unitman.visible_units_on(task.pos, self.agent.opp_team_id):
        self.opp_units_ram.add(u.id)
    debug(f"{unit}: {task} {''.join([dir2str[d] for d in path])}\n")

  def _generate_tasks(self, unit):
    tasks = []
    if self.agent.tactics.can_sap(unit):
      for dsap, sp in self._sap_tasks(unit):
        task = Task(Task.SAP, dsap, sp, 0)
        task.priority = self.agent.strategy.priority(unit, task)
        tasks.append((task, 0))
    found, visited = self._search_tasks(unit)
    for dest, cost, steps, vtasks, pcost in found:
      for ttype in (vtasks & self.allowed_tasks):
        task = Task(ttype, dest, cost, steps)
        task.priority = self.agent.strategy.priority(unit, task)
        tasks.append((task, pcost))
    return tasks, visited

  def _top_task(self, unit):
    tasks_list, visited = self._generate_tasks(unit)
    #if unit.id in { 4, 7 } and self.agent.env.step in { 50 }:
    #debug(f"{unit} tasks {sorted(tasks_list, key=lambda t: t[0].priority, reverse=True)}\n")
    if len(tasks_list) > 0:
      task, pcost = max(tasks_list, key=lambda t: t[0].priority)
      path = [] if task.type == Task.SAP else self._recover_path((task.pos, min(task.steps, self.maxdelsteps)), visited)
      return task, path, pcost
    return None, [], 0
  
  def _check_path(self, unit, path, old_pcost):
    # check for crowding interference from other paths or probing conflict
    steps = 0
    cost = 0
    pcost = 0
    x, y = unit.pos
    # for units without path
    ac = self.estimate_cost(unit.pos, 0) - self.agent.env.unit_move_cost
    pcost = self.agent.strategy.path_cost(ac, 0)
    pcost += self.agent.strategy.path_crowding(unit.pos, 0)
    # path
    for d in path:
      dx, dy = dir2delta[d]
      nx, ny, nt = x+dx, y+dy, steps+1
      if d == 0:
        ac = self.estimate_cost((nx, ny), nt) - self.agent.env.unit_move_cost
      else:
        ac = self.estimate_cost((nx, ny), nt)      
      pcost = self.agent.strategy.path_cost(cost+ac, nt)
      pcost += self.agent.strategy.path_crowding((nx, ny), nt)
      cost += ac
      x, y, steps = nx, ny, nt
      if steps in self.probing and (x,y) != self.probing[steps] and self.agent.env.is_unknown_fragment((x,y)):
        return False
    return old_pcost == pcost

  def _still_valid(self, unit, task):
    if task.type == Task.SAP:
      sp = self.agent.tactics._evaluate_sap(unit, task.pos, self.opp_units_ene, self.opp_units_ram)
      return sp >= task.cost
    return task.type in (self._valid_tasks(unit, task.pos, task.cost, task.steps) & self.allowed_tasks)    

  def task_kept(self, unit, task):
    if unit.old_task is None or task is None:
      return False
    return unit.old_task == task

  def assign_tasks(self, free_uids):
    self._reset_coordination()
    self._precalculate(free_uids)
    self.allowed_tasks = self.agent.strategy.phase_tasks.copy()
    # create units priority queue with top task for all units
    hpc = 0
    unit_queue = []
    for uid in free_uids:
      unit = self.agent.unitman.my_unit(uid)
      top_task, top_path, top_cost = self._top_task(unit)
      if top_task is not None:
        item = (-top_task.priority, hpc, unit, top_task, top_path, top_cost)
        heapq.heappush(unit_queue, item)
        hpc += 1
      else:  
        unit.set_task(None)
        self._track_no_task(unit)
        debug(f"{unit} without task\n")
    # assign tasks in order of priority
    while len(unit_queue) > 0:
      #if self.agent.env.step in { 21 }:
      # debug(f"uq={unit_queue}\n")
      _, _, unit, task, path, pcost = heapq.heappop(unit_queue)
      if self._still_valid(unit, task) and (task.type == Task.SAP or self._check_path(unit, path, pcost)):
        self._start_task(unit, task, path)
      else:
        # task not valid or path cost changed, recalculate top task for this unit and push back to queue
        top_task, top_path, top_cost = self._top_task(unit)
        if top_task is not None:
          item = (-top_task.priority, hpc, unit, top_task, top_path, top_cost)
          heapq.heappush(unit_queue, item)
          hpc += 1
        else:  
          unit.set_task(None)
          self._track_no_task(unit)
          debug(f"{unit} without task\n")

  def cstatus(self):
    s = "Crowding:\n"
    for dstep in range(3):
      s += f"Track map {dstep}:\n"
      for y in range(24):
        for x in range(24):
          s += str(self.crowding((x,y), dstep)).zfill(2) + " "
        s += "\n"
    return s
