import sys, os

def init_debug():
  if os.path.exists("C:\\"): 
    with open("debug.txt", "wt") as fd:
      fd.write("Debug started\n")
      fd.flush()
  else:
    sys.stderr.write("Debug started\n")

def debug(s):
  if os.path.exists("C:\\"): 
    with open("debug.txt", "at") as fd:
      fd.write(s)
      fd.flush()
  else:
    sys.stderr.write(s)

center = (12, 12)

spawn_points = [ (0, 0), (23, 23) ]

def mirror(pos):
  return (23-pos[1], 23-pos[0])

dir2str = { 0: "C", 1: "U", 2: "R", 3: "D", 4: "L", 5: "S" }

dir2delta = { 0: (0,0), 1: (0,-1), 2: (1,0), 3: (0,1), 4: (-1,0), 5: (0,0) }

DELTAS = [ (1, 0), (-1, 0), (0, 1), (0, -1) ]

delta2dir = { (0, 0):0, (1, 0): 2, (-1, 0): 4, (0, 1): 3, (0, -1): 1 }

def path2str(path):
  return "".join(dir2str[d] for d in path)

def manhattan(posa, posb):
    return abs(posa[0]-posb[0]) + abs(posa[1]-posb[1])

def euclid(posa, posb):
    return ((posa[0]-posb[0])**2 + (posa[1]-posb[1])**2) ** 0.5

def max_distance(posa, posb):
  return max(abs(posa[0]-posb[0]), abs(posa[1]-posb[1]))

def on_map(x, y):
  return 0 <= x < 24 and 0 <= y < 24

def clip(v, mi, ma):
  return min(max(v, mi), ma)

def _plus(x,y):
   return set((x+dx, y+dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1] if on_map(x+dx, y+dy) and abs(dx)+abs(dy) <= 1)

def _adjacent(x,y):
   return set((x+dx, y+dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1] if on_map(x+dx, y+dy) and abs(dx)+abs(dy) <= 2)

def _romb(x,y):
   return set((x+dx, y+dy) for dx in [-2, -1, 0, 1, 2] for dy in [-2, -1, 0, 1, 2] if on_map(x+dx, y+dy) and abs(dx)+abs(dy) <= 2)

def _square(x,y,d):
   return set((x+dx, y+dy) for dx in range(-d, d+1) for dy in range(-d, d+1) if on_map(x+dx, y+dy))

pos_to_plus = { (x,y): _plus(x,y) for x in range(24) for y in range(24) }
pos_to_3x3 = { (x,y): _adjacent(x,y) for x in range(24) for y in range(24) }
pos_to_romb = { (x,y): _romb(x,y) for x in range(24) for y in range(24) }
pos_to_5x5 = { (x,y): _square(x,y,2) for x in range(24) for y in range(24) }
pos_to_7x7 = { (x,y): _square(x,y,3) for x in range(24) for y in range(24) }
pos_to_9x9 = { (x,y): _square(x,y,4) for x in range(24) for y in range(24) }

pos_to_belt = { (x,y): _square(x,y,4) - _square(x,y,1) for x in range(24) for y in range(24) }

pos_to_srange = { 
  1: { (x,y): _square(x, y, 1) for x in range(24) for y in range(24) },
  2: { (x,y): _square(x, y, 2) for x in range(24) for y in range(24) },
  3: { (x,y): _square(x, y, 3) for x in range(24) for y in range(24) },
  4: { (x,y): _square(x, y, 4) for x in range(24) for y in range(24) }
 }

