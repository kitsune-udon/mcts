import mcts
from mcts import UCB, MCTS, TimeBudget
import math, random, os, argparse

class TwentyOneReward(mcts.Reward):
  def __init__(self, reward, player_id):
    self.reward = reward
    self.player_id = player_id
  def apply(self, node):
    if self.player_id == node.state.player_id:
      node.accumulated_reward -= self.reward
    else:
      node.accumulated_reward += self.reward

class TwentyOneState(mcts.State):
  n_playouts = 10
  def __init__(self, player_id, cursor=1):
    self.player_id = player_id
    self.cursor = cursor
  def next_state(self, action):
    next_cursor = self.cursor + action
    next_player_id = 3 - self.player_id
    return TwentyOneState(next_player_id, next_cursor)
  def actions(self):
    limit = min(22 - self.cursor, 3)
    return list(range(1, limit+1))
  def is_terminal(self):
    return len(self.actions()) == 0
  def playout(self):
    n_playouts = self.__class__.n_playouts
    s0 = self
    r = 0.
    for _ in range(n_playouts):
      s = s0
      while not s.is_terminal():
        a = random.choice(s.actions())
        s = s.next_state(a)
      r += 1. if s.player_id == s0.player_id else -1.
    return TwentyOneReward(r/n_playouts, s0.player_id)

def count_tree_size(node):
  if len(node.children()) == 0:
    return 1
  else:
    count = 0
    for v in node.children():
      count += count_tree_size(v)
    return count+1

def count_tree_depth(node):
  if len(node.children()) == 0:
    return 1
  else:
    count = []
    for v in node.children():
      count.append(count_tree_depth(v))
    return max(count) + 1

def ucb_of_children(ucb, node):
  d = {}
  for v in node.children():
    d[v] = ucb.calc(node, v)
  return d

def children_info(ucb, node):
  d = ucb_of_children(ucb, node)
  xs = []
  for v in node.children():
    xs.append((v.action, v.average_reward(), d[v], v.n_visits))
  xs.sort()
  ys = list(map(lambda x: "act:{} / R_ave:{:.2g} / ucb:{:.2g} / Nv:{}".format(*x), xs))
  return os.linesep.join(ys)

def report_result(r):
  d = {}
  d["tree_size"] = count_tree_size(r["head"])
  d["tree_depth"] = count_tree_depth(r["head"])
  d["children_info"] = children_info(r["ucb"], r["head"])
  d["elapsed_time"] = r["budget"].elapsed_time()
  return os.linesep.join(["tree_size : {} / tree_depth : {} / elapsed : {:.2g} secs".format(d["tree_size"], d["tree_depth"], d["elapsed_time"]),
    d["children_info"]])

def get_number(message):
  return int(input(message))

def main():
  parser = argparse.ArgumentParser(description="21 game with MCTS AI")
  parser.add_argument('--verbose', action='store_true', help="report evidences of AI behaviors")
  parser.add_argument('--time_limit', action='store', default=3, type=float, help="time of computing")
  parser.add_argument('--exploration_constant', action='store', default=0.1, type=float, help='UCB\'s exploration factor')
  parser.add_argument('--n_playouts', action='store', default=5, type=int, help="number of playouts")
  args = parser.parse_args()
  TwentyOneState.n_playouts = args.n_playouts
  scaler = lambda x: 0.5 * (x + 1)
  ucb = UCB(args.exploration_constant, scaler)
  solver = MCTS(ucb)
  human_id = get_number("select your player_id (1,2)>")
  assert(human_id in [1, 2])
  com_id = 3 - human_id
  s = TwentyOneState(1, cursor=1)
  while not s.is_terminal():
    print("current starting number:{}".format(s.cursor))
    print("player{}'s turn".format(s.player_id))
    if s.player_id == human_id:
      actions = s.actions()
      actions_str = ",".join(map(str, actions))
      c = get_number("take action ({})>".format(actions_str))
      assert(c in actions)
      a = c
    else:
      assert(s.player_id == com_id)
      r = solver.search(s, TimeBudget(args.time_limit))
      if args.verbose:
        print(report_result(r))
      a = r["best_action"]
    print("player{}'s action: take {} steps".format(s.player_id, a))
    s = s.next_state(a)
  if s.player_id == com_id:
    print("you lose")
  else:
    print("you win")

if __name__ == "__main__":
  main()
