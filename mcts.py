import math, random, time

''' Monte Carlo Tree Search Algorithm '''
class MCTS:
  def __init__(self, ucb):
    self.ucb = ucb

  def search(self, state, budget):
    v = v0 = Node(state)

    while not budget.is_exhausted():
      v = self.select_and_expand(v0)
      reward = v.state.playout()
      self.backup(v, reward)
      budget.consume()

    best_action = self.best_child(v0, mode="average_reward").action
    result = {"head":v0,
              "ucb":self.ucb,
              "budget":budget,
              "best_action":best_action}
    return result

  def select_and_expand(self, node):
    v = node

    while not v.is_terminal():
      if v.is_fully_expanded():
        v = self.best_child(v)
      else:
        return self.expand(v)

    return v

  def expand(self, node):
    v = node
    s = v.state
    a = random.choice(v.untried_actions())

    # create child node
    child_state = s.next_state(a)
    child_node = Node(child_state)
    child_node.action = a
    child_node.parent = v
    v.add_child(child_node)
    v.register_tried_action(a)

    return child_node

  def best_child(self, node, mode="ucb"):
    max_score = float('-inf')
    v_max = None

    for v in node.children():
      if v.n_visits == 0:
        score = float('inf')
      else:
        if mode == "ucb":
          score = self.ucb.calc(node, v)
        elif mode == "average_reward":
          score = v.average_reward()
        else:
          raise RuntimeError("unknown mode : {}".format(mode))

      if score > max_score:
        v_max = v
        max_score = score

    return v_max

  def backup(self, node, reward):
    v = node
    while v is not None:
      v.n_visits += 1
      reward.apply(v)
      v = v.parent

''' 探索木ノード '''
class Node:
  def __init__(self, state):
    self.state = state
    self.n_visits = 0
    self.accumulated_reward = 0.
    self.parent = None
    self.action = None
    self._children = []
    self._tried_actions = []

  def register_tried_action(self, action):
      self._tried_actions.append(action)

  def add_child(self, node):
    self._children.append(node)

  def untried_actions(self):
    return list(set(self.state.actions()) - set(self._tried_actions))

  def is_terminal(self):
    return self.state.is_terminal()

  def is_fully_expanded(self):
    return len(self.untried_actions()) == 0

  def children(self):
    return self._children

  def average_reward(self):
    return self.accumulated_reward / self.n_visits

''' 探索時の計算資源制約 '''
class Budget:
  def consume(self):
    pass
  def is_exhausted(self):
    raise NotImplementedError

class CountBudget(Budget):
  def __init__(self, n):
    self.n = n
    self.remain = n
  def consume(self):
    self.remain -= 1
  def is_exhausted(self):
    return self.remain <= 0

class TimeBudget(Budget):
  def __init__(self, secs):
    self.secs = secs
    self.created_time = time.time()
    self.updated_time = self.created_time
    self.limit = self.created_time + secs
  def consume(self):
    self.updated_time = time.time()
  def is_exhausted(self):
    return self.updated_time > self.limit
  def elapsed_time(self):
    return self.updated_time - self.created_time

''' プレイアウトによって得られる報酬 '''
class Reward:
  def apply(self, node):
    raise NotImplementedError

class State:
  def playout(self):
    raise NotImplementedError
  def next_state(self, action):
    raise NotImplementedError
  def actions(self):
    raise NotImplementedError
  def is_terminal(self):
    raise NotImplementedError

''' Upper Confidence Bound '''
class UCB:
  def __init__(self, exploration_constant, scaler=lambda x: x):
    self.c = exploration_constant
    self.scaler = scaler
  def calc(self, parent_node, child_node):
    x = self.scaler(child_node.average_reward())
    y = self.c * math.sqrt(2 * math.log(parent_node.n_visits) / child_node.n_visits)
    return x + y
