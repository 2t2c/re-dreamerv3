import collections
import threading
import elements

import numpy as np


class Fifo:

  def __init__(self):
    self.queue = collections.deque()

  def __call__(self):
    return self.queue[0]

  def __len__(self):
    return len(self.queue)

  def __setitem__(self, key, stepids):
    self.queue.append(key)

  def __delitem__(self, key):
    if self.queue[0] == key:
      self.queue.popleft()
    else:
      # This is very slow but typically not used.
      self.queue.remove(key)


class Uniform:

  def __init__(self, seed=0):
    self.indices = {}
    self.keys = []
    self.rng = np.random.default_rng(seed)
    self.lock = threading.Lock()

  def __len__(self):
    return len(self.keys)

  def __call__(self):
    with self.lock:
      index = self.rng.integers(0, len(self.keys)).item()
      return self.keys[index]

  def __setitem__(self, key, stepids):
    with self.lock:
      self.indices[key] = len(self.keys)
      self.keys.append(key)

  def __delitem__(self, key):
    with self.lock:
      assert 2 <= len(self), len(self)
      index = self.indices.pop(key)
      last = self.keys.pop()
      if index != len(self.keys):
        self.keys[index] = last
        self.indices[last] = index


class Recency:

  def __init__(self, uprobs, seed=0):
    assert uprobs[0] >= uprobs[-1], uprobs
    self.uprobs = uprobs
    self.tree = self._build(uprobs)
    self.rng = np.random.default_rng(seed)
    self.step = 0
    self.steps = {}
    self.items = {}

  def __len__(self):
    return len(self.items)

  def __call__(self):
    for retry in range(10):
      try:
        age = self._sample(self.tree, self.rng)
        if len(self.items) < len(self.uprobs):
          age = int(age / len(self.uprobs) * len(self.items))
        return self.items[self.step - 1 - age]
      except KeyError:
        # Item might have been deleted very recently.
        if retry < 9:
          import time
          time.sleep(0.01)
        else:
          raise

  def __setitem__(self, key, stepids):
    self.steps[key] = self.step
    self.items[self.step] = key
    self.step += 1

  def __delitem__(self, key):
    step = self.steps.pop(key)
    del self.items[step]

  def _sample(self, tree, rng, bfactor=16):
    path = []
    for level, prob in enumerate(tree):
      p = prob
      for segment in path:
        p = p[segment]
      index = rng.choice(len(segment), p=p)
      path.append(index)
    index = sum(
        index * bfactor ** (len(tree) - level - 1)
        for level, index in enumerate(path))
    return index

  def _build(self, uprobs, bfactor=16):
    assert np.isfinite(uprobs).all(), uprobs
    assert (uprobs >= 0).all(), uprobs
    depth = int(np.ceil(np.log(len(uprobs)) / np.log(bfactor)))
    size = bfactor ** depth
    uprobs = np.concatenate([uprobs, np.zeros(size - len(uprobs))])
    tree = [uprobs]
    for level in reversed(range(depth - 1)):
      tree.insert(0, tree[0].reshape((-1, bfactor)).sum(-1))
    for level, prob in enumerate(tree):
      prob = prob.reshape([bfactor] * (1 + level))
      total = prob.sum(-1, keepdims=True)
      with np.errstate(divide='ignore', invalid='ignore'):
        tree[level] = np.where(total, prob / total, prob)
    return tree


class Prioritized:

  def __init__(
      self, alpha, epsilon, seed, max_aggregation, branching=16):
    self.alpha = float(alpha)
    self.epsilon = float(epsilon)
    self.tree = SampleTree(branching, seed)
    assert alpha == 0.6, alpha
    assert epsilon == 1e-6, epsilon
    self.max_aggregation = max_aggregation
    self.lock = elements.RWLock()

    # Stores the priority of each step ID.
    # Used to compute the importance of full samples made up of these steps.
    self.prios = collections.defaultdict(int)

    # Maps each step ID to all sample keys (items) that include it.  
    # Enables efficient updates when a step's priority changes.
    self.stepitems = collections.defaultdict(list)             

    # Maps each sample key to the list of step IDs it contains.  
    # Defines the content of what gets sampled (e.g., a sequence of steps).
    self.items = {}                                           

  def prioritize(self, stepids, td_errors):
    """
    Update priorities of the provided step IDs and propagate changes to their associated keys.

    Args:
      stepids (list): List of step IDs (can be numpy arrays or bytes).
      td_errors (list): Corresponding list of TD-errors.
    """
    if not isinstance(stepids[0], bytes):
      stepids = [sid.tobytes() for sid in stepids]
    with self.lock.writing:
      for sid, td_error in zip(stepids, td_errors):
        try:
          self.prios[sid] = (np.abs(td_error) + self.epsilon) ** self.alpha
        except KeyError:
          print('Ignoring pirority update for removed timestep.')
          
      # Recompute item-level priorities in the sum-tree
      keys = []
      for sid in stepids:
        keys += self.stepitems.get(sid, [])
      for key in set(keys):
        try:
          self.tree.update(key, self._aggregate(key))
        except KeyError:
          print('Ignoring tree update for remoed timestep')

  def __len__(self):
    with self.lock.reading:
      return len(self.items)

  def __call__(self):
    with self.lock.reading:
      key = self.tree.sample()
      return key

  def __setitem__(self, key, stepids):
    if not isinstance(stepids[0], bytes):
      stepids = [sid.tobytes() for sid in stepids]
    with self.lock.writing:
      self.items[key] = stepids
      [self.stepitems[sid].append(key) for sid in stepids]
      self.tree.insert(key, self._aggregate(key))

  def __delitem__(self, key):
    with self.lock.writing:
      self.tree.remove(key)
      stepids = self.items.pop(key)
      for sid in stepids:
        stepitems = self.stepitems[sid]
        stepitems.remove(key)
        if not stepitems:
          del self.stepitems[sid]
          del self.prios[sid]
          del self.visit[sid]

  def _aggregate(self, key):
    prios = [self.prios[sid] for sid in self.items[key]]

    if self.max_aggregation:
      return max(prios)

    mean = sum(prios) / len(prios)
    return mean

class Curious:

  """
  A sampling buffer that prioritizes some data over others using both
  adversarial (loss-based) and count-based terms, with thread-safe
  read/write locking via elements.RWLock.
  """

  def __init__(self, alpha, beta, c, epsilon, seed, max_aggregation, branching=16):
    self.alpha   = float(alpha)
    self.beta    = float(beta)
    self.c       = float(c)
    self.eps     = float(epsilon)
    assert alpha == 0.7, alpha
    assert beta == 0.7, beta
    assert c == 1e4, c
    assert epsilon == 0.1, epsilon
    self.max_aggregation = max_aggregation
    self.tree  = SampleTree(branching, seed)
    self.prios = collections.defaultdict(int)
    self.visit = collections.defaultdict(int)
    self.stepitems = collections.defaultdict(list)
    self.items = {}

    self.lock = elements.RWLock()

  # ------------------------------------------------------------------
  # Update priorities given raw model-losses for the corresponding
  # step IDs.
  # ------------------------------------------------------------------
  def prioritize(self, stepids, losses):
    if not isinstance(stepids[0], bytes):
      stepids = [sid.tobytes() for sid in stepids]
    with self.lock.writing:
      for sid, loss in zip(stepids, losses):
        v = self.visit[sid]
        try:
          # https://github.com/AutonomousAgentsLab/cr-dv3/blob/main/dreamerv3/embodied/replay/curious_replay.py#L13
          adversarial_priority = (np.abs(float(loss)) + self.eps) ** self.alpha
          count_based_priority = self.c * (self.beta ** v)
          self.prios[sid] = adversarial_priority + count_based_priority

          self.visit[sid] = v + 1
        except KeyError:
          print('Ignoring pirority update for removed timestep.')
          
      # Recompute item-level priorities in the sum-tree
      keys = []
      for sid in stepids:
        keys += self.stepitems.get(sid, [])
      for key in set(keys):
        try:
          self.tree.update(key, self._aggregate(key))
        except KeyError:
          print('Ignoring tree update for remoed timestep')

  def __len__(self):
    with self.lock.reading:
      return len(self.items)

  def __call__(self):
    with self.lock.reading:
      key = self.tree.sample()
      return key

  def __setitem__(self, key, stepids):
    if not isinstance(stepids[0], bytes):
      stepids = [sid.tobytes() for sid in stepids]
    with self.lock.writing:
      self.items[key] = stepids
      [self.stepitems[sid].append(key) for sid in stepids]
      self.tree.insert(key, self._aggregate(key))

  def __delitem__(self, key):
    with self.lock.writing:
      self.tree.remove(key)
      stepids = self.items.pop(key)
      for sid in stepids:
        stepitems = self.stepitems[sid]
        stepitems.remove(key)
        if not stepitems:
          del self.stepitems[sid]
          del self.prios[sid]
          del self.visit[sid]

  def _aggregate(self, key):
    prios = [self.prios[sid] for sid in self.items[key]]

    if self.max_aggregation:
      return max(prios)

    mean = sum(prios) / len(prios)
    return mean


class Mixture:

  def __init__(self, selectors, fractions, seed=0):
    assert set(selectors.keys()) == set(fractions.keys())
    assert sum(fractions.values()) == 1, fractions
    for key, frac in list(fractions.items()):
      if not frac:
        selectors.pop(key)
        fractions.pop(key)
    keys = sorted(selectors.keys())
    self.selectors = [selectors[key] for key in keys]
    self.fractions = np.array([fractions[key] for key in keys], np.float32)
    self.rng = np.random.default_rng(seed)

  def __call__(self):
    return self.rng.choice(self.selectors, p=self.fractions)()

  def __setitem__(self, key, stepids):
    for selector in self.selectors:
      selector[key] = stepids

  def __delitem__(self, key):
    for selector in self.selectors:
      del selector[key]

  def prioritize(self, stepids, priorities):
    for selector in self.selectors:
      if hasattr(selector, 'prioritize'):
        selector.prioritize(stepids, priorities)
  
  def __len__(self):
    if not self.selectors:
          return 0
    # selectors may be a list or a dict; get the iterable view:
    children = (self.selectors.values()
                if isinstance(self.selectors, dict)
                else self.selectors)
    return min(len(sel) for sel in children)
    


class SampleTree:

  def __init__(self, branching=16, seed=0):
    assert 2 <= branching
    self.branching = branching
    self.root = SampleTreeNode()
    self.last = None
    self.entries = {}
    self.rng = np.random.default_rng(seed)

  def __len__(self):
    return len(self.entries)

  def insert(self, key, uprob):
    if not self.last:
      node = self.root
    else:
      ups = 0
      node = self.last.parent
      while node and len(node) >= self.branching:
        node = node.parent
        ups += 1
      if not node:
        node = SampleTreeNode()
        node.append(self.root)
        self.root = node
      for _ in range(ups):
        below = SampleTreeNode()
        node.append(below)
        node = below
    entry = SampleTreeEntry(key, uprob)
    node.append(entry)
    self.entries[key] = entry
    self.last = entry

  def remove(self, key):
    entry = self.entries.pop(key)
    entry_parent = entry.parent
    last_parent = self.last.parent
    entry.parent.remove(entry)
    if entry is not self.last:
      entry_parent.append(self.last)
    node = last_parent
    ups = 0
    while node.parent and not len(node):
      above = node.parent
      above.remove(node)
      node = above
      ups += 1
    if not len(node):
      self.last = None
      return
    while isinstance(node, SampleTreeNode):
      node = node.children[-1]
    self.last = node

  def update(self, key, uprob):
    entry = self.entries[key]
    entry.uprob = uprob
    entry.parent.recompute()

  def sample(self):
    node = self.root
    while isinstance(node, SampleTreeNode):
      uprobs = np.array([x.uprob for x in node.children])
      total = uprobs.sum()
      if not np.isfinite(total):
        finite = np.isinf(uprobs)
        probs = finite / finite.sum()
      elif total == 0:
        probs = np.ones(len(uprobs)) / len(uprobs)
      else:
        probs = uprobs / total
      choice = self.rng.choice(np.arange(len(uprobs)), p=probs)
      node = node.children[choice.item()]
    return node.key


class SampleTreeNode:

  __slots__ = ('parent', 'children', 'uprob')

  def __init__(self, parent=None):
    self.parent = parent
    self.children = []
    self.uprob = 0

  def __repr__(self):
    return (
        f'SampleTreeNode(uprob={self.uprob}, '
        f'children={[x.uprob for x in self.children]})'
    )

  def __len__(self):
    return len(self.children)

  def __bool__(self):
    return True

  def append(self, child):
    if child.parent:
      child.parent.remove(child)
    child.parent = self
    self.children.append(child)
    self.recompute()

  def remove(self, child):
    child.parent = None
    self.children.remove(child)
    self.recompute()

  def recompute(self):
    self.uprob = sum(x.uprob for x in self.children)
    self.parent and self.parent.recompute()


class SampleTreeEntry:

  __slots__ = ('parent', 'key', 'uprob')

  def __init__(self, key=None, uprob=None):
    self.parent = None
    self.key = key
    self.uprob = uprob
