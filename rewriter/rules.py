from nltk.tree import *
import en
import copy, re

class TreeUtil:
   @classmethod
   def has_left_sibling(cls, node, check):
      '''
      check if the node has a left sibling satisfying check
      '''
      while node.left_sibling():
         node = node.left_sibling()
         if check(node):
            return True
      return False

   @classmethod
   def get_all_toks(cls, node, check):
      '''
      return all toks satisfying check
      '''
      return [leaf[0][1] for leaf in node.subtrees(check)]

   @classmethod
   def clean_dangling_node(cls, node):
      if type(node) == tuple:
         return
      elif len(node) == 0:
         cls.delete(node)
      else:
         for child in node:
            cls.clean_dangling_node(child)

   @classmethod
   def remove_redundant_unary_chain(cls, node):
      if node.height() < 3:
         return
      if len(node) == 1 and node.label() == node[0].label():
         parent = node.parent()
         if parent:
            insert_idx = TreeUtil.delete(node)
            node[0]._parent = None
            parent.insert(insert_idx, node[0])
      for child in node:
         cls.remove_redundant_unary_chain(child)

   @classmethod
   def get_leftmost_leaf(cls, node):
      while not cls.is_leaf(node):
         node = node[0]
      return node

   @classmethod
   def token(cls, node):
      '''
      assumes node is a leaf
      '''
      if not cls.is_leaf(node):
         return None, None
      leaf = node.leaves()[0]
      return leaf[0], leaf[1].lower()


   @classmethod
   def is_leaf(cls, node):
      '''
      (A a) or (A ) or a
      '''
      if not isinstance(node, ParentedTree):
         print 'should not come to this level!'
         raise ValueError
      if len(node) == 0:
         return False
      if not isinstance(node[0], ParentedTree):
         return True
      return False


   @classmethod
   def has_children(cls, node, check):
      '''
      check is the node's children contain a certain label
      '''
      for c in node:
         if check(c):
            return True
      return False


   @classmethod
   def has_ancestor(cls, node, check):
      '''
      return the lowest ancestor satisfying check
      '''
      while node:
         parent = node.parent()
         if parent and check(parent):
            return parent
         else:
            node = parent
      return None


   @classmethod
   def label_idx(cls, tree):
      '''
      make (id, tok) tuple
      '''
      idx = [0]
      cls.label_idx_aux(tree, idx)
      assert idx[0] == len(tree.leaves())


   @classmethod
   def label_idx_aux(cls, tree, idx):
      for child in tree:
         if cls.is_leaf(child):
            tok = child[0]
            child[0] = (str(idx[0]), tok)
            idx[0] += 1
         else:
            cls.label_idx_aux(child, idx)

   @classmethod
   def set_idx(cls, tree, idx):
      '''
      set id in (id, tok)
      '''
      for child in tree:
         if cls.is_leaf(child):
            tok = child[0][1]
            child[0] = (str(idx), tok)
         else:
            cls.set_idx(child, idx)

   @classmethod
   def deepcopy_parentedtree(cls, parentedtree):
      tree = Tree.convert(parentedtree)
      new_tree = copy.deepcopy(tree)
      new_tree = ParentedTree.convert(new_tree)
      parentedtree = ParentedTree.convert(tree)
      return new_tree


   @classmethod
   def get_unary_chain_leaf(cls, node):
      '''
      get the leaf of a unary rule chain
      '''
      while not cls.is_leaf(node):
         if len(node) > 1:
            return None
         else:
            node = node[0]
      return node

   @classmethod
   def delete(cls, node):
      parent = node.parent()
      if not parent:
         return None
      id_ = node.treeposition()[-1]
      del parent[id_]
      return id_

class Util:
   be = ['being', 'be', 'am', 'is', 'are', 'was', 'were', 'been', "'s", "'re"]
   have = ['have', 'has', 'had']
   do = ['do', 'does', 'did']
   # pronoun_person = {'you':2, 'they':2, 'we':2, 'i':1, 'it':3, 'he':3, 'she':3}
   pronoun_person = {'you':2, 'they':2, 'we':2, 'i':1, 'it':3, 'he':3, 'she':3, 'themselves':3, '\'em':3, 'em':3, 'them':3, 'her': 3, 'him':3, 'ourselves': 2, 'me': 1}

   @classmethod
   def present(cls, verb):
      try:
         present_verb = en.verb.present(verb)
         if present_verb:
            return present_verb
         return verb
      except KeyError:
         return verb

   @classmethod
   def pronoun_to_tree(cls, pronoun):
      if pronoun == 'they':
         return ParentedTree('NP', [ParentedTree('PRP', [('-1', 'they')])])
      elif pronoun == 'it':
         return ParentedTree('NP', [ParentedTree('PRP', [('-1', 'it')])])
      else:
         raise ValueError

   @classmethod
   def np_to_pronoun(cls, np):
      for child in np:
         label = child.label()
         if label == 'NP':
            return cls.np_to_pronoun(child)
         elif label == 'CC' and child.leaves()[0] == 'and':
            return cls.pronoun_to_tree('they')
         elif label == 'NNS' and not child.right_sibling():
            return cls.pronoun_to_tree('they')
         elif (label == 'NN' or label == 'NNP') and not child.right_sibling():
            return cls.pronoun_to_tree('it')
      return cls.pronoun_to_tree('it')


   @classmethod
   def person(cls, np):
      '''
      decided by the left-most noun unless there is CC(and)
      '''
      person = 3
      for child in np:
         label = child.label()
         if label == 'NP':
            return cls.person(child)
         elif label == 'CC' and child.leaves()[0] == 'and':
            return 2
         elif label == 'NNS':
            person = 2
         elif label == 'NN' or label == 'NNP':
            person = 3
         elif label == 'PRP':
            tok = TreeUtil.token(child)[1]
            person = cls.pronoun_person[tok]
      return person


class Rule(object):
   def __init__(self, name):
      self.name = name
      self.num_applied = 0
      self.applied = False
      self.num_accepted = 0
      self.fail = False

   def apply(self, tree):
      raise NotImplementedError

class ArrayUtil:
   @classmethod
   def find(cls, array, check):
      for i, x in enumerate(array):
         if check(x):
            return i
      return None

class Clause(Rule):
   def __init__(self):
      super(Clause, self).__init__('clause')

   def apply(self, tree):
      self.applied = False
      self.it_is_to(tree)
      self.it_is_that(tree)
      return self.applied

   def it_is_that(self, tree):
      if TreeUtil.is_leaf(tree):
         return
      if tree.label() == 'S':
         for child in tree:
            if child.label() == 'NP' and len(child) == 1:
               leaf = TreeUtil.get_unary_chain_leaf(child)
               if leaf and leaf[0][1] == 'it':
                  it = child
                  right_sib = child.right_sibling()
                  if right_sib and right_sib.label() == 'VP' \
                        and len(right_sib) >= 3 \
                        and TreeUtil.token(right_sib[0])[1] in ['is', 'was'] \
                        and right_sib[1].label() == 'ADJP' \
                        and right_sib[2].label() == 'SBAR':
                     clause = right_sib[2]
                     self.move_clause(it, clause)
                     self.applied = True

      for child in tree:
         self.it_is_that(child)

   def it_is_to(self, tree):
      if TreeUtil.is_leaf(tree):
         return
      if tree.label() == 'S':
         for child in tree:
            if child.label() == 'NP' and len(child) == 1:
               leaf = TreeUtil.get_unary_chain_leaf(child)
               if leaf and leaf[0][1] == 'it':
                  it = child
                  right_sib = child.right_sibling()
                  if right_sib and right_sib.label() == 'VP' \
                        and len(right_sib) >= 2 \
                        and TreeUtil.token(right_sib[0])[1] in ['is', 'was'] \
                        and right_sib[1].label() == 'ADJP' \
                        and right_sib[1][0].label() == 'JJ':
                     # sometimes S is child of ADJP, sometimes is sibling
                     nodes = [right_sib[1], right_sib]
                     clause = self.find_clause(nodes)
                     if clause:
                        self.move_clause(it, clause)
                        self.applied = True
                        break

      for child in tree:
         self.it_is_to(child)

   def find_clause(self, nodes):
      for node in nodes:
         for i in range(1, len(node)):
            if node[i].label() == 'S' \
                  and node[i].leaves()[0][1] == 'to':
               return node[i]
      return None

   def move_clause(self, it, clause):
      root = it.parent()
      insert_id = TreeUtil.delete(it)
      TreeUtil.delete(clause)
      clause._parent = None
      root.insert(insert_id, clause)


class Verb(Rule):
   verbs = set(['say', \
         'think', \
         'believe', \
         'argue', \
         'feel', \
         'know', \
         'report', \
         'claim', \
         'add', \
         'see', \
         'assume', \
         'forecast', \
         'hypothesize', \
         'hear', \
         'observe', \
         'order', \
         'understand', \
         'suspect', \
         'deny', \
         'announce', \
         'indicate', \
         'fear', \
         'recommend', \
         'realize', \
         'expect', \
         'reveal', \
         'pretend', \
         # TODO: don't rewrite "allows for"
         'allow', \
         'ask', \
         # might be to-clause
         'agree', \
         'decide', \
         'appear', \
         'inform', \
         'write', \
         'expect', \
         'hope'])

   skip_verbs = Util.have + Util.do + Util.be

   def __init__(self):
      super(Verb, self).__init__('verb')

   def apply(self, tree):
      self.applied = False
      tok_set = set([Util.present(tok) for tok in TreeUtil.get_all_toks(tree, lambda x: x.label().startswith('VB'))])
      # apply twice anyway
      #num_verbs = min(2, len(tok_set & self.verbs))
      if len(tok_set & self.verbs) > 0:
         for i in range(2):
            self.move_verb(tree)
      return self.applied

   def get_verb_clause(self, vp):
      if TreeUtil.is_leaf(vp):
         return None, None
      for child in vp:
         label = child.label()
         if label.startswith('VB'):
            tok = TreeUtil.token(child)[1]
            if tok not in Verb.skip_verbs:
               tok = Util.present(tok)
               if tok not in self.verbs:
                  return None, None
               else:
                  verb = child
                  clause = None
                  # said NP: don't rewrite
                  if len(vp) == 2 and child.right_sibling() \
                        and child.right_sibling().label() == 'NP':
                     return verb, None
                  # search for clause to the right of the verb
                  verb_idx = child.treeposition()[-1]
                  i = verb_idx + 1
                  end = len(vp)
                  while i < end:
                     label = vp[i].label()
                     if label in ['S', 'SBAR']:
                        clause = vp[i]
                        break
                     # if PP is the last one, then it's the clause
                     elif label == 'PP' and i == len(vp) - 1:
                        clause = vp[i]
                        break
                     # leaves between vp and clause, e.g. said : ``
                     elif vp[i].label() == ':':
                        del vp[i]
                        end -= 1
                     i += 1
                  return child, clause
         elif label == 'VP':
            return self.get_verb_clause(child)

      return None, None


   def move_verb(self, tree):
      if TreeUtil.is_leaf(tree):
         return
      if tree.label() == 'S':
         np = None
         for child in tree:
            if child.label() == 'NP':
               np = child
            elif child.label() == 'VP':
               if np and len(child) > 1:
                  verb, clause = self.get_verb_clause(child)
                  if verb and clause:
                     if self.is_to_clause(clause):
                        self.move_backward_to(np, child, verb, clause)
                     else:
                        self.move_backward_that(np, child, verb, clause)
                     self.applied = True
                  break

      for child in tree:
         self.move_verb(child)

   def is_to_clause(self, clause):
      # (not) to do
      leaves = clause.leaves()
      if leaves[0][1] == 'to' or \
            (leaves[0][1] == 'not' and leaves[1][1] == 'to'):
         return True
      return False

   def move_backward_to(self, np, vp, verb, clause):
      '''
      (S (NP ) (VP ... (verb) ... (to clause
      '''
      # TODO: change verb to correct form
      # S/SBAR -> VP
      clause.set_label('VP')
      if clause.leaves()[0][1] == 'not':
         # delete to
         # get the second left leaf
         # sometimes (TO to) is parsed differently
         to = None
         for i, node in enumerate(clause.subtrees(lambda x: x.height() == 2)):
            if i == 1:
               to = node
               break
         if to:
            TreeUtil.delete(to)
         # add do
         person = Util.person(np)
         if person == 3:
            do_tok = 'does'
            do_tag = 'VBZ'
         else:
            do_tok = 'do'
            do_tag = 'VB'
         clause.insert(0, ParentedTree(do_tag, [('-1', do_tok)]))
      else:
         # delete to
         to = TreeUtil.get_leftmost_leaf(clause)
         TreeUtil.delete(to)

      # insert clause after NP
      parent = np.parent()
      TreeUtil.delete(clause)
      clause._parent = None
      insert_idx = np.treeposition()[-1] + 1
      parent.insert(insert_idx, clause)

      # delete VP
      TreeUtil.delete(vp)
      vp._parent = None

      # make new clause
      # parent is (S (NP) (VP ...
      # root is one level up of S
      # add S above parent
      root = parent.parent()
      insert_idx = TreeUtil.delete(parent)
      parent._parent = None
      clause = ParentedTree('S', [parent])
      root.insert(insert_idx, clause)

      # append ", NP VP"
      insert_idx = clause.treeposition()[-1] + 1
      root.insert(insert_idx, ParentedTree(',', [('-1', ',')]))
      # repeat NP, don't count these in alignments
      np = TreeUtil.deepcopy_parentedtree(np)
      # don't repeat too many words
      if len(np.leaves()) > 2:
         np = Util.np_to_pronoun(np)
      TreeUtil.set_idx(np, -1)
      np._parent = None
      root.insert(insert_idx+1, np)
      root.insert(insert_idx+2, vp)
      TreeUtil.remove_redundant_unary_chain(root)
      TreeUtil.clean_dangling_node(root)

   def move_backward_that(self, np, vp, verb, clause):
      '''
      (S (NP ) (VP ... (verb) ... (that clause
      '''
      # (SBAR (IN that)
      if clause and clause.label() == 'SBAR' and TreeUtil.token(clause[0])[1] == 'that':
         TreeUtil.delete(clause[0])

      # delete clause
      # TODO: set parent to None after delete
      TreeUtil.delete(clause)
      clause._parent = None
      clause = ParentedTree('S', [clause])
      insert_idx = np.treeposition()[-1]
      root = np.parent()
      root.insert(insert_idx, clause)
      root.insert(insert_idx+1, ParentedTree(',', [('-1', ',')]))

      TreeUtil.remove_redundant_unary_chain(root)
      TreeUtil.clean_dangling_node(root)


class Seem(Rule):
   seem = ['seem', 'seems', 'seemed']

   def __init__(self):
      super(Seem, self).__init__('seem')

   def apply(self, tree):
      self.applied = False
      self.move_seems(tree)
      return self.applied

   def move_seems(self, tree):
      if TreeUtil.is_leaf(tree):
         return
      if tree.label() == 'S':
         np = None
         for child in tree:
            if child.label() == 'NP':
               np = child
            elif child.label() == 'VP':
               if np and len(child) > 1 \
                     and TreeUtil.token(child[0])[1] in self.seem:
                  # insert before clause
                  leaves = []
                  # append after verb
                  attach = []
                  clause = None
                  # leaves between vp and clause, e.g. said : ``
                  i = 1
                  while i < len(child):
                     if TreeUtil.is_leaf(child[i]):
                        leaves.append(child[i])
                     elif child[i].label() not in ['S', 'SBAR', 'PP']:
                        attach.append(child[i])
                     else:
                        clause = child[i]
                        break
                     i += 1
                  if clause:
                     seem = child[0]
                     self.move_backward(np, seem, attach, leaves, clause)
                     self.applied = True
                     break

      for child in tree:
         self.move_seems(child)

   def move_backward(self, np, seem, attach, leaves, clause):
      np_idx = np.treeposition()[-1]
      root = np.parent()
      insert_idx = np_idx + 1
      # (S (VP (TO to)
      if clause.label() == 'S' and TreeUtil.token(clause[0][0])[1] == 'to':
         TreeUtil.delete(clause[0][0])
      # (SBAR (IN that)
      if clause.label() == 'SBAR' and TreeUtil.token(clause[0])[1] == 'that':
         TreeUtil.delete(clause[0])
         if len(np) == 1 and TreeUtil.get_unary_chain_leaf(np)[0][1].lower() == 'it':
            TreeUtil.delete(np)
            insert_idx = np_idx

      while len(clause) == 1 and clause.label() in ['S', 'SBAR']:
         clause = clause[0]
      TreeUtil.delete(seem.parent())
      clause._parent = None
      new_clause = ParentedTree('S', [clause])
      for leaf in leaves[::-1]:
         if TreeUtil.token(leaf)[1] == ':':
            continue
         leaf._parent = None
         new_clause.insert(0, leaf)
      root.insert(insert_idx, new_clause)
      clause.parent().append(ParentedTree(',', [('-1', ',')]))
      clause.parent().append(ParentedTree('NP', [ParentedTree('PRP', [('-1', 'it')])]))
      seem_id = seem[0][0]
      seem_tok = seem[0][1]
      seem_tree = ParentedTree(seem.label(), [(seem_id, seem_tok)])
      clause.parent().append(ParentedTree('VP', [seem_tree]))
      for a in attach:
         a._parent = None
         clause.parent().append(a)


class LooksLike(Rule):
   look = ['look', 'looks', 'looked']

   def __init__(self):
      super(LooksLike, self).__init__('lookslike')

   def apply(self, tree):
      self.applied = False
      self.looks_like(tree)
      return self.applied

   def looks_like(self, tree):
      if TreeUtil.is_leaf(tree):
         return
      if tree.label() == 'S':
         np = None
         for child in tree:
            if child.label() == 'NP':
               np = child
            elif child.label() == 'VP':
               if np and len(child) > 1 \
                  and TreeUtil.token(child[0])[1] in self.look \
                  and len(child[1]) > 1 \
                  and TreeUtil.token(child[1][0])[1] == 'like':
                  clause = child[1][1]
                  look = child[0]
                  like = child[1][0]
                  self.move_backward(np, look, like, clause)
                  self.applied = True
                  break

      for child in tree:
         self.looks_like(child)

   def move_backward(self, np, look, like, clause):
      if clause[0].label() == 'IN' and TreeUtil.token(clause[0])[1] == 'that':
         TreeUtil.delete(clause[0])
      vp = look.parent()
      root = vp.parent()
      if not np:
         insert_idx = look.treeposition()[-1]
      else:
         np_idx = np.treeposition()[-1]
         # it looks like
         if len(np) == 1 and TreeUtil.get_unary_chain_leaf(np)[0][1].lower() == 'it':
            TreeUtil.delete(np)
            insert_idx = np_idx
         else:
            insert_idx = np_idx + 1

      # vp contains clause
      #while len(clause) == 1:
      #   clause = clause[0]
      #TreeUtil.delete(vp)
      #clause._parent = None
      #root.insert(insert_idx, ParentedTree('S', [clause]))
      insert_idx = look.parent().treeposition()[-1] + 1
      root.insert(insert_idx, ParentedTree(',', [('-1', ',')]))
      root.insert(insert_idx+1, ParentedTree('NP', [ParentedTree('PRP', [('-1', 'it')])]))
      TreeUtil.delete(look)
      look._parent = None
      TreeUtil.delete(like)
      like._parent = None
      look_id = look[0][0]
      look_tok = look[0][1]
      look_tree = ParentedTree(look.label(), [(look_id, look_tok)])
      like_id = like[0][0]
      like_tree = ParentedTree('PP', [ParentedTree('IN', [(like_id, 'like')])])
      root.insert(insert_idx+2, ParentedTree('VP', [look_tree, like_tree]))


class Conjunction(Rule):
   '''
   change order of SBAR clauses
   '''
   def __init__(self):
      super(Conjunction, self).__init__('conjunction')

   def apply(self, tree):
      self.applied = False
      self.because(tree)
      if not self.applied:
         self.because_of(tree)
      if not self.applied:
         self.despite(tree)
      if not self.applied:
         self.in_order_to(tree)
      if not self.applied:
         self.even_though(tree)
      if not self.applied:
         self.as_a_result_of(tree)
      return self.applied

   def even_though(self, tree):
      if TreeUtil.is_leaf(tree):
         return
      if tree.label() == 'SBAR' and len(tree) == 3 \
            and TreeUtil.token(tree[0])[1] == 'even' \
            and TreeUtil.token(tree[1])[1] in ['if', 'though'] \
            and tree[2].label() == 'S':
         s2 = tree[2]
         s1 = TreeUtil.has_ancestor(tree, lambda x: x.label() == 'S' and x.leaves()[0][1] != 'to')
         if s2 and s1:
            toks = []
            tok_ids = []
            for node in [tree[0], tree[1]]:
               tok_id, tok = TreeUtil.token(node)
               tok_ids.append(tok_id)
               toks.append(tok)
            self.move_backward(tok_ids, toks, s1, s2)
            self.applied = True
            # tree has been deleted, don't iterate its children
            return

      for child in tree:
         self.even_though(child)
      return self.applied

   def as_a_result_of(self, tree):
      if TreeUtil.is_leaf(tree):
         return
      if tree.label() == 'PP' and len(tree) == 2 \
            and TreeUtil.token(tree[0])[1] == 'as' \
            and len(tree[1]) == 2 and tree[1].label() == 'NP' \
            and len(tree[1][0].leaves()) == 2 \
            and tree[1][0].leaves()[0][1] == 'a' \
            and tree[1][0].leaves()[1][1] == 'result' \
            and tree[1][1].leaves()[0][1] == 'of':
         # NP
         s2 = tree[1][1][1]
         s1 = TreeUtil.has_ancestor(tree, lambda x: x.label() == 'S' and x.leaves()[0][1] != 'to')
         if s2 and s1:
            toks = []
            tok_ids = []
            nodes = [tree[0], tree[1][0][0], tree[1][0][1], tree[1][1][0]]
            for node in nodes:
               tok_id, tok = TreeUtil.token(node)
               tok_ids.append(tok_id)
               toks.append(tok)
            # delete as a result
            root = tree.parent()
            tree_idx = TreeUtil.delete(tree)
            s2.parent()._parent = None
            root.insert(tree_idx, s2.parent())
            s2.parent().set_label('NP')
            self.move_backward(tok_ids, toks, s1, s2)
            self.applied = True
            # tree has been deleted, don't iterate its children
            return

      for child in tree:
         self.as_a_result_of(child)
      return self.applied


   def according_to(self, tree):
      if TreeUtil.is_leaf(tree):
         return
      if tree.label() == 'PP' and len(tree) == 2 \
            and TreeUtil.token(tree[0])[1] == 'according' \
            and tree[1].leaves()[0][1] == 'to':
         # TODO: in order to VP, VP to VP-ing when moving forward??
         s2 = tree[2]
         s1 = TreeUtil.has_ancestor(tree, lambda x: x.label() == 'S' and x.leaves()[0][1] != 'to')
         if s2 and s1:
            toks = []
            tok_ids = []
            for node in [tree[0], TreeUtil.get_leftmost_leaf(tree[1])]:
               tok_id, tok = TreeUtil.token(node)
               tok_ids.append(tok_id)
               toks.append(tok)
            self.move_backward(tok_ids, toks, s1, s2)
            self.applied = True
            # tree has been deleted, don't iterate its children
            return

      for child in tree:
         self.according_to(child)
      return self.applied


   def in_order_to(self, tree):
      if TreeUtil.is_leaf(tree):
         return
      if tree.label() == 'SBAR' and len(tree) == 3 \
            and TreeUtil.token(tree[0])[1] == 'in' \
            and TreeUtil.token(tree[1])[1] == 'order' \
            and tree[2].label() == 'S' \
            and tree[2].leaves()[0][1] == 'to':
         # TODO: in order to VP, VP to VP-ing when moving forward??
         s2 = tree[2]
         s1 = TreeUtil.has_ancestor(tree, lambda x: x.label() == 'S' and x.leaves()[0][1] != 'to')
         if s2 and s1:
            toks = []
            tok_ids = []
            for node in [tree[0], tree[1], TreeUtil.get_leftmost_leaf(tree[2])]:
               tok_id, tok = TreeUtil.token(node)
               tok_ids.append(tok_id)
               toks.append(tok)
            self.move_backward(tok_ids, toks, s1, s2)
            self.applied = True
            # tree has been deleted, don't iterate its children
            return

      for child in tree:
         self.in_order_to(child)
      return self.applied

   def despite(self, tree):
      if TreeUtil.is_leaf(tree):
         return
      if tree.label() == 'PP' and len(tree) == 2:
         tok_id, tok = TreeUtil.token(tree[0])
         # check "of" because there are parsing errors
         # TODO: add comma before the NP phrase is probably better
         if tok == 'despite':
            s2 = tree[1]
            s1 = TreeUtil.has_ancestor(tree, lambda x: x.label() == 'S' and x.leaves()[0][1] != 'to')
            if s2 and s1:
               self.move_backward([tok_id], [tok], s1, s2)
               self.applied = True
               # tree has been deleted, don't iterate its children
               return

      for child in tree:
         self.despite(child)
      return self.applied


   def because_of(self, tree):
      if TreeUtil.is_leaf(tree):
         return
      if tree.label() == 'PP' and len(tree) == 3:
         tok1_id, tok1 = TreeUtil.token(tree[0])
         if tok1 != 'because':
            return
         tok2_id, tok2 = TreeUtil.token(tree[1])
         if tok2 != 'of':
            return
         if tree[2].label() == 'NP':
            np = tree[2]
            # because of this/that: don't rewrite
            if np.leaves()[0] in ['this', 'that']:
               return
            root = TreeUtil.has_ancestor(tree, lambda x: x.label() == 'S' or x.label() == 'SINV')
            if root:
               parent = tree.parent()
               pp_id = tree.treeposition()[-1]
               del parent[pp_id]
               del tree[2]
               np._parent = None
               root.insert(0, np)
               root.insert(1, ParentedTree('IN', [('-1', ', because of this ,')]))
               self.applied = True

      for child in tree:
         self.because_of(child)

   def move_backward(self, tok_ids, toks, s1, s2):
      '''
      conjunction word (id and tok), clause before it, clause after it
      e.g. s1 because s2
      '''
      if len(toks) == 1:
         tok = toks[0]
         tok_id = tok_ids[0]
         if tok == 'because':
            inserted_tree = ParentedTree('IN', [(tok_id, ', because of this ,')])
         elif tok == 'despite':
            inserted_tree = ParentedTree('IN', [(tok_id, ', despite this ,')])
         elif tok == 'although':
            inserted_tree = ParentedTree('IN', [(tok_id, ', although this is the case ,')])
         elif tok == 'during':
            inserted_tree = ParentedTree('IN', [(tok_id, ', during this ,')])
         else:
            print 'cannot handle', tok
            raise ValueError
      else:
         s = ' '.join(toks)
         if s == 'in order to':
            # delete "to"
            TreeUtil.delete(TreeUtil.get_leftmost_leaf(s2))
            in_tree = ParentedTree('IN', [(tok_ids[0], ', in')])
            order_tree = ParentedTree('NN', [(tok_ids[1], 'order')])
            to_tree = ParentedTree('TO', [(tok_ids[2], 'to do this ,')])
            inserted_tree = ParentedTree('SBAR', [in_tree, order_tree, to_tree])
         elif s == 'according to':
            # delete "to"
            TreeUtil.delete(TreeUtil.get_leftmost_leaf(s2))
            according_tree = ParentedTree('VBG', [(tok_ids[0], ', according')])
            to_tree = ParentedTree('TO', [(tok_ids[1], 'to this ,')])
            inserted_tree = ParentedTree('SBAR', [according_tree, to_tree])
         elif s == 'even though':
            even_tree = ParentedTree('RB', [(tok_ids[0], ', even')])
            though_tree = ParentedTree('IN', [(tok_ids[1], 'though this is the case ,')])
            inserted_tree = ParentedTree('SBAR', [even_tree, though_tree])
         elif s == 'even if':
            even_tree = ParentedTree('RB', [(tok_ids[0], ', even')])
            if_tree = ParentedTree('IN', [(tok_ids[1], 'if this is the case ,')])
            inserted_tree = ParentedTree('SBAR', [even_tree, if_tree])
         elif s == 'as a result of':
            trees = []
            labels = ['IN', 'DT', 'NN', 'IN']
            new_toks = [', as', 'a', 'result', 'of this ,']
            for i, tok_id in enumerate(tok_ids):
               trees.append(ParentedTree(labels[i], [(tok_id, new_toks[i])]))
            inserted_tree = ParentedTree('SBAR', trees)
         else:
            print 'cannot handle', toks
            raise ValueError

      # conjunction word at the begining
      # e.g. because s2 ...
      conj_begin = True
      for i, tok in enumerate(toks):
         if s1.leaves()[i][1] != tok:
            conj_begin = False
      if conj_begin:
         root = s1
         insert_id = s2.parent().treeposition()[-1]
         s1 = None
      else:
         root = s1.parent()
         insert_id = TreeUtil.delete(s1)
         s1._parent = None

      # s2.parent(): clause that include the conjunction word
      TreeUtil.delete(s2.parent())
      s2._parent = None
      root.insert(insert_id, s2)
      root.insert(insert_id+1, inserted_tree)
      if s1:
         root.insert(insert_id+2, s1)

      return

   def because(self, tree):
      if TreeUtil.is_leaf(tree):
         return
      if tree.label() in ['SBAR', 'PP'] and len(tree) == 2:
         tok_id, tok = TreeUtil.token(tree[0])
         # check "of" because there are parsing errors
         if (tok == 'because' and tree[1].leaves()[0][1] != 'of') \
               or (tok == 'although') \
               or (tok == 'during'):
            s2 = tree[1]
            s1 = TreeUtil.has_ancestor(tree, \
                  lambda x: x.label() == 'S' \
                  and x.leaves()[0][1] != 'to' \
                  and (not x.parent() or x.parent().label() != 'SBAR') \
                  )
            if s2 and s1:
               self.move_backward([tok_id], [tok], s1, s2)
               self.applied = True
               # tree has been deleted, don't iterate its children
               return

      for child in tree:
         self.because(child)

class Possessive(Rule):
   '''
   change orders of nouns
   '''
   number = ['ten', 'hundred', 'thousand', 'million', 'billion', 'tens', 'hundreds', 'thousands', 'millions', 'billions']
   pronoun = ['these', 'those', 'that', 'this', 'some']
   date = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday', 'years', 'months', 'month']

   def __init__(self):
      super(Possessive, self).__init__('post_positive')

   def apply(self, tree):
      self.applied = False
      self.post_positivize(tree)
      return self.applied

   def post_positivize(self, tree):
      if TreeUtil.is_leaf(tree):
         return
      for child in tree:
         if child.label() == 'NP':
            np1, np2 = self.get_poss_nouns(child)
            if np1 and np2 and self.remove_article(np1):
               del child[0]  # NP
               del child[0]  # PP
               np1._parent = None
               np2._parent = None
               self.post_positivize(np1)
               self.post_positivize(np2)
               # sometimes we don't need to add 's
               if TreeUtil.token(np1)[1] not in ['province']:
                  if np2.leaves()[-1][1][-1] == 's':
                     pos = "'"
                  else:
                     pos = "'s"
                  np2.append(ParentedTree('POS', [('-1', pos)]))
               child.insert(0, np2)
               for i, c in enumerate(np1):
                  c._parent = None
                  child.insert(i+1, c)
               self.applied= True
         else:
            self.post_positivize(child)


   def can_possess(self, node):
      if node.label() in ['VP', 'SBAR', 'PRP']:
         return False
      return True


   def can_be_possessed(self, node):
      if node.label() in ['CD', 'POS', 'SBAR', 'NNP']:
         return False
      if TreeUtil.is_leaf(node):
         tok = TreeUtil.token(node)[1]
         if re.search(r'\d+', tok):
            return False
         if tok in self.number or tok in self.date or tok in self.pronoun:
            return False
      return True


   def remove_article(self, np):
      dt = None
      for i, child in enumerate(np):
         if child.label() == 'DT':
            if child[0][1] not in ['the', 'a']:
               return False
            dt = i
            break
      if dt is not None:
         del np[dt]
      return True


   def get_poss_nouns(self, np):
      np1 = None
      np2 = None
      if len(np) >= 2 and \
            np[0].label() == 'NP' and np[1].label() == 'PP' and \
            len(np[1]) == 2 and \
            np[1][0].label() == 'IN' and TreeUtil.token(np[1][0])[1] == 'of' \
            and np[1][1].label() == 'NP' and \
            not TreeUtil.has_children(np[0], lambda x: not self.can_be_possessed(x)) and \
            not TreeUtil.has_children(np[1][1], lambda x: not self.can_possess(x)):
         np1 = np[0]
         np2 = np[1][1]
      return np1, np2

class Voice(Rule):
   '''
   change voice of a sentence: passivize, depassivize
   '''

   be = ['being', 'be', 'am', 'is', 'are', 'was', 'were', 'been', "'s", "'re"]
   have = ['have', 'has', 'had']
   do = ['do', 'does', 'did']
   say = ['say', 'said']
   pronoun_subj2obj = {'they':'them', 'we':'us', 'i':'me', 'he':'him', 'she':'her', 'it':'it'}
   pronoun_obj2subj = {k: v for v, k in pronoun_subj2obj.items()}
   vowel = {'a', 'e', 'i', 'o', 'u'}
   skip_verbs = be + say + have + list(Verb.verbs)

   def __init__(self):
      super(Voice, self).__init__('voice')

   def apply(self, tree):
      self.applied = False
      self.change_voice(tree)
      return self.applied

   def present_participle(self, tok):
      '''
      tok is past particple
      '''
      if tok[-2:] == 'ed':
         if len(tok) > 3 and tok[-3:] == 'ied':
            return tok[:-3] + 'ing'
         return tok[:-2] + 'ing'
      else:
         return tok + 'ing'


   def past_participle(self, tok):
      '''
      very simple heuristic; used only when en.verb.past_participle fails
      '''
      if tok[-1] == 'e':
         return tok + 'd'
      elif len(tok) > 1 and tok[-2] not in self.vowel and tok[-1] == 'y':
         return tok[:-1] + 'ied'
      else:
         return tok + 'ed'

   def change_voice(self, tree):
      if TreeUtil.is_leaf(tree):
         return
      if tree.label() == 'S':
         subj = None
         verbs = None
         obj = None
         for child in tree:
            if child.label() == 'NP':
               subj = child
            elif child.label() == 'VP':
               # multiple (verb, obj) occurs with VP CC VP
               if subj:
                  pairs = []
                  self.get_verb_np(child, pairs)
                  for i, (be_verbs, verbs, obj, voice) in enumerate(pairs):
                     # add subject to the first verb only
                     if i > 0:
                        subj = None
                        #continue
                     if voice == 'active':
                        no_be = False
                        self.passivize(subj, be_verbs, verbs, obj)
                     else:
                        self.activize(subj, be_verbs, verbs, obj)
      # recurse
      for child in tree:
         self.change_voice(child)

   def personal_pronoun_subj2obj(self, subj):
      leaf = TreeUtil.get_unary_chain_leaf(subj)
      if leaf and leaf.label() == 'PRP':
         tok_id, tok = TreeUtil.token(leaf)
         if tok in self.pronoun_subj2obj:
            leaf[0] = (tok_id, self.pronoun_subj2obj[tok])
         # cannot handle!
         else:
            return False
      return True

   def personal_pronoun_obj2subj(self, obj):
      leaf = TreeUtil.get_unary_chain_leaf(obj)
      if leaf and leaf.label() == 'PRP':
         tok_id, tok = TreeUtil.token(leaf)
         if tok in self.pronoun_obj2subj:
            leaf[0] = (tok_id, self.pronoun_obj2subj[tok])
         # cannot handle! e.g. himself
         else:
            return False
      return True

   def activize(self, subj, be_verbs, verbs, obj):
      # replace pronoun
      # obj is (PP (IN by) ...)
      if len(obj[1]) == 1:
         if not self.personal_pronoun_obj2subj(obj[1]):
            return
      if subj and len(subj) == 1:
         if not self.personal_pronoun_subj2obj(subj):
            return

      # fix verbs
      p = Util.person(obj)
      be_verb = be_verbs[0]
      be_id, be_tok = TreeUtil.token(be_verb)

      # is being done -> is doing
      if len(be_verbs) == 2 and TreeUtil.token(be_verbs[1])[1] == 'being':
         TreeUtil.delete(be_verbs[1])
         if be_tok in ['was', 'were']:
            new_be_tok = 'were' if p == 2 else 'was'
         else:
            new_be_tok = 'am' if p == 1 else 'is' if p == 3 else 'are'
         be_verb[0] = (be_id, new_be_tok)
         for verb in verbs:
            verb_id, verb_tok = TreeUtil.token(verb)
            try:
               new_verb_tok = en.verb.present_participle(verb_tok)
               if not new_verb_tok:
                  new_verb_tok = self.present_participle(verb_tok)
            except KeyError:
               new_verb_tok = self.present_participle(verb_tok)
            new_verb_tag = 'VBG'
            verb.set_label(new_verb_tag)
            verb[0] = (verb_id, new_verb_tok)
      else:
         if be_tok in ['was', 'were']:
            for verb in verbs:
               verb_id, verb_tok = TreeUtil.token(verb)
               try:
                  new_verb_tok = en.verb.past(verb_tok)
                  if not new_verb_tok:
                     new_verb_tok = self.past_participle(verb_tok)
               except KeyError:
                  new_verb_tok = self.past_participle(verb_tok)
               new_verb_tag = 'VBD'
               verb.set_label(new_verb_tag)
               verb[0] = (verb_id, new_verb_tok)
         elif be_tok in ['am', 'is', 'are', "'s", "'re"]:
            for verb in verbs:
               verb_id, verb_tok = TreeUtil.token(verb)
               try:
                  new_verb_tok = en.verb.present(verb_tok, person=p)
                  if not new_verb_tok:
                     new_verb_tok = verb_tok
               except KeyError:
                  new_verb_tok = verb_tok
               new_verb_tag = 'VBZ' if p == 3 else 'VB'
               verb.set_label(new_verb_tag)
               verb[0] = (verb_id, new_verb_tok)
         elif be_tok in ['be']:
            for verb in verbs:
               verb_id, verb_tok = TreeUtil.token(verb)
               try:
                  new_verb_tok = en.verb.present(verb_tok)
                  if not new_verb_tok:
                     new_verb_tok = verb_tok
               except KeyError:
                  new_verb_tok = verb_tok
               new_verb_tag = 'VB'
               verb.set_label(new_verb_tag)
               verb[0] = (verb_id, new_verb_tok)
         else:
            for verb in verbs:
               verb_id, verb_tok = TreeUtil.token(verb)
               new_verb_tok = verb_tok
               new_verb_tag = verb.label()
               verb.set_label(new_verb_tag)
               verb[0] = (verb_id, new_verb_tok)

         TreeUtil.delete(be_verb)


      # swap subj and (by+) obj
      obj_idx = obj.treeposition()[-1]
      vp = obj.parent()  # PP's parent
      TreeUtil.delete(obj)
      obj = obj[1]
      obj._parent = None

      if subj:
         s = subj.parent()
         subj_idx = subj.treeposition()[-1]
         del s[subj_idx]
         subj._parent = None

         # swap and insert
         vp.insert(obj_idx, subj)
         s.insert(subj_idx, obj)

      self.applied = True


   def passivize(self, subj, be_verbs, verbs, obj):
      # replace pronoun
      if len(obj) == 1:
         if not self.personal_pronoun_obj2subj(obj):
            return
      if subj and len(subj) == 1:
         if not self.personal_pronoun_subj2obj(subj):
            return
      #print subj
      #print verbs
      #print obj

      # all change based on the first verb if there are multiple verbs
      verb = None
      for v in verbs:
         v_tok = TreeUtil.token(v)[1]
         if v_tok not in self.have:
            verb = v
            break
      if not verb:
         return
      vp = verb.parent()
      verb_idx = verb.treeposition()[-1]

      # add be-word
      if not be_verbs:

         # whether we have used special rules or general rules
         special = False

         # TODO: factor this into functions
         # special rules
         root = subj.parent() if subj else None
         if root and root.parent():
            left_sib = root.left_sibling()
            # don't add be
            if root.parent().label() in ['SBAR', 'S'] \
                  and left_sib and left_sib.leaves()[0][1] == 'what':
               TreeUtil.delete(left_sib)
               be_tok = None
               special = True
            elif root.parent().label() == 'VP' \
                  and left_sib.label().startswith('VB') \
                  and verb.label() == 'VB':
               #verb_tok = left_sib[0][1]
               #verb_tok = Util.present(verb_tok)
               #if verb_tok in ['help', 'make', 'let', 'have', 'suggest']:
               be_tok = 'be'
               be_tag = 'VB'
               special = True

         # general rules
         if not special:
            # have + been
            if TreeUtil.has_left_sibling(vp, \
                  lambda x: x.label()[:2] == 'VB' and TreeUtil.token(x)[1] in self.have):
               be_tok = 'been'
               be_tag = 'VBN'
            # may/can/... + be
            elif TreeUtil.has_left_sibling(vp, \
                  lambda x: x.label()[:2] == 'MD'):
               be_tok = 'be'
               be_tag = 'VB'
            else:
               p = Util.person(obj)
               if verb.label() == 'VBD':
                  be_tok = 'was' if p == 3 else 'were'
                  be_tag = 'VBD'
               else:
                  be_tok = 'is' if p == 3 else 'are' if p == 2 else 'am'
                  be_tag = 'VBZ' if p == 3 else 'VBP'

         # TODO: there should be a VP node above be_tok and verb
         if be_tok:
            be_tree = ParentedTree(be_tag, [('-1', be_tok)])
            vp.insert(verb_idx, be_tree)
      else:
         assert len(be_verbs) == 1
         be_verb = be_verbs[0]
         be_tok = TreeUtil.token(be_verb)[1]
         assert be_tok not in ['be', 'being']
         parent = be_verb.parent()
         be_verb_id = be_verb.treeposition()[-1]
         # TODO: there should be a VP node above being and verb
         # have been being is wrong..
         if be_tok != 'been':
            parent.insert(be_verb_id+1, ParentedTree('VBG', [('-1', 'being')]))

      # replace the verb
      for v in verbs:
         v_id, v_tok = TreeUtil.token(v)
         if v_tok in self.do and TreeUtil.token(v.right_sibling())[1] == 'not':
            TreeUtil.delete(v)
         elif v_tok in self.have:
            continue
         else:
            try:
               new_v_tok = en.verb.past_participle(v_tok)
               if not new_v_tok:
                  new_v_tok = self.past_participle(v_tok)
            except KeyError:
               new_v_tok = self.past_participle(v_tok)
            v.set_label('VBN')
            v[0] = (v_id, new_v_tok)

      # swap subj and obj
      # NOTE: delete before detach because it checks parents
      vp = obj.parent()
      obj_idx = TreeUtil.delete(obj)
      obj._parent = None

      if subj:
         s = subj.parent()
         subj_idx = subj.treeposition()[-1]
         del s[subj_idx]
         subj._parent = None

         # swap and insert
         vp.insert(obj_idx, ParentedTree('PP', [ParentedTree('IN', [('-1', 'by')]), subj]))
         s.insert(subj_idx, obj)
      else:
         if not be_verbs:
            be_idx = be_tree.treeposition()[-1]
            vp.insert(be_idx, obj)
         else:
            be_idx = be_verbs[0].treeposition()[-1]
            be_verbs[0].parent().insert(max(0, be_idx-1), obj)

      self.applied = True

   def get_verb_np(self, vp, result):
      self.get_verb_np_aux(vp, [], [], result)

   def get_verb_np_aux(self, vp, verbs, be_verbs, result):
      if TreeUtil.token(vp[0])[1] == 'to':
         return
      voice = None
      np = None
      for child in vp:
         label = child.label()
         # TODO: this assumes one verb to one np
         # extend to multiple verbs: VB and VB
         if label[:2] == 'VB':
            tok = TreeUtil.token(child)[1]
            if tok not in Voice.skip_verbs:
               verbs.append(child)
               if be_verbs and label == 'VBN':
                  voice = 'passive'
               elif not be_verbs or (len(be_verbs) == 1 and TreeUtil.token(be_verbs[0])[1] not in ['be', 'being'] and label == 'VBG'):
                  voice = 'active'
               else:
                  verbs = []
            elif tok in self.be:
               be_verbs.append(child)
         # active np (obj)
         elif label == 'NP':
            np = child
            # NOTE: passive can also satisfy this condition:
            # e.g. NP will be given NP
            # we must see PP(IN by..
            if verbs and np and voice == 'active':
               result.append((be_verbs[:], verbs[:], np, voice))
               verbs = []
               np = None
               be_verbs = []
            else:
               np = None
         # passive np (subj)
         elif label == 'PP':
            if len(child) == 2 and TreeUtil.token(child[0])[1] == 'by' and child[1].label() == 'NP':
               np = child
               if be_verbs and verbs and np and voice == 'passive':
                  result.append((be_verbs[:], verbs[:], np, voice))
                  verbs = []
                  np = None
                  be_verbs = []
               else:
                  np = None
         elif label == 'VP':
            self.get_verb_np_aux(child, verbs, be_verbs, result)
         elif label in ['CC', ',']:
            verbs = []

