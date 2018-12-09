from rules import *
from corpus import EnSentence, JaSentence, Sentence, Word
from collections import defaultdict

class RewriteSentence(EnSentence):
   def __init__(self, words, alignments, parse_tree, deps=None, labs=None):
      super(RewriteSentence, self).__init__(words)
      self.alignments = alignments
      self.parse_tree = parse_tree
      self.deps = deps
      self.labs = labs

   def __str__(self):
      return ' '.join([x[1] for x in self.parse_tree.leaves()])

   def get_ori_idx_map(self, tree=None):
      '''
      map[idx in original sent] = idx in rewrited sent
      '''
      if not tree:
         tree = self.parse_tree
      # do not include inserted words
      idx_map = {int(k[0]): v for v, k in enumerate(tree.leaves()) if k[0] != '-1'}
      return idx_map

   def get_new_alignments(self, tree=None):
      alignments = defaultdict(list)
      # map of en_idx
      idx_map = self.get_ori_idx_map(tree)
      for en_idx, fr_idx_list in self.alignments.items():
         # do not include deleted words
         if en_idx in idx_map:
            for fr_idx in fr_idx_list:
               alignments[idx_map[en_idx]].append(fr_idx)
      return alignments


   # this is the old delay, calculated from the source side
   def get_new_alignments2(self, tree=None):
      alignments = defaultdict(list)
      idx_map = self.get_ori_idx_map(tree)
      for fr_idx, en_idx_list in self.alignments.items():
         for en_idx in en_idx_list:
            # do not include deleted words
            if en_idx in idx_map:
               alignments[fr_idx].append(idx_map[en_idx])
      return alignments

   @classmethod
   def alignments_str(cls, alignments):
      s = []
      for fr_idx, en_idx_list in alignments.items():
         for en_idx in en_idx_list:
            s.append('%d-%d' % (fr_idx, en_idx))
      return ' '.join(s)


class Rewriter:

   def get_alignments(self, line):
      '''
      alignments[ja_idx] = en_idx
      '''
      alignments = defaultdict(list)
      for pair in line.split():
         fr_idx, en_idx = [int(x) for x in pair.split('-')]
         alignments[fr_idx].append(en_idx)
      return alignments

   def read_sent(self, fin):
      while True:
         line = fin.readline()
         if not line:
            break
         yield JaSentence([Word(feats) for feats in map(lambda x: x.split('|'), line.strip().split())])

   def read_sent_parse(self, fparse, falign=None, dep=False):
      while True:
         line = fparse.readline()
         if not line:
            break
         toks = line.strip().split()
         fparse.readline()
         parse_tree = ParentedTree.fromstring(fparse.readline().strip())
         tags = []
         for node in parse_tree.subtrees(filter=lambda x: x.height() == 2):
            tags.append(node.label())
         if dep:
            deps = defaultdict(list)
            labs = []
            line = fparse.readline()
            while line != '\n':
               line = line.strip()
               m = re.search(r'([\w\.-]+)\([\w\.-]+-(\d+), [\w\.-]+-(\d+)\)', line)
               label, hid, mid = m.group(1), int(m.group(2)), int(m.group(3))
               labs.append(label)
               # NOTE: hid, mid starts from 1; ROOT = 0
               # mid are automatically sorted here
               deps[hid].append(mid)
               line = fparse.readline()
         if falign:
            alignments = self.get_alignments(falign.readline().strip())
         else:
            alignments = None
         # don't need pos tags
         #words = [Word(feats) for feats in map(lambda x: x.split('|'), ftag.readline().strip().split())]
         words = [Word([tok, tag]) for tok, tag in zip(toks, tags)]
         yield RewriteSentence(words, alignments, parse_tree)

