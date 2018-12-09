import sys, argparse
from rules import *
from rewriter import *
import csv


APPLIED = 1
NOT_APPLICABLE = 0
FAILED = -1

if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('--src', dest='src', action='store', type=str, help='Path to parsed file', required=True)
   parser.add_argument('--save', dest='save', action='store', type=str, help='Path to save the rewritten file', required=True)
   parser.add_argument('--csv', dest='csv', action='store', type=str, help='Path to save the csv file', required=True)
   args = parser.parse_args()

   tgt_parse_file = args.src
   rewritten_file = args.save
   csv_file = args.csv

   rewriter = Rewriter()
   rules = [LooksLike(), Seem(), Clause(), Verb(), Conjunction(), Possessive(), Voice()]
   rules_log = {}
   
   rewritten_sents = []
   csv_header = ["original sent", "rewritten sent", 'rewritten']
   for rule in rules:
      rules_log[rule.name] = None
      csv_header.append(rule.name)

   with open(csv_file, 'w') as f:
      writer = csv.writer(f)
      writer.writerow(csv_header) 


   with open(tgt_parse_file, 'r') as fparse:
      tgt_sents = rewriter.read_sent_parse(fparse)
      for i, tgt_sent in enumerate(tgt_sents):
         # print i
         # print 'ORIGINAL:', tgt_sent.text()

         TreeUtil.label_idx(tgt_sent.parse_tree)

         # apply rules in fixed order greedily
         tmp_tree = TreeUtil.deepcopy_parentedtree(tgt_sent.parse_tree)
         rewritten = 0
         results = [tgt_sent.text(), None, None]
         for rule in rules:
             # print 'RULE:', rule.name
             # rule is applicable
             try:
               rule_apply = rule.apply(tmp_tree)
             except Exception as e:
               rule_apply = False
             if rule_apply:
                if not rule.fail:
                   tgt_sent.parse_tree = TreeUtil.deepcopy_parentedtree(tmp_tree)
                   # print 'REWRITTEN:', str(tgt_sent)
                   rules_log[rule.name] = APPLIED
                   rewritten = 1
                else:
                   tmp_tree = TreeUtil.deepcopy_parentedtree(tgt_sent.parse_tree)
                   print 'failed and reverted'
                   rules_log[rule.name] = FAILED
             else:
                # print 'not applicable'
                rules_log[rule.name] = NOT_APPLICABLE
             results.append(rules_log[rule.name])

         # post-processing
         new_sent = str(tgt_sent)
         # may have inserted redundant comma
         new_sent = re.sub(r'[\.,] [\.,]', r',', new_sent)
         new_sent = re.sub(r'^ , ', r'', new_sent)
         new_sent = re.sub(r'\'\'|``', r'"', new_sent)
         print 'NEW:', new_sent
         
         
         results[1] = new_sent
         results[2] = rewritten
         print results
         with open(csv_file, 'a+') as f:
             writer = csv.writer(f)
             writer.writerow(results)

         rewritten_sents.append(new_sent)
         

   with open(rewritten_file, 'w') as f:
      f.write('\n'.join(rewritten_sents))

   
