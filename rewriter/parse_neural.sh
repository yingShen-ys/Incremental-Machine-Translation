#!/bin/bash
files='/media/bighdd3/ying/tmp/incremental_mt/rewriter/data/test.en'
#sep=$2
PARSERDIR=/media/bighdd3/ying/stanford-parser-full-2018-10-17
for file in ${files}; do
   java -Xmx2g -cp "$PARSERDIR/*:" edu.stanford.nlp.parser.nndep.DependencyParser \
   -originalDependencies \
   -model edu/stanford/nlp/models/parser/nndep/english_UD.gz \
   -textFile ${file} -outFile ${file}.neu.parsed
done

