#!/bin/bash
files='/media/bighdd3/ying/tmp/incremental_mt/rewriter/data/train.en'
#sep=$2
PARSERDIR=/media/bighdd3/ying/stanford-parser-full-2015-01-30
for file in ${files}; do
   echo ${file}
   java -Xmx16g -cp "$PARSERDIR/*:" edu.stanford.nlp.parser.lexparser.LexicalizedParser \
   -nthreads 8 \
   -sentences newline \
   -tokenized \
   -escaper edu.stanford.nlp.process.PTBEscapingProcessor \
   -outputFormatOptions "basicDependencies" \
   -outputFormat "words,oneline" edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz $file > $file.parsed
done

