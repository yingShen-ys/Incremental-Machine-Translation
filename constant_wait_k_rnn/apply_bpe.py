import sentencepiece as spm
sp = spm.SentencePieceProcessor()
sp.Load("bpe.model")

src_file="train.ja.new"
tgt_file="train.ja.tok"
tgt_lines = []
with open(src_file, "r") as f:
    for line in f.readlines():
        new_pieces = sp.EncodeAsPieces(line)
        tgt_lines.append(' '.join(new_pieces))
with open(tgt_file, 'w') as f:
    f.write('\n'.join(tgt_lines))
