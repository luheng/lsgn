#!/bin/bash

ontonotes_path=/home/luheng/Data/conll-formatted-ontonotes-5.0
#bash conll-2012/v3/scripts/skeleton2conll.sh -D $ontonotes_path/data/files/data conll-2012

# $1: train/development/test
# $2: short for train development test
# $3: _gold_conll
# $4: lang (english/chinese/arabic)
function compile_partition() {
    rm -f $2.$4.v5_$3
    cat $ontonotes_path/data/$1/data/$4/annotations/*/*/*/*.$3 >> $2.$4.v5_$3
}

function compile_language() {
    compile_partition development dev gold_conll $1
    compile_partition train train gold_conll $1
    compile_partition test test gold_conll $1
    compile_partition conll-2012-test conll12test gold_conll $1
}

compile_language english
compile_language chinese
compile_language arabic

#python minimize.py
#python get_char_vocab.py
#python filter_embeddings.py glove.840B.300d.txt train.english.jsonlines dev.english.jsonlines test.english.jsonlines
