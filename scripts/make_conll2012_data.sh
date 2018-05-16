#!/bin/bash

#ONTONOTES_PATH=/home/luheng/Data/conll-formatted-ontonotes-5.0
ONTONOTES_PATH=$1
#bash conll-2012/v3/scripts/skeleton2conll.sh -D $ontonotes_path/data/files/data conll-2012

SRL_PATH="./data/srl"

if [ ! -d $SRLPATH ]; then
  mkdir -p $SRLPATH
fi

rm -f ${SRL_PATH}/train.english.v5_gold_conll
cat ${ONTONOTES_PATH}/data/train/data/english/annotations/*/*/*/*.gold_conll \
  >> ${SRL_PATH}/train.english.v5_gold_conll

rm -f ${SRL_PATH}/dev.english.v5_gold_conll
cat ${ONTONOTES_PATH}/data/development/data/english/annotations/*/*/*/*.gold_conll \
  >> ${SRL_PATH}/dev.english.v5_gold_conll

rm -f ${SRL_PATH}/conll12test.english.v5_gold_conll
cat ${ONTONOTES_PATH}/data/conll-2012-test/data/english/annotations/*/*/*/*.gold_conll \
  >> ${SRL_PATH}/conll12test.english.v5_gold_conll


python scripts/ontonotes5_to_json.py ${SRL_PATH}/train.english.v5_gold_conll \
  ${SRL_PATH}/train.english.v5.jsonlines
python scripts/ontonotes5_to_json.py ${SRL_PATH}/dev.english.v5_gold_conll \
  ${SRL_PATH}/dev.english.v5.jsonlines
python scripts/ontonotes5_to_json.py ${SRL_PATH}/conll12test.english.v5_gold_conll \
  ${SRL_PATH}/conll12test.english.v5.jsonlines

#python scripts/get_char_vocab.py
#python filter_embeddings.py glove.840B.300d.txt train.english.jsonlines dev.english.jsonlines test.english.jsonlines
