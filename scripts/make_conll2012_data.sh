#!/bin/bash

ONTONOTES_PATH=$1

SRL_PATH="./data/srl"
if [ ! -d $SRL_PATH ]; then
  mkdir -p $SRL_PATH
fi

EMB_PATH="./embeddings"
if [ ! -d $EMB_PATH ]; then
  mkdir -p $EMB_PATH
fi


# Preprocess CoNLL formatted files.

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


# Filter data for e2e experiments.

wget http://conll.cemantix.org/2012/download/ids/english/coref/train.id -O ${SRL_PATH}/conll12.train.id
wget http://conll.cemantix.org/2012/download/ids/english/coref/development.id -O ${SRL_PATH}/conll12.dev.id
wget http://conll.cemantix.org/2012/download/ids/english/coref/test.id -O ${SRL_PATH}/conll12.test.id

python scripts/filter_conll2012_data.py ${SRL_PATH}/train.english.v5.jsonlines \
  ${SRL_PATH}/conll12.train.id \
  ${SRL_PATH}/train.english.mtl.jsonlines

python scripts/filter_conll2012_data.py ${SRL_PATH}/dev.english.v5.jsonlines \
  ${SRL_PATH}/conll12.dev.id \
  ${SRL_PATH}/dev.english.mtl.jsonlines

python scripts/filter_conll2012_data.py ${SRL_PATH}/conll12test.english.v5.jsonlines \
  ${SRL_PATH}/conll12.test.id \
  ${SRL_PATH}/test.english.mtl.jsonlines

python scripts/filter_embeddings.py ${EMB_PATH}/glove.840B.300d.txt \
  ${EMB_PATH}/glove.840B.300d.v5.filtered \
  ${SRL_PATH}/train.english.v5.jsonlines ${SRL_PATH}/dev.english.v5.jsonlines 
