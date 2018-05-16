#!/bin/bash

if [ ! -d "data/embeddings" ]; then
  mkdir -p "data/embeddings"
fi

#cd data/embeddings
#wget http://nlp.stanford.edu/data/glove.840B.300d.zip
#unzip glove.840B.300d.zip
#rm glove.840B.300d.zip
#cd $OLDPWD

SRLPATH="./data/srl"
if [ ! -d $SRLPATH ]; then
  mkdir -p $SRLPATH
fi

# Get srl-conll package.
wget -O "${SRLPATH}/srlconll-1.1.tgz" http://www.lsi.upc.edu/~srlconll/srlconll-1.1.tgz
tar xf "${SRLPATH}/srlconll-1.1.tgz" -C "${SRLPATH}"
rm "${SRLPATH}/srlconll-1.1.tgz"



