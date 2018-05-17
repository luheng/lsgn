#! /bin/bash

SRL_PATH="./data/srl"

if [ ! -d $SRL_PATH ]; then
  mkdir -p $SRL_PATH
fi

EMB_PATH="./embeddings"
if [ ! -d $EMB_PATH ]; then
  mkdir -p $EMB_PATH
fi


export PERL5LIB="$SRL_PATH/srlconll-1.1/lib:$PERL5LIB"
export PATH="$SRL_PATH/srlconll-1.1/bin:$PATH"

WSJPATH=$1

TRAIN_SECTIONS=(02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21)
DEVEL_SECTIONS=(24)

# Fetch data
wget -O "${SRL_PATH}/conll05st-release.tar.gz" http://www.lsi.upc.edu/~srlconll/conll05st-release.tar.gz
wget -O "${SRL_PATH}/conll05st-tests.tar.gz" http://www.lsi.upc.edu/~srlconll/conll05st-tests.tar.gz
tar xf "${SRL_PATH}/conll05st-release.tar.gz" -C "${SRL_PATH}"
tar xf "${SRL_PATH}/conll05st-tests.tar.gz" -C "${SRL_PATH}"

CONLL05_PATH="${SRL_PATH}/conll05st-release"

if [ ! -d "${CONLL05_PATH}/train/words" ]; then
  mkdir -p "${CONLL05_PATH}/train/words"
fi

if [ ! -d "${CONLL05_PATH}/devel/words" ]; then
  mkdir -p "${CONLL05_PATH}/devel/words"
fi

# Retrieve words from PTB source.
for s in "${TRAIN_SECTIONS[@]}"
do
  echo $s
  cat ${WSJPATH}/parsed/mrg/wsj/$s/* | wsj-removetraces.pl | wsj-to-se.pl -w 1 | awk '{print $1}' | \
    gzip > "${CONLL05_PATH}/train/words/train.$s.words.gz"
done

for s in "${DEVEL_SECTIONS[@]}"
do
  echo $s
  cat ${WSJPATH}/parsed/mrg/wsj/$s/* | wsj-removetraces.pl | wsj-to-se.pl -w 1 | awk '{print $1}' | \
    gzip > "${CONLL05_PATH}/devel/words/devel.$s.words.gz"
done

rm "${SRL_PATH}/conll05st-release.tar.gz"
rm "${SRL_PATH}/conll05st-tests.tar.gz"

cd ${CONLL05_PATH}
./scripts/make-trainset.sh
./scripts/make-devset.sh

# Prepare test set.
zcat test.wsj/words/test.wsj.words.gz > /tmp/words
zcat test.wsj/props/test.wsj.props.gz > /tmp/props
paste -d ' ' /tmp/words /tmp/props  > "test-wsj"
echo Cleaning files
rm -f /tmp/$$*

zcat test.brown/words/test.brown.words.gz > /tmp/words
zcat test.brown/props/test.brown.props.gz > /tmp/props
paste -d ' ' /tmp/words /tmp/props  > "test-brown"
echo Cleaning files
rm -f /tmp/$$*

cd $OLDPWD

# Process CoNLL05 data
zcat "${CONLL05_PATH}/devel/props/devel.24.props.gz" > "${SRL_PATH}/conll05.devel.props.gold.txt"
zcat "${CONLL05_PATH}/test.wsj/props/test.wsj.props.gz" > "${SRL_PATH}/conll05.test.wsj.props.gold.txt"
zcat "${CONLL05_PATH}/test.brown/props/test.brown.props.gz" > "${SRL_PATH}/conll05.test.brown.props.gold.txt"

zcat "${CONLL05_PATH}/train-set.gz" > "${CONLL05_PATH}/train-set"
zcat "${CONLL05_PATH}/dev-set.gz" > "${CONLL05_PATH}/dev-set"

# Convert CoNLL to json format.
python scripts/conll05_to_json.py "${CONLL05_PATH}/train-set" \
  "${SRL_PATH}/train.english.conll05.jsonlines" 5
python scripts/conll05_to_json.py "${CONLL05_PATH}/dev-set" \
  "${SRL_PATH}/dev.english.conll05.jsonlines" 5
python scripts/conll05_to_json.py "${CONLL05_PATH}/test-wsj" \
  "${SRL_PATH}/test_wsj.english.conll05.jsonlines" 1
python scripts/conll05_to_json.py "${CONLL05_PATH}/test-brown" \
  "${SRL_PATH}/test_brown.english.conll05.jsonlines" 1


# Filter embeddings.
python scripts/filter_embeddings.py ${EMB_PATH}/glove.840B.300d.txt \
  ${EMB_PATH}/glove.840B.300d.05.filtered \
  ${SRL_PATH}/train.english.conll05.jsonlines ${SRL_PATH}/dev.english.conll05.jsonlines


