#! /bin/bash

MODELS=( "conll05_final_wsj" "conll05_goldprops_noelmo_final_wsj" 
         "conll2012_final" "conll2012_goldprops_noelmo_final"
         "conll05_goldprops_final_wsj" "conll05_noelmo_final_wsj"
         "conll2012_goldprops_final" "conll2012_noelmo_final" )

LOGS_PATH="./logs"
if [ ! -d $LOGS_PATH ]; then
  mkdir -p $LOGS_PATH
fi

cd $LOGS_PATH

for MD_NAME in "${MODELS[@]}"
do
if [ ! -d ${MD_NAME} ]; then
  echo "Downloading pretrained model: ${MD_NAME}"
  wget "https://dada.cs.washington.edu/qasrl/models/${MD_NAME}.tar.gz"
  tar xf "${MD_NAME}.tar.gz"
  rm "${MD_NAME}.tar.gz"
fi
done

cd $OLDPWD
