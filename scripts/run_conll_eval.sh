# Should point to the srlconll library.

CONLL_PATH="./data/srl/srlconll-1.1"

export PERL5LIB="${CONLL_PATH}/lib:$PERL5LIB"
export PATH="${CONLL_PATH}/bin:$PATH"

perl "${CONLL_PATH}/bin/srl-eval.pl" $1 $2

