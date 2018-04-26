# Should point to the srlconll library.

export PERL5LIB="./srlconll-1.1/lib:$PERL5LIB"
export PATH="./srlconll-1.1/bin:$PATH"

perl ./srlconll-1.1/bin/srl-eval.pl $1 $2

