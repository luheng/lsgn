#!/bin/bash

# Build custom kernels.
TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')


# Linux (pip)
#g++ -std=c++11 -shared coref_kernels.cc -o coref_kernels.so -I $TF_INC -I  "${TF_INC}/external/nsync/public" -fPIC -D_GLIBCXX_USE_CXX11_ABI=0

# Linux (build from source)
#g++ -std=c++11 -shared srl_kernels.cc -o srl_kernels.so -I $TF_INC -I "${TF_INC}/external/nsync/public" -fPIC

echo $TF_INC
echo $TF_LIB

g++ -std=c++11 -shared srl_kernels.cc -o srl_kernels.so -I$TF_INC -I$TF_INC/external/nsync/public -L$TF_LIB -ltensorflow_framework -fPIC -O2 

# On machines with gcc>=5.0
#g++ -std=c++11 -shared srl_kernels.cc -o srl_kernels.so -I$TF_INC -I$TF_INC/external/nsync/public -L$TF_LIB -ltensorflow_framework -fPIC -O2 -D_GLIBCXX_USE_CXX11_ABI=0

# Mac
#g++ -std=c++11 -shared coref_kernels.cc -o coref_kernels.so -I $TF_INC -fPIC -D_GLIBCXX_USE_CXX11_ABI=0  -undefined dynamic_lookup
