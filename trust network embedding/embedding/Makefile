CC=g++
PS=import tensorflow as tf; print(tf.sysconfig.get_include())
TF_INC=`python -c '$(PS)'`

all: linux

linux:
	$(CC) -std=c++11 -shared embedding/side_ops.cc embedding/side_kernels.cc -o embedding/side_ops.so -fPIC -I $(TF_INC) -O2 -D_GLIBCXX_USE_CXX11_ABI=0

mac:
	$(CC) -std=c++11 -shared embedding/side_ops.cc embedding/side_kernels.cc -o embedding/side_ops.so -fPIC -I $(TF_INC) -O2 -D_GLIBCXX_USE_CXX11_ABI=0 -undefined dynamic_lookup
