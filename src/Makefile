#compilers
CC=nvcc

#GLOBAL_PARAMETERS
MAT_VAL_TYPE = double
VALUE_TYPE = double

#CUDA_PARAMETERS
NVCC_FLAGS = -O3 -w -arch=compute_61 -code=sm_86 -gencode=arch=compute_61,code=sm_86
#-gencode=arch=compute_61,code=sm_75
# -m64 -Xptxas -dlcm=cg -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_61,code=compute_61
#-Xcompiler -Wall -D_FORCE_INLINES -DVERBOSE --expt-extended-lambda -use_fast_math --expt-relaxed-constexpr

#ENVIRONMENT_PARAMETERS
CUDA_INSTALL_PATH = /usr/local/cuda-11.4

#includes
INCLUDES = -I$(CUDA_INSTALL_PATH)/include

#libs
#CLANG_LIBS = -stdlib=libstdc++ -lstdc++
CUDA_LIBS = -L$(CUDA_INSTALL_PATH)/lib64  -lcudart  -lcusparse
LIBS = $(CUDA_LIBS)

#options
#OPTIONS = -std=c99

make:
	$(CC) $(NVCC_FLAGS) -Xcompiler -fopenmp -Xcompiler -mfma main.cu -o test $(INCLUDES) $(LIBS) $(OPTIONS) -D VALUE_TYPE=$(VALUE_TYPE)
