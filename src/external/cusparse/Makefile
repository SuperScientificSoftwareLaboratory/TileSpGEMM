#compilers
CC=nvcc

#GLOBAL_PARAMETERS
VALUE_TYPE = double

#CUDA_PARAMETERS
#-Xptxas -dlcm=cg
NVCC_FLAGS = -O3 -w -m64 -gencode=arch=compute_80,code=sm_86 -Xptxas -dlcm=cg
#ENVIRONMENT_PARAMETERS
CUDA_INSTALL_PATH = /usr/local/cuda-11.5
SEGSORTMACRO+=-D_ARCH80

#includes
INCLUDES = -I$(CUDA_INSTALL_PATH)/include

#libs
#CLANG_LIBS = -stdlib=libstdc++ -lstdc++
CUDA_LIBS = -L$(CUDA_INSTALL_PATH)/lib64  -lcudart -lcusparse
LIBS = $(CUDA_LIBS)

#options
#OPTIONS = -std=c99

make:
	$(CC) $(NVCC_FLAGS) -Xcompiler -fopenmp -Xcompiler -mavx -Xcompiler -mavx2 -Xcompiler -mfma -lm main.cu -o spgemm $(INCLUDES) $(LIBS) $(OPTIONS) -D VALUE_TYPE=$(VALUE_TYPE) $(SEGSORTMACRO)
