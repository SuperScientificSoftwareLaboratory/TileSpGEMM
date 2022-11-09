#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mm_malloc.h>
#include <x86intrin.h>
#include <immintrin.h>
#include <nmmintrin.h>


#include <omp.h>

#include <sys/time.h>
#include "cuda_fp16.h"

#include "utils.h"

#ifndef MAT_VAL_TYPE
#define MAT_VAL_TYPE double
#endif

#ifndef MAT_PTR_TYPE
#define MAT_PTR_TYPE int
#endif

#define WARP_SIZE 32
#define WARP_PER_BLOCK 4

#define HALFWARP_SIZE 16
#define HALFWARP_PER_BLOCK 8

#ifndef BLOCK_SIZE
#define BLOCK_SIZE  16
#endif


#define SMEM_TNY_TH 32
#define SMEM_SML_TH 32 //112
#define SMEM_LRG_TH 224
#define SMEM_DNS_TH 256

#define USE_HALFWARP 1
#define TILE_PER_WARP 16 // should not be larger than WARPSIZE
#define TILE_PER_HALFWARP 8 // should not be larger than HALFWARP_SIZE

//#define LOAD_MASKB_TH 4
#define VECTORIZE_NNZA_OR_NNZB_TH 8

#define SMEM_INTERSECTION_TH 16
#define SMEM_INTERSECTION_LEN 48

#define USE_GMEM_SPECULATIVE_INTERSECTION 1
#define GMEM_SPECULATIVE_INTERSECTION 1

#define SPECULATIVE_INTERSECTION 32

#define SPA_INT_PER_WARP 512
#define NUMCOLC_SPA_OR_HASH_TH     SPA_INT_PER_WARP * 32 // SPA_INT_PER_WARP int per warp

// e.g., INTERSECTION_SPARSE_OR_DNS_TH = 0.2 means when density is higher than 20%, use DNS for intersection
#define INTERSECTION_SPARSE_OR_DNS_TH 0.2
#define NNZTOTALA_FAST_TRACK_TH2 192

#define USE_DNS_THREAD 1

#define DEBUG 1

#define REPEAT_NUM 1

#ifndef TIMING
#define TIMING 1
#endif

#ifndef SPACE
#define SPACE 1
#endif


#ifndef CHECK_RESULT
#define CHECK_RESULT 1
#endif

#define SMEM_TNY_TH 32
#define SMEM_SML_TH 32 //112
#define SMEM_LRG_TH 224
#define SMEM_DNS_TH 256

#define USE_HALFWARP 1
#define TILE_PER_WARP 16 // should not be larger than WARPSIZE
#define TILE_PER_HALFWARP 8 // should not be larger than HALFWARP_SIZE

#define VECTORIZE_NNZA_OR_NNZB_TH 8

#define SMEM_INTERSECTION_TH 16
#define SMEM_INTERSECTION_LEN 48

#define USE_GMEM_SPECULATIVE_INTERSECTION 1
#define GMEM_SPECULATIVE_INTERSECTION 1

#define SPECULATIVE_INTERSECTION 32

#define SPA_INT_PER_WARP 512
#define NUMCOLC_SPA_OR_HASH_TH     SPA_INT_PER_WARP * 32 // SPA_INT_PER_WARP int per warp


// e.g., INTERSECTION_SPARSE_OR_DNS_TH = 0.2 means when density is higher than 20%, use DNS for intersection
#define INTERSECTION_SPARSE_OR_DNS_TH 0.2
#define NNZTOTALA_FAST_TRACK_TH2 192

#define USE_DNS_THREAD 0
#define HASH_SCALE 107

#ifndef SMATRIX
#define SMATRIX
typedef struct 
{
    int m;
    int n;
    int nnz;
    int isSymmetric;
	MAT_VAL_TYPE *value;
	int *columnindex;
	MAT_PTR_TYPE *rowpointer;
    int tilem;
    int tilen;
    MAT_PTR_TYPE *tile_ptr;
    int *tile_columnidx;
    int *tile_rowidx;
    int *tile_nnz;
    int numtile;
    MAT_VAL_TYPE *tile_csr_Value;
    unsigned char *tile_csr_Col;
    unsigned char *tile_csr_Ptr;
    unsigned short *mask;
    int *csc_tile_ptr;
    int *csc_tile_rowidx;
}SMatrix;
#endif

