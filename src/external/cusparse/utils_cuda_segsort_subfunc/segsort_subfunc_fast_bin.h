#ifndef _FAST_BIN
#define _FAST_BIN


#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <vector>
#include <algorithm>
//#include "fast_utils.h"
using namespace std;

// #define BINRULE1
// #define BINRULE2

#define SEGBIN_NUM 13

__global__
void fast_bin_step1(int *d_bin_counter, const int *d_segs, int length, int n)
{
    const int global_id = blockIdx.x * blockDim.x + threadIdx.x; 
    __shared__ int s_segbin_counter[SEGBIN_NUM];
    if (threadIdx.x < SEGBIN_NUM)
        s_segbin_counter[threadIdx.x] = 0;
    __syncthreads();

    if (global_id < length)
    {
        const int size = ((global_id == length-1)?n:d_segs[global_id+1]) - d_segs[global_id];

        if (size <= 1)
            atomicAdd((int *)&s_segbin_counter[0 ], 1);
        if (1 < size && size <= 2)
            atomicAdd((int *)&s_segbin_counter[1 ], 1);
        if (2 < size && size <= 4)
            atomicAdd((int *)&s_segbin_counter[2 ], 1);
        if (4 < size && size <= 8)
            atomicAdd((int *)&s_segbin_counter[3 ], 1);
        if (8 < size && size <= 16)
            atomicAdd((int *)&s_segbin_counter[4 ], 1);
        if (16 < size && size <= 32)
            atomicAdd((int *)&s_segbin_counter[5 ], 1);
        if (32 < size && size <= 64)
            atomicAdd((int *)&s_segbin_counter[6 ], 1);
        if (64 < size && size <= 128)
            atomicAdd((int *)&s_segbin_counter[7 ], 1);
        if (128 < size && size <= 256)
            atomicAdd((int *)&s_segbin_counter[8 ], 1);
        if (256 < size && size <= 512)
            atomicAdd((int *)&s_segbin_counter[9 ], 1);
        if (512 < size && size <= 1024)
            atomicAdd((int *)&s_segbin_counter[10], 1);
        if (1024 < size && size <= 2048)
            atomicAdd((int *)&s_segbin_counter[11], 1);
        if (2048 < size)
            atomicAdd((int *)&s_segbin_counter[12], 1);
    }
    __syncthreads();
    if (threadIdx.x < SEGBIN_NUM)
        atomicAdd((int *)&d_bin_counter[threadIdx.x], s_segbin_counter[threadIdx.x]);
}

template<class T>
void fast_excl_scan(T *in, const int length)
{
    thrust::device_ptr<int> d_array_thrust = thrust::device_pointer_cast(in);
    thrust::exclusive_scan(d_array_thrust, d_array_thrust + length, d_array_thrust);
}

__global__
void fast_bin_step2(int *d_bin_segs_id, int *d_bin_counter, 
        const int *d_segs, const int length, const int n)
{
    const int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (global_id < length)
    {
        const int size = ((global_id == length-1)?n:d_segs[global_id+1]) - d_segs[global_id];
        int position;
        if (size <= 1)
            position = atomicAdd((int *)&d_bin_counter[0 ], 1);
        else if (size <= 2)                              
            position = atomicAdd((int *)&d_bin_counter[1 ], 1);
        else if (size <= 4)                              
            position = atomicAdd((int *)&d_bin_counter[2 ], 1);
        else if (size <= 8)                              
            position = atomicAdd((int *)&d_bin_counter[3 ], 1);
        else if (size <= 16)                             
            position = atomicAdd((int *)&d_bin_counter[4 ], 1);
        else if (size <= 32)                             
            position = atomicAdd((int *)&d_bin_counter[5 ], 1);
        else if (size <= 64)                             
            position = atomicAdd((int *)&d_bin_counter[6 ], 1);
        else if (size <= 128)                            
            position = atomicAdd((int *)&d_bin_counter[7 ], 1);
        else if (size <= 256)                            
            position = atomicAdd((int *)&d_bin_counter[8 ], 1);
        else if (size <= 512)                            
            position = atomicAdd((int *)&d_bin_counter[9 ], 1);
        else if (size <= 1024)
            position = atomicAdd((int *)&d_bin_counter[10], 1);
        else if (size <= 2048)
            position = atomicAdd((int *)&d_bin_counter[11], 1);
        else
            position = atomicAdd((int *)&d_bin_counter[12], 1);
        d_bin_segs_id[position] = global_id;
    }

}

void fast_bin_cuda(int *d_bin_segs_id, int *d_bin_counter, const int *d_segs, 
        const int length, const int n, int *h_bin_counter)
{

    const int num_threads = 256;
    const int num_blocks = ceil((double)length/(double)num_threads);

#ifdef __PROF
    double time0, time1;
    time0 = dtime();
#endif

    fast_bin_step1<<< num_blocks, num_threads >>>(d_bin_counter, d_segs, length, n);

#ifdef __PROF
    cudaDeviceSynchronize();
    time1 = dtime();
    cout << "time bin_step1(ms): " << time1 - time0 << endl;
#endif


    // show_me_d(d_bin_counter, SEGBIN_NUM, "bin_counter:");

#ifdef __PROF
    time0 = dtime();
#endif

    fast_excl_scan(d_bin_counter, SEGBIN_NUM);

    cudaMemcpyAsync(h_bin_counter, d_bin_counter, SEGBIN_NUM*sizeof(int), cudaMemcpyDeviceToHost);
#ifdef __PROF
    cudaDeviceSynchronize();
    time1 = dtime();
    cout << "time bin_scan(ms): " << time1 - time0 << endl;
#endif

    // show_me_d(d_bin_counter, SEGBIN_NUM, "bin_counter(scan):");

#ifdef __PROF
    time0 = dtime();
#endif

    fast_bin_step2<<<num_blocks, num_threads>>>(d_bin_segs_id, d_bin_counter, d_segs, length, n);

#ifdef __PROF
    cudaDeviceSynchronize();
    time1 = dtime();
    cout << "time bin_step2(ms): " << time1 - time0 << endl;
#endif

}


void fast_bin_cpu(int *ref_bin_segs_id, const int *segs, 
        const int length, const int n)
{
    vector<int> bin_counter(SEGBIN_NUM, 0);
    
    for(int i = 0; i < length; i++)
    {
        const int size = ((i == length-1)?n:segs[i+1]) - segs[i];

        if (size <= 1)
            bin_counter[0]++;
        else if (size <= 2)
            bin_counter[1]++;
        else if (size <= 4)
            bin_counter[2]++;
        else if (size <= 8)
            bin_counter[3]++;
        else if (size <= 16)
            bin_counter[4]++;
        else if (size <= 32)
            bin_counter[5]++;
        else if (size <= 64)
            bin_counter[6]++;
        else if (size <= 128)
            bin_counter[7]++;
        else if (size <= 256)
            bin_counter[8]++;
        else if (size <= 512)
            bin_counter[9]++;
        else if (size <= 1024)
            bin_counter[10]++;
        else if (size <= 2048)
            bin_counter[11]++;
        else
            bin_counter[12]++;


    }

    // show_me(&bin_counter[0], SEGBIN_NUM, "bin_counter(cpu):");

    int sum = 0;
    for(int i = 0; i < SEGBIN_NUM; i++)
    {
        int tmp = bin_counter[i];
        bin_counter[i] = sum;
        sum += tmp;
    }
    cout << "ratio_sm: " << (double)bin_counter[9]/sum << endl;
    cout << "ratio_md: " << (double)(bin_counter[11]-bin_counter[9])/sum << endl;
    cout << "ratio_lg: " << (double)(sum-bin_counter[SEGBIN_NUM-1])/sum << endl;

    // show_me(&bin_counter[0], SEGBIN_NUM, "bin_counter_scan(cpu):");
    
    for(int i = 0; i < length; i++)
    {
        const int size = ((i == length-1)?n:segs[i+1]) - segs[i];
        int position;

        if (size <= 1)
            position = bin_counter[0 ]++;
        else if (size <= 2)          
            position = bin_counter[1 ]++;
        else if (size <= 4)          
            position = bin_counter[2 ]++;
        else if (size <= 8)          
            position = bin_counter[3 ]++;
        else if (size <= 16)         
            position = bin_counter[4 ]++;
        else if (size <= 32)         
            position = bin_counter[5 ]++;
        else if (size <= 64)         
            position = bin_counter[6 ]++;
        else if (size <= 128)        
            position = bin_counter[7 ]++;
        else if (size <= 256)        
            position = bin_counter[8 ]++;
        else if (size <= 512)        
            position = bin_counter[9 ]++;
        else if (size <= 1024)
            position = bin_counter[10]++;
        else if (size <= 2048)
            position = bin_counter[11]++;
        else
            position = bin_counter[12]++;

        ref_bin_segs_id[position] = i;
    }

}


#endif
