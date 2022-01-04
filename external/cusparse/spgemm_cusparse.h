#ifndef _SPGEMM_CUDA_CUSPARSE_
#define _SPGEMM_CUDA_CUSPARSE_

#include "common.h"
#include "utils.h"
#include <cuda_runtime.h>
#include <cusparse.h>

//#include "utils_cuda_sort.h"
//#include "utils_cuda_spgemm_subfunc.h"
//#include "utils_cuda_scan.h"
//#include "utils_cuda_segmerge.h"
//#include "utils_cuda_segsum.h"

int spgemm_cusparse_executor(cusparseHandle_t handle, cusparseSpMatDescr_t matA,
                             const int mA,
                             const int nA,
                             const int nnzA,
                             const int *d_csrRowPtrA,
                             const int *d_csrColIdxA,
                             const VALUE_TYPE *d_csrValA,
                             cusparseSpMatDescr_t matB,
                             const int mB,
                             const int nB,
                             const int nnzB,
                             const int *d_csrRowPtrB,
                             const int *d_csrColIdxB,
                             const VALUE_TYPE *d_csrValB,
                             cusparseSpMatDescr_t matC,
                             const int mC,
                             const int nC,
                             unsigned long long int *nnzC,
                             int **d_csrRowPtrC,
                             int **d_csrColIdxC,
                             VALUE_TYPE **d_csrValC)
{
    cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseOperation_t opB = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cudaDataType computeType = CUDA_R_32F;
    void *dBuffer1 = NULL, *dBuffer2 = NULL;
    size_t bufferSize1 = 0, bufferSize2 = 0;

    float alpha = 1.0f;
    float beta = 0.0f;

    cudaMalloc((void **)d_csrRowPtrC, (mC + 1) * sizeof(int));

    //--------------------------------------------------------------------------
    // SpGEMM Computation
    cusparseSpGEMMDescr_t spgemmDesc;
    cusparseSpGEMM_createDescr(&spgemmDesc);

    // ask bufferSize1 bytes for external memory
    cusparseSpGEMM_workEstimation(handle, opA, opB,
                                  &alpha, matA, matB, &beta, matC,
                                  computeType, CUSPARSE_SPGEMM_DEFAULT,
                                  spgemmDesc, &bufferSize1, NULL);
    cudaMalloc((void **)&dBuffer1, bufferSize1);
    // inspect the matrices A and B to understand the memory requiremnent for
    // the next step
    cusparseSpGEMM_workEstimation(handle, opA, opB,
                                  &alpha, matA, matB, &beta, matC,
                                  computeType, CUSPARSE_SPGEMM_DEFAULT,
                                  spgemmDesc, &bufferSize1, dBuffer1);

    // ask bufferSize2 bytes for external memory
    cusparseSpGEMM_compute(handle, opA, opB,
                           &alpha, matA, matB, &beta, matC,
                           computeType, CUSPARSE_SPGEMM_DEFAULT,
                           spgemmDesc, &bufferSize2, NULL);
    cudaMalloc((void **)&dBuffer2, bufferSize2);

    // compute the intermediate product of A * B
    cusparseSpGEMM_compute(handle, opA, opB,
                           &alpha, matA, matB, &beta, matC,
                           computeType, CUSPARSE_SPGEMM_DEFAULT,
                           spgemmDesc, &bufferSize2, dBuffer2);
    // get matrix C non-zero entries C_num_nnz1
    int64_t C_num_rows1, C_num_cols1, C_num_nnz1;
    cusparseSpMatGetSize(matC, &C_num_rows1, &C_num_cols1, &C_num_nnz1);
    // allocate matrix C
    cudaMalloc((void **)d_csrColIdxC, C_num_nnz1 * sizeof(int));
    cudaMalloc((void **)d_csrValC, C_num_nnz1 * sizeof(VALUE_TYPE));
    // update matC with the new pointers
    cusparseCsrSetPointers(matC, *d_csrRowPtrC, *d_csrColIdxC, *d_csrValC);

    // copy the final products to the matrix C
    cusparseSpGEMM_copy(handle, opA, opB,
                        &alpha, matA, matB, &beta, matC,
                        computeType, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc);

    *nnzC = C_num_nnz1;

    cusparseSpGEMM_destroyDescr(spgemmDesc);

    return 0;
}

int spgemm_cusparse(const int mA,
                    const int nA,
                    const int nnzA,
                    const int *h_csrRowPtrA,
                    const int *h_csrColIdxA,
                    const VALUE_TYPE *h_csrValA,
                    const int mB,
                    const int nB,
                    const int nnzB,
                    const int *h_csrRowPtrB,
                    const int *h_csrColIdxB,
                    const VALUE_TYPE *h_csrValB,
                    const int mC,
                    const int nC,
                    const int nnzC_golden,
                    const int *h_csrRowPtrC_golden,
                    const int *h_csrColIdxC_golden,
                    const VALUE_TYPE *h_csrValC_golden,
                    const bool check_result,
                    unsigned long long int nnzCub,
                    unsigned long long int *nnzC,
                    double *compression_rate,
                    double *time_segmerge,
                    double *gflops_segmerge)

{
    // transfer host mem to device mem
    int *d_csrRowPtrA;
    int *d_csrColIdxA;
    VALUE_TYPE *d_csrValA;
    int *d_csrRowPtrB;
    int *d_csrColIdxB;
    VALUE_TYPE *d_csrValB;
    //unsigned long long int nnzC = 0;
    int *d_csrRowPtrC;
    int *d_csrColIdxC;
    VALUE_TYPE *d_csrValC;

    // Matrix A in CSR
    cudaMalloc((void **)&d_csrRowPtrA, (mA + 1) * sizeof(int));
    cudaMalloc((void **)&d_csrColIdxA, nnzA * sizeof(int));
    cudaMalloc((void **)&d_csrValA, nnzA * sizeof(VALUE_TYPE));

    cudaMemcpy(d_csrRowPtrA, h_csrRowPtrA, (mA + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrColIdxA, h_csrColIdxA, nnzA * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrValA, h_csrValA, nnzA * sizeof(VALUE_TYPE), cudaMemcpyHostToDevice);

    // Matrix B in CSR
    cudaMalloc((void **)&d_csrRowPtrB, (mB + 1) * sizeof(int));
    cudaMalloc((void **)&d_csrColIdxB, nnzB * sizeof(int));
    cudaMalloc((void **)&d_csrValB, nnzB * sizeof(VALUE_TYPE));

    cudaMemcpy(d_csrRowPtrB, h_csrRowPtrB, (mB + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrColIdxB, h_csrColIdxB, nnzB * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrValB, h_csrValB, nnzB * sizeof(VALUE_TYPE), cudaMemcpyHostToDevice);

    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    cusparseHandle_t handle = NULL;
    cusparseSpMatDescr_t matA, matB, matC;

    cusparseCreate(&handle);
    // Create sparse matrix A in CSR format
    cusparseCreateCsr(&matA, mA, nA, nnzA,
                      d_csrRowPtrA, d_csrColIdxA, d_csrValA,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    cusparseCreateCsr(&matB, mB, nB, nnzB,
                      d_csrRowPtrB, d_csrColIdxB, d_csrValB,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    cusparseCreateCsr(&matC, mA, nB, 0,
                      NULL, NULL, NULL,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    //--------------------------------------------------------------------------

    //  - cuda SpGEMM start!
    printf(" - cuda SpGEMM start! Benchmark runs %i times.\n", BENCH_REPEAT);

    if (check_result && BENCH_REPEAT > 1)
    {
        printf("If check_result, Set BENCH_REPEAT to 1.\n");
        return -1;
    }
    //unsigned long long int nnzCub = 0;

    struct timeval t1, t2;

    cudaDeviceSynchronize();
    gettimeofday(&t1, NULL);

    for (int i = 0; i < BENCH_REPEAT; i++)
    {
        spgemm_cusparse_executor(handle, matA, mA, nA, nnzA, d_csrRowPtrA, d_csrColIdxA, d_csrValA,
                                 matB, mB, nB, nnzB, d_csrRowPtrB, d_csrColIdxB, d_csrValB,
                                 matC, mC, nC, nnzC, &d_csrRowPtrC, &d_csrColIdxC, &d_csrValC);

        if (check_result != 1 || i != BENCH_REPEAT - 1)
        {
            cudaFree(d_csrRowPtrC);
            cudaFree(d_csrColIdxC);
            cudaFree(d_csrValC);
        }
    }

    cudaDeviceSynchronize();
    gettimeofday(&t2, NULL);

    printf(" - cuda SpGEMM completed!\n\n");
    double time_cuda_spgemm = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    time_cuda_spgemm /= BENCH_REPEAT;
    *time_segmerge = time_cuda_spgemm;
    *compression_rate = (double)nnzCub / (double)*nnzC;
    *gflops_segmerge = 2 * (double)nnzCub / (1e6 * time_cuda_spgemm);
    printf("nnzC = %i, nnzCub = %lld, Compression rate = %4.2f\n",
           *nnzC, nnzCub, *compression_rate);
    printf("CUDA  cuSPARSE SpGEMM runtime is %4.4f ms, GFlops = %4.4f\n",
           time_cuda_spgemm, *gflops_segmerge);

    // validate C = AB

    if (check_result)
    {
        if (*nnzC <= 0)
        {
            printf("cuSPARSE failed!\n");
            return 0;
        }
        else
        {
            printf("\nValidating results...\n");
            if (*nnzC != nnzC_golden)
            {

                printf("[NOT PASSED] nnzC = %i, nnzC_golden = %i\n", *nnzC, nnzC_golden);
            }
            else
            {
                printf("[PASSED] nnzC = %i\n", *nnzC);
            }

            int *h_csrRowPtrC = (int *)malloc((mC + 1) * sizeof(int));
            int *h_csrColIdxC = (int *)malloc(*nnzC * sizeof(int));
            VALUE_TYPE *h_csrValC = (VALUE_TYPE *)malloc(*nnzC * sizeof(VALUE_TYPE));

            cudaMemcpy(h_csrRowPtrC, d_csrRowPtrC, (mC + 1) * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_csrColIdxC, d_csrColIdxC, *nnzC * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_csrValC, d_csrValC, *nnzC * sizeof(VALUE_TYPE), cudaMemcpyDeviceToHost);

            int errcounter = 0;
            for (int i = 0; i < mC + 1; i++)
            {
                if (h_csrRowPtrC[i] != h_csrRowPtrC_golden[i])
                {
                    if (h_csrRowPtrC[i] < 0)
                    {
                        printf("cuSPARSE failed!\n");
                        return 0;
                    }
                    else{
                    errcounter++;}
                }
            }
            if (errcounter != 0)
            {
                printf("[NOT PASSED] row_pointer, #err = %i\n", errcounter);
            }
            else
            {
                printf("[PASSED] row_pointer\n");
            }

            /*for (int i = 0; i < mC; i++)
        {
            quick_sort_key_val_pair<int, VALUE_TYPE>(&h_csrColIdxC[h_csrRowPtrC[i]],
                                                     &h_csrValC[h_csrRowPtrC[i]],
                                                     h_csrRowPtrC[i+1]-h_csrRowPtrC[i]);
        }*/

            errcounter = 0;
            for (int j = 0; j < *nnzC; j++)
            {
                if (h_csrColIdxC[j] != h_csrColIdxC_golden[j]) //|| h_csrValC[j] != h_csrValC_golden[j])
                {
                    //    printf("h_csrColIdxC[j] = %i,  h_csrColIdxC_golden[j] = %i\n",h_csrColIdxC[j] ,h_csrColIdxC_golden[j]);
                    errcounter++;
                }
            }

            if (errcounter != 0)
            {
                printf("[NOT PASSED] column_index & value, #err = %i (%4.2f%% #nnz)\n",
                       errcounter, 100.0 * (double)errcounter / (double)(*nnzC));
            }
            else
            {
                printf("[PASSED] column_index & value\n");
            }

            free(h_csrRowPtrC);
            free(h_csrColIdxC);
            free(h_csrValC);
        }
    }

    cudaFree(d_csrRowPtrA);
    cudaFree(d_csrColIdxA);
    cudaFree(d_csrValA);
    cudaFree(d_csrRowPtrB);
    cudaFree(d_csrColIdxB);
    cudaFree(d_csrValB);

    if (check_result)
    {
        cudaFree(d_csrRowPtrC);
        cudaFree(d_csrColIdxC);
        cudaFree(d_csrValC);
    }

    cusparseDestroySpMat(matA);
    cusparseDestroySpMat(matB);
    cusparseDestroySpMat(matC);
    cusparseDestroy(handle);

    return 0;
}

#endif
