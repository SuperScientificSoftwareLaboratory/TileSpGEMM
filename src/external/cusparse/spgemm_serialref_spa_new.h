#ifndef _SPGEMM_PARALLELREF_NEW_
#define _SPGEMM_PARALLELREF_NEW_

#include <stdbool.h>
#include "common.h"
#include "utils.h"
void spgemm_spa( const int           *d_csrRowPtrA,
                        const int           *d_csrColIdxA,
                        const VALUE_TYPE    *d_csrValA,
                        const int            mA,
                        const int            nA,
                        const int            nnzA,
                        const int           *d_csrRowPtrB,
                        const int           *d_csrColIdxB,
                        const VALUE_TYPE    *d_csrValB,
                        const int            mB,
                        const int            nB,
                        const int            nnzB,
                            int           *d_csrRowPtrC,
                            int           *d_csrColIdxC,
                           VALUE_TYPE    *d_csrValC,
                        const int            mC,
                        const int            nC,
                            int           *nnzC,
                        const int           get_nnzC_only)
{
    int nthreads = omp_get_max_threads();

    if (get_nnzC_only ==1 )
    {
        unsigned int *flag_g = (unsigned int *)malloc(nthreads * (nB / 32 + 1) * sizeof(unsigned int));

        #pragma omp parallel for
        for (int iid=0;iid<mA;iid++)
        {
            int thread_id = omp_get_thread_num();

            unsigned int *flag = flag_g + thread_id * (nB / 32 + 1); //(unsigned int *)malloc((nB/32+1)*sizeof(unsigned int));
            memset(flag, 0, sizeof(unsigned int) * (nB / 32 + 1));
            for (int blkj = d_csrRowPtrA[iid]; blkj < d_csrRowPtrA[iid + 1]; blkj++)
            {
                int col = d_csrColIdxA[blkj];
                for (int l = d_csrRowPtrB[col]; l < d_csrRowPtrB[col + 1]; l++)
                {
                    const int key = d_csrColIdxB[l];
                    int ind = key / 32;
                    flag[ind] |= (1 << (key % 32));
                }
            }
            //int nnzr_new=0;
            int nnzr_new1 = 0;
            for (int i = 0; i < (nB / 32) + 1; i++)
            {
                nnzr_new1 += _mm_popcnt_u32(flag[i]);
            }

            d_csrRowPtrC[iid] = nnzr_new1;
        }
        exclusive_scan(d_csrRowPtrC, mC +1);
        *nnzC = d_csrRowPtrC[mC];
        free(flag_g);
    }
    else
    {
        unsigned int *flag_g = (unsigned int *)malloc(nthreads * (nB / 32 + 1) * sizeof(unsigned int));
        #pragma omp parallel for
        for (int iid=0;iid<mA;iid++)
        {
            int thread_id = omp_get_thread_num();
            unsigned int *flag = flag_g + thread_id * (nB / 32 + 1);
            memset(flag, 0, sizeof(unsigned int) * (nB / 32 + 1));
            //   int pos=0;
            //    int j=bin[iid];
            for (int blkj = d_csrRowPtrA[iid]; blkj < d_csrRowPtrA[iid + 1]; blkj++)
            {
                int col = d_csrColIdxA[blkj];
                for (int l = d_csrRowPtrB[col]; l < d_csrRowPtrB[col + 1]; l++)
                {
                    const int key = d_csrColIdxB[l];
                    int ind = key / 32;
                    flag[ind] |= (1 << (key % 32));
                }
            }

            // int nnzr = d_csrRowPtrC[iid];
            int nnzr_new = d_csrRowPtrC[iid];
            for (int i = 0; i < (nB / 32) + 1; i++)
            {
                int count = 0;
                while (flag[i])
                {
                    count++;
                    if ((flag[i] & 1) != 0)
                    {
                        d_csrColIdxC[nnzr_new] = (i * 32 + count - 1);
                        nnzr_new++;
                    }
                    flag[i] = flag[i] >> 1;
                }
            }
        }
        free(flag_g);
    }

}

#endif


