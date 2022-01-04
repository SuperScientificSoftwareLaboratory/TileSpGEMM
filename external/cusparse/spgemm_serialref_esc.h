#ifndef _SPGEMM_SERIALREF_
#define _SPGEMM_SERIALREF_

#include "common.h"
#include "utils.h"

int spgemm_serialref(const int           *d_csrRowPtrA,
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
                     const bool           get_nnzC_only)
{
    if (nA != mB)
    {
        printf("Cannot multiply matrix A of size %i x %i and matrix B of size %i x %i, return.\n",
               mA, nA, mB, nB);
        return -1;
    }

    int *d_csrRowPtrCub = (int *)malloc((mC+1) * sizeof(int));
    memset(d_csrRowPtrCub, 0, (mC+1) * sizeof(int));
    for (int i = 0; i < mA; i++)
    {
        for (int j = d_csrRowPtrA[i]; j < d_csrRowPtrA[i+1]; j++)
        {
            int rowB = d_csrColIdxA[j];
            d_csrRowPtrCub[i] += d_csrRowPtrB[rowB + 1] - d_csrRowPtrB[rowB];
        }
    }

    exclusive_scan(d_csrRowPtrCub, mC+1);
    int nnzCub = d_csrRowPtrCub[mC];

    if (get_nnzC_only == true)
    {
        //printf("round 1, rid = %i\n", rid);
        int *d_csrColIdxCub = (int *)malloc(nnzCub * sizeof(int));
        memset(d_csrColIdxCub, 0, nnzCub * sizeof(int));
        memset(d_csrRowPtrC, 0, (mC+1) * sizeof(int));

        for (int rid = 0; rid < mC; rid++)
        {
            // collect indices
            int rsize = d_csrRowPtrCub[rid + 1] - d_csrRowPtrCub[rid];
            int offset = d_csrRowPtrCub[rid];
            for (int j = d_csrRowPtrA[rid]; j < d_csrRowPtrA[rid+1]; j++)
            {
                int rowB = d_csrColIdxA[j];
                int incr = 0;
                for (int k = d_csrRowPtrB[rowB]; k < d_csrRowPtrB[rowB + 1]; k++)
                {
                    d_csrColIdxCub[offset+incr] = d_csrColIdxB[k];
                    incr++;
                }
                offset += incr;
            }

            // sort
            quick_sort_key(&d_csrColIdxCub[d_csrRowPtrCub[rid]], rsize);

            // compress
            int nnzr = rsize > 0 ? 1 : 0;
            for (int i = d_csrRowPtrCub[rid]+1; i < d_csrRowPtrCub[rid + 1]; i++)
            {
                nnzr = d_csrColIdxCub[i] == d_csrColIdxCub[i-1] ? nnzr : nnzr+1;
            }

            d_csrRowPtrC[rid] = nnzr;
        }

        exclusive_scan(d_csrRowPtrC, mC+1);
        *nnzC = d_csrRowPtrC[mC];

        //printf("1st round nnzc = %i\n", *nnzC);

        free(d_csrColIdxCub);
    }
    else
    {
        //printf("round 2, rid = %i\n", rid);
        int *d_csrColIdxCub = (int *)malloc(nnzCub * sizeof(int));
        VALUE_TYPE *d_csrValCub = (VALUE_TYPE *)malloc(nnzCub * sizeof(VALUE_TYPE));
        bool *d_flagCub = (bool *)malloc(nnzCub * sizeof(bool));
        memset(d_csrColIdxCub, 0, nnzCub * sizeof(int));
        memset(d_csrValCub, 0, nnzCub * sizeof(VALUE_TYPE));
        memset(d_flagCub, 0, nnzCub * sizeof(bool));

        for (int rid = 0; rid < mC; rid++)
        {
            // collect indices
            int rsize = d_csrRowPtrCub[rid + 1] - d_csrRowPtrCub[rid];
            if (rsize == 0) continue;

            int offset = d_csrRowPtrCub[rid];
            for (int j = d_csrRowPtrA[rid]; j < d_csrRowPtrA[rid+1]; j++)
            {
                int rowB = d_csrColIdxA[j];
                int val = d_csrValA[j];
                int incr = 0;
                for (int k = d_csrRowPtrB[rowB]; k < d_csrRowPtrB[rowB + 1]; k++)
                {
                    d_csrColIdxCub[offset+incr] = d_csrColIdxB[k];
                    d_csrValCub[offset+incr] = val * d_csrValB[k];
                    incr++;
                }
                offset += incr;
            }

            // sort
            quick_sort_key_val_pair(&d_csrColIdxCub[d_csrRowPtrCub[rid]],
                                    &d_csrValCub[d_csrRowPtrCub[rid]], rsize);

            // compress
            d_flagCub[d_csrRowPtrCub[rid]] = 1;
            for (int i = d_csrRowPtrCub[rid]; i < d_csrRowPtrCub[rid + 1]-1; i++)
            {
                d_flagCub[1+i] = d_csrColIdxCub[1+i] == d_csrColIdxCub[i] ? 0 : 1;
            }
            segmented_sum<VALUE_TYPE, bool>(&d_csrValCub[d_csrRowPtrCub[rid]], &d_flagCub[d_csrRowPtrCub[rid]], rsize);

            int incr = 0;
            for (int i = d_csrRowPtrCub[rid]; i < d_csrRowPtrCub[rid + 1]; i++)
            {
                if (d_flagCub[i] == 1)
                {
                    d_csrColIdxC[d_csrRowPtrC[rid] + incr] = d_csrColIdxCub[i];
                    d_csrValC[d_csrRowPtrC[rid] + incr] = d_csrValCub[i];
                    incr++;
                }
            }
        }

        free(d_csrColIdxCub);
        free(d_csrValCub);
    }

    free(d_csrRowPtrCub);

    return 0;
}


#endif


