#ifndef _SPGEMM_SERIALREF_
#define _SPGEMM_SERIALREF_

#include "common.h"
#include "utils.h"

void compute_dense_row(const int        *d_csrRowPtrA,
                       const int        *d_csrColIdxA,
                       const VALUE_TYPE *d_csrValA,
                       const int        *d_csrRowPtrB,
                       const int        *d_csrColIdxB,
                       const VALUE_TYPE *d_csrValB,
                             int        *d_dense_row_column_flag,
                             VALUE_TYPE *d_dense_row_value,
                       const int         rid,
                       const bool        has_value)
{
    for (int rid_a = d_csrRowPtrA[rid]; rid_a < d_csrRowPtrA[rid+1]; rid_a++)
    {
        int rid_b = d_csrColIdxA[rid_a];
        VALUE_TYPE val_a = 0;
        if (has_value) val_a = d_csrValA[rid_a];

        for (int cid_b = d_csrRowPtrB[rid_b]; cid_b < d_csrRowPtrB[rid_b+1]; cid_b++)
        {
            d_dense_row_column_flag[d_csrColIdxB[cid_b]] = 1;
            if (has_value) d_dense_row_value[d_csrColIdxB[cid_b]] += val_a * d_csrValB[cid_b];
        }
    }
    return;
}

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

    // malloc column index of a dense row of C
    int *d_dense_row_column_flag = (int *)malloc(nC * sizeof(int));
    VALUE_TYPE *d_dense_row_value = (VALUE_TYPE *)malloc(nC * sizeof(VALUE_TYPE));

    if (get_nnzC_only == true)
    {
        for (int rid = 0; rid < mC; rid++)
        {
            //printf("round 1, rid = %i\n", rid);
            memset(d_dense_row_column_flag, 0, nC * sizeof(int));

            compute_dense_row(d_csrRowPtrA, d_csrColIdxA, d_csrValA,
                              d_csrRowPtrB, d_csrColIdxB, d_csrValB,
                              d_dense_row_column_flag, d_dense_row_value,
                              rid, !get_nnzC_only);

            int nnzr = 0;
            for (int cid = 0; cid < nC; cid++)
            {
                if (d_dense_row_column_flag[cid] == 1)
                {
                    nnzr++;
                }
            }
            d_csrRowPtrC[rid] = nnzr;
        }
        exclusive_scan(d_csrRowPtrC, mC+1);
        *nnzC = d_csrRowPtrC[mC];
    }
    else
    {
        for (int rid = 0; rid < mC; rid++)
        {
            //printf("round 2, rid = %i\n", rid);
            memset(d_dense_row_column_flag, 0, nC * sizeof(int));
            memset(d_dense_row_value, 0, nC * sizeof(VALUE_TYPE));

            compute_dense_row(d_csrRowPtrA, d_csrColIdxA, d_csrValA,
                              d_csrRowPtrB, d_csrColIdxB, d_csrValB,
                              d_dense_row_column_flag, d_dense_row_value,
                              rid, !get_nnzC_only);

            int nnzr = 0;
            for (int cid = 0; cid < nC; cid++)
            {
                if (d_dense_row_column_flag[cid] == 1)
                {
                    d_csrColIdxC[d_csrRowPtrC[rid] + nnzr] = cid;
                    d_csrValC[d_csrRowPtrC[rid] + nnzr] = d_dense_row_value[cid];
                    nnzr++;
                }
            }
        }
    }

    free(d_dense_row_column_flag);
    free(d_dense_row_value);

    return 0;
}


#endif


