#ifndef _MATRIX_INFO_UTILS_
#define _MATRIX_INFO_UTILS_

#include "common.h"
#include "utils.h"

double get_variation(const int *row_ptr,
                     const int  m)
{
    int nnz = row_ptr[m];
    double mean, stddev, skewness, variation, variance;

    mean       = double(nnz) / m;
    variance = 0.0;
    skewness = 0.0;
    for (int i = 0; i < m; i++)
    {
        int len = row_ptr[i + 1] - row_ptr[i];
        double delta = double(len) - mean;
        variance   += (delta * delta);
        skewness   += (delta * delta * delta);
    }
    variance  = variance / m;
    stddev    = sqrt(variance);
    skewness  = (skewness / m) / pow(stddev, 3.0);
    variation = stddev / mean;

    return variation;
}

double get_variation_trans(const int *row_ptr,
                           const int *col_idx,
                           const int  m,
                           const int  n)
{
    int nnz = row_ptr[m];
    int *col_ptr = (int *)malloc((n+1)*sizeof(int));
    for (int i = 0; i < n+1; i++) col_ptr[i] = 0;

    for(int i = 0; i < nnz; i++)
    {
        int j = col_idx[i];
        col_ptr[j]++;
    }
    exclusive_scan<int>(col_ptr, n+1);

    double variation = get_variation(col_ptr, n);
    free(col_ptr);

    return variation;
}

#endif
