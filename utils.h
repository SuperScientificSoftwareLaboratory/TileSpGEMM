#ifndef _UTILS_
#define _UTILS_

#include "common.h"

void binary_search_right_boundary_item_kernel(const MAT_PTR_TYPE *row_pointer, 
                                              const MAT_PTR_TYPE  key_input, 
                                              const int        size, 
                                                    int       *colpos, 
                                                    MAT_PTR_TYPE *nnzpos)
{
    int start = 0;
    int stop  = size - 1;
    MAT_PTR_TYPE median;
    MAT_PTR_TYPE key_median;

    while (stop >= start)
    {
        median = (stop + start) / 2;

        key_median = row_pointer[median];

        if (key_input >= key_median)
            start = median + 1;
        else
            stop = median - 1;
    }

    *colpos = start - 1;
    *nnzpos = key_input - row_pointer[*colpos];
}

// in-place exclusive scan
void exclusive_scan(MAT_PTR_TYPE *input, int length)
{
    if(length == 0 || length == 1)
        return;
    
    MAT_PTR_TYPE old_val, new_val;
    
    old_val = input[0];
    input[0] = 0;
    for (int i = 1; i < length; i++)
    {
        new_val = input[i];
        input[i] = old_val + input[i-1];
        old_val = new_val;
    }
}


// in-place exclusive scan
void exclusive_scan_char(unsigned char *input, int length)
{
    if(length == 0 || length == 1)
        return;
    
    unsigned char old_val, new_val;
    
    old_val = input[0];
    input[0] = 0;
    for (int i = 1; i < length; i++)
    {
        new_val = input[i];
        input[i] = old_val + input[i-1];
        old_val = new_val;
    }
}


void swap_key(int *a , int *b)
{
    int tmp = *a;
    *a = *b;
    *b = tmp;
}

void swap_val(MAT_VAL_TYPE *a , MAT_VAL_TYPE *b)
{
    MAT_VAL_TYPE tmp = *a;
    *a = *b;
    *b = tmp;
}

// quick sort key-value pair (child function)
int partition_key_val_pair(int *key, MAT_VAL_TYPE *val, int length, int pivot_index)
{
    int i  = 0 ;
    int small_length = pivot_index;

    int pivot = key[pivot_index];
    swap_key(&key[pivot_index], &key[pivot_index + (length - 1)]);
    swap_val(&val[pivot_index], &val[pivot_index + (length - 1)]);

    for(; i < length; i++)
    {
        if(key[pivot_index+i] < pivot)
        {
            swap_key(&key[pivot_index+i], &key[small_length]);
            swap_val(&val[pivot_index+i], &val[small_length]);
            small_length++;
        }
    }

    swap_key(&key[pivot_index + length - 1], &key[small_length]);
    swap_val(&val[pivot_index + length - 1], &val[small_length]);

    return small_length;
}

// quick sort key-value pair (main function)
void quick_sort_key_val_pair(int *key, MAT_VAL_TYPE *val, int length)
{
    if(length == 0 || length == 1)
        return;

    int small_length = partition_key_val_pair(key, val, length, 0) ;
    quick_sort_key_val_pair(key, val, small_length);
    quick_sort_key_val_pair(&key[small_length + 1], &val[small_length + 1], length - small_length - 1);
}

// quick sort key (child function)
int partition_key(int *key, int length, int pivot_index)
{
    int i  = 0 ;
    int small_length = pivot_index;

    int pivot = key[pivot_index];
    swap_key(&key[pivot_index], &key[pivot_index + (length - 1)]);

    for(; i < length; i++)
    {
        if(key[pivot_index+i] < pivot)
        {
            swap_key(&key[pivot_index+i], &key[small_length]);
            small_length++;
        }
    }

    swap_key(&key[pivot_index + length - 1], &key[small_length]);

    return small_length;
}

// quick sort key (main function)
void quick_sort_key(int *key, int length)
{
    if(length == 0 || length == 1)
        return;

    int small_length = partition_key(key, length, 0) ;
    quick_sort_key(key, small_length);
    quick_sort_key(&key[small_length + 1], length - small_length - 1);
}



void matrix_transposition(const int           m,
                          const int           n,
                          const MAT_PTR_TYPE     nnz,
                          const MAT_PTR_TYPE    *csrRowPtr,
                          const int          *csrColIdx,
                          const MAT_VAL_TYPE *csrVal,
                                int          *cscRowIdx,
                                MAT_PTR_TYPE    *cscColPtr,
                                MAT_VAL_TYPE *cscVal)
{
    // histogram in column pointer
    memset (cscColPtr, 0, sizeof(MAT_PTR_TYPE) * (n+1));
    for (MAT_PTR_TYPE i = 0; i < nnz; i++)
    {
        cscColPtr[csrColIdx[i]]++;
    }

    // prefix-sum scan to get the column pointer
    exclusive_scan(cscColPtr, n + 1);

    MAT_PTR_TYPE *cscColIncr = (MAT_PTR_TYPE *)malloc(sizeof(MAT_PTR_TYPE) * (n+1));
    memcpy (cscColIncr, cscColPtr, sizeof(MAT_PTR_TYPE) * (n+1));

    // insert nnz to csc
    for (int row = 0; row < m; row++)
    {
        for (MAT_PTR_TYPE j = csrRowPtr[row]; j < csrRowPtr[row+1]; j++)
        {
            int col = csrColIdx[j];

            cscRowIdx[cscColIncr[col]] = row;
            cscVal[cscColIncr[col]] = csrVal[j];
            cscColIncr[col]++;
        }
    }

    free (cscColIncr);
}

#endif
