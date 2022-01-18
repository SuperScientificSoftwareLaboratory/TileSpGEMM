

#ifndef _TILETOCSR_
#define _TILETOCSR_
#include <stdio.h>
#include <stdlib.h>
#include "common.h"
void Tile_csr_to_csr_PTR(unsigned char *Tile_csr_Ptr,
                         MAT_VAL_TYPE *Tile_csr_Val,
                         int tilennz,
                         int tilem,
                         int m,
                         int tile_row,
                         int *csrRowPtr,
                         int csr_ptr_offset,
                         int tile_csr_offset,
                         int tile_csrptr_offset)
{

    int rowlen = tile_row == tilem - 1 ? m - (tilem - 1) * BLOCK_SIZE : BLOCK_SIZE;
    for (int i = 0; i < rowlen; i++)
    {
        int temp = i == rowlen - 1 ? tilennz : Tile_csr_Ptr[tile_csrptr_offset + i + 1];
        int cnt = 0;
        for (int j = Tile_csr_Ptr[tile_csrptr_offset + i]; j < temp; j++)
        {
            // if (Tile_csr_Val[tile_csr_offset + j] == 0)
            //     cnt++;
        }
        csrRowPtr[csr_ptr_offset + i] += temp - Tile_csr_Ptr[tile_csrptr_offset + i] - cnt;
    }
}

void Tile_csr_to_csr(unsigned char *Tile_csr_Ptr,
                     unsigned char *Tile_csr_Col,
                     MAT_VAL_TYPE *Tile_csr_Val,
                     int tilennz,
                     int tilem,
                     int m,
                     int tile_row,
                     int tile_col,
                     int *csrRowPtr,
                     int *csrColIdx,
                     MAT_VAL_TYPE *csrVal,
                     int csr_ptr_offset,
                     int tile_csrptr_offset,
                     int tile_csr_index_offset,
                     int *row_nnz_offset)

{
    int rowlen = tile_row == tilem - 1 ? m - (tilem - 1) * BLOCK_SIZE : BLOCK_SIZE;
    for (int i = 0; i < rowlen; i++)
    {
        int start = Tile_csr_Ptr[tile_csrptr_offset + i];
        int end = i == rowlen - 1 ? tilennz : Tile_csr_Ptr[tile_csrptr_offset + i + 1];
        for (int j = start; j < end; j++)
        {
            // if (Tile_csr_Val[tile_csr_index_offset + j] != 0)
            // {
                int temp = csrRowPtr[csr_ptr_offset + i] + row_nnz_offset[tile_row * BLOCK_SIZE + i];
                csrColIdx[temp] = tile_col * BLOCK_SIZE + Tile_csr_Col[tile_csr_index_offset + j];
                csrVal[temp] = Tile_csr_Val[tile_csr_index_offset + j];
                row_nnz_offset[tile_row * BLOCK_SIZE + i]++;
            // }
        }
    }
}



void tile2csr(SMatrix *matrix)
{

    matrix->rowpointer = (MAT_PTR_TYPE *)malloc((matrix->m + 1) *sizeof(MAT_PTR_TYPE));
    MAT_PTR_TYPE *csrRowPtr = matrix->rowpointer;
    memset(csrRowPtr, 0, (matrix->m + 1) * sizeof(MAT_PTR_TYPE));

#pragma omp parallel for
    for (int i = 0; i < matrix->tilem; i++)
    {
        for (int j = matrix->tile_ptr[i]; j < matrix->tile_ptr[i + 1]; j++)
        {
            int csr_ptr_offset = i * BLOCK_SIZE;
            int tilennz = matrix->tile_nnz[j + 1] - matrix->tile_nnz[j];
            int m = matrix->m;
            int n = matrix->n;
            int tilem = matrix->tilem;
            int tilen = matrix->tilen;
            int tile_id = j;
            int tile_row = i;
            int tile_col = matrix->tile_columnidx[j];
            int tile_csr_offset = matrix->tile_nnz[j];
            int tile_csrptr_offset = j * BLOCK_SIZE;

            Tile_csr_to_csr_PTR(matrix->tile_csr_Ptr, matrix->tile_csr_Value, tilennz, tilem, m, tile_row, csrRowPtr,
                                csr_ptr_offset, tile_csr_offset, tile_csrptr_offset);
        }
    }
    exclusive_scan(csrRowPtr, matrix->m + 1);
    
    int nnzc_real = csrRowPtr[matrix->m];
    matrix->nnz = nnzc_real;

    matrix->value = (MAT_VAL_TYPE *)malloc(nnzc_real * sizeof(MAT_VAL_TYPE));
    memset(matrix->value, 0, nnzc_real * sizeof(MAT_VAL_TYPE));
    matrix->columnindex = (int *)malloc(nnzc_real * sizeof(int));
    memset(matrix->columnindex, 0, nnzc_real * sizeof(int));

    int *csrColIdx = matrix->columnindex;
    MAT_VAL_TYPE *csrVal = matrix->value;

    int *row_nnz_offset = (int *)malloc(sizeof(int) * matrix->m);
    memset(row_nnz_offset, 0, sizeof(int) * matrix->m);

#pragma omp parallel for
    for (int i = 0; i < matrix->tilem; i++)
    {
        for (int j = matrix->tile_ptr[i]; j < matrix->tile_ptr[i + 1]; j++)
        {
            int csr_ptr_offset = i * BLOCK_SIZE;
            int tilennz = matrix->tile_nnz[j + 1] - matrix->tile_nnz[j];
            int m = matrix->m;
            int n = matrix->n;
            int tilem = matrix->tilem;
            int tilen = matrix->tilen;
            int tile_id = j;
            int tile_row = i;
            int tile_col = matrix->tile_columnidx[j];
            int tile_csr_index_offset = matrix->tile_nnz[j];
            int tile_csrptr_offset = j * BLOCK_SIZE;

            Tile_csr_to_csr(matrix->tile_csr_Ptr, matrix->tile_csr_Col, matrix->tile_csr_Value,
                            tilennz, tilem, m, tile_row, tile_col, csrRowPtr, csrColIdx, csrVal,
                            csr_ptr_offset, tile_csrptr_offset, tile_csr_index_offset, row_nnz_offset);
        }
    }
}

#endif
