#include "common.h"

void step1(int *blkrowptrA, int *blkcolidxA, int blkmA, int blknA,
           int *blkcolptrB, int *blkrowidxB, int blkmB, int blknB,
           int *blkrowptrC, int *numblkC)
{
    struct timeval t1, t2;

    gettimeofday(&t1, NULL);

    memset(blkrowptrC, 0, (blkmA + 1) * sizeof(MAT_PTR_TYPE));

    for (int blki = 0; blki < blkmA; blki++)
    {
        int blknnz_C = 0;
        for (int blkj = 0; blkj < blknB; blkj++)
        {
            int posA = blkrowptrA[blki];
            int posB = blkcolptrB[blkj];
            int idxA = 0;
            int idxB = 0;
            int posa_updated = 1;
            int posb_updated = 1;
            int flag = 0;

            while (posA < blkrowptrA[blki + 1] && posB < blkcolptrB[blkj + 1])
            {

                idxA = posa_updated ? blkcolidxA[posA] : idxA;
                idxB = posb_updated ? blkrowidxB[posB] : idxB;
                if (idxA == idxB) // do spgemm of this pair
                {
                    flag = 1;
                    break;
                }
                else
                {
                    posA = idxA < idxB ? posA + 1 : posA;
                    posa_updated = idxA < idxB ? 1 : 0;
                    posB = idxA > idxB ? posB + 1 : posB;
                    posb_updated = idxA > idxB ? 1 : 0;
                }
            }
            if (flag == 1)
            {
                blknnz_C++;
            }
        }
        blkrowptrC[blki] = blknnz_C;
    }

    exclusive_scan(blkrowptrC, blkmA + 1);
    *numblkC = blkrowptrC[blkmA];
}

void step2(int *blkrowptrA, int *blkcolidxA, int blkmA, int blknA,
           int *blkcolptrB, int *blkrowidxB, int blkmB, int blknB,
           int *blkrowptrC, int *blkcolidxC)
{
    struct timeval t1, t2;

    gettimeofday(&t1, NULL);

    int blkcolcount = 0;
    for (int blki = 0; blki < blkmA; blki++)
    {
        for (int blkj = 0; blkj < blknB; blkj++)
        {
            int posA = blkrowptrA[blki];
            int posB = blkcolptrB[blkj];
            int idxA = 0;
            int idxB = 0;
            int posa_updated = 1;
            int posb_updated = 1;
            while (posA < blkrowptrA[blki + 1] && posB < blkcolptrB[blkj + 1])
            {

                idxA = posa_updated ? blkcolidxA[posA] : idxA;
                idxB = posb_updated ? blkrowidxB[posB] : idxB;
                if (idxA == idxB) // do spgemm of this pair
                {
                    blkcolidxC[blkcolcount] = blkj;
                    blkcolcount++;
                    break;
                }
                else
                {
                    posA = idxA < idxB ? posA + 1 : posA;
                    posa_updated = idxA < idxB ? 1 : 0;
                    posB = idxA > idxB ? posB + 1 : posB;
                    posb_updated = idxA > idxB ? 1 : 0;
                }
            }
        }
    }
}

// void step3(int *blkrowptrA, int *blkcolidxA, int blkmA, int blknA, int *nnzb_A, int mA,
//            MAT_VAL_TYPE *blkcsr_Val_A, unsigned char *blkcsr_Col_A, unsigned char *blkcsr_Ptr_A,
//            int *blkcolptrB, int *blkrowidxB, int blkmB, int blknB, int *nnzb_B,
//            MAT_VAL_TYPE *blkcsr_Val_B, unsigned char *blkcsr_Col_B, unsigned char *blkcsr_Ptr_B,
//            int *blkrowptrC, int *blkcolidxC, int *nnzb_C, int *nnzC)
// {

//     struct timeval t1, t2;

//     gettimeofday(&t1, NULL);

//     char *blkc = (char *)malloc((BLOCK_SIZE * BLOCK_SIZE) * sizeof(char));

//     for (int blki = 0; blki < blkmA; blki++)
//     {
//         int rowlen = blki == blkmA - 1 ? mA - (blkmA - 1) * BLOCK_SIZE : BLOCK_SIZE;
//         for (int blkj = blkrowptrC[blki]; blkj < blkrowptrC[blki + 1]; blkj++)
//         {
//             int count = 0;
//             int blkccolid = blkcolidxC[blkj];
//             memset(blkc, 0, (BLOCK_SIZE * BLOCK_SIZE) * sizeof(char));

//             int posA = blkrowptrA[blki];
//             int posB = blkcolptrB[blkccolid];
//             int idxA = 0;
//             int idxB = 0;
//             int posa_updated = 1;
//             int posb_updated = 1;
//             while (posA < blkrowptrA[blki + 1] && posB < blkcolptrB[blkccolid + 1])
//             {
//                 idxA = posa_updated ? blkcolidxA[posA] : idxA;
//                 idxB = posb_updated ? blkrowidxB[posB] : idxB;
//                 if (idxA == idxB) // do spgemm of this pair
//                 {
//                     for (int ri = 0; ri < BLOCK_SIZE; ri++)
//                     {
//                         if (ri == rowlen)
//                             break;
//                         int stopa = ri == BLOCK_SIZE - 1 ? nnzb_A[posA + 1] - nnzb_A[posA] : blkcsr_Ptr_A[posA * BLOCK_SIZE + ri + 1];

//                         for (int i = blkcsr_Ptr_A[posA * BLOCK_SIZE + ri]; i < stopa; i++)
//                         {
//                             int cola = blkcsr_Col_A[nnzb_A[posA] + i];
//                             int stopb = cola == BLOCK_SIZE - 1 ? nnzb_B[posB + 1] - nnzb_B[posB] : blkcsr_Ptr_B[posB * BLOCK_SIZE + cola + 1];
//                             for (int bi = blkcsr_Ptr_B[posB * BLOCK_SIZE + cola]; bi < stopb; bi++)
//                             {
//                                 const int colb = blkcsr_Col_B[nnzb_B[posB] + bi];
//                                 if (blkc[ri * BLOCK_SIZE + colb] == 0)
//                                 {
//                                     blkc[ri * BLOCK_SIZE + colb] = 1;
//                                 }
//                             }
//                         }
//                     }
//                     posA++;
//                     posa_updated = 1;
//                     posB++;
//                     posb_updated = 1;
//                 }
//                 else
//                 {
//                     posA = idxA < idxB ? posA + 1 : posA;
//                     posa_updated = idxA < idxB ? 1 : 0;
//                     posB = idxA > idxB ? posB + 1 : posB;
//                     posb_updated = idxA > idxB ? 1 : 0;
//                 }
//             }
//             for (int ci = 0; ci < BLOCK_SIZE * BLOCK_SIZE; ci++)
//             {
//                 if (blkc[ci] == 1)
//                 {
//                     count++;
//                 }
//             }
//             nnzb_C[blkj] = count;
//         }
//     }

//     exclusive_scan(nnzb_C, blkrowptrC[blkmA] + 1);
//     *nnzC = nnzb_C[blkrowptrC[blkmA]];
//         for (int i=0; i< blkrowptrC[blkmA] + 1; i ++)
//     {
//         printf("i= %i, nnz = %i\n", i, nnzb_C[i]);
//     }

// }

void step3 (int *d_blkrowptrA, int *d_blkcolidxA, int blkmA, int blknA,int *nnzb_A ,int mA,
            MAT_VAL_TYPE *blkcsr_Val_A , unsigned char  *blkcsr_Col_A , unsigned char *blkcsr_Ptr_A ,
            int *d_blkcolptrB, int *d_blkrowidxB, int blkmB, int blknB , int *nnzb_B ,int nB,
            MAT_VAL_TYPE *blkcsr_Val_B , unsigned char  *blkcsr_Col_B , unsigned char *blkcsr_Ptr_B ,
            int *d_blkrowptrC, int *d_blkcolidxC,int *nnzb_C, int *nnzC)
{

    struct timeval t1, t2;

    gettimeofday(&t1, NULL);

    char * blkc = (char *)malloc((BLOCK_SIZE * BLOCK_SIZE) *sizeof(char));

    for (int blki =0 ;blki <blkmA ;blki++)
    {
        int rowlen = blki == blkmA -1 ? mA- (blkmA -1 ) * BLOCK_SIZE : BLOCK_SIZE ;
        for (int blkj =d_blkrowptrC[blki]; blkj <d_blkrowptrC[blki + 1]; blkj++)
        {
            int count =0 ;
            int blkccolid = d_blkcolidxC[blkj];
        //    int rowlen = blki == blkmA -1 ? mA- (blkmA -1 ) * BLOCK_SIZE : BLOCK_SIZE ;
           int collen = blkccolid == blknB -1 ? nB - (blknB -1) *BLOCK_SIZE : BLOCK_SIZE ;
            memset (blkc , 0, (BLOCK_SIZE * BLOCK_SIZE) *sizeof(char));

            int posA = d_blkrowptrA[blki];
            int posB = d_blkcolptrB[blkccolid];
            int idxA= 0;
            int idxB =0;
            int posa_updated =1;
            int posb_updated =1;
            while (posA < d_blkrowptrA[blki +1] && posB <d_blkcolptrB[blkccolid + 1])
            {
                idxA = posa_updated ? d_blkcolidxA[posA] : idxA ;
                idxB = posb_updated ? d_blkrowidxB[posB] : idxB ;
                if (idxA == idxB)  // do spgemm of this pair
                {
                //        printf ("do spgemm of this pair, idxa = %i, idxb = %i\n", idxA, idxB);
                //for each row of block
                    for (int ri =0;ri <BLOCK_SIZE ;ri ++ )
                    {
                        if (ri == rowlen)
                            break;
                        int stopa = ri == rowlen -1 ? nnzb_A[posA +1] - nnzb_A[posA] : blkcsr_Ptr_A[posA * BLOCK_SIZE + ri + 1] ;
                
                        for (int i=blkcsr_Ptr_A[ posA * BLOCK_SIZE+ ri];i<stopa;i++)
                        {
                            int cola= blkcsr_Col_A[nnzb_A[posA]+i] ;
                            int stopb = cola == collen -1  ? nnzb_B[posB +1]- nnzb_B[posB] : blkcsr_Ptr_B[posB * BLOCK_SIZE+cola +1] ;
                            for (int bi= blkcsr_Ptr_B[posB * BLOCK_SIZE +cola ];bi< stopb; bi++)
                            {
                                const int colb = blkcsr_Col_B[nnzb_B[posB] + bi];
                                if (blkc[ri * BLOCK_SIZE + colb] == 0)
                                {
                                    blkc[ri * BLOCK_SIZE + colb] = 1;
                                }
                            }

                        }
                    }
                    posA++;
                    posa_updated = 1;
                    posB++;
                    posb_updated = 1;
                }
                else 
                {
                //    printf ("the smaller index goes forward (before), posa = %i, posb = %i\n", posA, posA);
                    posA = idxA < idxB ? posA + 1 : posA;
                    posa_updated = idxA < idxB ? 1 : 0;
                    posB = idxA > idxB ? posB + 1 : posB;
                    posb_updated = idxA > idxB ? 1 : 0;
                //    printf ("the smaller index goes forward (after),  posa = %i, posb = %i\n", posA, posA);
                }

            }  
            for (int ci=0;ci< BLOCK_SIZE * BLOCK_SIZE ; ci ++)
            {
                if (blkc[ci]== 1)
                {
                    count ++ ;
                }
            }
            nnzb_C[blkj] = count ;
        //    printf("count = %d\n",count);
        }
    }

    exclusive_scan(nnzb_C,d_blkrowptrC[blkmA] + 1);

    *nnzC = nnzb_C[d_blkrowptrC[blkmA]];
    printf("nnzc = %i\n", *nnzC);

        gettimeofday(&t2, NULL);

    double time_kernel = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    printf("CPU  step3 kernel = %.2f ms\n", time_kernel);
}


void step4(int *blkrowptrA, int *blkcolidxA, int blkmA, int blknA, int *nnzb_A, int mA,
           MAT_VAL_TYPE *blkcsr_Val_A, unsigned char *blkcsr_Col_A, unsigned char *blkcsr_Ptr_A,
           int *blkcolptrB, int *blkrowidxB, int blkmB, int blknB, int *nnzb_B,
           MAT_VAL_TYPE *blkcsr_Val_B, unsigned char *blkcsr_Col_B, unsigned char *blkcsr_Ptr_B,
           int *blkrowptrC, int *blkcolidxC, int *nnzb_C,
           MAT_VAL_TYPE *blkcsr_Val_C, unsigned char *blkcsr_Col_C, unsigned char *blkcsr_Ptr_C)

{
    MAT_VAL_TYPE *blkcval = (MAT_VAL_TYPE *)malloc((BLOCK_SIZE * BLOCK_SIZE) * sizeof(MAT_VAL_TYPE));
    char *blkc = (char *)malloc((BLOCK_SIZE * BLOCK_SIZE) * sizeof(char));
    for (int blki = 0; blki < blkmA; blki++)
    {
        int rowlen = blki == blkmA - 1 ? mA - (blkmA - 1) * BLOCK_SIZE : BLOCK_SIZE;
        for (int blkj = blkrowptrC[blki]; blkj < blkrowptrC[blki + 1]; blkj++)
        {
            int count = 0;
            int blkccolid = blkcolidxC[blkj];
            memset(blkc, 0, (BLOCK_SIZE * BLOCK_SIZE) * sizeof(char));
            memset(blkcval, 0, (BLOCK_SIZE * BLOCK_SIZE) * sizeof(MAT_VAL_TYPE));

            int posA = blkrowptrA[blki];
            int posB = blkcolptrB[blkccolid];
            int idxA = 0;
            int idxB = 0;
            int posa_updated = 1;
            int posb_updated = 1;
            while (posA < blkrowptrA[blki + 1] && posB < blkcolptrB[blkccolid + 1])
            {
                idxA = posa_updated ? blkcolidxA[posA] : idxA;
                idxB = posb_updated ? blkrowidxB[posB] : idxB;
                if (idxA == idxB) // do spgemm of this pair
                {
                    for (int ri = 0; ri < BLOCK_SIZE; ri++)
                    {
                        if (ri == rowlen)
                            break;
                        int stopa = ri == BLOCK_SIZE - 1 ? nnzb_A[posA + 1] - nnzb_A[posA] : blkcsr_Ptr_A[posA * BLOCK_SIZE + ri + 1];

                        for (int i = blkcsr_Ptr_A[posA * BLOCK_SIZE + ri]; i < stopa; i++)
                        {
                            int cola = blkcsr_Col_A[nnzb_A[posA] + i];
                            int stopb = cola == BLOCK_SIZE - 1 ? nnzb_B[posB + 1] - nnzb_B[posB] : blkcsr_Ptr_B[posB * BLOCK_SIZE + cola + 1];
                            for (int bi = blkcsr_Ptr_B[posB * BLOCK_SIZE + cola]; bi < stopb; bi++)
                            {
                                const int colb = blkcsr_Col_B[nnzb_B[posB] + bi];

                                blkcval[ri * BLOCK_SIZE + colb] += blkcsr_Val_A[nnzb_A[posA] + i] * blkcsr_Val_B[nnzb_B[posB] + bi];
                                if (blkc[ri * BLOCK_SIZE + colb] == 0)
                                {
                                    blkc[ri * BLOCK_SIZE + colb] = 1;
                                }
                            }
                        }
                    }
                    posA++;
                    posa_updated = 1;
                    posB++;
                    posb_updated = 1;
                }
                else
                {
                    posA = idxA < idxB ? posA + 1 : posA;
                    posa_updated = idxA < idxB ? 1 : 0;
                    posB = idxA > idxB ? posB + 1 : posB;
                    posb_updated = idxA > idxB ? 1 : 0;
                }
            }
            for (int ri = 0; ri < BLOCK_SIZE; ri++)
            {
                for (int ci = 0; ci < BLOCK_SIZE; ci++)
                {
                    if (blkc[ri * BLOCK_SIZE + ci] == 1)
                    {
                        blkcsr_Val_C[nnzb_C[blkj] + count] = blkcval[ri * BLOCK_SIZE + ci];
                        blkcsr_Col_C[nnzb_C[blkj] + count] = ci;
                        count++;
                    }
                }
                if (ri < BLOCK_SIZE - 1)
                    blkcsr_Ptr_C[BLOCK_SIZE * blkj + ri + 1] = count;
            }
        }
    }
}

void spgemm_cpu(SMatrix *A,
                SMatrix *B,
                SMatrix *C)
{
    int blkmA = A->tilem;
    int blknA = A->tilen;
    int mA = A->m;
    int nA = A->n;
    int nnzA = A->nnz;
    int numblkA = A->numtile;
    int *blkrowptrA = A->tile_ptr;
    int *blkcolidxA = A->tile_columnidx;
    int *nnzb_A = A->tile_nnz;
    MAT_VAL_TYPE *blkcsr_Val_A = A->tile_csr_Value;
    unsigned char *blkcsr_Col_A = A->tile_csr_Col;
    unsigned char *blkcsr_Ptr_A = A->tile_csr_Ptr;

    int blkmB = B->tilem;
    int blknB = B->tilen;
    int mB = B->m;
    int nB = B->n;
    int nnzB = B->nnz;
    int numblkB = B->numtile;
    int *blkcolptrB = B->csc_tile_ptr;
    int *blkrowidxB = B->csc_tile_rowidx;
    int *nnzb_B = B->tile_nnz;
    MAT_VAL_TYPE *blkcsr_Val_B = B->tile_csr_Value;
    unsigned char *blkcsr_Col_B = B->tile_csr_Col;
    unsigned char *blkcsr_Ptr_B = B->tile_csr_Ptr;

    int *blkrowptrC = (int *)malloc((blkmA + 1) * sizeof(int));
    memset(blkrowptrC, 0, (blkmA + 1) * sizeof(int));
    int numtileC;

    step1(blkrowptrA, blkcolidxA, blkmA, blknA,
          blkcolptrB, blkrowidxB, blkmB, blknB,
          blkrowptrC, &numtileC);

    int *blkcolidxC = (int *)malloc(numtileC * sizeof(int));
    memset(blkcolidxC, 0, (numtileC) * sizeof(int));

    step2(blkrowptrA, blkcolidxA, blkmA, blknA,
          blkcolptrB, blkrowidxB, blkmB, blknB,
          blkrowptrC, blkcolidxC);

    int *nnzb_C = (int *)malloc((numtileC + 1) * sizeof(int));
    memset(nnzb_C, 0, (numtileC + 1) * sizeof(int));
    int nnzC =0;
    step3(blkrowptrA, blkcolidxA, blkmA, blknA, nnzb_A, mA,
          blkcsr_Val_A, blkcsr_Col_A, blkcsr_Ptr_A,
          blkcolptrB, blkrowidxB, blkmB, blknB, nnzb_B,nB,
          blkcsr_Val_B, blkcsr_Col_B, blkcsr_Ptr_B,
          blkrowptrC, blkcolidxC, nnzb_C, &nnzC);
    // for (int i=0; i< numtileC + 1; i ++)
    // {
    //     printf("i= %i, nnz = %i\n", i, nnzb_C[i]);
    // }


    MAT_VAL_TYPE *blkcsr_Val_C = (MAT_VAL_TYPE *)malloc(nnzC * sizeof(MAT_VAL_TYPE));
    memset(blkcsr_Val_C, 0, nnzC * sizeof(MAT_VAL_TYPE));
    unsigned char *blkcsr_Col_C = (unsigned char *)malloc(nnzC * sizeof(unsigned char));
    memset(blkcsr_Col_C, 0, nnzC * sizeof(unsigned char));
    unsigned char *blkcsr_Ptr_C = (unsigned char *)malloc(numtileC * BLOCK_SIZE * sizeof(unsigned char));
    memset(blkcsr_Ptr_C, 0, numtileC * BLOCK_SIZE * sizeof(unsigned char));

    step4(blkrowptrA, blkcolidxA, blkmA, blknA, nnzb_A, mA,
          blkcsr_Val_A, blkcsr_Col_A, blkcsr_Ptr_A,
          blkcolptrB, blkrowidxB, blkmB, blknB, nnzb_B,
          blkcsr_Val_B, blkcsr_Col_B, blkcsr_Ptr_B,
          blkrowptrC, blkcolidxC, nnzb_C,
          blkcsr_Val_C, blkcsr_Col_C, blkcsr_Ptr_C);

    printf("spgemm-cpu complete\n");

    C->tile_csr_Ptr = blkcsr_Ptr_C;
    C->tile_csr_Value = blkcsr_Val_C;
    C->tile_csr_Col = blkcsr_Col_C;
    C->tile_ptr = blkrowptrC;
    C->tile_columnidx = blkcolidxC;
    C->tile_nnz = nnzb_C;
   // C->tile_ptr = blkrowptrC;
   // C->tile_columnidx = blkcolidxC;
    C->nnz = nnzC;
    C->numtile = numtileC;

}