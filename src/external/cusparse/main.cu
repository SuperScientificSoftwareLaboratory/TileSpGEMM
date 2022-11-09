#include "common.h"
#include "mmio_highlevel.h"
#include "utils.h"
#include "tranpose.h"

#include "spgemm_serialref_esc.h"
#include "spgemm_cusparse.h"
#include "spgemm_serialref_spa_new.h"

int main(int argc, char ** argv)
{
    // report precision of floating-point
    printf("---------------------------------------------------------------\n");
    char  *precision;
    if (sizeof(VALUE_TYPE) == 4)
    {
        precision = (char *)"32-bit Single Precision";
    }
    else if (sizeof(VALUE_TYPE) == 8)
    {
        precision = (char *)"64-bit Double Precision";
    }
    else
    {
        printf("Wrong precision. Program exit!\n");
        return 0;
    }

    printf("PRECISION = %s\n", precision);
    printf("Benchmark REPEAT = %i\n", BENCH_REPEAT);
    printf("---------------------------------------------------------------\n");

    int mA, nA, nnzA, isSymmetricA;
    int *csrRowPtrA;
    int *csrColIdxA;
    VALUE_TYPE *csrValA;

    int mB, nB, nnzB, isSymmetricB;
    int *csrRowPtrB;
    int *csrColIdxB;
    VALUE_TYPE *csrValB;

    int device_id = 0;
    bool check_result = 0;

    // "Usage: ``./spgemm -d 0 -check 0 A.mtx B.mtx'' for AB=C on device 0, no check"
    int argi = 1;

    // load device id
    char *devstr;
    if(argc > argi)
    {
        devstr = argv[argi];
        argi++;
    }

    if (strcmp(devstr, "-d") != 0) return 0;

    if(argc > argi)
    {
        device_id = atoi(argv[argi]);
        argi++;
    }
    printf("device_id = %i\n", device_id);

    // load device id
    char *checkstr;
    if(argc > argi)
    {
        checkstr = argv[argi];
        argi++;
    }

    if (strcmp(checkstr, "-check") != 0) return 0;

    if(argc > argi)
    {
        check_result = atoi(argv[argi]);
        argi++;
    }
    printf("check_result = %i\n", check_result);

    // load matrix A data from file
    char  *filenameA;
    if(argc > argi)
    {
        filenameA = argv[argi];
        argi++;
    }
    printf("A: -------------- %s --------------\n", filenameA);

    // load mtx A data to the csr format
    srand(time(NULL));
    mmio_info(&mA, &nA, &nnzA, &isSymmetricA, filenameA);
    csrRowPtrA = (int *)malloc((mA+1) * sizeof(int));
    csrColIdxA = (int *)malloc(nnzA * sizeof(int));
    csrValA    = (VALUE_TYPE *)malloc(nnzA * sizeof(VALUE_TYPE));
    mmio_data(csrRowPtrA, csrColIdxA, csrValA, filenameA);
    for (int i = 0; i < nnzA; i++) csrValA[i] = ( rand() % 9 ) + 1;
    printf("input matrix A: ( %i, %i ) nnz = %i\n", mA, nA, nnzA);

    // keep each column sort
    /*for (int i = 0; i < mA; i++)
    {
        quick_sort_key_val_pair<int, VALUE_TYPE>(&csrColIdxA[csrRowPtrA[i]],
                                                 &csrValA[csrRowPtrA[i]],
                                                 csrRowPtrA[i+1]-csrRowPtrA[i]);
    }*/

    // load matrix B data from file
    char  *filenameB;
    if(argc > argi)
    {
        filenameB = argv[argi];
        argi++;
    }
    printf("B: -------------- %s --------------\n", filenameB);

    // load mtx B data to the csr format
    mmio_info(&mB, &nB, &nnzB, &isSymmetricB, filenameB);
    csrRowPtrB = (int *)malloc((mB+1) * sizeof(int));
    csrColIdxB = (int *)malloc(nnzB * sizeof(int));
    csrValB    = (VALUE_TYPE *)malloc(nnzB * sizeof(VALUE_TYPE));
    mmio_data(csrRowPtrB, csrColIdxB, csrValB, filenameB);
    for (int i = 0; i < nnzB; i++) csrValB[i] = ( rand() % 9 ) + 1;
    printf("input matrix B: ( %i, %i ) nnz = %i\n", mB, nB, nnzB);
    
    //if (isSymmetricB) {printf("B is symmetric, no need to compute AA. Exit\n"); return 0;}
    //    if (nA != mB) {printf("nA != mB, cannot compute AA. Exit\n"); return 0;}

    // keep each column sort
    /*for (int i = 0; i < mB; i++)
    {
        quick_sort_key_val_pair<int, VALUE_TYPE>(&csrColIdxB[csrRowPtrB[i]],
                                                 &csrValB[csrRowPtrB[i]],
                                                 csrRowPtrB[i+1]-csrRowPtrB[i]);
    }*/

        int *csrRowPtrBT = (int *)malloc((nB+1) * sizeof(int));
    int *csrColIdxBT = (int *)malloc(nnzB * sizeof(int));
    VALUE_TYPE *csrValBT    = (VALUE_TYPE *)malloc(nnzB * sizeof(VALUE_TYPE));

   // matrix_transposition(mB, nB, nnzB, csrRowPtrB, csrColIdxB, csrValB,
   //                      csrColIdxBT, csrRowPtrBT, csrValBT);
   // mB = nA;
   // nB = mA;
   
   
   // free(csrColIdxB);
   // free(csrValB);
   // free(csrRowPtrB);

   // csrColIdxB = csrColIdxBT;
   // csrValB = csrValBT;
   // csrRowPtrB = csrRowPtrBT;
    

    // calculate bytes and flops consumed
    unsigned long long int nnzCub = 0;
    for (int i = 0; i < nnzA; i++)
    {
        int rowB = csrColIdxA[i];
        nnzCub += csrRowPtrB[rowB + 1] - csrRowPtrB[rowB];
    }
    double flops = 2 * nnzCub; // flop mul-add for each nonzero entry
    printf("SpGEMM flops = %lld.\n", nnzCub);

    int mC = mA;
    int nC = nB;
    int nnzC_golden = 0;
    int *csrRowPtrC_golden;
    int *csrColIdxC_golden;
    VALUE_TYPE *csrValC_golden;

    struct timeval t1, t2;

    // run serial (ESC) SpGEMM as a reference
    if (check_result)
    {
        // printf("--------------------------------------------ESC-SPGEMM-SERIAL--\n");
        printf("--------------------------------------------SPA-SPGEMM-PARALLEL--\n");

        mC = mA;
        nC = nB;
        nnzC_golden = 0;

        // malloc d_csrRowPtrC
        csrRowPtrC_golden = (int *)malloc((mC+1) * sizeof(int));
        memset(csrRowPtrC_golden, 0, (mC+1) * sizeof(int));

        gettimeofday(&t1, NULL);

        // spgemm_serialref(csrRowPtrA, csrColIdxA, csrValA, mA, nA, nnzA,
        //                  csrRowPtrB, csrColIdxB, csrValB, mB, nB, nnzB,
        //                  csrRowPtrC_golden, csrColIdxC_golden, csrValC_golden, mC, nC, &nnzC_golden, true);
        spgemm_spa(csrRowPtrA, csrColIdxA, csrValA, mA, nA, nnzA,
                         csrRowPtrB, csrColIdxB, csrValB, mB, nB, nnzB,
                         csrRowPtrC_golden, csrColIdxC_golden, csrValC_golden, mC, nC, &nnzC_golden, 1);

        printf("Serial ref nnzC = %i, compression rate is %f\n",
               nnzC_golden, (double)nnzCub/(double)nnzC_golden);
        csrColIdxC_golden = (int *)malloc(nnzC_golden * sizeof(int));
        csrValC_golden    = (VALUE_TYPE *)malloc(nnzC_golden * sizeof(VALUE_TYPE));

        double bytes =
                sizeof(int) * (mA+1) + (sizeof(int) + sizeof(VALUE_TYPE)) * nnzA +       // data loaded from A
                sizeof(int) * (mB+1) + (sizeof(int) + sizeof(VALUE_TYPE)) * nnzB +       // data loaded from B
                sizeof(int) * (mC+1) + (sizeof(int) + sizeof(VALUE_TYPE)) * nnzC_golden; // data written back for C

        spgemm_spa(csrRowPtrA, csrColIdxA, csrValA, mA, nA, nnzA,
                        csrRowPtrB, csrColIdxB, csrValB, mB, nB, nnzB,
                        csrRowPtrC_golden, csrColIdxC_golden, csrValC_golden, mC, nC, &nnzC_golden, 0);

        // spgemm_serialref(csrRowPtrA, csrColIdxA, csrValA, mA, nA, nnzA,
        //                  csrRowPtrB, csrColIdxB, csrValB, mB, nB, nnzB,
        //                  csrRowPtrC_golden, csrColIdxC_golden, csrValC_golden, mC, nC, &nnzC_golden, false);

        gettimeofday(&t2, NULL);
        double time_spgemm_serialref = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
        printf("Serial SpGEMM takes %4.2f ms, %4.2f GFlop/s, %4.2f GB/s\n",
               time_spgemm_serialref, (1e-6*flops)/time_spgemm_serialref,
               (1e-6*bytes)/time_spgemm_serialref);
    }

    // set device
    cudaSetDevice(device_id);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device_id);

    printf("---------------------------------------------------------------\n");
    printf("Device [ %i ] %s @ %4.2f MHz\n",
           device_id, deviceProp.name, deviceProp.clockRate * 1e-3f);

    // run cuda SpGEMM (using cuSPARSE)
    printf("\n--------------- SpGEMM (using cuSPARSE) ---------------\n");
        unsigned long long int nnzC = 0;
    double compression_rate1 = 0;
        double time_cusparse = 0;
    double gflops_cusparse = 0;
    int flag =0;
    spgemm_cusparse(mA, nA, nnzA, csrRowPtrA, csrColIdxA, csrValA,
                 mB, nB, nnzB, csrRowPtrB, csrColIdxB, csrValB,
                 mC, nC, nnzC_golden, csrRowPtrC_golden, csrColIdxC_golden, csrValC_golden,
                 check_result, nnzCub, &nnzC, &compression_rate1, &time_cusparse, &gflops_cusparse,&flag);
    printf("---------------------------------------------------------------\n");

    // write results to text (scv) file
    if (gflops_cusparse > 0 && gflops_cusparse > 0)
    {
    FILE *fout = fopen("results.csv", "a");
    if (fout == NULL)
        printf("Writing results fails.\n");
    fprintf(fout, "%s,%s,%i,%i,%i,%lld,%lld,%f,%f,%f,%i\n",
            filenameA, filenameB, mA, nA, nnzA, nnzCub, nnzC, compression_rate1, time_cusparse, gflops_cusparse,flag);
    fclose(fout);
    }

    // done!
    free(csrColIdxA);
    free(csrValA);
    free(csrRowPtrA);

    free(csrColIdxB);
    free(csrValB);
    free(csrRowPtrB);

    if (check_result)
    {
        free(csrRowPtrC_golden);
        free(csrColIdxC_golden);
        free(csrValC_golden);
    }

    return 0;
}
