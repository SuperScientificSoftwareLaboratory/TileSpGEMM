template<class T>
__global__
void gen_copy( 
    uintT *key, T *val, uintT *keyB, T *valB, int n, int *segs, int *bin, int bin_size, int length) {

    const int gid = threadIdx.x + blockIdx.x * blockDim.x;
    const int bin_it = gid;
    int k;
    if(bin_it < bin_size) {
        k = segs[bin[bin_it]];
        keyB[k] = key[k];
        valB[k] = val[k];
    }
}
