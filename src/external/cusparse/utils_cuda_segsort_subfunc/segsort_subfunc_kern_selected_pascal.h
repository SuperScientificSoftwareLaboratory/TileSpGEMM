
#ifdef _ARCH61
/* block tcf subwarp coalesced quiet real_kern */
/*   256   1       2     false  true      true */
template<class T>
__global__
void gen_bk256_wp2_tc1_r2_r2_orig( 
    uintT *key, T *val, uintT *keyB, T *valB, int n, int *segs, int *bin, int bin_size, int length) {

    const int gid = threadIdx.x + blockIdx.x * blockDim.x;
    const int bin_it = (gid>>1);
    const int tid = (threadIdx.x & 1);
    const int bit1 = (tid>>0)&0x1;
    uintT rg_k0 ;
    int rg_v0 ;
    int k;
    int seg_size;
    if(bin_it < bin_size) {
        k = segs[bin[bin_it]];
        seg_size = ((bin[bin_it]==length-1)?n:segs[bin[bin_it]+1])-segs[bin[bin_it]];
        rg_k0  = (tid+0   <seg_size)?key[k+tid+0   ]:INT_MAX;
        if(tid+0   <seg_size) rg_v0  = tid+0   ;
        // sort 2 elements
        // exch_intxn: generate exch_intxn()
        exch_intxn(rg_k0 ,
                   rg_v0 ,
                   0x1,bit1);
        if((tid<<0)+0 <seg_size) keyB[k+(tid<<0)+0 ] = rg_k0 ;
        if((tid<<0)+0 <seg_size) valB[k+(tid<<0)+0 ] = val[k+rg_v0 ];
    }
}
/* block tcf subwarp coalesced quiet real_kern */
/*   128   2       2     false  true      true */
template<class T>
__global__
void gen_bk128_wp2_tc2_r3_r4_orig( 
    uintT *key, T *val, uintT *keyB, T *valB, int n, int *segs, int *bin, int bin_size, int length) {

    const int gid = threadIdx.x + blockIdx.x * blockDim.x;
    const int bin_it = (gid>>1);
    const int tid = (threadIdx.x & 1);
    const int bit1 = (tid>>0)&0x1;
    uintT rg_k0 ;
    uintT rg_k1 ;
    int rg_v0 ;
    int rg_v1 ;
    int k;
    int seg_size;
    if(bin_it < bin_size) {
        k = segs[bin[bin_it]];
        seg_size = ((bin[bin_it]==length-1)?n:segs[bin[bin_it]+1])-segs[bin[bin_it]];
        rg_k0  = (tid+0   <seg_size)?key[k+tid+0   ]:INT_MAX;
        rg_k1  = (tid+2   <seg_size)?key[k+tid+2   ]:INT_MAX;
        if(tid+0   <seg_size) rg_v0  = tid+0   ;
        if(tid+2   <seg_size) rg_v1  = tid+2   ;
        // sort 4 elements
        // exch_intxn: switch to exch_local()
        CMP_SWP(uintT,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
        // exch_intxn: generate exch_intxn()
        exch_intxn(rg_k0 ,rg_k1 ,
                   rg_v0 ,rg_v1 ,
                   0x1,bit1);
        // exch_paral: switch to exch_local()
        CMP_SWP(uintT,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
        if((tid<<1)+0 <seg_size) keyB[k+(tid<<1)+0 ] = rg_k0 ;
        if((tid<<1)+1 <seg_size) keyB[k+(tid<<1)+1 ] = rg_k1 ;
        if((tid<<1)+0 <seg_size) valB[k+(tid<<1)+0 ] = val[k+rg_v0 ];
        if((tid<<1)+1 <seg_size) valB[k+(tid<<1)+1 ] = val[k+rg_v1 ];
    }
}
/* block tcf subwarp coalesced quiet real_kern */
/*   128   4       2     false  true      true */
template<class T>
__global__
void gen_bk128_wp2_tc4_r5_r8_orig( 
    uintT *key, T *val, uintT *keyB, T *valB, int n, int *segs, int *bin, int bin_size, int length) {

    const int gid = threadIdx.x + blockIdx.x * blockDim.x;
    const int bin_it = (gid>>1);
    const int tid = (threadIdx.x & 1);
    const int bit1 = (tid>>0)&0x1;
    uintT rg_k0 ;
    uintT rg_k1 ;
    uintT rg_k2 ;
    uintT rg_k3 ;
    int rg_v0 ;
    int rg_v1 ;
    int rg_v2 ;
    int rg_v3 ;
    int k;
    int seg_size;
    if(bin_it < bin_size) {
        k = segs[bin[bin_it]];
        seg_size = ((bin[bin_it]==length-1)?n:segs[bin[bin_it]+1])-segs[bin[bin_it]];
        rg_k0  = (tid+0   <seg_size)?key[k+tid+0   ]:INT_MAX;
        rg_k1  = (tid+2   <seg_size)?key[k+tid+2   ]:INT_MAX;
        rg_k2  = (tid+4   <seg_size)?key[k+tid+4   ]:INT_MAX;
        rg_k3  = (tid+6   <seg_size)?key[k+tid+6   ]:INT_MAX;
        if(tid+0   <seg_size) rg_v0  = tid+0   ;
        if(tid+2   <seg_size) rg_v1  = tid+2   ;
        if(tid+4   <seg_size) rg_v2  = tid+4   ;
        if(tid+6   <seg_size) rg_v3  = tid+6   ;
        // sort 8 elements
        // exch_intxn: switch to exch_local()
        CMP_SWP(uintT,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
        CMP_SWP(uintT,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
        // exch_intxn: switch to exch_local()
        CMP_SWP(uintT,rg_k0 ,rg_k3 ,int,rg_v0 ,rg_v3 );
        CMP_SWP(uintT,rg_k1 ,rg_k2 ,int,rg_v1 ,rg_v2 );
        // exch_paral: switch to exch_local()
        CMP_SWP(uintT,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
        CMP_SWP(uintT,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
        // exch_intxn: generate exch_intxn()
        exch_intxn(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                   rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                   0x1,bit1);
        // exch_paral: switch to exch_local()
        CMP_SWP(uintT,rg_k0 ,rg_k2 ,int,rg_v0 ,rg_v2 );
        CMP_SWP(uintT,rg_k1 ,rg_k3 ,int,rg_v1 ,rg_v3 );
        // exch_paral: switch to exch_local()
        CMP_SWP(uintT,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
        CMP_SWP(uintT,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
        if((tid<<2)+0 <seg_size) keyB[k+(tid<<2)+0 ] = rg_k0 ;
        if((tid<<2)+1 <seg_size) keyB[k+(tid<<2)+1 ] = rg_k1 ;
        if((tid<<2)+2 <seg_size) keyB[k+(tid<<2)+2 ] = rg_k2 ;
        if((tid<<2)+3 <seg_size) keyB[k+(tid<<2)+3 ] = rg_k3 ;
        if((tid<<2)+0 <seg_size) valB[k+(tid<<2)+0 ] = val[k+rg_v0 ];
        if((tid<<2)+1 <seg_size) valB[k+(tid<<2)+1 ] = val[k+rg_v1 ];
        if((tid<<2)+2 <seg_size) valB[k+(tid<<2)+2 ] = val[k+rg_v2 ];
        if((tid<<2)+3 <seg_size) valB[k+(tid<<2)+3 ] = val[k+rg_v3 ];
    }
}
/* block tcf subwarp coalesced quiet real_kern */
/*   128   4       4      true  true      true */
template<class T>
__global__
void gen_bk128_wp4_tc4_r9_r16_strd( 
    uintT *key, T *val, uintT *keyB, T *valB, int n, int *segs, int *bin, int bin_size, int length) {

    const int gid = threadIdx.x + blockIdx.x * blockDim.x;
    const int bin_it = (gid>>2);
    const int tid = (threadIdx.x & 3);
    const int bit1 = (tid>>0)&0x1;
    const int bit2 = (tid>>1)&0x1;
    uintT rg_k0 ;
    uintT rg_k1 ;
    uintT rg_k2 ;
    uintT rg_k3 ;
    int rg_v0 ;
    int rg_v1 ;
    int rg_v2 ;
    int rg_v3 ;
    int normalized_bin_size = (bin_size/8)*8;
    int k;
    int seg_size;
    if(bin_it < bin_size) {
        k = segs[bin[bin_it]];
        seg_size = ((bin[bin_it]==length-1)?n:segs[bin[bin_it]+1])-segs[bin[bin_it]];
        rg_k0  = (tid+0   <seg_size)?key[k+tid+0   ]:INT_MAX;
        rg_k1  = (tid+4   <seg_size)?key[k+tid+4   ]:INT_MAX;
        rg_k2  = (tid+8   <seg_size)?key[k+tid+8   ]:INT_MAX;
        rg_k3  = (tid+12  <seg_size)?key[k+tid+12  ]:INT_MAX;
        if(tid+0   <seg_size) rg_v0  = tid+0   ;
        if(tid+4   <seg_size) rg_v1  = tid+4   ;
        if(tid+8   <seg_size) rg_v2  = tid+8   ;
        if(tid+12  <seg_size) rg_v3  = tid+12  ;
        // sort 16 elements
        // exch_intxn: switch to exch_local()
        CMP_SWP(uintT,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
        CMP_SWP(uintT,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
        // exch_intxn: switch to exch_local()
        CMP_SWP(uintT,rg_k0 ,rg_k3 ,int,rg_v0 ,rg_v3 );
        CMP_SWP(uintT,rg_k1 ,rg_k2 ,int,rg_v1 ,rg_v2 );
        // exch_paral: switch to exch_local()
        CMP_SWP(uintT,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
        CMP_SWP(uintT,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
        // exch_intxn: generate exch_intxn()
        exch_intxn(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                   rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                   0x1,bit1);
        // exch_paral: switch to exch_local()
        CMP_SWP(uintT,rg_k0 ,rg_k2 ,int,rg_v0 ,rg_v2 );
        CMP_SWP(uintT,rg_k1 ,rg_k3 ,int,rg_v1 ,rg_v3 );
        // exch_paral: switch to exch_local()
        CMP_SWP(uintT,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
        CMP_SWP(uintT,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
        // exch_intxn: generate exch_intxn()
        exch_intxn(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                   rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                   0x3,bit2);
        // exch_paral: generate exch_paral()
        exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                   rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                   0x1,bit1);
        // exch_paral: switch to exch_local()
        CMP_SWP(uintT,rg_k0 ,rg_k2 ,int,rg_v0 ,rg_v2 );
        CMP_SWP(uintT,rg_k1 ,rg_k3 ,int,rg_v1 ,rg_v3 );
        // exch_paral: switch to exch_local()
        CMP_SWP(uintT,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
        CMP_SWP(uintT,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
    }

    if(bin_it < normalized_bin_size) {
        // store back the results
        int lane_id = threadIdx.x & 31;
        rg_k1  = __shfl_xor(rg_k1 , 0x1 );
        rg_k3  = __shfl_xor(rg_k3 , 0x1 );
        rg_v1  = __shfl_xor(rg_v1 , 0x1 );
        rg_v3  = __shfl_xor(rg_v3 , 0x1 );
        if(lane_id&0x1 ) SWP(uintT, rg_k0 , rg_k1 , int, rg_v0 , rg_v1 );
        if(lane_id&0x1 ) SWP(uintT, rg_k2 , rg_k3 , int, rg_v2 , rg_v3 );
        rg_k1  = __shfl_xor(rg_k1 , 0x1 );
        rg_k3  = __shfl_xor(rg_k3 , 0x1 );
        rg_v1  = __shfl_xor(rg_v1 , 0x1 );
        rg_v3  = __shfl_xor(rg_v3 , 0x1 );
        rg_k2  = __shfl_xor(rg_k2 , 0x2 );
        rg_k3  = __shfl_xor(rg_k3 , 0x2 );
        rg_v2  = __shfl_xor(rg_v2 , 0x2 );
        rg_v3  = __shfl_xor(rg_v3 , 0x2 );
        if(lane_id&0x2 ) SWP(uintT, rg_k0 , rg_k2 , int, rg_v0 , rg_v2 );
        if(lane_id&0x2 ) SWP(uintT, rg_k1 , rg_k3 , int, rg_v1 , rg_v3 );
        rg_k2  = __shfl_xor(rg_k2 , 0x2 );
        rg_k3  = __shfl_xor(rg_k3 , 0x2 );
        rg_v2  = __shfl_xor(rg_v2 , 0x2 );
        rg_v3  = __shfl_xor(rg_v3 , 0x2 );
        rg_k1  = __shfl_xor(rg_k1 , 0x4 );
        rg_k3  = __shfl_xor(rg_k3 , 0x4 );
        rg_v1  = __shfl_xor(rg_v1 , 0x4 );
        rg_v3  = __shfl_xor(rg_v3 , 0x4 );
        if(lane_id&0x4 ) SWP(uintT, rg_k0 , rg_k1 , int, rg_v0 , rg_v1 );
        if(lane_id&0x4 ) SWP(uintT, rg_k2 , rg_k3 , int, rg_v2 , rg_v3 );
        rg_k1  = __shfl_xor(rg_k1 , 0x4 );
        rg_k3  = __shfl_xor(rg_k3 , 0x4 );
        rg_v1  = __shfl_xor(rg_v1 , 0x4 );
        rg_v3  = __shfl_xor(rg_v3 , 0x4 );
        rg_k2  = __shfl_xor(rg_k2 , 0x8 );
        rg_k3  = __shfl_xor(rg_k3 , 0x8 );
        rg_v2  = __shfl_xor(rg_v2 , 0x8 );
        rg_v3  = __shfl_xor(rg_v3 , 0x8 );
        if(lane_id&0x8 ) SWP(uintT, rg_k0 , rg_k2 , int, rg_v0 , rg_v2 );
        if(lane_id&0x8 ) SWP(uintT, rg_k1 , rg_k3 , int, rg_v1 , rg_v3 );
        rg_k2  = __shfl_xor(rg_k2 , 0x8 );
        rg_k3  = __shfl_xor(rg_k3 , 0x8 );
        rg_v2  = __shfl_xor(rg_v2 , 0x8 );
        rg_v3  = __shfl_xor(rg_v3 , 0x8 );
        rg_k1  = __shfl_xor(rg_k1 , 0x10);
        rg_k3  = __shfl_xor(rg_k3 , 0x10);
        rg_v1  = __shfl_xor(rg_v1 , 0x10);
        rg_v3  = __shfl_xor(rg_v3 , 0x10);
        if(lane_id&0x10) SWP(uintT, rg_k0 , rg_k1 , int, rg_v0 , rg_v1 );
        if(lane_id&0x10) SWP(uintT, rg_k2 , rg_k3 , int, rg_v2 , rg_v3 );
        rg_k1  = __shfl_xor(rg_k1 , 0x10);
        rg_k3  = __shfl_xor(rg_k3 , 0x10);
        rg_v1  = __shfl_xor(rg_v1 , 0x10);
        rg_v3  = __shfl_xor(rg_v3 , 0x10);
        int kk;
        int ss;
        int base = (lane_id/16)*16;
        kk = __shfl(k, 0 );
        ss = __shfl(seg_size, 0 );
        if((lane_id>>4)==0&&lane_id-base<ss) keyB[kk+lane_id-base] = rg_k0 ;
        kk = __shfl(k, 4 );
        ss = __shfl(seg_size, 4 );
        if((lane_id>>4)==1&&lane_id-base<ss) keyB[kk+lane_id-base] = rg_k0 ;
        kk = __shfl(k, 8 );
        ss = __shfl(seg_size, 8 );
        if((lane_id>>4)==0&&lane_id-base<ss) keyB[kk+lane_id-base] = rg_k2 ;
        kk = __shfl(k, 12);
        ss = __shfl(seg_size, 12);
        if((lane_id>>4)==1&&lane_id-base<ss) keyB[kk+lane_id-base] = rg_k2 ;
        kk = __shfl(k, 16);
        ss = __shfl(seg_size, 16);
        if((lane_id>>4)==0&&lane_id-base<ss) keyB[kk+lane_id-base] = rg_k1 ;
        kk = __shfl(k, 20);
        ss = __shfl(seg_size, 20);
        if((lane_id>>4)==1&&lane_id-base<ss) keyB[kk+lane_id-base] = rg_k1 ;
        kk = __shfl(k, 24);
        ss = __shfl(seg_size, 24);
        if((lane_id>>4)==0&&lane_id-base<ss) keyB[kk+lane_id-base] = rg_k3 ;
        kk = __shfl(k, 28);
        ss = __shfl(seg_size, 28);
        if((lane_id>>4)==1&&lane_id-base<ss) keyB[kk+lane_id-base] = rg_k3 ;
        kk = __shfl(k, 0 );
        ss = __shfl(seg_size, 0 );
        if((lane_id>>4)==0&lane_id-base<ss) valB[kk+lane_id-base] = val[kk+rg_v0 ];
        kk = __shfl(k, 4 );
        ss = __shfl(seg_size, 4 );
        if((lane_id>>4)==1&lane_id-base<ss) valB[kk+lane_id-base] = val[kk+rg_v0 ];
        kk = __shfl(k, 8 );
        ss = __shfl(seg_size, 8 );
        if((lane_id>>4)==0&lane_id-base<ss) valB[kk+lane_id-base] = val[kk+rg_v2 ];
        kk = __shfl(k, 12);
        ss = __shfl(seg_size, 12);
        if((lane_id>>4)==1&lane_id-base<ss) valB[kk+lane_id-base] = val[kk+rg_v2 ];
        kk = __shfl(k, 16);
        ss = __shfl(seg_size, 16);
        if((lane_id>>4)==0&lane_id-base<ss) valB[kk+lane_id-base] = val[kk+rg_v1 ];
        kk = __shfl(k, 20);
        ss = __shfl(seg_size, 20);
        if((lane_id>>4)==1&lane_id-base<ss) valB[kk+lane_id-base] = val[kk+rg_v1 ];
        kk = __shfl(k, 24);
        ss = __shfl(seg_size, 24);
        if((lane_id>>4)==0&lane_id-base<ss) valB[kk+lane_id-base] = val[kk+rg_v3 ];
        kk = __shfl(k, 28);
        ss = __shfl(seg_size, 28);
        if((lane_id>>4)==1&lane_id-base<ss) valB[kk+lane_id-base] = val[kk+rg_v3 ];
    } else if(bin_it < bin_size) {
        if((tid<<2)+0 <seg_size) keyB[k+(tid<<2)+0 ] = rg_k0 ;
        if((tid<<2)+1 <seg_size) keyB[k+(tid<<2)+1 ] = rg_k1 ;
        if((tid<<2)+2 <seg_size) keyB[k+(tid<<2)+2 ] = rg_k2 ;
        if((tid<<2)+3 <seg_size) keyB[k+(tid<<2)+3 ] = rg_k3 ;
        if((tid<<2)+0 <seg_size) valB[k+(tid<<2)+0 ] = val[k+rg_v0 ];
        if((tid<<2)+1 <seg_size) valB[k+(tid<<2)+1 ] = val[k+rg_v1 ];
        if((tid<<2)+2 <seg_size) valB[k+(tid<<2)+2 ] = val[k+rg_v2 ];
        if((tid<<2)+3 <seg_size) valB[k+(tid<<2)+3 ] = val[k+rg_v3 ];
    }
}
/* block tcf subwarp coalesced quiet real_kern */
/*   128   4       8      true  true      true */
template<class T>
__global__
void gen_bk128_wp8_tc4_r17_r32_strd( 
    uintT *key, T *val, uintT *keyB, T *valB, int n, int *segs, int *bin, int bin_size, int length) {

    const int gid = threadIdx.x + blockIdx.x * blockDim.x;
    const int bin_it = (gid>>3);
    const int tid = (threadIdx.x & 7);
    const int bit1 = (tid>>0)&0x1;
    const int bit2 = (tid>>1)&0x1;
    const int bit3 = (tid>>2)&0x1;
    uintT rg_k0 ;
    uintT rg_k1 ;
    uintT rg_k2 ;
    uintT rg_k3 ;
    int rg_v0 ;
    int rg_v1 ;
    int rg_v2 ;
    int rg_v3 ;
    int normalized_bin_size = (bin_size/4)*4;
    int k;
    int seg_size;
    if(bin_it < bin_size) {
        k = segs[bin[bin_it]];
        seg_size = ((bin[bin_it]==length-1)?n:segs[bin[bin_it]+1])-segs[bin[bin_it]];
        rg_k0  = (tid+0   <seg_size)?key[k+tid+0   ]:INT_MAX;
        rg_k1  = (tid+8   <seg_size)?key[k+tid+8   ]:INT_MAX;
        rg_k2  = (tid+16  <seg_size)?key[k+tid+16  ]:INT_MAX;
        rg_k3  = (tid+24  <seg_size)?key[k+tid+24  ]:INT_MAX;
        if(tid+0   <seg_size) rg_v0  = tid+0   ;
        if(tid+8   <seg_size) rg_v1  = tid+8   ;
        if(tid+16  <seg_size) rg_v2  = tid+16  ;
        if(tid+24  <seg_size) rg_v3  = tid+24  ;
        // sort 32 elements
        // exch_intxn: switch to exch_local()
        CMP_SWP(uintT,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
        CMP_SWP(uintT,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
        // exch_intxn: switch to exch_local()
        CMP_SWP(uintT,rg_k0 ,rg_k3 ,int,rg_v0 ,rg_v3 );
        CMP_SWP(uintT,rg_k1 ,rg_k2 ,int,rg_v1 ,rg_v2 );
        // exch_paral: switch to exch_local()
        CMP_SWP(uintT,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
        CMP_SWP(uintT,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
        // exch_intxn: generate exch_intxn()
        exch_intxn(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                   rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                   0x1,bit1);
        // exch_paral: switch to exch_local()
        CMP_SWP(uintT,rg_k0 ,rg_k2 ,int,rg_v0 ,rg_v2 );
        CMP_SWP(uintT,rg_k1 ,rg_k3 ,int,rg_v1 ,rg_v3 );
        // exch_paral: switch to exch_local()
        CMP_SWP(uintT,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
        CMP_SWP(uintT,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
        // exch_intxn: generate exch_intxn()
        exch_intxn(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                   rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                   0x3,bit2);
        // exch_paral: generate exch_paral()
        exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                   rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                   0x1,bit1);
        // exch_paral: switch to exch_local()
        CMP_SWP(uintT,rg_k0 ,rg_k2 ,int,rg_v0 ,rg_v2 );
        CMP_SWP(uintT,rg_k1 ,rg_k3 ,int,rg_v1 ,rg_v3 );
        // exch_paral: switch to exch_local()
        CMP_SWP(uintT,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
        CMP_SWP(uintT,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
        // exch_intxn: generate exch_intxn()
        exch_intxn(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                   rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                   0x7,bit3);
        // exch_paral: generate exch_paral()
        exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                   rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                   0x2,bit2);
        // exch_paral: generate exch_paral()
        exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                   rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                   0x1,bit1);
        // exch_paral: switch to exch_local()
        CMP_SWP(uintT,rg_k0 ,rg_k2 ,int,rg_v0 ,rg_v2 );
        CMP_SWP(uintT,rg_k1 ,rg_k3 ,int,rg_v1 ,rg_v3 );
        // exch_paral: switch to exch_local()
        CMP_SWP(uintT,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
        CMP_SWP(uintT,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
    }

    if(bin_it < normalized_bin_size) {
        // store back the results
        int lane_id = threadIdx.x & 31;
        rg_k1  = __shfl_xor(rg_k1 , 0x1 );
        rg_k3  = __shfl_xor(rg_k3 , 0x1 );
        rg_v1  = __shfl_xor(rg_v1 , 0x1 );
        rg_v3  = __shfl_xor(rg_v3 , 0x1 );
        if(lane_id&0x1 ) SWP(uintT, rg_k0 , rg_k1 , int, rg_v0 , rg_v1 );
        if(lane_id&0x1 ) SWP(uintT, rg_k2 , rg_k3 , int, rg_v2 , rg_v3 );
        rg_k1  = __shfl_xor(rg_k1 , 0x1 );
        rg_k3  = __shfl_xor(rg_k3 , 0x1 );
        rg_v1  = __shfl_xor(rg_v1 , 0x1 );
        rg_v3  = __shfl_xor(rg_v3 , 0x1 );
        rg_k2  = __shfl_xor(rg_k2 , 0x2 );
        rg_k3  = __shfl_xor(rg_k3 , 0x2 );
        rg_v2  = __shfl_xor(rg_v2 , 0x2 );
        rg_v3  = __shfl_xor(rg_v3 , 0x2 );
        if(lane_id&0x2 ) SWP(uintT, rg_k0 , rg_k2 , int, rg_v0 , rg_v2 );
        if(lane_id&0x2 ) SWP(uintT, rg_k1 , rg_k3 , int, rg_v1 , rg_v3 );
        rg_k2  = __shfl_xor(rg_k2 , 0x2 );
        rg_k3  = __shfl_xor(rg_k3 , 0x2 );
        rg_v2  = __shfl_xor(rg_v2 , 0x2 );
        rg_v3  = __shfl_xor(rg_v3 , 0x2 );
        rg_k1  = __shfl_xor(rg_k1 , 0x4 );
        rg_k3  = __shfl_xor(rg_k3 , 0x4 );
        rg_v1  = __shfl_xor(rg_v1 , 0x4 );
        rg_v3  = __shfl_xor(rg_v3 , 0x4 );
        if(lane_id&0x4 ) SWP(uintT, rg_k0 , rg_k1 , int, rg_v0 , rg_v1 );
        if(lane_id&0x4 ) SWP(uintT, rg_k2 , rg_k3 , int, rg_v2 , rg_v3 );
        rg_k1  = __shfl_xor(rg_k1 , 0x4 );
        rg_k3  = __shfl_xor(rg_k3 , 0x4 );
        rg_v1  = __shfl_xor(rg_v1 , 0x4 );
        rg_v3  = __shfl_xor(rg_v3 , 0x4 );
        rg_k2  = __shfl_xor(rg_k2 , 0x8 );
        rg_k3  = __shfl_xor(rg_k3 , 0x8 );
        rg_v2  = __shfl_xor(rg_v2 , 0x8 );
        rg_v3  = __shfl_xor(rg_v3 , 0x8 );
        if(lane_id&0x8 ) SWP(uintT, rg_k0 , rg_k2 , int, rg_v0 , rg_v2 );
        if(lane_id&0x8 ) SWP(uintT, rg_k1 , rg_k3 , int, rg_v1 , rg_v3 );
        rg_k2  = __shfl_xor(rg_k2 , 0x8 );
        rg_k3  = __shfl_xor(rg_k3 , 0x8 );
        rg_v2  = __shfl_xor(rg_v2 , 0x8 );
        rg_v3  = __shfl_xor(rg_v3 , 0x8 );
        rg_k1  = __shfl_xor(rg_k1 , 0x10);
        rg_k3  = __shfl_xor(rg_k3 , 0x10);
        rg_v1  = __shfl_xor(rg_v1 , 0x10);
        rg_v3  = __shfl_xor(rg_v3 , 0x10);
        if(lane_id&0x10) SWP(uintT, rg_k0 , rg_k1 , int, rg_v0 , rg_v1 );
        if(lane_id&0x10) SWP(uintT, rg_k2 , rg_k3 , int, rg_v2 , rg_v3 );
        rg_k1  = __shfl_xor(rg_k1 , 0x10);
        rg_k3  = __shfl_xor(rg_k3 , 0x10);
        rg_v1  = __shfl_xor(rg_v1 , 0x10);
        rg_v3  = __shfl_xor(rg_v3 , 0x10);
        int kk;
        int ss;
        kk = __shfl(k, 0 );
        ss = __shfl(seg_size, 0 );
        if(lane_id+0  <ss) keyB[kk+lane_id+0  ] = rg_k0 ;
        kk = __shfl(k, 8 );
        ss = __shfl(seg_size, 8 );
        if(lane_id+0  <ss) keyB[kk+lane_id+0  ] = rg_k2 ;
        kk = __shfl(k, 16);
        ss = __shfl(seg_size, 16);
        if(lane_id+0  <ss) keyB[kk+lane_id+0  ] = rg_k1 ;
        kk = __shfl(k, 24);
        ss = __shfl(seg_size, 24);
        if(lane_id+0  <ss) keyB[kk+lane_id+0  ] = rg_k3 ;
        kk = __shfl(k, 0 );
        ss = __shfl(seg_size, 0 );
        if(lane_id+0  <ss) valB[kk+lane_id+0  ] = val[kk+rg_v0 ];
        kk = __shfl(k, 8 );
        ss = __shfl(seg_size, 8 );
        if(lane_id+0  <ss) valB[kk+lane_id+0  ] = val[kk+rg_v2 ];
        kk = __shfl(k, 16);
        ss = __shfl(seg_size, 16);
        if(lane_id+0  <ss) valB[kk+lane_id+0  ] = val[kk+rg_v1 ];
        kk = __shfl(k, 24);
        ss = __shfl(seg_size, 24);
        if(lane_id+0  <ss) valB[kk+lane_id+0  ] = val[kk+rg_v3 ];
    } else if(bin_it < bin_size) {
        if((tid<<2)+0 <seg_size) keyB[k+(tid<<2)+0 ] = rg_k0 ;
        if((tid<<2)+1 <seg_size) keyB[k+(tid<<2)+1 ] = rg_k1 ;
        if((tid<<2)+2 <seg_size) keyB[k+(tid<<2)+2 ] = rg_k2 ;
        if((tid<<2)+3 <seg_size) keyB[k+(tid<<2)+3 ] = rg_k3 ;
        if((tid<<2)+0 <seg_size) valB[k+(tid<<2)+0 ] = val[k+rg_v0 ];
        if((tid<<2)+1 <seg_size) valB[k+(tid<<2)+1 ] = val[k+rg_v1 ];
        if((tid<<2)+2 <seg_size) valB[k+(tid<<2)+2 ] = val[k+rg_v2 ];
        if((tid<<2)+3 <seg_size) valB[k+(tid<<2)+3 ] = val[k+rg_v3 ];
    }
}
/* block tcf subwarp coalesced quiet real_kern */
/*   128   4      16      true  true      true */
template<class T>
__global__
void gen_bk128_wp16_tc4_r33_r64_strd( 
    uintT *key, T *val, uintT *keyB, T *valB, int n, int *segs, int *bin, int bin_size, int length) {

    const int gid = threadIdx.x + blockIdx.x * blockDim.x;
    const int bin_it = (gid>>4);
    const int tid = (threadIdx.x & 15);
    const int bit1 = (tid>>0)&0x1;
    const int bit2 = (tid>>1)&0x1;
    const int bit3 = (tid>>2)&0x1;
    const int bit4 = (tid>>3)&0x1;
    uintT rg_k0 ;
    uintT rg_k1 ;
    uintT rg_k2 ;
    uintT rg_k3 ;
    int rg_v0 ;
    int rg_v1 ;
    int rg_v2 ;
    int rg_v3 ;
    int normalized_bin_size = (bin_size/2)*2;
    int k;
    int seg_size;
    if(bin_it < bin_size) {
        k = segs[bin[bin_it]];
        seg_size = ((bin[bin_it]==length-1)?n:segs[bin[bin_it]+1])-segs[bin[bin_it]];
        rg_k0  = (tid+0   <seg_size)?key[k+tid+0   ]:INT_MAX;
        rg_k1  = (tid+16  <seg_size)?key[k+tid+16  ]:INT_MAX;
        rg_k2  = (tid+32  <seg_size)?key[k+tid+32  ]:INT_MAX;
        rg_k3  = (tid+48  <seg_size)?key[k+tid+48  ]:INT_MAX;
        if(tid+0   <seg_size) rg_v0  = tid+0   ;
        if(tid+16  <seg_size) rg_v1  = tid+16  ;
        if(tid+32  <seg_size) rg_v2  = tid+32  ;
        if(tid+48  <seg_size) rg_v3  = tid+48  ;
        // sort 64 elements
        // exch_intxn: switch to exch_local()
        CMP_SWP(uintT,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
        CMP_SWP(uintT,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
        // exch_intxn: switch to exch_local()
        CMP_SWP(uintT,rg_k0 ,rg_k3 ,int,rg_v0 ,rg_v3 );
        CMP_SWP(uintT,rg_k1 ,rg_k2 ,int,rg_v1 ,rg_v2 );
        // exch_paral: switch to exch_local()
        CMP_SWP(uintT,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
        CMP_SWP(uintT,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
        // exch_intxn: generate exch_intxn()
        exch_intxn(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                   rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                   0x1,bit1);
        // exch_paral: switch to exch_local()
        CMP_SWP(uintT,rg_k0 ,rg_k2 ,int,rg_v0 ,rg_v2 );
        CMP_SWP(uintT,rg_k1 ,rg_k3 ,int,rg_v1 ,rg_v3 );
        // exch_paral: switch to exch_local()
        CMP_SWP(uintT,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
        CMP_SWP(uintT,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
        // exch_intxn: generate exch_intxn()
        exch_intxn(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                   rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                   0x3,bit2);
        // exch_paral: generate exch_paral()
        exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                   rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                   0x1,bit1);
        // exch_paral: switch to exch_local()
        CMP_SWP(uintT,rg_k0 ,rg_k2 ,int,rg_v0 ,rg_v2 );
        CMP_SWP(uintT,rg_k1 ,rg_k3 ,int,rg_v1 ,rg_v3 );
        // exch_paral: switch to exch_local()
        CMP_SWP(uintT,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
        CMP_SWP(uintT,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
        // exch_intxn: generate exch_intxn()
        exch_intxn(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                   rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                   0x7,bit3);
        // exch_paral: generate exch_paral()
        exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                   rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                   0x2,bit2);
        // exch_paral: generate exch_paral()
        exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                   rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                   0x1,bit1);
        // exch_paral: switch to exch_local()
        CMP_SWP(uintT,rg_k0 ,rg_k2 ,int,rg_v0 ,rg_v2 );
        CMP_SWP(uintT,rg_k1 ,rg_k3 ,int,rg_v1 ,rg_v3 );
        // exch_paral: switch to exch_local()
        CMP_SWP(uintT,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
        CMP_SWP(uintT,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
        // exch_intxn: generate exch_intxn()
        exch_intxn(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                   rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                   0xf,bit4);
        // exch_paral: generate exch_paral()
        exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                   rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                   0x4,bit3);
        // exch_paral: generate exch_paral()
        exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                   rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                   0x2,bit2);
        // exch_paral: generate exch_paral()
        exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                   rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                   0x1,bit1);
        // exch_paral: switch to exch_local()
        CMP_SWP(uintT,rg_k0 ,rg_k2 ,int,rg_v0 ,rg_v2 );
        CMP_SWP(uintT,rg_k1 ,rg_k3 ,int,rg_v1 ,rg_v3 );
        // exch_paral: switch to exch_local()
        CMP_SWP(uintT,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
        CMP_SWP(uintT,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
    }

    if(bin_it < normalized_bin_size) {
        // store back the results
        int lane_id = threadIdx.x & 31;
        rg_k1  = __shfl_xor(rg_k1 , 0x1 );
        rg_k3  = __shfl_xor(rg_k3 , 0x1 );
        rg_v1  = __shfl_xor(rg_v1 , 0x1 );
        rg_v3  = __shfl_xor(rg_v3 , 0x1 );
        if(lane_id&0x1 ) SWP(uintT, rg_k0 , rg_k1 , int, rg_v0 , rg_v1 );
        if(lane_id&0x1 ) SWP(uintT, rg_k2 , rg_k3 , int, rg_v2 , rg_v3 );
        rg_k1  = __shfl_xor(rg_k1 , 0x1 );
        rg_k3  = __shfl_xor(rg_k3 , 0x1 );
        rg_v1  = __shfl_xor(rg_v1 , 0x1 );
        rg_v3  = __shfl_xor(rg_v3 , 0x1 );
        rg_k2  = __shfl_xor(rg_k2 , 0x2 );
        rg_k3  = __shfl_xor(rg_k3 , 0x2 );
        rg_v2  = __shfl_xor(rg_v2 , 0x2 );
        rg_v3  = __shfl_xor(rg_v3 , 0x2 );
        if(lane_id&0x2 ) SWP(uintT, rg_k0 , rg_k2 , int, rg_v0 , rg_v2 );
        if(lane_id&0x2 ) SWP(uintT, rg_k1 , rg_k3 , int, rg_v1 , rg_v3 );
        rg_k2  = __shfl_xor(rg_k2 , 0x2 );
        rg_k3  = __shfl_xor(rg_k3 , 0x2 );
        rg_v2  = __shfl_xor(rg_v2 , 0x2 );
        rg_v3  = __shfl_xor(rg_v3 , 0x2 );
        rg_k1  = __shfl_xor(rg_k1 , 0x4 );
        rg_k3  = __shfl_xor(rg_k3 , 0x4 );
        rg_v1  = __shfl_xor(rg_v1 , 0x4 );
        rg_v3  = __shfl_xor(rg_v3 , 0x4 );
        if(lane_id&0x4 ) SWP(uintT, rg_k0 , rg_k1 , int, rg_v0 , rg_v1 );
        if(lane_id&0x4 ) SWP(uintT, rg_k2 , rg_k3 , int, rg_v2 , rg_v3 );
        rg_k1  = __shfl_xor(rg_k1 , 0x4 );
        rg_k3  = __shfl_xor(rg_k3 , 0x4 );
        rg_v1  = __shfl_xor(rg_v1 , 0x4 );
        rg_v3  = __shfl_xor(rg_v3 , 0x4 );
        rg_k2  = __shfl_xor(rg_k2 , 0x8 );
        rg_k3  = __shfl_xor(rg_k3 , 0x8 );
        rg_v2  = __shfl_xor(rg_v2 , 0x8 );
        rg_v3  = __shfl_xor(rg_v3 , 0x8 );
        if(lane_id&0x8 ) SWP(uintT, rg_k0 , rg_k2 , int, rg_v0 , rg_v2 );
        if(lane_id&0x8 ) SWP(uintT, rg_k1 , rg_k3 , int, rg_v1 , rg_v3 );
        rg_k2  = __shfl_xor(rg_k2 , 0x8 );
        rg_k3  = __shfl_xor(rg_k3 , 0x8 );
        rg_v2  = __shfl_xor(rg_v2 , 0x8 );
        rg_v3  = __shfl_xor(rg_v3 , 0x8 );
        rg_k1  = __shfl_xor(rg_k1 , 0x10);
        rg_k3  = __shfl_xor(rg_k3 , 0x10);
        rg_v1  = __shfl_xor(rg_v1 , 0x10);
        rg_v3  = __shfl_xor(rg_v3 , 0x10);
        if(lane_id&0x10) SWP(uintT, rg_k0 , rg_k1 , int, rg_v0 , rg_v1 );
        if(lane_id&0x10) SWP(uintT, rg_k2 , rg_k3 , int, rg_v2 , rg_v3 );
        rg_k1  = __shfl_xor(rg_k1 , 0x10);
        rg_k3  = __shfl_xor(rg_k3 , 0x10);
        rg_v1  = __shfl_xor(rg_v1 , 0x10);
        rg_v3  = __shfl_xor(rg_v3 , 0x10);
        int kk;
        int ss;
        kk = __shfl(k, 0 );
        ss = __shfl(seg_size, 0 );
        if(lane_id+0  <ss) keyB[kk+lane_id+0  ] = rg_k0 ;
        if(lane_id+32 <ss) keyB[kk+lane_id+32 ] = rg_k2 ;
        kk = __shfl(k, 16);
        ss = __shfl(seg_size, 16);
        if(lane_id+0  <ss) keyB[kk+lane_id+0  ] = rg_k1 ;
        if(lane_id+32 <ss) keyB[kk+lane_id+32 ] = rg_k3 ;
        kk = __shfl(k, 0 );
        ss = __shfl(seg_size, 0 );
        if(lane_id+0  <ss) valB[kk+lane_id+0  ] = val[kk+rg_v0 ];
        if(lane_id+32 <ss) valB[kk+lane_id+32 ] = val[kk+rg_v2 ];
        kk = __shfl(k, 16);
        ss = __shfl(seg_size, 16);
        if(lane_id+0  <ss) valB[kk+lane_id+0  ] = val[kk+rg_v1 ];
        if(lane_id+32 <ss) valB[kk+lane_id+32 ] = val[kk+rg_v3 ];
    } else if(bin_it < bin_size) {
        if((tid<<2)+0 <seg_size) keyB[k+(tid<<2)+0 ] = rg_k0 ;
        if((tid<<2)+1 <seg_size) keyB[k+(tid<<2)+1 ] = rg_k1 ;
        if((tid<<2)+2 <seg_size) keyB[k+(tid<<2)+2 ] = rg_k2 ;
        if((tid<<2)+3 <seg_size) keyB[k+(tid<<2)+3 ] = rg_k3 ;
        if((tid<<2)+0 <seg_size) valB[k+(tid<<2)+0 ] = val[k+rg_v0 ];
        if((tid<<2)+1 <seg_size) valB[k+(tid<<2)+1 ] = val[k+rg_v1 ];
        if((tid<<2)+2 <seg_size) valB[k+(tid<<2)+2 ] = val[k+rg_v2 ];
        if((tid<<2)+3 <seg_size) valB[k+(tid<<2)+3 ] = val[k+rg_v3 ];
    }
}
/* block tcf subwarp coalesced quiet real_kern */
/*   256  16       8      true  true      true */
template<class T>
__global__
void gen_bk256_wp8_tc16_r65_r128_strd( 
    uintT *key, T *val, uintT *keyB, T *valB, int n, int *segs, int *bin, int bin_size, int length) {

    const int gid = threadIdx.x + blockIdx.x * blockDim.x;
    const int bin_it = (gid>>3);
    const int tid = (threadIdx.x & 7);
    const int bit1 = (tid>>0)&0x1;
    const int bit2 = (tid>>1)&0x1;
    const int bit3 = (tid>>2)&0x1;
    uintT rg_k0 ;
    uintT rg_k1 ;
    uintT rg_k2 ;
    uintT rg_k3 ;
    uintT rg_k4 ;
    uintT rg_k5 ;
    uintT rg_k6 ;
    uintT rg_k7 ;
    uintT rg_k8 ;
    uintT rg_k9 ;
    uintT rg_k10;
    uintT rg_k11;
    uintT rg_k12;
    uintT rg_k13;
    uintT rg_k14;
    uintT rg_k15;
    int rg_v0 ;
    int rg_v1 ;
    int rg_v2 ;
    int rg_v3 ;
    int rg_v4 ;
    int rg_v5 ;
    int rg_v6 ;
    int rg_v7 ;
    int rg_v8 ;
    int rg_v9 ;
    int rg_v10;
    int rg_v11;
    int rg_v12;
    int rg_v13;
    int rg_v14;
    int rg_v15;
    int normalized_bin_size = (bin_size/4)*4;
    int k;
    int seg_size;
    if(bin_it < bin_size) {
        k = segs[bin[bin_it]];
        seg_size = ((bin[bin_it]==length-1)?n:segs[bin[bin_it]+1])-segs[bin[bin_it]];
        rg_k0  = (tid+0   <seg_size)?key[k+tid+0   ]:INT_MAX;
        rg_k1  = (tid+8   <seg_size)?key[k+tid+8   ]:INT_MAX;
        rg_k2  = (tid+16  <seg_size)?key[k+tid+16  ]:INT_MAX;
        rg_k3  = (tid+24  <seg_size)?key[k+tid+24  ]:INT_MAX;
        rg_k4  = (tid+32  <seg_size)?key[k+tid+32  ]:INT_MAX;
        rg_k5  = (tid+40  <seg_size)?key[k+tid+40  ]:INT_MAX;
        rg_k6  = (tid+48  <seg_size)?key[k+tid+48  ]:INT_MAX;
        rg_k7  = (tid+56  <seg_size)?key[k+tid+56  ]:INT_MAX;
        rg_k8  = (tid+64  <seg_size)?key[k+tid+64  ]:INT_MAX;
        rg_k9  = (tid+72  <seg_size)?key[k+tid+72  ]:INT_MAX;
        rg_k10 = (tid+80  <seg_size)?key[k+tid+80  ]:INT_MAX;
        rg_k11 = (tid+88  <seg_size)?key[k+tid+88  ]:INT_MAX;
        rg_k12 = (tid+96  <seg_size)?key[k+tid+96  ]:INT_MAX;
        rg_k13 = (tid+104 <seg_size)?key[k+tid+104 ]:INT_MAX;
        rg_k14 = (tid+112 <seg_size)?key[k+tid+112 ]:INT_MAX;
        rg_k15 = (tid+120 <seg_size)?key[k+tid+120 ]:INT_MAX;
        if(tid+0   <seg_size) rg_v0  = tid+0   ;
        if(tid+8   <seg_size) rg_v1  = tid+8   ;
        if(tid+16  <seg_size) rg_v2  = tid+16  ;
        if(tid+24  <seg_size) rg_v3  = tid+24  ;
        if(tid+32  <seg_size) rg_v4  = tid+32  ;
        if(tid+40  <seg_size) rg_v5  = tid+40  ;
        if(tid+48  <seg_size) rg_v6  = tid+48  ;
        if(tid+56  <seg_size) rg_v7  = tid+56  ;
        if(tid+64  <seg_size) rg_v8  = tid+64  ;
        if(tid+72  <seg_size) rg_v9  = tid+72  ;
        if(tid+80  <seg_size) rg_v10 = tid+80  ;
        if(tid+88  <seg_size) rg_v11 = tid+88  ;
        if(tid+96  <seg_size) rg_v12 = tid+96  ;
        if(tid+104 <seg_size) rg_v13 = tid+104 ;
        if(tid+112 <seg_size) rg_v14 = tid+112 ;
        if(tid+120 <seg_size) rg_v15 = tid+120 ;
        // sort 128 elements
        // exch_intxn: switch to exch_local()
        CMP_SWP(uintT,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
        CMP_SWP(uintT,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
        CMP_SWP(uintT,rg_k4 ,rg_k5 ,int,rg_v4 ,rg_v5 );
        CMP_SWP(uintT,rg_k6 ,rg_k7 ,int,rg_v6 ,rg_v7 );
        CMP_SWP(uintT,rg_k8 ,rg_k9 ,int,rg_v8 ,rg_v9 );
        CMP_SWP(uintT,rg_k10,rg_k11,int,rg_v10,rg_v11);
        CMP_SWP(uintT,rg_k12,rg_k13,int,rg_v12,rg_v13);
        CMP_SWP(uintT,rg_k14,rg_k15,int,rg_v14,rg_v15);
        // exch_intxn: switch to exch_local()
        CMP_SWP(uintT,rg_k0 ,rg_k3 ,int,rg_v0 ,rg_v3 );
        CMP_SWP(uintT,rg_k1 ,rg_k2 ,int,rg_v1 ,rg_v2 );
        CMP_SWP(uintT,rg_k4 ,rg_k7 ,int,rg_v4 ,rg_v7 );
        CMP_SWP(uintT,rg_k5 ,rg_k6 ,int,rg_v5 ,rg_v6 );
        CMP_SWP(uintT,rg_k8 ,rg_k11,int,rg_v8 ,rg_v11);
        CMP_SWP(uintT,rg_k9 ,rg_k10,int,rg_v9 ,rg_v10);
        CMP_SWP(uintT,rg_k12,rg_k15,int,rg_v12,rg_v15);
        CMP_SWP(uintT,rg_k13,rg_k14,int,rg_v13,rg_v14);
        // exch_paral: switch to exch_local()
        CMP_SWP(uintT,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
        CMP_SWP(uintT,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
        CMP_SWP(uintT,rg_k4 ,rg_k5 ,int,rg_v4 ,rg_v5 );
        CMP_SWP(uintT,rg_k6 ,rg_k7 ,int,rg_v6 ,rg_v7 );
        CMP_SWP(uintT,rg_k8 ,rg_k9 ,int,rg_v8 ,rg_v9 );
        CMP_SWP(uintT,rg_k10,rg_k11,int,rg_v10,rg_v11);
        CMP_SWP(uintT,rg_k12,rg_k13,int,rg_v12,rg_v13);
        CMP_SWP(uintT,rg_k14,rg_k15,int,rg_v14,rg_v15);
        // exch_intxn: switch to exch_local()
        CMP_SWP(uintT,rg_k0 ,rg_k7 ,int,rg_v0 ,rg_v7 );
        CMP_SWP(uintT,rg_k1 ,rg_k6 ,int,rg_v1 ,rg_v6 );
        CMP_SWP(uintT,rg_k2 ,rg_k5 ,int,rg_v2 ,rg_v5 );
        CMP_SWP(uintT,rg_k3 ,rg_k4 ,int,rg_v3 ,rg_v4 );
        CMP_SWP(uintT,rg_k8 ,rg_k15,int,rg_v8 ,rg_v15);
        CMP_SWP(uintT,rg_k9 ,rg_k14,int,rg_v9 ,rg_v14);
        CMP_SWP(uintT,rg_k10,rg_k13,int,rg_v10,rg_v13);
        CMP_SWP(uintT,rg_k11,rg_k12,int,rg_v11,rg_v12);
        // exch_paral: switch to exch_local()
        CMP_SWP(uintT,rg_k0 ,rg_k2 ,int,rg_v0 ,rg_v2 );
        CMP_SWP(uintT,rg_k1 ,rg_k3 ,int,rg_v1 ,rg_v3 );
        CMP_SWP(uintT,rg_k4 ,rg_k6 ,int,rg_v4 ,rg_v6 );
        CMP_SWP(uintT,rg_k5 ,rg_k7 ,int,rg_v5 ,rg_v7 );
        CMP_SWP(uintT,rg_k8 ,rg_k10,int,rg_v8 ,rg_v10);
        CMP_SWP(uintT,rg_k9 ,rg_k11,int,rg_v9 ,rg_v11);
        CMP_SWP(uintT,rg_k12,rg_k14,int,rg_v12,rg_v14);
        CMP_SWP(uintT,rg_k13,rg_k15,int,rg_v13,rg_v15);
        // exch_paral: switch to exch_local()
        CMP_SWP(uintT,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
        CMP_SWP(uintT,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
        CMP_SWP(uintT,rg_k4 ,rg_k5 ,int,rg_v4 ,rg_v5 );
        CMP_SWP(uintT,rg_k6 ,rg_k7 ,int,rg_v6 ,rg_v7 );
        CMP_SWP(uintT,rg_k8 ,rg_k9 ,int,rg_v8 ,rg_v9 );
        CMP_SWP(uintT,rg_k10,rg_k11,int,rg_v10,rg_v11);
        CMP_SWP(uintT,rg_k12,rg_k13,int,rg_v12,rg_v13);
        CMP_SWP(uintT,rg_k14,rg_k15,int,rg_v14,rg_v15);
        // exch_intxn: switch to exch_local()
        CMP_SWP(uintT,rg_k0 ,rg_k15,int,rg_v0 ,rg_v15);
        CMP_SWP(uintT,rg_k1 ,rg_k14,int,rg_v1 ,rg_v14);
        CMP_SWP(uintT,rg_k2 ,rg_k13,int,rg_v2 ,rg_v13);
        CMP_SWP(uintT,rg_k3 ,rg_k12,int,rg_v3 ,rg_v12);
        CMP_SWP(uintT,rg_k4 ,rg_k11,int,rg_v4 ,rg_v11);
        CMP_SWP(uintT,rg_k5 ,rg_k10,int,rg_v5 ,rg_v10);
        CMP_SWP(uintT,rg_k6 ,rg_k9 ,int,rg_v6 ,rg_v9 );
        CMP_SWP(uintT,rg_k7 ,rg_k8 ,int,rg_v7 ,rg_v8 );
        // exch_paral: switch to exch_local()
        CMP_SWP(uintT,rg_k0 ,rg_k4 ,int,rg_v0 ,rg_v4 );
        CMP_SWP(uintT,rg_k1 ,rg_k5 ,int,rg_v1 ,rg_v5 );
        CMP_SWP(uintT,rg_k2 ,rg_k6 ,int,rg_v2 ,rg_v6 );
        CMP_SWP(uintT,rg_k3 ,rg_k7 ,int,rg_v3 ,rg_v7 );
        CMP_SWP(uintT,rg_k8 ,rg_k12,int,rg_v8 ,rg_v12);
        CMP_SWP(uintT,rg_k9 ,rg_k13,int,rg_v9 ,rg_v13);
        CMP_SWP(uintT,rg_k10,rg_k14,int,rg_v10,rg_v14);
        CMP_SWP(uintT,rg_k11,rg_k15,int,rg_v11,rg_v15);
        // exch_paral: switch to exch_local()
        CMP_SWP(uintT,rg_k0 ,rg_k2 ,int,rg_v0 ,rg_v2 );
        CMP_SWP(uintT,rg_k1 ,rg_k3 ,int,rg_v1 ,rg_v3 );
        CMP_SWP(uintT,rg_k4 ,rg_k6 ,int,rg_v4 ,rg_v6 );
        CMP_SWP(uintT,rg_k5 ,rg_k7 ,int,rg_v5 ,rg_v7 );
        CMP_SWP(uintT,rg_k8 ,rg_k10,int,rg_v8 ,rg_v10);
        CMP_SWP(uintT,rg_k9 ,rg_k11,int,rg_v9 ,rg_v11);
        CMP_SWP(uintT,rg_k12,rg_k14,int,rg_v12,rg_v14);
        CMP_SWP(uintT,rg_k13,rg_k15,int,rg_v13,rg_v15);
        // exch_paral: switch to exch_local()
        CMP_SWP(uintT,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
        CMP_SWP(uintT,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
        CMP_SWP(uintT,rg_k4 ,rg_k5 ,int,rg_v4 ,rg_v5 );
        CMP_SWP(uintT,rg_k6 ,rg_k7 ,int,rg_v6 ,rg_v7 );
        CMP_SWP(uintT,rg_k8 ,rg_k9 ,int,rg_v8 ,rg_v9 );
        CMP_SWP(uintT,rg_k10,rg_k11,int,rg_v10,rg_v11);
        CMP_SWP(uintT,rg_k12,rg_k13,int,rg_v12,rg_v13);
        CMP_SWP(uintT,rg_k14,rg_k15,int,rg_v14,rg_v15);
        // exch_intxn: generate exch_intxn()
        exch_intxn(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,rg_k4 ,rg_k5 ,rg_k6 ,rg_k7 ,
                   rg_k8 ,rg_k9 ,rg_k10,rg_k11,rg_k12,rg_k13,rg_k14,rg_k15,
                   rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,rg_v4 ,rg_v5 ,rg_v6 ,rg_v7 ,
                   rg_v8 ,rg_v9 ,rg_v10,rg_v11,rg_v12,rg_v13,rg_v14,rg_v15,
                   0x1,bit1);
        // exch_paral: switch to exch_local()
        CMP_SWP(uintT,rg_k0 ,rg_k8 ,int,rg_v0 ,rg_v8 );
        CMP_SWP(uintT,rg_k1 ,rg_k9 ,int,rg_v1 ,rg_v9 );
        CMP_SWP(uintT,rg_k2 ,rg_k10,int,rg_v2 ,rg_v10);
        CMP_SWP(uintT,rg_k3 ,rg_k11,int,rg_v3 ,rg_v11);
        CMP_SWP(uintT,rg_k4 ,rg_k12,int,rg_v4 ,rg_v12);
        CMP_SWP(uintT,rg_k5 ,rg_k13,int,rg_v5 ,rg_v13);
        CMP_SWP(uintT,rg_k6 ,rg_k14,int,rg_v6 ,rg_v14);
        CMP_SWP(uintT,rg_k7 ,rg_k15,int,rg_v7 ,rg_v15);
        // exch_paral: switch to exch_local()
        CMP_SWP(uintT,rg_k0 ,rg_k4 ,int,rg_v0 ,rg_v4 );
        CMP_SWP(uintT,rg_k1 ,rg_k5 ,int,rg_v1 ,rg_v5 );
        CMP_SWP(uintT,rg_k2 ,rg_k6 ,int,rg_v2 ,rg_v6 );
        CMP_SWP(uintT,rg_k3 ,rg_k7 ,int,rg_v3 ,rg_v7 );
        CMP_SWP(uintT,rg_k8 ,rg_k12,int,rg_v8 ,rg_v12);
        CMP_SWP(uintT,rg_k9 ,rg_k13,int,rg_v9 ,rg_v13);
        CMP_SWP(uintT,rg_k10,rg_k14,int,rg_v10,rg_v14);
        CMP_SWP(uintT,rg_k11,rg_k15,int,rg_v11,rg_v15);
        // exch_paral: switch to exch_local()
        CMP_SWP(uintT,rg_k0 ,rg_k2 ,int,rg_v0 ,rg_v2 );
        CMP_SWP(uintT,rg_k1 ,rg_k3 ,int,rg_v1 ,rg_v3 );
        CMP_SWP(uintT,rg_k4 ,rg_k6 ,int,rg_v4 ,rg_v6 );
        CMP_SWP(uintT,rg_k5 ,rg_k7 ,int,rg_v5 ,rg_v7 );
        CMP_SWP(uintT,rg_k8 ,rg_k10,int,rg_v8 ,rg_v10);
        CMP_SWP(uintT,rg_k9 ,rg_k11,int,rg_v9 ,rg_v11);
        CMP_SWP(uintT,rg_k12,rg_k14,int,rg_v12,rg_v14);
        CMP_SWP(uintT,rg_k13,rg_k15,int,rg_v13,rg_v15);
        // exch_paral: switch to exch_local()
        CMP_SWP(uintT,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
        CMP_SWP(uintT,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
        CMP_SWP(uintT,rg_k4 ,rg_k5 ,int,rg_v4 ,rg_v5 );
        CMP_SWP(uintT,rg_k6 ,rg_k7 ,int,rg_v6 ,rg_v7 );
        CMP_SWP(uintT,rg_k8 ,rg_k9 ,int,rg_v8 ,rg_v9 );
        CMP_SWP(uintT,rg_k10,rg_k11,int,rg_v10,rg_v11);
        CMP_SWP(uintT,rg_k12,rg_k13,int,rg_v12,rg_v13);
        CMP_SWP(uintT,rg_k14,rg_k15,int,rg_v14,rg_v15);
        // exch_intxn: generate exch_intxn()
        exch_intxn(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,rg_k4 ,rg_k5 ,rg_k6 ,rg_k7 ,
                   rg_k8 ,rg_k9 ,rg_k10,rg_k11,rg_k12,rg_k13,rg_k14,rg_k15,
                   rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,rg_v4 ,rg_v5 ,rg_v6 ,rg_v7 ,
                   rg_v8 ,rg_v9 ,rg_v10,rg_v11,rg_v12,rg_v13,rg_v14,rg_v15,
                   0x3,bit2);
        // exch_paral: generate exch_paral()
        exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,rg_k4 ,rg_k5 ,rg_k6 ,rg_k7 ,
                   rg_k8 ,rg_k9 ,rg_k10,rg_k11,rg_k12,rg_k13,rg_k14,rg_k15,
                   rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,rg_v4 ,rg_v5 ,rg_v6 ,rg_v7 ,
                   rg_v8 ,rg_v9 ,rg_v10,rg_v11,rg_v12,rg_v13,rg_v14,rg_v15,
                   0x1,bit1);
        // exch_paral: switch to exch_local()
        CMP_SWP(uintT,rg_k0 ,rg_k8 ,int,rg_v0 ,rg_v8 );
        CMP_SWP(uintT,rg_k1 ,rg_k9 ,int,rg_v1 ,rg_v9 );
        CMP_SWP(uintT,rg_k2 ,rg_k10,int,rg_v2 ,rg_v10);
        CMP_SWP(uintT,rg_k3 ,rg_k11,int,rg_v3 ,rg_v11);
        CMP_SWP(uintT,rg_k4 ,rg_k12,int,rg_v4 ,rg_v12);
        CMP_SWP(uintT,rg_k5 ,rg_k13,int,rg_v5 ,rg_v13);
        CMP_SWP(uintT,rg_k6 ,rg_k14,int,rg_v6 ,rg_v14);
        CMP_SWP(uintT,rg_k7 ,rg_k15,int,rg_v7 ,rg_v15);
        // exch_paral: switch to exch_local()
        CMP_SWP(uintT,rg_k0 ,rg_k4 ,int,rg_v0 ,rg_v4 );
        CMP_SWP(uintT,rg_k1 ,rg_k5 ,int,rg_v1 ,rg_v5 );
        CMP_SWP(uintT,rg_k2 ,rg_k6 ,int,rg_v2 ,rg_v6 );
        CMP_SWP(uintT,rg_k3 ,rg_k7 ,int,rg_v3 ,rg_v7 );
        CMP_SWP(uintT,rg_k8 ,rg_k12,int,rg_v8 ,rg_v12);
        CMP_SWP(uintT,rg_k9 ,rg_k13,int,rg_v9 ,rg_v13);
        CMP_SWP(uintT,rg_k10,rg_k14,int,rg_v10,rg_v14);
        CMP_SWP(uintT,rg_k11,rg_k15,int,rg_v11,rg_v15);
        // exch_paral: switch to exch_local()
        CMP_SWP(uintT,rg_k0 ,rg_k2 ,int,rg_v0 ,rg_v2 );
        CMP_SWP(uintT,rg_k1 ,rg_k3 ,int,rg_v1 ,rg_v3 );
        CMP_SWP(uintT,rg_k4 ,rg_k6 ,int,rg_v4 ,rg_v6 );
        CMP_SWP(uintT,rg_k5 ,rg_k7 ,int,rg_v5 ,rg_v7 );
        CMP_SWP(uintT,rg_k8 ,rg_k10,int,rg_v8 ,rg_v10);
        CMP_SWP(uintT,rg_k9 ,rg_k11,int,rg_v9 ,rg_v11);
        CMP_SWP(uintT,rg_k12,rg_k14,int,rg_v12,rg_v14);
        CMP_SWP(uintT,rg_k13,rg_k15,int,rg_v13,rg_v15);
        // exch_paral: switch to exch_local()
        CMP_SWP(uintT,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
        CMP_SWP(uintT,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
        CMP_SWP(uintT,rg_k4 ,rg_k5 ,int,rg_v4 ,rg_v5 );
        CMP_SWP(uintT,rg_k6 ,rg_k7 ,int,rg_v6 ,rg_v7 );
        CMP_SWP(uintT,rg_k8 ,rg_k9 ,int,rg_v8 ,rg_v9 );
        CMP_SWP(uintT,rg_k10,rg_k11,int,rg_v10,rg_v11);
        CMP_SWP(uintT,rg_k12,rg_k13,int,rg_v12,rg_v13);
        CMP_SWP(uintT,rg_k14,rg_k15,int,rg_v14,rg_v15);
        // exch_intxn: generate exch_intxn()
        exch_intxn(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,rg_k4 ,rg_k5 ,rg_k6 ,rg_k7 ,
                   rg_k8 ,rg_k9 ,rg_k10,rg_k11,rg_k12,rg_k13,rg_k14,rg_k15,
                   rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,rg_v4 ,rg_v5 ,rg_v6 ,rg_v7 ,
                   rg_v8 ,rg_v9 ,rg_v10,rg_v11,rg_v12,rg_v13,rg_v14,rg_v15,
                   0x7,bit3);
        // exch_paral: generate exch_paral()
        exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,rg_k4 ,rg_k5 ,rg_k6 ,rg_k7 ,
                   rg_k8 ,rg_k9 ,rg_k10,rg_k11,rg_k12,rg_k13,rg_k14,rg_k15,
                   rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,rg_v4 ,rg_v5 ,rg_v6 ,rg_v7 ,
                   rg_v8 ,rg_v9 ,rg_v10,rg_v11,rg_v12,rg_v13,rg_v14,rg_v15,
                   0x2,bit2);
        // exch_paral: generate exch_paral()
        exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,rg_k4 ,rg_k5 ,rg_k6 ,rg_k7 ,
                   rg_k8 ,rg_k9 ,rg_k10,rg_k11,rg_k12,rg_k13,rg_k14,rg_k15,
                   rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,rg_v4 ,rg_v5 ,rg_v6 ,rg_v7 ,
                   rg_v8 ,rg_v9 ,rg_v10,rg_v11,rg_v12,rg_v13,rg_v14,rg_v15,
                   0x1,bit1);
        // exch_paral: switch to exch_local()
        CMP_SWP(uintT,rg_k0 ,rg_k8 ,int,rg_v0 ,rg_v8 );
        CMP_SWP(uintT,rg_k1 ,rg_k9 ,int,rg_v1 ,rg_v9 );
        CMP_SWP(uintT,rg_k2 ,rg_k10,int,rg_v2 ,rg_v10);
        CMP_SWP(uintT,rg_k3 ,rg_k11,int,rg_v3 ,rg_v11);
        CMP_SWP(uintT,rg_k4 ,rg_k12,int,rg_v4 ,rg_v12);
        CMP_SWP(uintT,rg_k5 ,rg_k13,int,rg_v5 ,rg_v13);
        CMP_SWP(uintT,rg_k6 ,rg_k14,int,rg_v6 ,rg_v14);
        CMP_SWP(uintT,rg_k7 ,rg_k15,int,rg_v7 ,rg_v15);
        // exch_paral: switch to exch_local()
        CMP_SWP(uintT,rg_k0 ,rg_k4 ,int,rg_v0 ,rg_v4 );
        CMP_SWP(uintT,rg_k1 ,rg_k5 ,int,rg_v1 ,rg_v5 );
        CMP_SWP(uintT,rg_k2 ,rg_k6 ,int,rg_v2 ,rg_v6 );
        CMP_SWP(uintT,rg_k3 ,rg_k7 ,int,rg_v3 ,rg_v7 );
        CMP_SWP(uintT,rg_k8 ,rg_k12,int,rg_v8 ,rg_v12);
        CMP_SWP(uintT,rg_k9 ,rg_k13,int,rg_v9 ,rg_v13);
        CMP_SWP(uintT,rg_k10,rg_k14,int,rg_v10,rg_v14);
        CMP_SWP(uintT,rg_k11,rg_k15,int,rg_v11,rg_v15);
        // exch_paral: switch to exch_local()
        CMP_SWP(uintT,rg_k0 ,rg_k2 ,int,rg_v0 ,rg_v2 );
        CMP_SWP(uintT,rg_k1 ,rg_k3 ,int,rg_v1 ,rg_v3 );
        CMP_SWP(uintT,rg_k4 ,rg_k6 ,int,rg_v4 ,rg_v6 );
        CMP_SWP(uintT,rg_k5 ,rg_k7 ,int,rg_v5 ,rg_v7 );
        CMP_SWP(uintT,rg_k8 ,rg_k10,int,rg_v8 ,rg_v10);
        CMP_SWP(uintT,rg_k9 ,rg_k11,int,rg_v9 ,rg_v11);
        CMP_SWP(uintT,rg_k12,rg_k14,int,rg_v12,rg_v14);
        CMP_SWP(uintT,rg_k13,rg_k15,int,rg_v13,rg_v15);
        // exch_paral: switch to exch_local()
        CMP_SWP(uintT,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
        CMP_SWP(uintT,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
        CMP_SWP(uintT,rg_k4 ,rg_k5 ,int,rg_v4 ,rg_v5 );
        CMP_SWP(uintT,rg_k6 ,rg_k7 ,int,rg_v6 ,rg_v7 );
        CMP_SWP(uintT,rg_k8 ,rg_k9 ,int,rg_v8 ,rg_v9 );
        CMP_SWP(uintT,rg_k10,rg_k11,int,rg_v10,rg_v11);
        CMP_SWP(uintT,rg_k12,rg_k13,int,rg_v12,rg_v13);
        CMP_SWP(uintT,rg_k14,rg_k15,int,rg_v14,rg_v15);
    }

    if(bin_it < normalized_bin_size) {
        // store back the results
        int lane_id = threadIdx.x & 31;
        rg_k1  = __shfl_xor(rg_k1 , 0x1 );
        rg_k3  = __shfl_xor(rg_k3 , 0x1 );
        rg_k5  = __shfl_xor(rg_k5 , 0x1 );
        rg_k7  = __shfl_xor(rg_k7 , 0x1 );
        rg_k9  = __shfl_xor(rg_k9 , 0x1 );
        rg_k11 = __shfl_xor(rg_k11, 0x1 );
        rg_k13 = __shfl_xor(rg_k13, 0x1 );
        rg_k15 = __shfl_xor(rg_k15, 0x1 );
        rg_v1  = __shfl_xor(rg_v1 , 0x1 );
        rg_v3  = __shfl_xor(rg_v3 , 0x1 );
        rg_v5  = __shfl_xor(rg_v5 , 0x1 );
        rg_v7  = __shfl_xor(rg_v7 , 0x1 );
        rg_v9  = __shfl_xor(rg_v9 , 0x1 );
        rg_v11 = __shfl_xor(rg_v11, 0x1 );
        rg_v13 = __shfl_xor(rg_v13, 0x1 );
        rg_v15 = __shfl_xor(rg_v15, 0x1 );
        if(lane_id&0x1 ) SWP(uintT, rg_k0 , rg_k1 , int, rg_v0 , rg_v1 );
        if(lane_id&0x1 ) SWP(uintT, rg_k2 , rg_k3 , int, rg_v2 , rg_v3 );
        if(lane_id&0x1 ) SWP(uintT, rg_k4 , rg_k5 , int, rg_v4 , rg_v5 );
        if(lane_id&0x1 ) SWP(uintT, rg_k6 , rg_k7 , int, rg_v6 , rg_v7 );
        if(lane_id&0x1 ) SWP(uintT, rg_k8 , rg_k9 , int, rg_v8 , rg_v9 );
        if(lane_id&0x1 ) SWP(uintT, rg_k10, rg_k11, int, rg_v10, rg_v11);
        if(lane_id&0x1 ) SWP(uintT, rg_k12, rg_k13, int, rg_v12, rg_v13);
        if(lane_id&0x1 ) SWP(uintT, rg_k14, rg_k15, int, rg_v14, rg_v15);
        rg_k1  = __shfl_xor(rg_k1 , 0x1 );
        rg_k3  = __shfl_xor(rg_k3 , 0x1 );
        rg_k5  = __shfl_xor(rg_k5 , 0x1 );
        rg_k7  = __shfl_xor(rg_k7 , 0x1 );
        rg_k9  = __shfl_xor(rg_k9 , 0x1 );
        rg_k11 = __shfl_xor(rg_k11, 0x1 );
        rg_k13 = __shfl_xor(rg_k13, 0x1 );
        rg_k15 = __shfl_xor(rg_k15, 0x1 );
        rg_v1  = __shfl_xor(rg_v1 , 0x1 );
        rg_v3  = __shfl_xor(rg_v3 , 0x1 );
        rg_v5  = __shfl_xor(rg_v5 , 0x1 );
        rg_v7  = __shfl_xor(rg_v7 , 0x1 );
        rg_v9  = __shfl_xor(rg_v9 , 0x1 );
        rg_v11 = __shfl_xor(rg_v11, 0x1 );
        rg_v13 = __shfl_xor(rg_v13, 0x1 );
        rg_v15 = __shfl_xor(rg_v15, 0x1 );
        rg_k2  = __shfl_xor(rg_k2 , 0x2 );
        rg_k3  = __shfl_xor(rg_k3 , 0x2 );
        rg_k6  = __shfl_xor(rg_k6 , 0x2 );
        rg_k7  = __shfl_xor(rg_k7 , 0x2 );
        rg_k10 = __shfl_xor(rg_k10, 0x2 );
        rg_k11 = __shfl_xor(rg_k11, 0x2 );
        rg_k14 = __shfl_xor(rg_k14, 0x2 );
        rg_k15 = __shfl_xor(rg_k15, 0x2 );
        rg_v2  = __shfl_xor(rg_v2 , 0x2 );
        rg_v3  = __shfl_xor(rg_v3 , 0x2 );
        rg_v6  = __shfl_xor(rg_v6 , 0x2 );
        rg_v7  = __shfl_xor(rg_v7 , 0x2 );
        rg_v10 = __shfl_xor(rg_v10, 0x2 );
        rg_v11 = __shfl_xor(rg_v11, 0x2 );
        rg_v14 = __shfl_xor(rg_v14, 0x2 );
        rg_v15 = __shfl_xor(rg_v15, 0x2 );
        if(lane_id&0x2 ) SWP(uintT, rg_k0 , rg_k2 , int, rg_v0 , rg_v2 );
        if(lane_id&0x2 ) SWP(uintT, rg_k1 , rg_k3 , int, rg_v1 , rg_v3 );
        if(lane_id&0x2 ) SWP(uintT, rg_k4 , rg_k6 , int, rg_v4 , rg_v6 );
        if(lane_id&0x2 ) SWP(uintT, rg_k5 , rg_k7 , int, rg_v5 , rg_v7 );
        if(lane_id&0x2 ) SWP(uintT, rg_k8 , rg_k10, int, rg_v8 , rg_v10);
        if(lane_id&0x2 ) SWP(uintT, rg_k9 , rg_k11, int, rg_v9 , rg_v11);
        if(lane_id&0x2 ) SWP(uintT, rg_k12, rg_k14, int, rg_v12, rg_v14);
        if(lane_id&0x2 ) SWP(uintT, rg_k13, rg_k15, int, rg_v13, rg_v15);
        rg_k2  = __shfl_xor(rg_k2 , 0x2 );
        rg_k3  = __shfl_xor(rg_k3 , 0x2 );
        rg_k6  = __shfl_xor(rg_k6 , 0x2 );
        rg_k7  = __shfl_xor(rg_k7 , 0x2 );
        rg_k10 = __shfl_xor(rg_k10, 0x2 );
        rg_k11 = __shfl_xor(rg_k11, 0x2 );
        rg_k14 = __shfl_xor(rg_k14, 0x2 );
        rg_k15 = __shfl_xor(rg_k15, 0x2 );
        rg_v2  = __shfl_xor(rg_v2 , 0x2 );
        rg_v3  = __shfl_xor(rg_v3 , 0x2 );
        rg_v6  = __shfl_xor(rg_v6 , 0x2 );
        rg_v7  = __shfl_xor(rg_v7 , 0x2 );
        rg_v10 = __shfl_xor(rg_v10, 0x2 );
        rg_v11 = __shfl_xor(rg_v11, 0x2 );
        rg_v14 = __shfl_xor(rg_v14, 0x2 );
        rg_v15 = __shfl_xor(rg_v15, 0x2 );
        rg_k4  = __shfl_xor(rg_k4 , 0x4 );
        rg_k5  = __shfl_xor(rg_k5 , 0x4 );
        rg_k6  = __shfl_xor(rg_k6 , 0x4 );
        rg_k7  = __shfl_xor(rg_k7 , 0x4 );
        rg_k12 = __shfl_xor(rg_k12, 0x4 );
        rg_k13 = __shfl_xor(rg_k13, 0x4 );
        rg_k14 = __shfl_xor(rg_k14, 0x4 );
        rg_k15 = __shfl_xor(rg_k15, 0x4 );
        rg_v4  = __shfl_xor(rg_v4 , 0x4 );
        rg_v5  = __shfl_xor(rg_v5 , 0x4 );
        rg_v6  = __shfl_xor(rg_v6 , 0x4 );
        rg_v7  = __shfl_xor(rg_v7 , 0x4 );
        rg_v12 = __shfl_xor(rg_v12, 0x4 );
        rg_v13 = __shfl_xor(rg_v13, 0x4 );
        rg_v14 = __shfl_xor(rg_v14, 0x4 );
        rg_v15 = __shfl_xor(rg_v15, 0x4 );
        if(lane_id&0x4 ) SWP(uintT, rg_k0 , rg_k4 , int, rg_v0 , rg_v4 );
        if(lane_id&0x4 ) SWP(uintT, rg_k1 , rg_k5 , int, rg_v1 , rg_v5 );
        if(lane_id&0x4 ) SWP(uintT, rg_k2 , rg_k6 , int, rg_v2 , rg_v6 );
        if(lane_id&0x4 ) SWP(uintT, rg_k3 , rg_k7 , int, rg_v3 , rg_v7 );
        if(lane_id&0x4 ) SWP(uintT, rg_k8 , rg_k12, int, rg_v8 , rg_v12);
        if(lane_id&0x4 ) SWP(uintT, rg_k9 , rg_k13, int, rg_v9 , rg_v13);
        if(lane_id&0x4 ) SWP(uintT, rg_k10, rg_k14, int, rg_v10, rg_v14);
        if(lane_id&0x4 ) SWP(uintT, rg_k11, rg_k15, int, rg_v11, rg_v15);
        rg_k4  = __shfl_xor(rg_k4 , 0x4 );
        rg_k5  = __shfl_xor(rg_k5 , 0x4 );
        rg_k6  = __shfl_xor(rg_k6 , 0x4 );
        rg_k7  = __shfl_xor(rg_k7 , 0x4 );
        rg_k12 = __shfl_xor(rg_k12, 0x4 );
        rg_k13 = __shfl_xor(rg_k13, 0x4 );
        rg_k14 = __shfl_xor(rg_k14, 0x4 );
        rg_k15 = __shfl_xor(rg_k15, 0x4 );
        rg_v4  = __shfl_xor(rg_v4 , 0x4 );
        rg_v5  = __shfl_xor(rg_v5 , 0x4 );
        rg_v6  = __shfl_xor(rg_v6 , 0x4 );
        rg_v7  = __shfl_xor(rg_v7 , 0x4 );
        rg_v12 = __shfl_xor(rg_v12, 0x4 );
        rg_v13 = __shfl_xor(rg_v13, 0x4 );
        rg_v14 = __shfl_xor(rg_v14, 0x4 );
        rg_v15 = __shfl_xor(rg_v15, 0x4 );
        rg_k8  = __shfl_xor(rg_k8 , 0x8 );
        rg_k9  = __shfl_xor(rg_k9 , 0x8 );
        rg_k10 = __shfl_xor(rg_k10, 0x8 );
        rg_k11 = __shfl_xor(rg_k11, 0x8 );
        rg_k12 = __shfl_xor(rg_k12, 0x8 );
        rg_k13 = __shfl_xor(rg_k13, 0x8 );
        rg_k14 = __shfl_xor(rg_k14, 0x8 );
        rg_k15 = __shfl_xor(rg_k15, 0x8 );
        rg_v8  = __shfl_xor(rg_v8 , 0x8 );
        rg_v9  = __shfl_xor(rg_v9 , 0x8 );
        rg_v10 = __shfl_xor(rg_v10, 0x8 );
        rg_v11 = __shfl_xor(rg_v11, 0x8 );
        rg_v12 = __shfl_xor(rg_v12, 0x8 );
        rg_v13 = __shfl_xor(rg_v13, 0x8 );
        rg_v14 = __shfl_xor(rg_v14, 0x8 );
        rg_v15 = __shfl_xor(rg_v15, 0x8 );
        if(lane_id&0x8 ) SWP(uintT, rg_k0 , rg_k8 , int, rg_v0 , rg_v8 );
        if(lane_id&0x8 ) SWP(uintT, rg_k1 , rg_k9 , int, rg_v1 , rg_v9 );
        if(lane_id&0x8 ) SWP(uintT, rg_k2 , rg_k10, int, rg_v2 , rg_v10);
        if(lane_id&0x8 ) SWP(uintT, rg_k3 , rg_k11, int, rg_v3 , rg_v11);
        if(lane_id&0x8 ) SWP(uintT, rg_k4 , rg_k12, int, rg_v4 , rg_v12);
        if(lane_id&0x8 ) SWP(uintT, rg_k5 , rg_k13, int, rg_v5 , rg_v13);
        if(lane_id&0x8 ) SWP(uintT, rg_k6 , rg_k14, int, rg_v6 , rg_v14);
        if(lane_id&0x8 ) SWP(uintT, rg_k7 , rg_k15, int, rg_v7 , rg_v15);
        rg_k8  = __shfl_xor(rg_k8 , 0x8 );
        rg_k9  = __shfl_xor(rg_k9 , 0x8 );
        rg_k10 = __shfl_xor(rg_k10, 0x8 );
        rg_k11 = __shfl_xor(rg_k11, 0x8 );
        rg_k12 = __shfl_xor(rg_k12, 0x8 );
        rg_k13 = __shfl_xor(rg_k13, 0x8 );
        rg_k14 = __shfl_xor(rg_k14, 0x8 );
        rg_k15 = __shfl_xor(rg_k15, 0x8 );
        rg_v8  = __shfl_xor(rg_v8 , 0x8 );
        rg_v9  = __shfl_xor(rg_v9 , 0x8 );
        rg_v10 = __shfl_xor(rg_v10, 0x8 );
        rg_v11 = __shfl_xor(rg_v11, 0x8 );
        rg_v12 = __shfl_xor(rg_v12, 0x8 );
        rg_v13 = __shfl_xor(rg_v13, 0x8 );
        rg_v14 = __shfl_xor(rg_v14, 0x8 );
        rg_v15 = __shfl_xor(rg_v15, 0x8 );
        rg_k1  = __shfl_xor(rg_k1 , 0x10);
        rg_k3  = __shfl_xor(rg_k3 , 0x10);
        rg_k5  = __shfl_xor(rg_k5 , 0x10);
        rg_k7  = __shfl_xor(rg_k7 , 0x10);
        rg_k9  = __shfl_xor(rg_k9 , 0x10);
        rg_k11 = __shfl_xor(rg_k11, 0x10);
        rg_k13 = __shfl_xor(rg_k13, 0x10);
        rg_k15 = __shfl_xor(rg_k15, 0x10);
        rg_v1  = __shfl_xor(rg_v1 , 0x10);
        rg_v3  = __shfl_xor(rg_v3 , 0x10);
        rg_v5  = __shfl_xor(rg_v5 , 0x10);
        rg_v7  = __shfl_xor(rg_v7 , 0x10);
        rg_v9  = __shfl_xor(rg_v9 , 0x10);
        rg_v11 = __shfl_xor(rg_v11, 0x10);
        rg_v13 = __shfl_xor(rg_v13, 0x10);
        rg_v15 = __shfl_xor(rg_v15, 0x10);
        if(lane_id&0x10) SWP(uintT, rg_k0 , rg_k1 , int, rg_v0 , rg_v1 );
        if(lane_id&0x10) SWP(uintT, rg_k2 , rg_k3 , int, rg_v2 , rg_v3 );
        if(lane_id&0x10) SWP(uintT, rg_k4 , rg_k5 , int, rg_v4 , rg_v5 );
        if(lane_id&0x10) SWP(uintT, rg_k6 , rg_k7 , int, rg_v6 , rg_v7 );
        if(lane_id&0x10) SWP(uintT, rg_k8 , rg_k9 , int, rg_v8 , rg_v9 );
        if(lane_id&0x10) SWP(uintT, rg_k10, rg_k11, int, rg_v10, rg_v11);
        if(lane_id&0x10) SWP(uintT, rg_k12, rg_k13, int, rg_v12, rg_v13);
        if(lane_id&0x10) SWP(uintT, rg_k14, rg_k15, int, rg_v14, rg_v15);
        rg_k1  = __shfl_xor(rg_k1 , 0x10);
        rg_k3  = __shfl_xor(rg_k3 , 0x10);
        rg_k5  = __shfl_xor(rg_k5 , 0x10);
        rg_k7  = __shfl_xor(rg_k7 , 0x10);
        rg_k9  = __shfl_xor(rg_k9 , 0x10);
        rg_k11 = __shfl_xor(rg_k11, 0x10);
        rg_k13 = __shfl_xor(rg_k13, 0x10);
        rg_k15 = __shfl_xor(rg_k15, 0x10);
        rg_v1  = __shfl_xor(rg_v1 , 0x10);
        rg_v3  = __shfl_xor(rg_v3 , 0x10);
        rg_v5  = __shfl_xor(rg_v5 , 0x10);
        rg_v7  = __shfl_xor(rg_v7 , 0x10);
        rg_v9  = __shfl_xor(rg_v9 , 0x10);
        rg_v11 = __shfl_xor(rg_v11, 0x10);
        rg_v13 = __shfl_xor(rg_v13, 0x10);
        rg_v15 = __shfl_xor(rg_v15, 0x10);
        int kk;
        int ss;
        kk = __shfl(k, 0 );
        ss = __shfl(seg_size, 0 );
        if(lane_id+0  <ss) keyB[kk+lane_id+0  ] = rg_k0 ;
        if(lane_id+32 <ss) keyB[kk+lane_id+32 ] = rg_k2 ;
        if(lane_id+64 <ss) keyB[kk+lane_id+64 ] = rg_k4 ;
        if(lane_id+96 <ss) keyB[kk+lane_id+96 ] = rg_k6 ;
        kk = __shfl(k, 8 );
        ss = __shfl(seg_size, 8 );
        if(lane_id+0  <ss) keyB[kk+lane_id+0  ] = rg_k8 ;
        if(lane_id+32 <ss) keyB[kk+lane_id+32 ] = rg_k10;
        if(lane_id+64 <ss) keyB[kk+lane_id+64 ] = rg_k12;
        if(lane_id+96 <ss) keyB[kk+lane_id+96 ] = rg_k14;
        kk = __shfl(k, 16);
        ss = __shfl(seg_size, 16);
        if(lane_id+0  <ss) keyB[kk+lane_id+0  ] = rg_k1 ;
        if(lane_id+32 <ss) keyB[kk+lane_id+32 ] = rg_k3 ;
        if(lane_id+64 <ss) keyB[kk+lane_id+64 ] = rg_k5 ;
        if(lane_id+96 <ss) keyB[kk+lane_id+96 ] = rg_k7 ;
        kk = __shfl(k, 24);
        ss = __shfl(seg_size, 24);
        if(lane_id+0  <ss) keyB[kk+lane_id+0  ] = rg_k9 ;
        if(lane_id+32 <ss) keyB[kk+lane_id+32 ] = rg_k11;
        if(lane_id+64 <ss) keyB[kk+lane_id+64 ] = rg_k13;
        if(lane_id+96 <ss) keyB[kk+lane_id+96 ] = rg_k15;
        kk = __shfl(k, 0 );
        ss = __shfl(seg_size, 0 );
        if(lane_id+0  <ss) valB[kk+lane_id+0  ] = val[kk+rg_v0 ];
        if(lane_id+32 <ss) valB[kk+lane_id+32 ] = val[kk+rg_v2 ];
        if(lane_id+64 <ss) valB[kk+lane_id+64 ] = val[kk+rg_v4 ];
        if(lane_id+96 <ss) valB[kk+lane_id+96 ] = val[kk+rg_v6 ];
        kk = __shfl(k, 8 );
        ss = __shfl(seg_size, 8 );
        if(lane_id+0  <ss) valB[kk+lane_id+0  ] = val[kk+rg_v8 ];
        if(lane_id+32 <ss) valB[kk+lane_id+32 ] = val[kk+rg_v10];
        if(lane_id+64 <ss) valB[kk+lane_id+64 ] = val[kk+rg_v12];
        if(lane_id+96 <ss) valB[kk+lane_id+96 ] = val[kk+rg_v14];
        kk = __shfl(k, 16);
        ss = __shfl(seg_size, 16);
        if(lane_id+0  <ss) valB[kk+lane_id+0  ] = val[kk+rg_v1 ];
        if(lane_id+32 <ss) valB[kk+lane_id+32 ] = val[kk+rg_v3 ];
        if(lane_id+64 <ss) valB[kk+lane_id+64 ] = val[kk+rg_v5 ];
        if(lane_id+96 <ss) valB[kk+lane_id+96 ] = val[kk+rg_v7 ];
        kk = __shfl(k, 24);
        ss = __shfl(seg_size, 24);
        if(lane_id+0  <ss) valB[kk+lane_id+0  ] = val[kk+rg_v9 ];
        if(lane_id+32 <ss) valB[kk+lane_id+32 ] = val[kk+rg_v11];
        if(lane_id+64 <ss) valB[kk+lane_id+64 ] = val[kk+rg_v13];
        if(lane_id+96 <ss) valB[kk+lane_id+96 ] = val[kk+rg_v15];
    } else if(bin_it < bin_size) {
        if((tid<<4)+0 <seg_size) keyB[k+(tid<<4)+0 ] = rg_k0 ;
        if((tid<<4)+1 <seg_size) keyB[k+(tid<<4)+1 ] = rg_k1 ;
        if((tid<<4)+2 <seg_size) keyB[k+(tid<<4)+2 ] = rg_k2 ;
        if((tid<<4)+3 <seg_size) keyB[k+(tid<<4)+3 ] = rg_k3 ;
        if((tid<<4)+4 <seg_size) keyB[k+(tid<<4)+4 ] = rg_k4 ;
        if((tid<<4)+5 <seg_size) keyB[k+(tid<<4)+5 ] = rg_k5 ;
        if((tid<<4)+6 <seg_size) keyB[k+(tid<<4)+6 ] = rg_k6 ;
        if((tid<<4)+7 <seg_size) keyB[k+(tid<<4)+7 ] = rg_k7 ;
        if((tid<<4)+8 <seg_size) keyB[k+(tid<<4)+8 ] = rg_k8 ;
        if((tid<<4)+9 <seg_size) keyB[k+(tid<<4)+9 ] = rg_k9 ;
        if((tid<<4)+10<seg_size) keyB[k+(tid<<4)+10] = rg_k10;
        if((tid<<4)+11<seg_size) keyB[k+(tid<<4)+11] = rg_k11;
        if((tid<<4)+12<seg_size) keyB[k+(tid<<4)+12] = rg_k12;
        if((tid<<4)+13<seg_size) keyB[k+(tid<<4)+13] = rg_k13;
        if((tid<<4)+14<seg_size) keyB[k+(tid<<4)+14] = rg_k14;
        if((tid<<4)+15<seg_size) keyB[k+(tid<<4)+15] = rg_k15;
        if((tid<<4)+0 <seg_size) valB[k+(tid<<4)+0 ] = val[k+rg_v0 ];
        if((tid<<4)+1 <seg_size) valB[k+(tid<<4)+1 ] = val[k+rg_v1 ];
        if((tid<<4)+2 <seg_size) valB[k+(tid<<4)+2 ] = val[k+rg_v2 ];
        if((tid<<4)+3 <seg_size) valB[k+(tid<<4)+3 ] = val[k+rg_v3 ];
        if((tid<<4)+4 <seg_size) valB[k+(tid<<4)+4 ] = val[k+rg_v4 ];
        if((tid<<4)+5 <seg_size) valB[k+(tid<<4)+5 ] = val[k+rg_v5 ];
        if((tid<<4)+6 <seg_size) valB[k+(tid<<4)+6 ] = val[k+rg_v6 ];
        if((tid<<4)+7 <seg_size) valB[k+(tid<<4)+7 ] = val[k+rg_v7 ];
        if((tid<<4)+8 <seg_size) valB[k+(tid<<4)+8 ] = val[k+rg_v8 ];
        if((tid<<4)+9 <seg_size) valB[k+(tid<<4)+9 ] = val[k+rg_v9 ];
        if((tid<<4)+10<seg_size) valB[k+(tid<<4)+10] = val[k+rg_v10];
        if((tid<<4)+11<seg_size) valB[k+(tid<<4)+11] = val[k+rg_v11];
        if((tid<<4)+12<seg_size) valB[k+(tid<<4)+12] = val[k+rg_v12];
        if((tid<<4)+13<seg_size) valB[k+(tid<<4)+13] = val[k+rg_v13];
        if((tid<<4)+14<seg_size) valB[k+(tid<<4)+14] = val[k+rg_v14];
        if((tid<<4)+15<seg_size) valB[k+(tid<<4)+15] = val[k+rg_v15];
    }
}
/* block tcf subwarp coalesced quiet real_kern */
/*   256   8      32      true  true      true */
template<class T>
__global__
void gen_bk256_wp32_tc8_r129_r256_strd( 
    uintT *key, T *val, uintT *keyB, T *valB, int n, int *segs, int *bin, int bin_size, int length) {

    const int gid = threadIdx.x + blockIdx.x * blockDim.x;
    const int bin_it = (gid>>5);
    const int tid = (threadIdx.x & 31);
    const int bit1 = (tid>>0)&0x1;
    const int bit2 = (tid>>1)&0x1;
    const int bit3 = (tid>>2)&0x1;
    const int bit4 = (tid>>3)&0x1;
    const int bit5 = (tid>>4)&0x1;
    uintT rg_k0 ;
    uintT rg_k1 ;
    uintT rg_k2 ;
    uintT rg_k3 ;
    uintT rg_k4 ;
    uintT rg_k5 ;
    uintT rg_k6 ;
    uintT rg_k7 ;
    int rg_v0 ;
    int rg_v1 ;
    int rg_v2 ;
    int rg_v3 ;
    int rg_v4 ;
    int rg_v5 ;
    int rg_v6 ;
    int rg_v7 ;
    int normalized_bin_size = (bin_size/1)*1;
    int k;
    int seg_size;
    if(bin_it < bin_size) {
        k = segs[bin[bin_it]];
        seg_size = ((bin[bin_it]==length-1)?n:segs[bin[bin_it]+1])-segs[bin[bin_it]];
        rg_k0  = (tid+0   <seg_size)?key[k+tid+0   ]:INT_MAX;
        rg_k1  = (tid+32  <seg_size)?key[k+tid+32  ]:INT_MAX;
        rg_k2  = (tid+64  <seg_size)?key[k+tid+64  ]:INT_MAX;
        rg_k3  = (tid+96  <seg_size)?key[k+tid+96  ]:INT_MAX;
        rg_k4  = (tid+128 <seg_size)?key[k+tid+128 ]:INT_MAX;
        rg_k5  = (tid+160 <seg_size)?key[k+tid+160 ]:INT_MAX;
        rg_k6  = (tid+192 <seg_size)?key[k+tid+192 ]:INT_MAX;
        rg_k7  = (tid+224 <seg_size)?key[k+tid+224 ]:INT_MAX;
        if(tid+0   <seg_size) rg_v0  = tid+0   ;
        if(tid+32  <seg_size) rg_v1  = tid+32  ;
        if(tid+64  <seg_size) rg_v2  = tid+64  ;
        if(tid+96  <seg_size) rg_v3  = tid+96  ;
        if(tid+128 <seg_size) rg_v4  = tid+128 ;
        if(tid+160 <seg_size) rg_v5  = tid+160 ;
        if(tid+192 <seg_size) rg_v6  = tid+192 ;
        if(tid+224 <seg_size) rg_v7  = tid+224 ;
        // sort 256 elements
        // exch_intxn: switch to exch_local()
        CMP_SWP(uintT,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
        CMP_SWP(uintT,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
        CMP_SWP(uintT,rg_k4 ,rg_k5 ,int,rg_v4 ,rg_v5 );
        CMP_SWP(uintT,rg_k6 ,rg_k7 ,int,rg_v6 ,rg_v7 );
        // exch_intxn: switch to exch_local()
        CMP_SWP(uintT,rg_k0 ,rg_k3 ,int,rg_v0 ,rg_v3 );
        CMP_SWP(uintT,rg_k1 ,rg_k2 ,int,rg_v1 ,rg_v2 );
        CMP_SWP(uintT,rg_k4 ,rg_k7 ,int,rg_v4 ,rg_v7 );
        CMP_SWP(uintT,rg_k5 ,rg_k6 ,int,rg_v5 ,rg_v6 );
        // exch_paral: switch to exch_local()
        CMP_SWP(uintT,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
        CMP_SWP(uintT,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
        CMP_SWP(uintT,rg_k4 ,rg_k5 ,int,rg_v4 ,rg_v5 );
        CMP_SWP(uintT,rg_k6 ,rg_k7 ,int,rg_v6 ,rg_v7 );
        // exch_intxn: switch to exch_local()
        CMP_SWP(uintT,rg_k0 ,rg_k7 ,int,rg_v0 ,rg_v7 );
        CMP_SWP(uintT,rg_k1 ,rg_k6 ,int,rg_v1 ,rg_v6 );
        CMP_SWP(uintT,rg_k2 ,rg_k5 ,int,rg_v2 ,rg_v5 );
        CMP_SWP(uintT,rg_k3 ,rg_k4 ,int,rg_v3 ,rg_v4 );
        // exch_paral: switch to exch_local()
        CMP_SWP(uintT,rg_k0 ,rg_k2 ,int,rg_v0 ,rg_v2 );
        CMP_SWP(uintT,rg_k1 ,rg_k3 ,int,rg_v1 ,rg_v3 );
        CMP_SWP(uintT,rg_k4 ,rg_k6 ,int,rg_v4 ,rg_v6 );
        CMP_SWP(uintT,rg_k5 ,rg_k7 ,int,rg_v5 ,rg_v7 );
        // exch_paral: switch to exch_local()
        CMP_SWP(uintT,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
        CMP_SWP(uintT,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
        CMP_SWP(uintT,rg_k4 ,rg_k5 ,int,rg_v4 ,rg_v5 );
        CMP_SWP(uintT,rg_k6 ,rg_k7 ,int,rg_v6 ,rg_v7 );
        // exch_intxn: generate exch_intxn()
        exch_intxn(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,rg_k4 ,rg_k5 ,rg_k6 ,rg_k7 ,
                   rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,rg_v4 ,rg_v5 ,rg_v6 ,rg_v7 ,
                   0x1,bit1);
        // exch_paral: switch to exch_local()
        CMP_SWP(uintT,rg_k0 ,rg_k4 ,int,rg_v0 ,rg_v4 );
        CMP_SWP(uintT,rg_k1 ,rg_k5 ,int,rg_v1 ,rg_v5 );
        CMP_SWP(uintT,rg_k2 ,rg_k6 ,int,rg_v2 ,rg_v6 );
        CMP_SWP(uintT,rg_k3 ,rg_k7 ,int,rg_v3 ,rg_v7 );
        // exch_paral: switch to exch_local()
        CMP_SWP(uintT,rg_k0 ,rg_k2 ,int,rg_v0 ,rg_v2 );
        CMP_SWP(uintT,rg_k1 ,rg_k3 ,int,rg_v1 ,rg_v3 );
        CMP_SWP(uintT,rg_k4 ,rg_k6 ,int,rg_v4 ,rg_v6 );
        CMP_SWP(uintT,rg_k5 ,rg_k7 ,int,rg_v5 ,rg_v7 );
        // exch_paral: switch to exch_local()
        CMP_SWP(uintT,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
        CMP_SWP(uintT,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
        CMP_SWP(uintT,rg_k4 ,rg_k5 ,int,rg_v4 ,rg_v5 );
        CMP_SWP(uintT,rg_k6 ,rg_k7 ,int,rg_v6 ,rg_v7 );
        // exch_intxn: generate exch_intxn()
        exch_intxn(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,rg_k4 ,rg_k5 ,rg_k6 ,rg_k7 ,
                   rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,rg_v4 ,rg_v5 ,rg_v6 ,rg_v7 ,
                   0x3,bit2);
        // exch_paral: generate exch_paral()
        exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,rg_k4 ,rg_k5 ,rg_k6 ,rg_k7 ,
                   rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,rg_v4 ,rg_v5 ,rg_v6 ,rg_v7 ,
                   0x1,bit1);
        // exch_paral: switch to exch_local()
        CMP_SWP(uintT,rg_k0 ,rg_k4 ,int,rg_v0 ,rg_v4 );
        CMP_SWP(uintT,rg_k1 ,rg_k5 ,int,rg_v1 ,rg_v5 );
        CMP_SWP(uintT,rg_k2 ,rg_k6 ,int,rg_v2 ,rg_v6 );
        CMP_SWP(uintT,rg_k3 ,rg_k7 ,int,rg_v3 ,rg_v7 );
        // exch_paral: switch to exch_local()
        CMP_SWP(uintT,rg_k0 ,rg_k2 ,int,rg_v0 ,rg_v2 );
        CMP_SWP(uintT,rg_k1 ,rg_k3 ,int,rg_v1 ,rg_v3 );
        CMP_SWP(uintT,rg_k4 ,rg_k6 ,int,rg_v4 ,rg_v6 );
        CMP_SWP(uintT,rg_k5 ,rg_k7 ,int,rg_v5 ,rg_v7 );
        // exch_paral: switch to exch_local()
        CMP_SWP(uintT,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
        CMP_SWP(uintT,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
        CMP_SWP(uintT,rg_k4 ,rg_k5 ,int,rg_v4 ,rg_v5 );
        CMP_SWP(uintT,rg_k6 ,rg_k7 ,int,rg_v6 ,rg_v7 );
        // exch_intxn: generate exch_intxn()
        exch_intxn(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,rg_k4 ,rg_k5 ,rg_k6 ,rg_k7 ,
                   rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,rg_v4 ,rg_v5 ,rg_v6 ,rg_v7 ,
                   0x7,bit3);
        // exch_paral: generate exch_paral()
        exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,rg_k4 ,rg_k5 ,rg_k6 ,rg_k7 ,
                   rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,rg_v4 ,rg_v5 ,rg_v6 ,rg_v7 ,
                   0x2,bit2);
        // exch_paral: generate exch_paral()
        exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,rg_k4 ,rg_k5 ,rg_k6 ,rg_k7 ,
                   rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,rg_v4 ,rg_v5 ,rg_v6 ,rg_v7 ,
                   0x1,bit1);
        // exch_paral: switch to exch_local()
        CMP_SWP(uintT,rg_k0 ,rg_k4 ,int,rg_v0 ,rg_v4 );
        CMP_SWP(uintT,rg_k1 ,rg_k5 ,int,rg_v1 ,rg_v5 );
        CMP_SWP(uintT,rg_k2 ,rg_k6 ,int,rg_v2 ,rg_v6 );
        CMP_SWP(uintT,rg_k3 ,rg_k7 ,int,rg_v3 ,rg_v7 );
        // exch_paral: switch to exch_local()
        CMP_SWP(uintT,rg_k0 ,rg_k2 ,int,rg_v0 ,rg_v2 );
        CMP_SWP(uintT,rg_k1 ,rg_k3 ,int,rg_v1 ,rg_v3 );
        CMP_SWP(uintT,rg_k4 ,rg_k6 ,int,rg_v4 ,rg_v6 );
        CMP_SWP(uintT,rg_k5 ,rg_k7 ,int,rg_v5 ,rg_v7 );
        // exch_paral: switch to exch_local()
        CMP_SWP(uintT,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
        CMP_SWP(uintT,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
        CMP_SWP(uintT,rg_k4 ,rg_k5 ,int,rg_v4 ,rg_v5 );
        CMP_SWP(uintT,rg_k6 ,rg_k7 ,int,rg_v6 ,rg_v7 );
        // exch_intxn: generate exch_intxn()
        exch_intxn(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,rg_k4 ,rg_k5 ,rg_k6 ,rg_k7 ,
                   rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,rg_v4 ,rg_v5 ,rg_v6 ,rg_v7 ,
                   0xf,bit4);
        // exch_paral: generate exch_paral()
        exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,rg_k4 ,rg_k5 ,rg_k6 ,rg_k7 ,
                   rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,rg_v4 ,rg_v5 ,rg_v6 ,rg_v7 ,
                   0x4,bit3);
        // exch_paral: generate exch_paral()
        exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,rg_k4 ,rg_k5 ,rg_k6 ,rg_k7 ,
                   rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,rg_v4 ,rg_v5 ,rg_v6 ,rg_v7 ,
                   0x2,bit2);
        // exch_paral: generate exch_paral()
        exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,rg_k4 ,rg_k5 ,rg_k6 ,rg_k7 ,
                   rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,rg_v4 ,rg_v5 ,rg_v6 ,rg_v7 ,
                   0x1,bit1);
        // exch_paral: switch to exch_local()
        CMP_SWP(uintT,rg_k0 ,rg_k4 ,int,rg_v0 ,rg_v4 );
        CMP_SWP(uintT,rg_k1 ,rg_k5 ,int,rg_v1 ,rg_v5 );
        CMP_SWP(uintT,rg_k2 ,rg_k6 ,int,rg_v2 ,rg_v6 );
        CMP_SWP(uintT,rg_k3 ,rg_k7 ,int,rg_v3 ,rg_v7 );
        // exch_paral: switch to exch_local()
        CMP_SWP(uintT,rg_k0 ,rg_k2 ,int,rg_v0 ,rg_v2 );
        CMP_SWP(uintT,rg_k1 ,rg_k3 ,int,rg_v1 ,rg_v3 );
        CMP_SWP(uintT,rg_k4 ,rg_k6 ,int,rg_v4 ,rg_v6 );
        CMP_SWP(uintT,rg_k5 ,rg_k7 ,int,rg_v5 ,rg_v7 );
        // exch_paral: switch to exch_local()
        CMP_SWP(uintT,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
        CMP_SWP(uintT,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
        CMP_SWP(uintT,rg_k4 ,rg_k5 ,int,rg_v4 ,rg_v5 );
        CMP_SWP(uintT,rg_k6 ,rg_k7 ,int,rg_v6 ,rg_v7 );
        // exch_intxn: generate exch_intxn()
        exch_intxn(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,rg_k4 ,rg_k5 ,rg_k6 ,rg_k7 ,
                   rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,rg_v4 ,rg_v5 ,rg_v6 ,rg_v7 ,
                   0x1f,bit5);
        // exch_paral: generate exch_paral()
        exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,rg_k4 ,rg_k5 ,rg_k6 ,rg_k7 ,
                   rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,rg_v4 ,rg_v5 ,rg_v6 ,rg_v7 ,
                   0x8,bit4);
        // exch_paral: generate exch_paral()
        exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,rg_k4 ,rg_k5 ,rg_k6 ,rg_k7 ,
                   rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,rg_v4 ,rg_v5 ,rg_v6 ,rg_v7 ,
                   0x4,bit3);
        // exch_paral: generate exch_paral()
        exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,rg_k4 ,rg_k5 ,rg_k6 ,rg_k7 ,
                   rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,rg_v4 ,rg_v5 ,rg_v6 ,rg_v7 ,
                   0x2,bit2);
        // exch_paral: generate exch_paral()
        exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,rg_k4 ,rg_k5 ,rg_k6 ,rg_k7 ,
                   rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,rg_v4 ,rg_v5 ,rg_v6 ,rg_v7 ,
                   0x1,bit1);
        // exch_paral: switch to exch_local()
        CMP_SWP(uintT,rg_k0 ,rg_k4 ,int,rg_v0 ,rg_v4 );
        CMP_SWP(uintT,rg_k1 ,rg_k5 ,int,rg_v1 ,rg_v5 );
        CMP_SWP(uintT,rg_k2 ,rg_k6 ,int,rg_v2 ,rg_v6 );
        CMP_SWP(uintT,rg_k3 ,rg_k7 ,int,rg_v3 ,rg_v7 );
        // exch_paral: switch to exch_local()
        CMP_SWP(uintT,rg_k0 ,rg_k2 ,int,rg_v0 ,rg_v2 );
        CMP_SWP(uintT,rg_k1 ,rg_k3 ,int,rg_v1 ,rg_v3 );
        CMP_SWP(uintT,rg_k4 ,rg_k6 ,int,rg_v4 ,rg_v6 );
        CMP_SWP(uintT,rg_k5 ,rg_k7 ,int,rg_v5 ,rg_v7 );
        // exch_paral: switch to exch_local()
        CMP_SWP(uintT,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
        CMP_SWP(uintT,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
        CMP_SWP(uintT,rg_k4 ,rg_k5 ,int,rg_v4 ,rg_v5 );
        CMP_SWP(uintT,rg_k6 ,rg_k7 ,int,rg_v6 ,rg_v7 );
    }

    if(bin_it < normalized_bin_size) {
        // store back the results
        int lane_id = threadIdx.x & 31;
        rg_k1  = __shfl_xor(rg_k1 , 0x1 );
        rg_k3  = __shfl_xor(rg_k3 , 0x1 );
        rg_k5  = __shfl_xor(rg_k5 , 0x1 );
        rg_k7  = __shfl_xor(rg_k7 , 0x1 );
        rg_v1  = __shfl_xor(rg_v1 , 0x1 );
        rg_v3  = __shfl_xor(rg_v3 , 0x1 );
        rg_v5  = __shfl_xor(rg_v5 , 0x1 );
        rg_v7  = __shfl_xor(rg_v7 , 0x1 );
        if(lane_id&0x1 ) SWP(uintT, rg_k0 , rg_k1 , int, rg_v0 , rg_v1 );
        if(lane_id&0x1 ) SWP(uintT, rg_k2 , rg_k3 , int, rg_v2 , rg_v3 );
        if(lane_id&0x1 ) SWP(uintT, rg_k4 , rg_k5 , int, rg_v4 , rg_v5 );
        if(lane_id&0x1 ) SWP(uintT, rg_k6 , rg_k7 , int, rg_v6 , rg_v7 );
        rg_k1  = __shfl_xor(rg_k1 , 0x1 );
        rg_k3  = __shfl_xor(rg_k3 , 0x1 );
        rg_k5  = __shfl_xor(rg_k5 , 0x1 );
        rg_k7  = __shfl_xor(rg_k7 , 0x1 );
        rg_v1  = __shfl_xor(rg_v1 , 0x1 );
        rg_v3  = __shfl_xor(rg_v3 , 0x1 );
        rg_v5  = __shfl_xor(rg_v5 , 0x1 );
        rg_v7  = __shfl_xor(rg_v7 , 0x1 );
        rg_k2  = __shfl_xor(rg_k2 , 0x2 );
        rg_k3  = __shfl_xor(rg_k3 , 0x2 );
        rg_k6  = __shfl_xor(rg_k6 , 0x2 );
        rg_k7  = __shfl_xor(rg_k7 , 0x2 );
        rg_v2  = __shfl_xor(rg_v2 , 0x2 );
        rg_v3  = __shfl_xor(rg_v3 , 0x2 );
        rg_v6  = __shfl_xor(rg_v6 , 0x2 );
        rg_v7  = __shfl_xor(rg_v7 , 0x2 );
        if(lane_id&0x2 ) SWP(uintT, rg_k0 , rg_k2 , int, rg_v0 , rg_v2 );
        if(lane_id&0x2 ) SWP(uintT, rg_k1 , rg_k3 , int, rg_v1 , rg_v3 );
        if(lane_id&0x2 ) SWP(uintT, rg_k4 , rg_k6 , int, rg_v4 , rg_v6 );
        if(lane_id&0x2 ) SWP(uintT, rg_k5 , rg_k7 , int, rg_v5 , rg_v7 );
        rg_k2  = __shfl_xor(rg_k2 , 0x2 );
        rg_k3  = __shfl_xor(rg_k3 , 0x2 );
        rg_k6  = __shfl_xor(rg_k6 , 0x2 );
        rg_k7  = __shfl_xor(rg_k7 , 0x2 );
        rg_v2  = __shfl_xor(rg_v2 , 0x2 );
        rg_v3  = __shfl_xor(rg_v3 , 0x2 );
        rg_v6  = __shfl_xor(rg_v6 , 0x2 );
        rg_v7  = __shfl_xor(rg_v7 , 0x2 );
        rg_k4  = __shfl_xor(rg_k4 , 0x4 );
        rg_k5  = __shfl_xor(rg_k5 , 0x4 );
        rg_k6  = __shfl_xor(rg_k6 , 0x4 );
        rg_k7  = __shfl_xor(rg_k7 , 0x4 );
        rg_v4  = __shfl_xor(rg_v4 , 0x4 );
        rg_v5  = __shfl_xor(rg_v5 , 0x4 );
        rg_v6  = __shfl_xor(rg_v6 , 0x4 );
        rg_v7  = __shfl_xor(rg_v7 , 0x4 );
        if(lane_id&0x4 ) SWP(uintT, rg_k0 , rg_k4 , int, rg_v0 , rg_v4 );
        if(lane_id&0x4 ) SWP(uintT, rg_k1 , rg_k5 , int, rg_v1 , rg_v5 );
        if(lane_id&0x4 ) SWP(uintT, rg_k2 , rg_k6 , int, rg_v2 , rg_v6 );
        if(lane_id&0x4 ) SWP(uintT, rg_k3 , rg_k7 , int, rg_v3 , rg_v7 );
        rg_k4  = __shfl_xor(rg_k4 , 0x4 );
        rg_k5  = __shfl_xor(rg_k5 , 0x4 );
        rg_k6  = __shfl_xor(rg_k6 , 0x4 );
        rg_k7  = __shfl_xor(rg_k7 , 0x4 );
        rg_v4  = __shfl_xor(rg_v4 , 0x4 );
        rg_v5  = __shfl_xor(rg_v5 , 0x4 );
        rg_v6  = __shfl_xor(rg_v6 , 0x4 );
        rg_v7  = __shfl_xor(rg_v7 , 0x4 );
        rg_k1  = __shfl_xor(rg_k1 , 0x8 );
        rg_k3  = __shfl_xor(rg_k3 , 0x8 );
        rg_k5  = __shfl_xor(rg_k5 , 0x8 );
        rg_k7  = __shfl_xor(rg_k7 , 0x8 );
        rg_v1  = __shfl_xor(rg_v1 , 0x8 );
        rg_v3  = __shfl_xor(rg_v3 , 0x8 );
        rg_v5  = __shfl_xor(rg_v5 , 0x8 );
        rg_v7  = __shfl_xor(rg_v7 , 0x8 );
        if(lane_id&0x8 ) SWP(uintT, rg_k0 , rg_k1 , int, rg_v0 , rg_v1 );
        if(lane_id&0x8 ) SWP(uintT, rg_k2 , rg_k3 , int, rg_v2 , rg_v3 );
        if(lane_id&0x8 ) SWP(uintT, rg_k4 , rg_k5 , int, rg_v4 , rg_v5 );
        if(lane_id&0x8 ) SWP(uintT, rg_k6 , rg_k7 , int, rg_v6 , rg_v7 );
        rg_k1  = __shfl_xor(rg_k1 , 0x8 );
        rg_k3  = __shfl_xor(rg_k3 , 0x8 );
        rg_k5  = __shfl_xor(rg_k5 , 0x8 );
        rg_k7  = __shfl_xor(rg_k7 , 0x8 );
        rg_v1  = __shfl_xor(rg_v1 , 0x8 );
        rg_v3  = __shfl_xor(rg_v3 , 0x8 );
        rg_v5  = __shfl_xor(rg_v5 , 0x8 );
        rg_v7  = __shfl_xor(rg_v7 , 0x8 );
        rg_k2  = __shfl_xor(rg_k2 , 0x10);
        rg_k3  = __shfl_xor(rg_k3 , 0x10);
        rg_k6  = __shfl_xor(rg_k6 , 0x10);
        rg_k7  = __shfl_xor(rg_k7 , 0x10);
        rg_v2  = __shfl_xor(rg_v2 , 0x10);
        rg_v3  = __shfl_xor(rg_v3 , 0x10);
        rg_v6  = __shfl_xor(rg_v6 , 0x10);
        rg_v7  = __shfl_xor(rg_v7 , 0x10);
        if(lane_id&0x10) SWP(uintT, rg_k0 , rg_k2 , int, rg_v0 , rg_v2 );
        if(lane_id&0x10) SWP(uintT, rg_k1 , rg_k3 , int, rg_v1 , rg_v3 );
        if(lane_id&0x10) SWP(uintT, rg_k4 , rg_k6 , int, rg_v4 , rg_v6 );
        if(lane_id&0x10) SWP(uintT, rg_k5 , rg_k7 , int, rg_v5 , rg_v7 );
        rg_k2  = __shfl_xor(rg_k2 , 0x10);
        rg_k3  = __shfl_xor(rg_k3 , 0x10);
        rg_k6  = __shfl_xor(rg_k6 , 0x10);
        rg_k7  = __shfl_xor(rg_k7 , 0x10);
        rg_v2  = __shfl_xor(rg_v2 , 0x10);
        rg_v3  = __shfl_xor(rg_v3 , 0x10);
        rg_v6  = __shfl_xor(rg_v6 , 0x10);
        rg_v7  = __shfl_xor(rg_v7 , 0x10);
        int kk;
        int ss;
        kk = __shfl(k, 0 );
        ss = __shfl(seg_size, 0 );
        if(lane_id+0  <ss) keyB[kk+lane_id+0  ] = rg_k0 ;
        if(lane_id+32 <ss) keyB[kk+lane_id+32 ] = rg_k4 ;
        if(lane_id+64 <ss) keyB[kk+lane_id+64 ] = rg_k1 ;
        if(lane_id+96 <ss) keyB[kk+lane_id+96 ] = rg_k5 ;
        if(lane_id+128<ss) keyB[kk+lane_id+128] = rg_k2 ;
        if(lane_id+160<ss) keyB[kk+lane_id+160] = rg_k6 ;
        if(lane_id+192<ss) keyB[kk+lane_id+192] = rg_k3 ;
        if(lane_id+224<ss) keyB[kk+lane_id+224] = rg_k7 ;
        kk = __shfl(k, 0 );
        ss = __shfl(seg_size, 0 );
        if(lane_id+0  <ss) valB[kk+lane_id+0  ] = val[kk+rg_v0 ];
        if(lane_id+32 <ss) valB[kk+lane_id+32 ] = val[kk+rg_v4 ];
        if(lane_id+64 <ss) valB[kk+lane_id+64 ] = val[kk+rg_v1 ];
        if(lane_id+96 <ss) valB[kk+lane_id+96 ] = val[kk+rg_v5 ];
        if(lane_id+128<ss) valB[kk+lane_id+128] = val[kk+rg_v2 ];
        if(lane_id+160<ss) valB[kk+lane_id+160] = val[kk+rg_v6 ];
        if(lane_id+192<ss) valB[kk+lane_id+192] = val[kk+rg_v3 ];
        if(lane_id+224<ss) valB[kk+lane_id+224] = val[kk+rg_v7 ];
    } else if(bin_it < bin_size) {
        if((tid<<3)+0 <seg_size) keyB[k+(tid<<3)+0 ] = rg_k0 ;
        if((tid<<3)+1 <seg_size) keyB[k+(tid<<3)+1 ] = rg_k1 ;
        if((tid<<3)+2 <seg_size) keyB[k+(tid<<3)+2 ] = rg_k2 ;
        if((tid<<3)+3 <seg_size) keyB[k+(tid<<3)+3 ] = rg_k3 ;
        if((tid<<3)+4 <seg_size) keyB[k+(tid<<3)+4 ] = rg_k4 ;
        if((tid<<3)+5 <seg_size) keyB[k+(tid<<3)+5 ] = rg_k5 ;
        if((tid<<3)+6 <seg_size) keyB[k+(tid<<3)+6 ] = rg_k6 ;
        if((tid<<3)+7 <seg_size) keyB[k+(tid<<3)+7 ] = rg_k7 ;
        if((tid<<3)+0 <seg_size) valB[k+(tid<<3)+0 ] = val[k+rg_v0 ];
        if((tid<<3)+1 <seg_size) valB[k+(tid<<3)+1 ] = val[k+rg_v1 ];
        if((tid<<3)+2 <seg_size) valB[k+(tid<<3)+2 ] = val[k+rg_v2 ];
        if((tid<<3)+3 <seg_size) valB[k+(tid<<3)+3 ] = val[k+rg_v3 ];
        if((tid<<3)+4 <seg_size) valB[k+(tid<<3)+4 ] = val[k+rg_v4 ];
        if((tid<<3)+5 <seg_size) valB[k+(tid<<3)+5 ] = val[k+rg_v5 ];
        if((tid<<3)+6 <seg_size) valB[k+(tid<<3)+6 ] = val[k+rg_v6 ];
        if((tid<<3)+7 <seg_size) valB[k+(tid<<3)+7 ] = val[k+rg_v7 ];
    }
}
/* block tcf1 tcf2 quiet real_kern */
/*   128    2    4  true      true */
template<class T>
__global__
void gen_bk128_tc4_r257_r512_orig(
    uintT *key, T *val, uintT *keyB, T *valB, int n, int *segs, int *bin, int bin_size, int length) {

    const int tid = threadIdx.x;
    const int bin_it = blockIdx.x;
    __shared__ uintT smem[512];
    __shared__ int tmem[512];
    const int bit1 = (tid>>0)&0x1;
    const int bit2 = (tid>>1)&0x1;
    const int bit3 = (tid>>2)&0x1;
    const int bit4 = (tid>>3)&0x1;
    const int bit5 = (tid>>4)&0x1;
    const int tid1 = threadIdx.x & 31;
    const int warp_id = threadIdx.x / 32;
    uintT rg_k0 ;
    uintT rg_k1 ;
    uintT rg_k2 ;
    uintT rg_k3 ;
    int rg_v0 ;
    int rg_v1 ;
    int rg_v2 ;
    int rg_v3 ;
    int k;
    int seg_size;
    int ext_seg_size;
    if(bin_it < bin_size) {
        k = segs[bin[bin_it]];
        seg_size = ((bin[bin_it]==length-1)?n:segs[bin[bin_it]+1])-segs[bin[bin_it]];
        ext_seg_size = ((seg_size + 63) / 64) * 64;
        int big_wp = (ext_seg_size - blockDim.x * 2) / 64;
        int sml_wp = blockDim.x / 32 - big_wp;
        int sml_len = sml_wp * 64;
        const int big_warp_id = (warp_id - sml_wp < 0)? 0: warp_id - sml_wp;
        bool sml_warp = warp_id < sml_wp;
        if(sml_warp) {
            rg_k0 = key[k+(warp_id<<6)+tid1+0   ];
            rg_k1 = key[k+(warp_id<<6)+tid1+32  ];
            rg_v0 = (warp_id<<6)+tid1+0   ;
            rg_v1 = (warp_id<<6)+tid1+32  ;
            // exch_intxn: switch to exch_local()
            CMP_SWP(uintT,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
            // exch_intxn: generate exch_intxn()
            exch_intxn(rg_k0 ,rg_k1 ,
                       rg_v0 ,rg_v1 ,
                       0x1,bit1);
            // exch_paral: switch to exch_local()
            CMP_SWP(uintT,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
            // exch_intxn: generate exch_intxn()
            exch_intxn(rg_k0 ,rg_k1 ,
                       rg_v0 ,rg_v1 ,
                       0x3,bit2);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,
                       rg_v0 ,rg_v1 ,
                       0x1,bit1);
            // exch_paral: switch to exch_local()
            CMP_SWP(uintT,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
            // exch_intxn: generate exch_intxn()
            exch_intxn(rg_k0 ,rg_k1 ,
                       rg_v0 ,rg_v1 ,
                       0x7,bit3);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,
                       rg_v0 ,rg_v1 ,
                       0x2,bit2);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,
                       rg_v0 ,rg_v1 ,
                       0x1,bit1);
            // exch_paral: switch to exch_local()
            CMP_SWP(uintT,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
            // exch_intxn: generate exch_intxn()
            exch_intxn(rg_k0 ,rg_k1 ,
                       rg_v0 ,rg_v1 ,
                       0xf,bit4);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,
                       rg_v0 ,rg_v1 ,
                       0x4,bit3);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,
                       rg_v0 ,rg_v1 ,
                       0x2,bit2);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,
                       rg_v0 ,rg_v1 ,
                       0x1,bit1);
            // exch_paral: switch to exch_local()
            CMP_SWP(uintT,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
            // exch_intxn: generate exch_intxn()
            exch_intxn(rg_k0 ,rg_k1 ,
                       rg_v0 ,rg_v1 ,
                       0x1f,bit5);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,
                       rg_v0 ,rg_v1 ,
                       0x8,bit4);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,
                       rg_v0 ,rg_v1 ,
                       0x4,bit3);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,
                       rg_v0 ,rg_v1 ,
                       0x2,bit2);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,
                       rg_v0 ,rg_v1 ,
                       0x1,bit1);
            // exch_paral: switch to exch_local()
            CMP_SWP(uintT,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
        } else {
            rg_k0  = (sml_len+tid1+(big_warp_id<<7)+0   <seg_size)?key[k+sml_len+tid1+(big_warp_id<<7)+0   ]:INT_MAX;
            rg_k1  = (sml_len+tid1+(big_warp_id<<7)+32  <seg_size)?key[k+sml_len+tid1+(big_warp_id<<7)+32  ]:INT_MAX;
            rg_k2  = (sml_len+tid1+(big_warp_id<<7)+64  <seg_size)?key[k+sml_len+tid1+(big_warp_id<<7)+64  ]:INT_MAX;
            rg_k3  = (sml_len+tid1+(big_warp_id<<7)+96  <seg_size)?key[k+sml_len+tid1+(big_warp_id<<7)+96  ]:INT_MAX;
            if(sml_len+tid1+(big_warp_id<<7)+0   <seg_size) rg_v0  = sml_len+tid1+(big_warp_id<<7)+0   ;
            if(sml_len+tid1+(big_warp_id<<7)+32  <seg_size) rg_v1  = sml_len+tid1+(big_warp_id<<7)+32  ;
            if(sml_len+tid1+(big_warp_id<<7)+64  <seg_size) rg_v2  = sml_len+tid1+(big_warp_id<<7)+64  ;
            if(sml_len+tid1+(big_warp_id<<7)+96  <seg_size) rg_v3  = sml_len+tid1+(big_warp_id<<7)+96  ;
            // exch_intxn: switch to exch_local()
            CMP_SWP(uintT,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
            CMP_SWP(uintT,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
            // exch_intxn: switch to exch_local()
            CMP_SWP(uintT,rg_k0 ,rg_k3 ,int,rg_v0 ,rg_v3 );
            CMP_SWP(uintT,rg_k1 ,rg_k2 ,int,rg_v1 ,rg_v2 );
            // exch_paral: switch to exch_local()
            CMP_SWP(uintT,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
            CMP_SWP(uintT,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
            // exch_intxn: generate exch_intxn()
            exch_intxn(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                       rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                       0x1,bit1);
            // exch_paral: switch to exch_local()
            CMP_SWP(uintT,rg_k0 ,rg_k2 ,int,rg_v0 ,rg_v2 );
            CMP_SWP(uintT,rg_k1 ,rg_k3 ,int,rg_v1 ,rg_v3 );
            // exch_paral: switch to exch_local()
            CMP_SWP(uintT,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
            CMP_SWP(uintT,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
            // exch_intxn: generate exch_intxn()
            exch_intxn(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                       rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                       0x3,bit2);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                       rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                       0x1,bit1);
            // exch_paral: switch to exch_local()
            CMP_SWP(uintT,rg_k0 ,rg_k2 ,int,rg_v0 ,rg_v2 );
            CMP_SWP(uintT,rg_k1 ,rg_k3 ,int,rg_v1 ,rg_v3 );
            // exch_paral: switch to exch_local()
            CMP_SWP(uintT,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
            CMP_SWP(uintT,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
            // exch_intxn: generate exch_intxn()
            exch_intxn(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                       rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                       0x7,bit3);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                       rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                       0x2,bit2);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                       rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                       0x1,bit1);
            // exch_paral: switch to exch_local()
            CMP_SWP(uintT,rg_k0 ,rg_k2 ,int,rg_v0 ,rg_v2 );
            CMP_SWP(uintT,rg_k1 ,rg_k3 ,int,rg_v1 ,rg_v3 );
            // exch_paral: switch to exch_local()
            CMP_SWP(uintT,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
            CMP_SWP(uintT,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
            // exch_intxn: generate exch_intxn()
            exch_intxn(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                       rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                       0xf,bit4);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                       rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                       0x4,bit3);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                       rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                       0x2,bit2);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                       rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                       0x1,bit1);
            // exch_paral: switch to exch_local()
            CMP_SWP(uintT,rg_k0 ,rg_k2 ,int,rg_v0 ,rg_v2 );
            CMP_SWP(uintT,rg_k1 ,rg_k3 ,int,rg_v1 ,rg_v3 );
            // exch_paral: switch to exch_local()
            CMP_SWP(uintT,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
            CMP_SWP(uintT,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
            // exch_intxn: generate exch_intxn()
            exch_intxn(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                       rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                       0x1f,bit5);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                       rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                       0x8,bit4);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                       rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                       0x4,bit3);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                       rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                       0x2,bit2);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                       rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                       0x1,bit1);
            // exch_paral: switch to exch_local()
            CMP_SWP(uintT,rg_k0 ,rg_k2 ,int,rg_v0 ,rg_v2 );
            CMP_SWP(uintT,rg_k1 ,rg_k3 ,int,rg_v1 ,rg_v3 );
            // exch_paral: switch to exch_local()
            CMP_SWP(uintT,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
            CMP_SWP(uintT,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
        }
        // Store register results to shared memory
        if(sml_warp) {
            smem[(warp_id<<6)+(tid1<<1)+0 ] = rg_k0 ;
            smem[(warp_id<<6)+(tid1<<1)+1 ] = rg_k1 ;
            tmem[(warp_id<<6)+(tid1<<1)+0 ] = rg_v0 ;
            tmem[(warp_id<<6)+(tid1<<1)+1 ] = rg_v1 ;
        } else {
            smem[sml_len+(big_warp_id<<7)+(tid1<<2)+0 ] = rg_k0 ;
            smem[sml_len+(big_warp_id<<7)+(tid1<<2)+1 ] = rg_k1 ;
            smem[sml_len+(big_warp_id<<7)+(tid1<<2)+2 ] = rg_k2 ;
            smem[sml_len+(big_warp_id<<7)+(tid1<<2)+3 ] = rg_k3 ;
            tmem[sml_len+(big_warp_id<<7)+(tid1<<2)+0 ] = rg_v0 ;
            tmem[sml_len+(big_warp_id<<7)+(tid1<<2)+1 ] = rg_v1 ;
            tmem[sml_len+(big_warp_id<<7)+(tid1<<2)+2 ] = rg_v2 ;
            tmem[sml_len+(big_warp_id<<7)+(tid1<<2)+3 ] = rg_v3 ;
        }
        __syncthreads();
        // Merge in 2 steps
        int grp_start_wp_id;
        int grp_start_off;
        int tmp_wp_id;
        int lhs_len;
        int rhs_len;
        int gran;
        int s_a;
        int s_b;
        bool p;
        uintT tmp_k0;
        uintT tmp_k1;
        int tmp_v0;
        int tmp_v1;
        uintT *start;
        // Step 0
        grp_start_wp_id = ((warp_id>>1)<<1);
        grp_start_off = (grp_start_wp_id<sml_wp)?grp_start_wp_id*64:sml_len+(grp_start_wp_id-sml_wp)*128;
        tmp_wp_id = grp_start_wp_id;
        lhs_len = ((tmp_wp_id+0<sml_wp)?64  :128 );
        rhs_len = ((tmp_wp_id+1<sml_wp)?64  :128 );
        gran = (warp_id<sml_wp)?(tid1<<1): (tid1<<2);
        if((warp_id&1)==0){
            gran += 0;
        }
        if((warp_id&1)==1){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 );
        }
        start = smem + grp_start_off;
        s_a = find_kth3(start, lhs_len, start+lhs_len, rhs_len, gran);
        s_b = lhs_len + gran - s_a;
        if(sml_warp){
            tmp_k0 = (s_a<lhs_len        )?smem[grp_start_off+s_a]:INT_MAX;
            tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
            if(s_a<lhs_len        ) tmp_v0 = tmem[grp_start_off+s_a];
            if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k0 = p ? tmp_k0 : tmp_k1;
            rg_v0 = p ? tmp_v0 : tmp_v1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:INT_MAX;
                if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
                if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k1 = p ? tmp_k0 : tmp_k1;
            rg_v1 = p ? tmp_v0 : tmp_v1;
        } else {
            tmp_k0 = (s_a<lhs_len        )?smem[grp_start_off+s_a]:INT_MAX;
            tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
            if(s_a<lhs_len        ) tmp_v0 = tmem[grp_start_off+s_a];
            if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k0 = p ? tmp_k0 : tmp_k1;
            rg_v0 = p ? tmp_v0 : tmp_v1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:INT_MAX;
                if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
                if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k1 = p ? tmp_k0 : tmp_k1;
            rg_v1 = p ? tmp_v0 : tmp_v1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:INT_MAX;
                if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
                if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k2 = p ? tmp_k0 : tmp_k1;
            rg_v2 = p ? tmp_v0 : tmp_v1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:INT_MAX;
                if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
                if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k3 = p ? tmp_k0 : tmp_k1;
            rg_v3 = p ? tmp_v0 : tmp_v1;
        }
        __syncthreads();
        // Store merged results back to shared memory
        if(sml_warp){
            smem[grp_start_off+gran+0 ] = rg_k0 ;
            smem[grp_start_off+gran+1 ] = rg_k1 ;
            tmem[grp_start_off+gran+0 ] = rg_v0 ;
            tmem[grp_start_off+gran+1 ] = rg_v1 ;
        } else {
            smem[grp_start_off+gran+0 ] = rg_k0 ;
            smem[grp_start_off+gran+1 ] = rg_k1 ;
            smem[grp_start_off+gran+2 ] = rg_k2 ;
            smem[grp_start_off+gran+3 ] = rg_k3 ;
            tmem[grp_start_off+gran+0 ] = rg_v0 ;
            tmem[grp_start_off+gran+1 ] = rg_v1 ;
            tmem[grp_start_off+gran+2 ] = rg_v2 ;
            tmem[grp_start_off+gran+3 ] = rg_v3 ;
        }
        __syncthreads();
        // Step 1
        grp_start_wp_id = ((warp_id>>2)<<2);
        grp_start_off = (grp_start_wp_id<sml_wp)?grp_start_wp_id*64:sml_len+(grp_start_wp_id-sml_wp)*128;
        tmp_wp_id = grp_start_wp_id;
        lhs_len = ((tmp_wp_id+0<sml_wp)?64  :128 )+
                  ((tmp_wp_id+1<sml_wp)?64  :128 );
        rhs_len = ((tmp_wp_id+2<sml_wp)?64  :128 )+
                  ((tmp_wp_id+3<sml_wp)?64  :128 );
        gran = (warp_id<sml_wp)?(tid1<<1): (tid1<<2);
        if((warp_id&3)==0){
            gran += 0;
        }
        if((warp_id&3)==1){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 );
        }
        if((warp_id&3)==2){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 )+
                    ((tmp_wp_id+1<sml_wp)?64  :128 );
        }
        if((warp_id&3)==3){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 )+
                    ((tmp_wp_id+1<sml_wp)?64  :128 )+
                    ((tmp_wp_id+2<sml_wp)?64  :128 );
        }
        start = smem + grp_start_off;
        s_a = find_kth3(start, lhs_len, start+lhs_len, rhs_len, gran);
        s_b = lhs_len + gran - s_a;
        if(sml_warp){
            tmp_k0 = (s_a<lhs_len        )?smem[grp_start_off+s_a]:INT_MAX;
            tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
            if(s_a<lhs_len        ) tmp_v0 = tmem[grp_start_off+s_a];
            if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k0 = p ? tmp_k0 : tmp_k1;
            rg_v0 = p ? tmp_v0 : tmp_v1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:INT_MAX;
                if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
                if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k1 = p ? tmp_k0 : tmp_k1;
            rg_v1 = p ? tmp_v0 : tmp_v1;
        } else {
            tmp_k0 = (s_a<lhs_len        )?smem[grp_start_off+s_a]:INT_MAX;
            tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
            if(s_a<lhs_len        ) tmp_v0 = tmem[grp_start_off+s_a];
            if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k0 = p ? tmp_k0 : tmp_k1;
            rg_v0 = p ? tmp_v0 : tmp_v1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:INT_MAX;
                if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
                if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k1 = p ? tmp_k0 : tmp_k1;
            rg_v1 = p ? tmp_v0 : tmp_v1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:INT_MAX;
                if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
                if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k2 = p ? tmp_k0 : tmp_k1;
            rg_v2 = p ? tmp_v0 : tmp_v1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:INT_MAX;
                if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
                if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k3 = p ? tmp_k0 : tmp_k1;
            rg_v3 = p ? tmp_v0 : tmp_v1;
        }
        if(sml_warp){
        } else {
        }
        if(sml_warp){
            if((tid<<1)+0 <seg_size) keyB[k+(tid<<1)+0 ] = rg_k0 ;
            if((tid<<1)+1 <seg_size) keyB[k+(tid<<1)+1 ] = rg_k1 ;
            if((tid<<1)+0 <seg_size) valB[k+(tid<<1)+0 ] = val[k+rg_v0 ];
            if((tid<<1)+1 <seg_size) valB[k+(tid<<1)+1 ] = val[k+rg_v1 ];
        } else {
            if((tid<<2)+0 -sml_len<seg_size) keyB[k+(tid<<2)+0 -sml_len] = rg_k0 ;
            if((tid<<2)+1 -sml_len<seg_size) keyB[k+(tid<<2)+1 -sml_len] = rg_k1 ;
            if((tid<<2)+2 -sml_len<seg_size) keyB[k+(tid<<2)+2 -sml_len] = rg_k2 ;
            if((tid<<2)+3 -sml_len<seg_size) keyB[k+(tid<<2)+3 -sml_len] = rg_k3 ;
            if((tid<<2)+0 -sml_len<seg_size) valB[k+(tid<<2)+0 -sml_len] = val[k+rg_v0 ];
            if((tid<<2)+1 -sml_len<seg_size) valB[k+(tid<<2)+1 -sml_len] = val[k+rg_v1 ];
            if((tid<<2)+2 -sml_len<seg_size) valB[k+(tid<<2)+2 -sml_len] = val[k+rg_v2 ];
            if((tid<<2)+3 -sml_len<seg_size) valB[k+(tid<<2)+3 -sml_len] = val[k+rg_v3 ];
        }
    }
}
/* block tcf1 tcf2 quiet real_kern */
/*   256    2    4  true      true */
template<class T>
__global__
void gen_bk256_tc4_r513_r1024_orig(
    uintT *key, T *val, uintT *keyB, T *valB, int n, int *segs, int *bin, int bin_size, int length) {

    const int tid = threadIdx.x;
    const int bin_it = blockIdx.x;
    __shared__ uintT smem[1024];
    __shared__ int tmem[1024];
    const int bit1 = (tid>>0)&0x1;
    const int bit2 = (tid>>1)&0x1;
    const int bit3 = (tid>>2)&0x1;
    const int bit4 = (tid>>3)&0x1;
    const int bit5 = (tid>>4)&0x1;
    const int tid1 = threadIdx.x & 31;
    const int warp_id = threadIdx.x / 32;
    uintT rg_k0 ;
    uintT rg_k1 ;
    uintT rg_k2 ;
    uintT rg_k3 ;
    int rg_v0 ;
    int rg_v1 ;
    int rg_v2 ;
    int rg_v3 ;
    int k;
    int seg_size;
    int ext_seg_size;
    if(bin_it < bin_size) {
        k = segs[bin[bin_it]];
        seg_size = ((bin[bin_it]==length-1)?n:segs[bin[bin_it]+1])-segs[bin[bin_it]];
        ext_seg_size = ((seg_size + 63) / 64) * 64;
        int big_wp = (ext_seg_size - blockDim.x * 2) / 64;
        int sml_wp = blockDim.x / 32 - big_wp;
        int sml_len = sml_wp * 64;
        const int big_warp_id = (warp_id - sml_wp < 0)? 0: warp_id - sml_wp;
        bool sml_warp = warp_id < sml_wp;
        if(sml_warp) {
            rg_k0 = key[k+(warp_id<<6)+tid1+0   ];
            rg_k1 = key[k+(warp_id<<6)+tid1+32  ];
            rg_v0 = (warp_id<<6)+tid1+0   ;
            rg_v1 = (warp_id<<6)+tid1+32  ;
            // exch_intxn: switch to exch_local()
            CMP_SWP(uintT,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
            // exch_intxn: generate exch_intxn()
            exch_intxn(rg_k0 ,rg_k1 ,
                       rg_v0 ,rg_v1 ,
                       0x1,bit1);
            // exch_paral: switch to exch_local()
            CMP_SWP(uintT,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
            // exch_intxn: generate exch_intxn()
            exch_intxn(rg_k0 ,rg_k1 ,
                       rg_v0 ,rg_v1 ,
                       0x3,bit2);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,
                       rg_v0 ,rg_v1 ,
                       0x1,bit1);
            // exch_paral: switch to exch_local()
            CMP_SWP(uintT,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
            // exch_intxn: generate exch_intxn()
            exch_intxn(rg_k0 ,rg_k1 ,
                       rg_v0 ,rg_v1 ,
                       0x7,bit3);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,
                       rg_v0 ,rg_v1 ,
                       0x2,bit2);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,
                       rg_v0 ,rg_v1 ,
                       0x1,bit1);
            // exch_paral: switch to exch_local()
            CMP_SWP(uintT,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
            // exch_intxn: generate exch_intxn()
            exch_intxn(rg_k0 ,rg_k1 ,
                       rg_v0 ,rg_v1 ,
                       0xf,bit4);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,
                       rg_v0 ,rg_v1 ,
                       0x4,bit3);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,
                       rg_v0 ,rg_v1 ,
                       0x2,bit2);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,
                       rg_v0 ,rg_v1 ,
                       0x1,bit1);
            // exch_paral: switch to exch_local()
            CMP_SWP(uintT,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
            // exch_intxn: generate exch_intxn()
            exch_intxn(rg_k0 ,rg_k1 ,
                       rg_v0 ,rg_v1 ,
                       0x1f,bit5);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,
                       rg_v0 ,rg_v1 ,
                       0x8,bit4);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,
                       rg_v0 ,rg_v1 ,
                       0x4,bit3);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,
                       rg_v0 ,rg_v1 ,
                       0x2,bit2);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,
                       rg_v0 ,rg_v1 ,
                       0x1,bit1);
            // exch_paral: switch to exch_local()
            CMP_SWP(uintT,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
        } else {
            rg_k0  = (sml_len+tid1+(big_warp_id<<7)+0   <seg_size)?key[k+sml_len+tid1+(big_warp_id<<7)+0   ]:INT_MAX;
            rg_k1  = (sml_len+tid1+(big_warp_id<<7)+32  <seg_size)?key[k+sml_len+tid1+(big_warp_id<<7)+32  ]:INT_MAX;
            rg_k2  = (sml_len+tid1+(big_warp_id<<7)+64  <seg_size)?key[k+sml_len+tid1+(big_warp_id<<7)+64  ]:INT_MAX;
            rg_k3  = (sml_len+tid1+(big_warp_id<<7)+96  <seg_size)?key[k+sml_len+tid1+(big_warp_id<<7)+96  ]:INT_MAX;
            if(sml_len+tid1+(big_warp_id<<7)+0   <seg_size) rg_v0  = sml_len+tid1+(big_warp_id<<7)+0   ;
            if(sml_len+tid1+(big_warp_id<<7)+32  <seg_size) rg_v1  = sml_len+tid1+(big_warp_id<<7)+32  ;
            if(sml_len+tid1+(big_warp_id<<7)+64  <seg_size) rg_v2  = sml_len+tid1+(big_warp_id<<7)+64  ;
            if(sml_len+tid1+(big_warp_id<<7)+96  <seg_size) rg_v3  = sml_len+tid1+(big_warp_id<<7)+96  ;
            // exch_intxn: switch to exch_local()
            CMP_SWP(uintT,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
            CMP_SWP(uintT,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
            // exch_intxn: switch to exch_local()
            CMP_SWP(uintT,rg_k0 ,rg_k3 ,int,rg_v0 ,rg_v3 );
            CMP_SWP(uintT,rg_k1 ,rg_k2 ,int,rg_v1 ,rg_v2 );
            // exch_paral: switch to exch_local()
            CMP_SWP(uintT,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
            CMP_SWP(uintT,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
            // exch_intxn: generate exch_intxn()
            exch_intxn(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                       rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                       0x1,bit1);
            // exch_paral: switch to exch_local()
            CMP_SWP(uintT,rg_k0 ,rg_k2 ,int,rg_v0 ,rg_v2 );
            CMP_SWP(uintT,rg_k1 ,rg_k3 ,int,rg_v1 ,rg_v3 );
            // exch_paral: switch to exch_local()
            CMP_SWP(uintT,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
            CMP_SWP(uintT,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
            // exch_intxn: generate exch_intxn()
            exch_intxn(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                       rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                       0x3,bit2);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                       rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                       0x1,bit1);
            // exch_paral: switch to exch_local()
            CMP_SWP(uintT,rg_k0 ,rg_k2 ,int,rg_v0 ,rg_v2 );
            CMP_SWP(uintT,rg_k1 ,rg_k3 ,int,rg_v1 ,rg_v3 );
            // exch_paral: switch to exch_local()
            CMP_SWP(uintT,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
            CMP_SWP(uintT,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
            // exch_intxn: generate exch_intxn()
            exch_intxn(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                       rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                       0x7,bit3);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                       rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                       0x2,bit2);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                       rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                       0x1,bit1);
            // exch_paral: switch to exch_local()
            CMP_SWP(uintT,rg_k0 ,rg_k2 ,int,rg_v0 ,rg_v2 );
            CMP_SWP(uintT,rg_k1 ,rg_k3 ,int,rg_v1 ,rg_v3 );
            // exch_paral: switch to exch_local()
            CMP_SWP(uintT,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
            CMP_SWP(uintT,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
            // exch_intxn: generate exch_intxn()
            exch_intxn(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                       rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                       0xf,bit4);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                       rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                       0x4,bit3);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                       rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                       0x2,bit2);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                       rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                       0x1,bit1);
            // exch_paral: switch to exch_local()
            CMP_SWP(uintT,rg_k0 ,rg_k2 ,int,rg_v0 ,rg_v2 );
            CMP_SWP(uintT,rg_k1 ,rg_k3 ,int,rg_v1 ,rg_v3 );
            // exch_paral: switch to exch_local()
            CMP_SWP(uintT,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
            CMP_SWP(uintT,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
            // exch_intxn: generate exch_intxn()
            exch_intxn(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                       rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                       0x1f,bit5);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                       rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                       0x8,bit4);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                       rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                       0x4,bit3);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                       rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                       0x2,bit2);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                       rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                       0x1,bit1);
            // exch_paral: switch to exch_local()
            CMP_SWP(uintT,rg_k0 ,rg_k2 ,int,rg_v0 ,rg_v2 );
            CMP_SWP(uintT,rg_k1 ,rg_k3 ,int,rg_v1 ,rg_v3 );
            // exch_paral: switch to exch_local()
            CMP_SWP(uintT,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
            CMP_SWP(uintT,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
        }
        // Store register results to shared memory
        if(sml_warp) {
            smem[(warp_id<<6)+(tid1<<1)+0 ] = rg_k0 ;
            smem[(warp_id<<6)+(tid1<<1)+1 ] = rg_k1 ;
            tmem[(warp_id<<6)+(tid1<<1)+0 ] = rg_v0 ;
            tmem[(warp_id<<6)+(tid1<<1)+1 ] = rg_v1 ;
        } else {
            smem[sml_len+(big_warp_id<<7)+(tid1<<2)+0 ] = rg_k0 ;
            smem[sml_len+(big_warp_id<<7)+(tid1<<2)+1 ] = rg_k1 ;
            smem[sml_len+(big_warp_id<<7)+(tid1<<2)+2 ] = rg_k2 ;
            smem[sml_len+(big_warp_id<<7)+(tid1<<2)+3 ] = rg_k3 ;
            tmem[sml_len+(big_warp_id<<7)+(tid1<<2)+0 ] = rg_v0 ;
            tmem[sml_len+(big_warp_id<<7)+(tid1<<2)+1 ] = rg_v1 ;
            tmem[sml_len+(big_warp_id<<7)+(tid1<<2)+2 ] = rg_v2 ;
            tmem[sml_len+(big_warp_id<<7)+(tid1<<2)+3 ] = rg_v3 ;
        }
        __syncthreads();
        // Merge in 3 steps
        int grp_start_wp_id;
        int grp_start_off;
        int tmp_wp_id;
        int lhs_len;
        int rhs_len;
        int gran;
        int s_a;
        int s_b;
        bool p;
        uintT tmp_k0;
        uintT tmp_k1;
        int tmp_v0;
        int tmp_v1;
        uintT *start;
        // Step 0
        grp_start_wp_id = ((warp_id>>1)<<1);
        grp_start_off = (grp_start_wp_id<sml_wp)?grp_start_wp_id*64:sml_len+(grp_start_wp_id-sml_wp)*128;
        tmp_wp_id = grp_start_wp_id;
        lhs_len = ((tmp_wp_id+0<sml_wp)?64  :128 );
        rhs_len = ((tmp_wp_id+1<sml_wp)?64  :128 );
        gran = (warp_id<sml_wp)?(tid1<<1): (tid1<<2);
        if((warp_id&1)==0){
            gran += 0;
        }
        if((warp_id&1)==1){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 );
        }
        start = smem + grp_start_off;
        s_a = find_kth3(start, lhs_len, start+lhs_len, rhs_len, gran);
        s_b = lhs_len + gran - s_a;
        if(sml_warp){
            tmp_k0 = (s_a<lhs_len        )?smem[grp_start_off+s_a]:INT_MAX;
            tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
            if(s_a<lhs_len        ) tmp_v0 = tmem[grp_start_off+s_a];
            if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k0 = p ? tmp_k0 : tmp_k1;
            rg_v0 = p ? tmp_v0 : tmp_v1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:INT_MAX;
                if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
                if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k1 = p ? tmp_k0 : tmp_k1;
            rg_v1 = p ? tmp_v0 : tmp_v1;
        } else {
            tmp_k0 = (s_a<lhs_len        )?smem[grp_start_off+s_a]:INT_MAX;
            tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
            if(s_a<lhs_len        ) tmp_v0 = tmem[grp_start_off+s_a];
            if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k0 = p ? tmp_k0 : tmp_k1;
            rg_v0 = p ? tmp_v0 : tmp_v1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:INT_MAX;
                if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
                if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k1 = p ? tmp_k0 : tmp_k1;
            rg_v1 = p ? tmp_v0 : tmp_v1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:INT_MAX;
                if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
                if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k2 = p ? tmp_k0 : tmp_k1;
            rg_v2 = p ? tmp_v0 : tmp_v1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:INT_MAX;
                if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
                if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k3 = p ? tmp_k0 : tmp_k1;
            rg_v3 = p ? tmp_v0 : tmp_v1;
        }
        __syncthreads();
        // Store merged results back to shared memory
        if(sml_warp){
            smem[grp_start_off+gran+0 ] = rg_k0 ;
            smem[grp_start_off+gran+1 ] = rg_k1 ;
            tmem[grp_start_off+gran+0 ] = rg_v0 ;
            tmem[grp_start_off+gran+1 ] = rg_v1 ;
        } else {
            smem[grp_start_off+gran+0 ] = rg_k0 ;
            smem[grp_start_off+gran+1 ] = rg_k1 ;
            smem[grp_start_off+gran+2 ] = rg_k2 ;
            smem[grp_start_off+gran+3 ] = rg_k3 ;
            tmem[grp_start_off+gran+0 ] = rg_v0 ;
            tmem[grp_start_off+gran+1 ] = rg_v1 ;
            tmem[grp_start_off+gran+2 ] = rg_v2 ;
            tmem[grp_start_off+gran+3 ] = rg_v3 ;
        }
        __syncthreads();
        // Step 1
        grp_start_wp_id = ((warp_id>>2)<<2);
        grp_start_off = (grp_start_wp_id<sml_wp)?grp_start_wp_id*64:sml_len+(grp_start_wp_id-sml_wp)*128;
        tmp_wp_id = grp_start_wp_id;
        lhs_len = ((tmp_wp_id+0<sml_wp)?64  :128 )+
                  ((tmp_wp_id+1<sml_wp)?64  :128 );
        rhs_len = ((tmp_wp_id+2<sml_wp)?64  :128 )+
                  ((tmp_wp_id+3<sml_wp)?64  :128 );
        gran = (warp_id<sml_wp)?(tid1<<1): (tid1<<2);
        if((warp_id&3)==0){
            gran += 0;
        }
        if((warp_id&3)==1){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 );
        }
        if((warp_id&3)==2){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 )+
                    ((tmp_wp_id+1<sml_wp)?64  :128 );
        }
        if((warp_id&3)==3){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 )+
                    ((tmp_wp_id+1<sml_wp)?64  :128 )+
                    ((tmp_wp_id+2<sml_wp)?64  :128 );
        }
        start = smem + grp_start_off;
        s_a = find_kth3(start, lhs_len, start+lhs_len, rhs_len, gran);
        s_b = lhs_len + gran - s_a;
        if(sml_warp){
            tmp_k0 = (s_a<lhs_len        )?smem[grp_start_off+s_a]:INT_MAX;
            tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
            if(s_a<lhs_len        ) tmp_v0 = tmem[grp_start_off+s_a];
            if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k0 = p ? tmp_k0 : tmp_k1;
            rg_v0 = p ? tmp_v0 : tmp_v1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:INT_MAX;
                if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
                if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k1 = p ? tmp_k0 : tmp_k1;
            rg_v1 = p ? tmp_v0 : tmp_v1;
        } else {
            tmp_k0 = (s_a<lhs_len        )?smem[grp_start_off+s_a]:INT_MAX;
            tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
            if(s_a<lhs_len        ) tmp_v0 = tmem[grp_start_off+s_a];
            if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k0 = p ? tmp_k0 : tmp_k1;
            rg_v0 = p ? tmp_v0 : tmp_v1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:INT_MAX;
                if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
                if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k1 = p ? tmp_k0 : tmp_k1;
            rg_v1 = p ? tmp_v0 : tmp_v1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:INT_MAX;
                if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
                if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k2 = p ? tmp_k0 : tmp_k1;
            rg_v2 = p ? tmp_v0 : tmp_v1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:INT_MAX;
                if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
                if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k3 = p ? tmp_k0 : tmp_k1;
            rg_v3 = p ? tmp_v0 : tmp_v1;
        }
        __syncthreads();
        // Store merged results back to shared memory
        if(sml_warp){
            smem[grp_start_off+gran+0 ] = rg_k0 ;
            smem[grp_start_off+gran+1 ] = rg_k1 ;
            tmem[grp_start_off+gran+0 ] = rg_v0 ;
            tmem[grp_start_off+gran+1 ] = rg_v1 ;
        } else {
            smem[grp_start_off+gran+0 ] = rg_k0 ;
            smem[grp_start_off+gran+1 ] = rg_k1 ;
            smem[grp_start_off+gran+2 ] = rg_k2 ;
            smem[grp_start_off+gran+3 ] = rg_k3 ;
            tmem[grp_start_off+gran+0 ] = rg_v0 ;
            tmem[grp_start_off+gran+1 ] = rg_v1 ;
            tmem[grp_start_off+gran+2 ] = rg_v2 ;
            tmem[grp_start_off+gran+3 ] = rg_v3 ;
        }
        __syncthreads();
        // Step 2
        grp_start_wp_id = ((warp_id>>3)<<3);
        grp_start_off = (grp_start_wp_id<sml_wp)?grp_start_wp_id*64:sml_len+(grp_start_wp_id-sml_wp)*128;
        tmp_wp_id = grp_start_wp_id;
        lhs_len = ((tmp_wp_id+0<sml_wp)?64  :128 )+
                  ((tmp_wp_id+1<sml_wp)?64  :128 )+
                  ((tmp_wp_id+2<sml_wp)?64  :128 )+
                  ((tmp_wp_id+3<sml_wp)?64  :128 );
        rhs_len = ((tmp_wp_id+4<sml_wp)?64  :128 )+
                  ((tmp_wp_id+5<sml_wp)?64  :128 )+
                  ((tmp_wp_id+6<sml_wp)?64  :128 )+
                  ((tmp_wp_id+7<sml_wp)?64  :128 );
        gran = (warp_id<sml_wp)?(tid1<<1): (tid1<<2);
        if((warp_id&7)==0){
            gran += 0;
        }
        if((warp_id&7)==1){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 );
        }
        if((warp_id&7)==2){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 )+
                    ((tmp_wp_id+1<sml_wp)?64  :128 );
        }
        if((warp_id&7)==3){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 )+
                    ((tmp_wp_id+1<sml_wp)?64  :128 )+
                    ((tmp_wp_id+2<sml_wp)?64  :128 );
        }
        if((warp_id&7)==4){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 )+
                    ((tmp_wp_id+1<sml_wp)?64  :128 )+
                    ((tmp_wp_id+2<sml_wp)?64  :128 )+
                    ((tmp_wp_id+3<sml_wp)?64  :128 );
        }
        if((warp_id&7)==5){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 )+
                    ((tmp_wp_id+1<sml_wp)?64  :128 )+
                    ((tmp_wp_id+2<sml_wp)?64  :128 )+
                    ((tmp_wp_id+3<sml_wp)?64  :128 )+
                    ((tmp_wp_id+4<sml_wp)?64  :128 );
        }
        if((warp_id&7)==6){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 )+
                    ((tmp_wp_id+1<sml_wp)?64  :128 )+
                    ((tmp_wp_id+2<sml_wp)?64  :128 )+
                    ((tmp_wp_id+3<sml_wp)?64  :128 )+
                    ((tmp_wp_id+4<sml_wp)?64  :128 )+
                    ((tmp_wp_id+5<sml_wp)?64  :128 );
        }
        if((warp_id&7)==7){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 )+
                    ((tmp_wp_id+1<sml_wp)?64  :128 )+
                    ((tmp_wp_id+2<sml_wp)?64  :128 )+
                    ((tmp_wp_id+3<sml_wp)?64  :128 )+
                    ((tmp_wp_id+4<sml_wp)?64  :128 )+
                    ((tmp_wp_id+5<sml_wp)?64  :128 )+
                    ((tmp_wp_id+6<sml_wp)?64  :128 );
        }
        start = smem + grp_start_off;
        s_a = find_kth3(start, lhs_len, start+lhs_len, rhs_len, gran);
        s_b = lhs_len + gran - s_a;
        if(sml_warp){
            tmp_k0 = (s_a<lhs_len        )?smem[grp_start_off+s_a]:INT_MAX;
            tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
            if(s_a<lhs_len        ) tmp_v0 = tmem[grp_start_off+s_a];
            if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k0 = p ? tmp_k0 : tmp_k1;
            rg_v0 = p ? tmp_v0 : tmp_v1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:INT_MAX;
                if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
                if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k1 = p ? tmp_k0 : tmp_k1;
            rg_v1 = p ? tmp_v0 : tmp_v1;
        } else {
            tmp_k0 = (s_a<lhs_len        )?smem[grp_start_off+s_a]:INT_MAX;
            tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
            if(s_a<lhs_len        ) tmp_v0 = tmem[grp_start_off+s_a];
            if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k0 = p ? tmp_k0 : tmp_k1;
            rg_v0 = p ? tmp_v0 : tmp_v1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:INT_MAX;
                if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
                if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k1 = p ? tmp_k0 : tmp_k1;
            rg_v1 = p ? tmp_v0 : tmp_v1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:INT_MAX;
                if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
                if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k2 = p ? tmp_k0 : tmp_k1;
            rg_v2 = p ? tmp_v0 : tmp_v1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:INT_MAX;
                if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
                if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k3 = p ? tmp_k0 : tmp_k1;
            rg_v3 = p ? tmp_v0 : tmp_v1;
        }
        if(sml_warp){
        } else {
        }
        if(sml_warp){
            if((tid<<1)+0 <seg_size) keyB[k+(tid<<1)+0 ] = rg_k0 ;
            if((tid<<1)+1 <seg_size) keyB[k+(tid<<1)+1 ] = rg_k1 ;
            if((tid<<1)+0 <seg_size) valB[k+(tid<<1)+0 ] = val[k+rg_v0 ];
            if((tid<<1)+1 <seg_size) valB[k+(tid<<1)+1 ] = val[k+rg_v1 ];
        } else {
            if((tid<<2)+0 -sml_len<seg_size) keyB[k+(tid<<2)+0 -sml_len] = rg_k0 ;
            if((tid<<2)+1 -sml_len<seg_size) keyB[k+(tid<<2)+1 -sml_len] = rg_k1 ;
            if((tid<<2)+2 -sml_len<seg_size) keyB[k+(tid<<2)+2 -sml_len] = rg_k2 ;
            if((tid<<2)+3 -sml_len<seg_size) keyB[k+(tid<<2)+3 -sml_len] = rg_k3 ;
            if((tid<<2)+0 -sml_len<seg_size) valB[k+(tid<<2)+0 -sml_len] = val[k+rg_v0 ];
            if((tid<<2)+1 -sml_len<seg_size) valB[k+(tid<<2)+1 -sml_len] = val[k+rg_v1 ];
            if((tid<<2)+2 -sml_len<seg_size) valB[k+(tid<<2)+2 -sml_len] = val[k+rg_v2 ];
            if((tid<<2)+3 -sml_len<seg_size) valB[k+(tid<<2)+3 -sml_len] = val[k+rg_v3 ];
        }
    }
}
/* block tcf1 tcf2 quiet real_kern */
/*   512    2    4  true      true */
template<class T>
__global__
void gen_bk512_tc4_r1025_r2048_orig(
    uintT *key, T *val, uintT *keyB, T *valB, int n, int *segs, int *bin, int bin_size, int length) {

    const int tid = threadIdx.x;
    const int bin_it = blockIdx.x;
    __shared__ uintT smem[2048];
    __shared__ int tmem[2048];
    const int bit1 = (tid>>0)&0x1;
    const int bit2 = (tid>>1)&0x1;
    const int bit3 = (tid>>2)&0x1;
    const int bit4 = (tid>>3)&0x1;
    const int bit5 = (tid>>4)&0x1;
    const int tid1 = threadIdx.x & 31;
    const int warp_id = threadIdx.x / 32;
    uintT rg_k0 ;
    uintT rg_k1 ;
    uintT rg_k2 ;
    uintT rg_k3 ;
    int rg_v0 ;
    int rg_v1 ;
    int rg_v2 ;
    int rg_v3 ;
    int k;
    int seg_size;
    int ext_seg_size;
    if(bin_it < bin_size) {
        k = segs[bin[bin_it]];
        seg_size = ((bin[bin_it]==length-1)?n:segs[bin[bin_it]+1])-segs[bin[bin_it]];
        ext_seg_size = ((seg_size + 63) / 64) * 64;
        int big_wp = (ext_seg_size - blockDim.x * 2) / 64;
        int sml_wp = blockDim.x / 32 - big_wp;
        int sml_len = sml_wp * 64;
        const int big_warp_id = (warp_id - sml_wp < 0)? 0: warp_id - sml_wp;
        bool sml_warp = warp_id < sml_wp;
        if(sml_warp) {
            rg_k0 = key[k+(warp_id<<6)+tid1+0   ];
            rg_k1 = key[k+(warp_id<<6)+tid1+32  ];
            rg_v0 = (warp_id<<6)+tid1+0   ;
            rg_v1 = (warp_id<<6)+tid1+32  ;
            // exch_intxn: switch to exch_local()
            CMP_SWP(uintT,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
            // exch_intxn: generate exch_intxn()
            exch_intxn(rg_k0 ,rg_k1 ,
                       rg_v0 ,rg_v1 ,
                       0x1,bit1);
            // exch_paral: switch to exch_local()
            CMP_SWP(uintT,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
            // exch_intxn: generate exch_intxn()
            exch_intxn(rg_k0 ,rg_k1 ,
                       rg_v0 ,rg_v1 ,
                       0x3,bit2);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,
                       rg_v0 ,rg_v1 ,
                       0x1,bit1);
            // exch_paral: switch to exch_local()
            CMP_SWP(uintT,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
            // exch_intxn: generate exch_intxn()
            exch_intxn(rg_k0 ,rg_k1 ,
                       rg_v0 ,rg_v1 ,
                       0x7,bit3);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,
                       rg_v0 ,rg_v1 ,
                       0x2,bit2);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,
                       rg_v0 ,rg_v1 ,
                       0x1,bit1);
            // exch_paral: switch to exch_local()
            CMP_SWP(uintT,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
            // exch_intxn: generate exch_intxn()
            exch_intxn(rg_k0 ,rg_k1 ,
                       rg_v0 ,rg_v1 ,
                       0xf,bit4);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,
                       rg_v0 ,rg_v1 ,
                       0x4,bit3);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,
                       rg_v0 ,rg_v1 ,
                       0x2,bit2);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,
                       rg_v0 ,rg_v1 ,
                       0x1,bit1);
            // exch_paral: switch to exch_local()
            CMP_SWP(uintT,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
            // exch_intxn: generate exch_intxn()
            exch_intxn(rg_k0 ,rg_k1 ,
                       rg_v0 ,rg_v1 ,
                       0x1f,bit5);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,
                       rg_v0 ,rg_v1 ,
                       0x8,bit4);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,
                       rg_v0 ,rg_v1 ,
                       0x4,bit3);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,
                       rg_v0 ,rg_v1 ,
                       0x2,bit2);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,
                       rg_v0 ,rg_v1 ,
                       0x1,bit1);
            // exch_paral: switch to exch_local()
            CMP_SWP(uintT,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
        } else {
            rg_k0  = (sml_len+tid1+(big_warp_id<<7)+0   <seg_size)?key[k+sml_len+tid1+(big_warp_id<<7)+0   ]:INT_MAX;
            rg_k1  = (sml_len+tid1+(big_warp_id<<7)+32  <seg_size)?key[k+sml_len+tid1+(big_warp_id<<7)+32  ]:INT_MAX;
            rg_k2  = (sml_len+tid1+(big_warp_id<<7)+64  <seg_size)?key[k+sml_len+tid1+(big_warp_id<<7)+64  ]:INT_MAX;
            rg_k3  = (sml_len+tid1+(big_warp_id<<7)+96  <seg_size)?key[k+sml_len+tid1+(big_warp_id<<7)+96  ]:INT_MAX;
            if(sml_len+tid1+(big_warp_id<<7)+0   <seg_size) rg_v0  = sml_len+tid1+(big_warp_id<<7)+0   ;
            if(sml_len+tid1+(big_warp_id<<7)+32  <seg_size) rg_v1  = sml_len+tid1+(big_warp_id<<7)+32  ;
            if(sml_len+tid1+(big_warp_id<<7)+64  <seg_size) rg_v2  = sml_len+tid1+(big_warp_id<<7)+64  ;
            if(sml_len+tid1+(big_warp_id<<7)+96  <seg_size) rg_v3  = sml_len+tid1+(big_warp_id<<7)+96  ;
            // exch_intxn: switch to exch_local()
            CMP_SWP(uintT,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
            CMP_SWP(uintT,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
            // exch_intxn: switch to exch_local()
            CMP_SWP(uintT,rg_k0 ,rg_k3 ,int,rg_v0 ,rg_v3 );
            CMP_SWP(uintT,rg_k1 ,rg_k2 ,int,rg_v1 ,rg_v2 );
            // exch_paral: switch to exch_local()
            CMP_SWP(uintT,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
            CMP_SWP(uintT,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
            // exch_intxn: generate exch_intxn()
            exch_intxn(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                       rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                       0x1,bit1);
            // exch_paral: switch to exch_local()
            CMP_SWP(uintT,rg_k0 ,rg_k2 ,int,rg_v0 ,rg_v2 );
            CMP_SWP(uintT,rg_k1 ,rg_k3 ,int,rg_v1 ,rg_v3 );
            // exch_paral: switch to exch_local()
            CMP_SWP(uintT,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
            CMP_SWP(uintT,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
            // exch_intxn: generate exch_intxn()
            exch_intxn(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                       rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                       0x3,bit2);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                       rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                       0x1,bit1);
            // exch_paral: switch to exch_local()
            CMP_SWP(uintT,rg_k0 ,rg_k2 ,int,rg_v0 ,rg_v2 );
            CMP_SWP(uintT,rg_k1 ,rg_k3 ,int,rg_v1 ,rg_v3 );
            // exch_paral: switch to exch_local()
            CMP_SWP(uintT,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
            CMP_SWP(uintT,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
            // exch_intxn: generate exch_intxn()
            exch_intxn(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                       rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                       0x7,bit3);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                       rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                       0x2,bit2);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                       rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                       0x1,bit1);
            // exch_paral: switch to exch_local()
            CMP_SWP(uintT,rg_k0 ,rg_k2 ,int,rg_v0 ,rg_v2 );
            CMP_SWP(uintT,rg_k1 ,rg_k3 ,int,rg_v1 ,rg_v3 );
            // exch_paral: switch to exch_local()
            CMP_SWP(uintT,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
            CMP_SWP(uintT,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
            // exch_intxn: generate exch_intxn()
            exch_intxn(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                       rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                       0xf,bit4);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                       rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                       0x4,bit3);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                       rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                       0x2,bit2);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                       rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                       0x1,bit1);
            // exch_paral: switch to exch_local()
            CMP_SWP(uintT,rg_k0 ,rg_k2 ,int,rg_v0 ,rg_v2 );
            CMP_SWP(uintT,rg_k1 ,rg_k3 ,int,rg_v1 ,rg_v3 );
            // exch_paral: switch to exch_local()
            CMP_SWP(uintT,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
            CMP_SWP(uintT,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
            // exch_intxn: generate exch_intxn()
            exch_intxn(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                       rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                       0x1f,bit5);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                       rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                       0x8,bit4);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                       rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                       0x4,bit3);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                       rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                       0x2,bit2);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                       rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                       0x1,bit1);
            // exch_paral: switch to exch_local()
            CMP_SWP(uintT,rg_k0 ,rg_k2 ,int,rg_v0 ,rg_v2 );
            CMP_SWP(uintT,rg_k1 ,rg_k3 ,int,rg_v1 ,rg_v3 );
            // exch_paral: switch to exch_local()
            CMP_SWP(uintT,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
            CMP_SWP(uintT,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
        }
        // Store register results to shared memory
        if(sml_warp) {
            smem[(warp_id<<6)+(tid1<<1)+0 ] = rg_k0 ;
            smem[(warp_id<<6)+(tid1<<1)+1 ] = rg_k1 ;
            tmem[(warp_id<<6)+(tid1<<1)+0 ] = rg_v0 ;
            tmem[(warp_id<<6)+(tid1<<1)+1 ] = rg_v1 ;
        } else {
            smem[sml_len+(big_warp_id<<7)+(tid1<<2)+0 ] = rg_k0 ;
            smem[sml_len+(big_warp_id<<7)+(tid1<<2)+1 ] = rg_k1 ;
            smem[sml_len+(big_warp_id<<7)+(tid1<<2)+2 ] = rg_k2 ;
            smem[sml_len+(big_warp_id<<7)+(tid1<<2)+3 ] = rg_k3 ;
            tmem[sml_len+(big_warp_id<<7)+(tid1<<2)+0 ] = rg_v0 ;
            tmem[sml_len+(big_warp_id<<7)+(tid1<<2)+1 ] = rg_v1 ;
            tmem[sml_len+(big_warp_id<<7)+(tid1<<2)+2 ] = rg_v2 ;
            tmem[sml_len+(big_warp_id<<7)+(tid1<<2)+3 ] = rg_v3 ;
        }
        __syncthreads();
        // Merge in 4 steps
        int grp_start_wp_id;
        int grp_start_off;
        int tmp_wp_id;
        int lhs_len;
        int rhs_len;
        int gran;
        int s_a;
        int s_b;
        bool p;
        uintT tmp_k0;
        uintT tmp_k1;
        int tmp_v0;
        int tmp_v1;
        uintT *start;
        // Step 0
        grp_start_wp_id = ((warp_id>>1)<<1);
        grp_start_off = (grp_start_wp_id<sml_wp)?grp_start_wp_id*64:sml_len+(grp_start_wp_id-sml_wp)*128;
        tmp_wp_id = grp_start_wp_id;
        lhs_len = ((tmp_wp_id+0<sml_wp)?64  :128 );
        rhs_len = ((tmp_wp_id+1<sml_wp)?64  :128 );
        gran = (warp_id<sml_wp)?(tid1<<1): (tid1<<2);
        if((warp_id&1)==0){
            gran += 0;
        }
        if((warp_id&1)==1){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 );
        }
        start = smem + grp_start_off;
        s_a = find_kth3(start, lhs_len, start+lhs_len, rhs_len, gran);
        s_b = lhs_len + gran - s_a;
        if(sml_warp){
            tmp_k0 = (s_a<lhs_len        )?smem[grp_start_off+s_a]:INT_MAX;
            tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
            if(s_a<lhs_len        ) tmp_v0 = tmem[grp_start_off+s_a];
            if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k0 = p ? tmp_k0 : tmp_k1;
            rg_v0 = p ? tmp_v0 : tmp_v1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:INT_MAX;
                if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
                if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k1 = p ? tmp_k0 : tmp_k1;
            rg_v1 = p ? tmp_v0 : tmp_v1;
        } else {
            tmp_k0 = (s_a<lhs_len        )?smem[grp_start_off+s_a]:INT_MAX;
            tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
            if(s_a<lhs_len        ) tmp_v0 = tmem[grp_start_off+s_a];
            if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k0 = p ? tmp_k0 : tmp_k1;
            rg_v0 = p ? tmp_v0 : tmp_v1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:INT_MAX;
                if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
                if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k1 = p ? tmp_k0 : tmp_k1;
            rg_v1 = p ? tmp_v0 : tmp_v1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:INT_MAX;
                if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
                if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k2 = p ? tmp_k0 : tmp_k1;
            rg_v2 = p ? tmp_v0 : tmp_v1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:INT_MAX;
                if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
                if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k3 = p ? tmp_k0 : tmp_k1;
            rg_v3 = p ? tmp_v0 : tmp_v1;
        }
        __syncthreads();
        // Store merged results back to shared memory
        if(sml_warp){
            smem[grp_start_off+gran+0 ] = rg_k0 ;
            smem[grp_start_off+gran+1 ] = rg_k1 ;
            tmem[grp_start_off+gran+0 ] = rg_v0 ;
            tmem[grp_start_off+gran+1 ] = rg_v1 ;
        } else {
            smem[grp_start_off+gran+0 ] = rg_k0 ;
            smem[grp_start_off+gran+1 ] = rg_k1 ;
            smem[grp_start_off+gran+2 ] = rg_k2 ;
            smem[grp_start_off+gran+3 ] = rg_k3 ;
            tmem[grp_start_off+gran+0 ] = rg_v0 ;
            tmem[grp_start_off+gran+1 ] = rg_v1 ;
            tmem[grp_start_off+gran+2 ] = rg_v2 ;
            tmem[grp_start_off+gran+3 ] = rg_v3 ;
        }
        __syncthreads();
        // Step 1
        grp_start_wp_id = ((warp_id>>2)<<2);
        grp_start_off = (grp_start_wp_id<sml_wp)?grp_start_wp_id*64:sml_len+(grp_start_wp_id-sml_wp)*128;
        tmp_wp_id = grp_start_wp_id;
        lhs_len = ((tmp_wp_id+0<sml_wp)?64  :128 )+
                  ((tmp_wp_id+1<sml_wp)?64  :128 );
        rhs_len = ((tmp_wp_id+2<sml_wp)?64  :128 )+
                  ((tmp_wp_id+3<sml_wp)?64  :128 );
        gran = (warp_id<sml_wp)?(tid1<<1): (tid1<<2);
        if((warp_id&3)==0){
            gran += 0;
        }
        if((warp_id&3)==1){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 );
        }
        if((warp_id&3)==2){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 )+
                    ((tmp_wp_id+1<sml_wp)?64  :128 );
        }
        if((warp_id&3)==3){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 )+
                    ((tmp_wp_id+1<sml_wp)?64  :128 )+
                    ((tmp_wp_id+2<sml_wp)?64  :128 );
        }
        start = smem + grp_start_off;
        s_a = find_kth3(start, lhs_len, start+lhs_len, rhs_len, gran);
        s_b = lhs_len + gran - s_a;
        if(sml_warp){
            tmp_k0 = (s_a<lhs_len        )?smem[grp_start_off+s_a]:INT_MAX;
            tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
            if(s_a<lhs_len        ) tmp_v0 = tmem[grp_start_off+s_a];
            if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k0 = p ? tmp_k0 : tmp_k1;
            rg_v0 = p ? tmp_v0 : tmp_v1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:INT_MAX;
                if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
                if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k1 = p ? tmp_k0 : tmp_k1;
            rg_v1 = p ? tmp_v0 : tmp_v1;
        } else {
            tmp_k0 = (s_a<lhs_len        )?smem[grp_start_off+s_a]:INT_MAX;
            tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
            if(s_a<lhs_len        ) tmp_v0 = tmem[grp_start_off+s_a];
            if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k0 = p ? tmp_k0 : tmp_k1;
            rg_v0 = p ? tmp_v0 : tmp_v1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:INT_MAX;
                if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
                if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k1 = p ? tmp_k0 : tmp_k1;
            rg_v1 = p ? tmp_v0 : tmp_v1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:INT_MAX;
                if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
                if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k2 = p ? tmp_k0 : tmp_k1;
            rg_v2 = p ? tmp_v0 : tmp_v1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:INT_MAX;
                if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
                if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k3 = p ? tmp_k0 : tmp_k1;
            rg_v3 = p ? tmp_v0 : tmp_v1;
        }
        __syncthreads();
        // Store merged results back to shared memory
        if(sml_warp){
            smem[grp_start_off+gran+0 ] = rg_k0 ;
            smem[grp_start_off+gran+1 ] = rg_k1 ;
            tmem[grp_start_off+gran+0 ] = rg_v0 ;
            tmem[grp_start_off+gran+1 ] = rg_v1 ;
        } else {
            smem[grp_start_off+gran+0 ] = rg_k0 ;
            smem[grp_start_off+gran+1 ] = rg_k1 ;
            smem[grp_start_off+gran+2 ] = rg_k2 ;
            smem[grp_start_off+gran+3 ] = rg_k3 ;
            tmem[grp_start_off+gran+0 ] = rg_v0 ;
            tmem[grp_start_off+gran+1 ] = rg_v1 ;
            tmem[grp_start_off+gran+2 ] = rg_v2 ;
            tmem[grp_start_off+gran+3 ] = rg_v3 ;
        }
        __syncthreads();
        // Step 2
        grp_start_wp_id = ((warp_id>>3)<<3);
        grp_start_off = (grp_start_wp_id<sml_wp)?grp_start_wp_id*64:sml_len+(grp_start_wp_id-sml_wp)*128;
        tmp_wp_id = grp_start_wp_id;
        lhs_len = ((tmp_wp_id+0<sml_wp)?64  :128 )+
                  ((tmp_wp_id+1<sml_wp)?64  :128 )+
                  ((tmp_wp_id+2<sml_wp)?64  :128 )+
                  ((tmp_wp_id+3<sml_wp)?64  :128 );
        rhs_len = ((tmp_wp_id+4<sml_wp)?64  :128 )+
                  ((tmp_wp_id+5<sml_wp)?64  :128 )+
                  ((tmp_wp_id+6<sml_wp)?64  :128 )+
                  ((tmp_wp_id+7<sml_wp)?64  :128 );
        gran = (warp_id<sml_wp)?(tid1<<1): (tid1<<2);
        if((warp_id&7)==0){
            gran += 0;
        }
        if((warp_id&7)==1){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 );
        }
        if((warp_id&7)==2){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 )+
                    ((tmp_wp_id+1<sml_wp)?64  :128 );
        }
        if((warp_id&7)==3){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 )+
                    ((tmp_wp_id+1<sml_wp)?64  :128 )+
                    ((tmp_wp_id+2<sml_wp)?64  :128 );
        }
        if((warp_id&7)==4){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 )+
                    ((tmp_wp_id+1<sml_wp)?64  :128 )+
                    ((tmp_wp_id+2<sml_wp)?64  :128 )+
                    ((tmp_wp_id+3<sml_wp)?64  :128 );
        }
        if((warp_id&7)==5){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 )+
                    ((tmp_wp_id+1<sml_wp)?64  :128 )+
                    ((tmp_wp_id+2<sml_wp)?64  :128 )+
                    ((tmp_wp_id+3<sml_wp)?64  :128 )+
                    ((tmp_wp_id+4<sml_wp)?64  :128 );
        }
        if((warp_id&7)==6){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 )+
                    ((tmp_wp_id+1<sml_wp)?64  :128 )+
                    ((tmp_wp_id+2<sml_wp)?64  :128 )+
                    ((tmp_wp_id+3<sml_wp)?64  :128 )+
                    ((tmp_wp_id+4<sml_wp)?64  :128 )+
                    ((tmp_wp_id+5<sml_wp)?64  :128 );
        }
        if((warp_id&7)==7){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 )+
                    ((tmp_wp_id+1<sml_wp)?64  :128 )+
                    ((tmp_wp_id+2<sml_wp)?64  :128 )+
                    ((tmp_wp_id+3<sml_wp)?64  :128 )+
                    ((tmp_wp_id+4<sml_wp)?64  :128 )+
                    ((tmp_wp_id+5<sml_wp)?64  :128 )+
                    ((tmp_wp_id+6<sml_wp)?64  :128 );
        }
        start = smem + grp_start_off;
        s_a = find_kth3(start, lhs_len, start+lhs_len, rhs_len, gran);
        s_b = lhs_len + gran - s_a;
        if(sml_warp){
            tmp_k0 = (s_a<lhs_len        )?smem[grp_start_off+s_a]:INT_MAX;
            tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
            if(s_a<lhs_len        ) tmp_v0 = tmem[grp_start_off+s_a];
            if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k0 = p ? tmp_k0 : tmp_k1;
            rg_v0 = p ? tmp_v0 : tmp_v1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:INT_MAX;
                if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
                if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k1 = p ? tmp_k0 : tmp_k1;
            rg_v1 = p ? tmp_v0 : tmp_v1;
        } else {
            tmp_k0 = (s_a<lhs_len        )?smem[grp_start_off+s_a]:INT_MAX;
            tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
            if(s_a<lhs_len        ) tmp_v0 = tmem[grp_start_off+s_a];
            if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k0 = p ? tmp_k0 : tmp_k1;
            rg_v0 = p ? tmp_v0 : tmp_v1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:INT_MAX;
                if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
                if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k1 = p ? tmp_k0 : tmp_k1;
            rg_v1 = p ? tmp_v0 : tmp_v1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:INT_MAX;
                if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
                if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k2 = p ? tmp_k0 : tmp_k1;
            rg_v2 = p ? tmp_v0 : tmp_v1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:INT_MAX;
                if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
                if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k3 = p ? tmp_k0 : tmp_k1;
            rg_v3 = p ? tmp_v0 : tmp_v1;
        }
        __syncthreads();
        // Store merged results back to shared memory
        if(sml_warp){
            smem[grp_start_off+gran+0 ] = rg_k0 ;
            smem[grp_start_off+gran+1 ] = rg_k1 ;
            tmem[grp_start_off+gran+0 ] = rg_v0 ;
            tmem[grp_start_off+gran+1 ] = rg_v1 ;
        } else {
            smem[grp_start_off+gran+0 ] = rg_k0 ;
            smem[grp_start_off+gran+1 ] = rg_k1 ;
            smem[grp_start_off+gran+2 ] = rg_k2 ;
            smem[grp_start_off+gran+3 ] = rg_k3 ;
            tmem[grp_start_off+gran+0 ] = rg_v0 ;
            tmem[grp_start_off+gran+1 ] = rg_v1 ;
            tmem[grp_start_off+gran+2 ] = rg_v2 ;
            tmem[grp_start_off+gran+3 ] = rg_v3 ;
        }
        __syncthreads();
        // Step 3
        grp_start_wp_id = ((warp_id>>4)<<4);
        grp_start_off = (grp_start_wp_id<sml_wp)?grp_start_wp_id*64:sml_len+(grp_start_wp_id-sml_wp)*128;
        tmp_wp_id = grp_start_wp_id;
        lhs_len = ((tmp_wp_id+0<sml_wp)?64  :128 )+
                  ((tmp_wp_id+1<sml_wp)?64  :128 )+
                  ((tmp_wp_id+2<sml_wp)?64  :128 )+
                  ((tmp_wp_id+3<sml_wp)?64  :128 )+
                  ((tmp_wp_id+4<sml_wp)?64  :128 )+
                  ((tmp_wp_id+5<sml_wp)?64  :128 )+
                  ((tmp_wp_id+6<sml_wp)?64  :128 )+
                  ((tmp_wp_id+7<sml_wp)?64  :128 );
        rhs_len = ((tmp_wp_id+8<sml_wp)?64  :128 )+
                  ((tmp_wp_id+9<sml_wp)?64  :128 )+
                  ((tmp_wp_id+10<sml_wp)?64  :128 )+
                  ((tmp_wp_id+11<sml_wp)?64  :128 )+
                  ((tmp_wp_id+12<sml_wp)?64  :128 )+
                  ((tmp_wp_id+13<sml_wp)?64  :128 )+
                  ((tmp_wp_id+14<sml_wp)?64  :128 )+
                  ((tmp_wp_id+15<sml_wp)?64  :128 );
        gran = (warp_id<sml_wp)?(tid1<<1): (tid1<<2);
        if((warp_id&15)==0){
            gran += 0;
        }
        if((warp_id&15)==1){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 );
        }
        if((warp_id&15)==2){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 )+
                    ((tmp_wp_id+1<sml_wp)?64  :128 );
        }
        if((warp_id&15)==3){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 )+
                    ((tmp_wp_id+1<sml_wp)?64  :128 )+
                    ((tmp_wp_id+2<sml_wp)?64  :128 );
        }
        if((warp_id&15)==4){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 )+
                    ((tmp_wp_id+1<sml_wp)?64  :128 )+
                    ((tmp_wp_id+2<sml_wp)?64  :128 )+
                    ((tmp_wp_id+3<sml_wp)?64  :128 );
        }
        if((warp_id&15)==5){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 )+
                    ((tmp_wp_id+1<sml_wp)?64  :128 )+
                    ((tmp_wp_id+2<sml_wp)?64  :128 )+
                    ((tmp_wp_id+3<sml_wp)?64  :128 )+
                    ((tmp_wp_id+4<sml_wp)?64  :128 );
        }
        if((warp_id&15)==6){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 )+
                    ((tmp_wp_id+1<sml_wp)?64  :128 )+
                    ((tmp_wp_id+2<sml_wp)?64  :128 )+
                    ((tmp_wp_id+3<sml_wp)?64  :128 )+
                    ((tmp_wp_id+4<sml_wp)?64  :128 )+
                    ((tmp_wp_id+5<sml_wp)?64  :128 );
        }
        if((warp_id&15)==7){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 )+
                    ((tmp_wp_id+1<sml_wp)?64  :128 )+
                    ((tmp_wp_id+2<sml_wp)?64  :128 )+
                    ((tmp_wp_id+3<sml_wp)?64  :128 )+
                    ((tmp_wp_id+4<sml_wp)?64  :128 )+
                    ((tmp_wp_id+5<sml_wp)?64  :128 )+
                    ((tmp_wp_id+6<sml_wp)?64  :128 );
        }
        if((warp_id&15)==8){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 )+
                    ((tmp_wp_id+1<sml_wp)?64  :128 )+
                    ((tmp_wp_id+2<sml_wp)?64  :128 )+
                    ((tmp_wp_id+3<sml_wp)?64  :128 )+
                    ((tmp_wp_id+4<sml_wp)?64  :128 )+
                    ((tmp_wp_id+5<sml_wp)?64  :128 )+
                    ((tmp_wp_id+6<sml_wp)?64  :128 )+
                    ((tmp_wp_id+7<sml_wp)?64  :128 );
        }
        if((warp_id&15)==9){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 )+
                    ((tmp_wp_id+1<sml_wp)?64  :128 )+
                    ((tmp_wp_id+2<sml_wp)?64  :128 )+
                    ((tmp_wp_id+3<sml_wp)?64  :128 )+
                    ((tmp_wp_id+4<sml_wp)?64  :128 )+
                    ((tmp_wp_id+5<sml_wp)?64  :128 )+
                    ((tmp_wp_id+6<sml_wp)?64  :128 )+
                    ((tmp_wp_id+7<sml_wp)?64  :128 )+
                    ((tmp_wp_id+8<sml_wp)?64  :128 );
        }
        if((warp_id&15)==10){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 )+
                    ((tmp_wp_id+1<sml_wp)?64  :128 )+
                    ((tmp_wp_id+2<sml_wp)?64  :128 )+
                    ((tmp_wp_id+3<sml_wp)?64  :128 )+
                    ((tmp_wp_id+4<sml_wp)?64  :128 )+
                    ((tmp_wp_id+5<sml_wp)?64  :128 )+
                    ((tmp_wp_id+6<sml_wp)?64  :128 )+
                    ((tmp_wp_id+7<sml_wp)?64  :128 )+
                    ((tmp_wp_id+8<sml_wp)?64  :128 )+
                    ((tmp_wp_id+9<sml_wp)?64  :128 );
        }
        if((warp_id&15)==11){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 )+
                    ((tmp_wp_id+1<sml_wp)?64  :128 )+
                    ((tmp_wp_id+2<sml_wp)?64  :128 )+
                    ((tmp_wp_id+3<sml_wp)?64  :128 )+
                    ((tmp_wp_id+4<sml_wp)?64  :128 )+
                    ((tmp_wp_id+5<sml_wp)?64  :128 )+
                    ((tmp_wp_id+6<sml_wp)?64  :128 )+
                    ((tmp_wp_id+7<sml_wp)?64  :128 )+
                    ((tmp_wp_id+8<sml_wp)?64  :128 )+
                    ((tmp_wp_id+9<sml_wp)?64  :128 )+
                    ((tmp_wp_id+10<sml_wp)?64  :128 );
        }
        if((warp_id&15)==12){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 )+
                    ((tmp_wp_id+1<sml_wp)?64  :128 )+
                    ((tmp_wp_id+2<sml_wp)?64  :128 )+
                    ((tmp_wp_id+3<sml_wp)?64  :128 )+
                    ((tmp_wp_id+4<sml_wp)?64  :128 )+
                    ((tmp_wp_id+5<sml_wp)?64  :128 )+
                    ((tmp_wp_id+6<sml_wp)?64  :128 )+
                    ((tmp_wp_id+7<sml_wp)?64  :128 )+
                    ((tmp_wp_id+8<sml_wp)?64  :128 )+
                    ((tmp_wp_id+9<sml_wp)?64  :128 )+
                    ((tmp_wp_id+10<sml_wp)?64  :128 )+
                    ((tmp_wp_id+11<sml_wp)?64  :128 );
        }
        if((warp_id&15)==13){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 )+
                    ((tmp_wp_id+1<sml_wp)?64  :128 )+
                    ((tmp_wp_id+2<sml_wp)?64  :128 )+
                    ((tmp_wp_id+3<sml_wp)?64  :128 )+
                    ((tmp_wp_id+4<sml_wp)?64  :128 )+
                    ((tmp_wp_id+5<sml_wp)?64  :128 )+
                    ((tmp_wp_id+6<sml_wp)?64  :128 )+
                    ((tmp_wp_id+7<sml_wp)?64  :128 )+
                    ((tmp_wp_id+8<sml_wp)?64  :128 )+
                    ((tmp_wp_id+9<sml_wp)?64  :128 )+
                    ((tmp_wp_id+10<sml_wp)?64  :128 )+
                    ((tmp_wp_id+11<sml_wp)?64  :128 )+
                    ((tmp_wp_id+12<sml_wp)?64  :128 );
        }
        if((warp_id&15)==14){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 )+
                    ((tmp_wp_id+1<sml_wp)?64  :128 )+
                    ((tmp_wp_id+2<sml_wp)?64  :128 )+
                    ((tmp_wp_id+3<sml_wp)?64  :128 )+
                    ((tmp_wp_id+4<sml_wp)?64  :128 )+
                    ((tmp_wp_id+5<sml_wp)?64  :128 )+
                    ((tmp_wp_id+6<sml_wp)?64  :128 )+
                    ((tmp_wp_id+7<sml_wp)?64  :128 )+
                    ((tmp_wp_id+8<sml_wp)?64  :128 )+
                    ((tmp_wp_id+9<sml_wp)?64  :128 )+
                    ((tmp_wp_id+10<sml_wp)?64  :128 )+
                    ((tmp_wp_id+11<sml_wp)?64  :128 )+
                    ((tmp_wp_id+12<sml_wp)?64  :128 )+
                    ((tmp_wp_id+13<sml_wp)?64  :128 );
        }
        if((warp_id&15)==15){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 )+
                    ((tmp_wp_id+1<sml_wp)?64  :128 )+
                    ((tmp_wp_id+2<sml_wp)?64  :128 )+
                    ((tmp_wp_id+3<sml_wp)?64  :128 )+
                    ((tmp_wp_id+4<sml_wp)?64  :128 )+
                    ((tmp_wp_id+5<sml_wp)?64  :128 )+
                    ((tmp_wp_id+6<sml_wp)?64  :128 )+
                    ((tmp_wp_id+7<sml_wp)?64  :128 )+
                    ((tmp_wp_id+8<sml_wp)?64  :128 )+
                    ((tmp_wp_id+9<sml_wp)?64  :128 )+
                    ((tmp_wp_id+10<sml_wp)?64  :128 )+
                    ((tmp_wp_id+11<sml_wp)?64  :128 )+
                    ((tmp_wp_id+12<sml_wp)?64  :128 )+
                    ((tmp_wp_id+13<sml_wp)?64  :128 )+
                    ((tmp_wp_id+14<sml_wp)?64  :128 );
        }
        start = smem + grp_start_off;
        s_a = find_kth3(start, lhs_len, start+lhs_len, rhs_len, gran);
        s_b = lhs_len + gran - s_a;
        if(sml_warp){
            tmp_k0 = (s_a<lhs_len        )?smem[grp_start_off+s_a]:INT_MAX;
            tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
            if(s_a<lhs_len        ) tmp_v0 = tmem[grp_start_off+s_a];
            if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k0 = p ? tmp_k0 : tmp_k1;
            rg_v0 = p ? tmp_v0 : tmp_v1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:INT_MAX;
                if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
                if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k1 = p ? tmp_k0 : tmp_k1;
            rg_v1 = p ? tmp_v0 : tmp_v1;
        } else {
            tmp_k0 = (s_a<lhs_len        )?smem[grp_start_off+s_a]:INT_MAX;
            tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
            if(s_a<lhs_len        ) tmp_v0 = tmem[grp_start_off+s_a];
            if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k0 = p ? tmp_k0 : tmp_k1;
            rg_v0 = p ? tmp_v0 : tmp_v1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:INT_MAX;
                if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
                if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k1 = p ? tmp_k0 : tmp_k1;
            rg_v1 = p ? tmp_v0 : tmp_v1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:INT_MAX;
                if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
                if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k2 = p ? tmp_k0 : tmp_k1;
            rg_v2 = p ? tmp_v0 : tmp_v1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:INT_MAX;
                if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
                if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k3 = p ? tmp_k0 : tmp_k1;
            rg_v3 = p ? tmp_v0 : tmp_v1;
        }
        if(sml_warp){
        } else {
        }
        if(sml_warp){
            if((tid<<1)+0 <seg_size) keyB[k+(tid<<1)+0 ] = rg_k0 ;
            if((tid<<1)+1 <seg_size) keyB[k+(tid<<1)+1 ] = rg_k1 ;
            if((tid<<1)+0 <seg_size) valB[k+(tid<<1)+0 ] = val[k+rg_v0 ];
            if((tid<<1)+1 <seg_size) valB[k+(tid<<1)+1 ] = val[k+rg_v1 ];
        } else {
            if((tid<<2)+0 -sml_len<seg_size) keyB[k+(tid<<2)+0 -sml_len] = rg_k0 ;
            if((tid<<2)+1 -sml_len<seg_size) keyB[k+(tid<<2)+1 -sml_len] = rg_k1 ;
            if((tid<<2)+2 -sml_len<seg_size) keyB[k+(tid<<2)+2 -sml_len] = rg_k2 ;
            if((tid<<2)+3 -sml_len<seg_size) keyB[k+(tid<<2)+3 -sml_len] = rg_k3 ;
            if((tid<<2)+0 -sml_len<seg_size) valB[k+(tid<<2)+0 -sml_len] = val[k+rg_v0 ];
            if((tid<<2)+1 -sml_len<seg_size) valB[k+(tid<<2)+1 -sml_len] = val[k+rg_v1 ];
            if((tid<<2)+2 -sml_len<seg_size) valB[k+(tid<<2)+2 -sml_len] = val[k+rg_v2 ];
            if((tid<<2)+3 -sml_len<seg_size) valB[k+(tid<<2)+3 -sml_len] = val[k+rg_v3 ];
        }
    }
}
#endif
