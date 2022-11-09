
#ifdef _ARCH30
/* block tcf subwarp coalesced quiet real_kern */
/*   128   1       2     false  true      true */
template<class T>
__global__
void gen_bk128_wp2_tc1_r2_r2_orig( 
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
/*   128   2       4     false  true      true */
template<class T>
__global__
void gen_bk128_wp4_tc2_r5_r8_orig( 
    uintT *key, T *val, uintT *keyB, T *valB, int n, int *segs, int *bin, int bin_size, int length) {

    const int gid = threadIdx.x + blockIdx.x * blockDim.x;
    const int bin_it = (gid>>2);
    const int tid = (threadIdx.x & 3);
    const int bit1 = (tid>>0)&0x1;
    const int bit2 = (tid>>1)&0x1;
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
        rg_k1  = (tid+4   <seg_size)?key[k+tid+4   ]:INT_MAX;
        if(tid+0   <seg_size) rg_v0  = tid+0   ;
        if(tid+4   <seg_size) rg_v1  = tid+4   ;
        // sort 8 elements
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
        if((tid<<1)+0 <seg_size) keyB[k+(tid<<1)+0 ] = rg_k0 ;
        if((tid<<1)+1 <seg_size) keyB[k+(tid<<1)+1 ] = rg_k1 ;
        if((tid<<1)+0 <seg_size) valB[k+(tid<<1)+0 ] = val[k+rg_v0 ];
        if((tid<<1)+1 <seg_size) valB[k+(tid<<1)+1 ] = val[k+rg_v1 ];
    }
}
/* block tcf subwarp coalesced quiet real_kern */
/*   128   2       8     false  true      true */
template<class T>
__global__
void gen_bk128_wp8_tc2_r9_r16_orig( 
    uintT *key, T *val, uintT *keyB, T *valB, int n, int *segs, int *bin, int bin_size, int length) {

    const int gid = threadIdx.x + blockIdx.x * blockDim.x;
    const int bin_it = (gid>>3);
    const int tid = (threadIdx.x & 7);
    const int bit1 = (tid>>0)&0x1;
    const int bit2 = (tid>>1)&0x1;
    const int bit3 = (tid>>2)&0x1;
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
        rg_k1  = (tid+8   <seg_size)?key[k+tid+8   ]:INT_MAX;
        if(tid+0   <seg_size) rg_v0  = tid+0   ;
        if(tid+8   <seg_size) rg_v1  = tid+8   ;
        // sort 16 elements
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
        if((tid<<1)+0 <seg_size) keyB[k+(tid<<1)+0 ] = rg_k0 ;
        if((tid<<1)+1 <seg_size) keyB[k+(tid<<1)+1 ] = rg_k1 ;
        if((tid<<1)+0 <seg_size) valB[k+(tid<<1)+0 ] = val[k+rg_v0 ];
        if((tid<<1)+1 <seg_size) valB[k+(tid<<1)+1 ] = val[k+rg_v1 ];
    }
}
/* block tcf subwarp coalesced quiet real_kern */
/*   128   2      16     false  true      true */
template<class T>
__global__
void gen_bk128_wp16_tc2_r17_r32_orig( 
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
    int rg_v0 ;
    int rg_v1 ;
    int k;
    int seg_size;
    if(bin_it < bin_size) {
        k = segs[bin[bin_it]];
        seg_size = ((bin[bin_it]==length-1)?n:segs[bin[bin_it]+1])-segs[bin[bin_it]];
        rg_k0  = (tid+0   <seg_size)?key[k+tid+0   ]:INT_MAX;
        rg_k1  = (tid+16  <seg_size)?key[k+tid+16  ]:INT_MAX;
        if(tid+0   <seg_size) rg_v0  = tid+0   ;
        if(tid+16  <seg_size) rg_v1  = tid+16  ;
        // sort 32 elements
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
        if((tid<<1)+0 <seg_size) keyB[k+(tid<<1)+0 ] = rg_k0 ;
        if((tid<<1)+1 <seg_size) keyB[k+(tid<<1)+1 ] = rg_k1 ;
        if((tid<<1)+0 <seg_size) valB[k+(tid<<1)+0 ] = val[k+rg_v0 ];
        if((tid<<1)+1 <seg_size) valB[k+(tid<<1)+1 ] = val[k+rg_v1 ];
    }
}
/* block tcf subwarp coalesced quiet real_kern */
/*   256  16       4      true  true      true */
template<class T>
__global__
void gen_bk256_wp4_tc16_r33_r64_strd( 
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
        rg_k4  = (tid+16  <seg_size)?key[k+tid+16  ]:INT_MAX;
        rg_k5  = (tid+20  <seg_size)?key[k+tid+20  ]:INT_MAX;
        rg_k6  = (tid+24  <seg_size)?key[k+tid+24  ]:INT_MAX;
        rg_k7  = (tid+28  <seg_size)?key[k+tid+28  ]:INT_MAX;
        rg_k8  = (tid+32  <seg_size)?key[k+tid+32  ]:INT_MAX;
        rg_k9  = (tid+36  <seg_size)?key[k+tid+36  ]:INT_MAX;
        rg_k10 = (tid+40  <seg_size)?key[k+tid+40  ]:INT_MAX;
        rg_k11 = (tid+44  <seg_size)?key[k+tid+44  ]:INT_MAX;
        rg_k12 = (tid+48  <seg_size)?key[k+tid+48  ]:INT_MAX;
        rg_k13 = (tid+52  <seg_size)?key[k+tid+52  ]:INT_MAX;
        rg_k14 = (tid+56  <seg_size)?key[k+tid+56  ]:INT_MAX;
        rg_k15 = (tid+60  <seg_size)?key[k+tid+60  ]:INT_MAX;
        if(tid+0   <seg_size) rg_v0  = tid+0   ;
        if(tid+4   <seg_size) rg_v1  = tid+4   ;
        if(tid+8   <seg_size) rg_v2  = tid+8   ;
        if(tid+12  <seg_size) rg_v3  = tid+12  ;
        if(tid+16  <seg_size) rg_v4  = tid+16  ;
        if(tid+20  <seg_size) rg_v5  = tid+20  ;
        if(tid+24  <seg_size) rg_v6  = tid+24  ;
        if(tid+28  <seg_size) rg_v7  = tid+28  ;
        if(tid+32  <seg_size) rg_v8  = tid+32  ;
        if(tid+36  <seg_size) rg_v9  = tid+36  ;
        if(tid+40  <seg_size) rg_v10 = tid+40  ;
        if(tid+44  <seg_size) rg_v11 = tid+44  ;
        if(tid+48  <seg_size) rg_v12 = tid+48  ;
        if(tid+52  <seg_size) rg_v13 = tid+52  ;
        if(tid+56  <seg_size) rg_v14 = tid+56  ;
        if(tid+60  <seg_size) rg_v15 = tid+60  ;
        // sort 64 elements
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
        kk = __shfl(k, 4 );
        ss = __shfl(seg_size, 4 );
        if(lane_id+0  <ss) keyB[kk+lane_id+0  ] = rg_k4 ;
        if(lane_id+32 <ss) keyB[kk+lane_id+32 ] = rg_k6 ;
        kk = __shfl(k, 8 );
        ss = __shfl(seg_size, 8 );
        if(lane_id+0  <ss) keyB[kk+lane_id+0  ] = rg_k8 ;
        if(lane_id+32 <ss) keyB[kk+lane_id+32 ] = rg_k10;
        kk = __shfl(k, 12);
        ss = __shfl(seg_size, 12);
        if(lane_id+0  <ss) keyB[kk+lane_id+0  ] = rg_k12;
        if(lane_id+32 <ss) keyB[kk+lane_id+32 ] = rg_k14;
        kk = __shfl(k, 16);
        ss = __shfl(seg_size, 16);
        if(lane_id+0  <ss) keyB[kk+lane_id+0  ] = rg_k1 ;
        if(lane_id+32 <ss) keyB[kk+lane_id+32 ] = rg_k3 ;
        kk = __shfl(k, 20);
        ss = __shfl(seg_size, 20);
        if(lane_id+0  <ss) keyB[kk+lane_id+0  ] = rg_k5 ;
        if(lane_id+32 <ss) keyB[kk+lane_id+32 ] = rg_k7 ;
        kk = __shfl(k, 24);
        ss = __shfl(seg_size, 24);
        if(lane_id+0  <ss) keyB[kk+lane_id+0  ] = rg_k9 ;
        if(lane_id+32 <ss) keyB[kk+lane_id+32 ] = rg_k11;
        kk = __shfl(k, 28);
        ss = __shfl(seg_size, 28);
        if(lane_id+0  <ss) keyB[kk+lane_id+0  ] = rg_k13;
        if(lane_id+32 <ss) keyB[kk+lane_id+32 ] = rg_k15;
        kk = __shfl(k, 0 );
        ss = __shfl(seg_size, 0 );
        if(lane_id+0  <ss) valB[kk+lane_id+0  ] = val[kk+rg_v0 ];
        if(lane_id+32 <ss) valB[kk+lane_id+32 ] = val[kk+rg_v2 ];
        kk = __shfl(k, 4 );
        ss = __shfl(seg_size, 4 );
        if(lane_id+0  <ss) valB[kk+lane_id+0  ] = val[kk+rg_v4 ];
        if(lane_id+32 <ss) valB[kk+lane_id+32 ] = val[kk+rg_v6 ];
        kk = __shfl(k, 8 );
        ss = __shfl(seg_size, 8 );
        if(lane_id+0  <ss) valB[kk+lane_id+0  ] = val[kk+rg_v8 ];
        if(lane_id+32 <ss) valB[kk+lane_id+32 ] = val[kk+rg_v10];
        kk = __shfl(k, 12);
        ss = __shfl(seg_size, 12);
        if(lane_id+0  <ss) valB[kk+lane_id+0  ] = val[kk+rg_v12];
        if(lane_id+32 <ss) valB[kk+lane_id+32 ] = val[kk+rg_v14];
        kk = __shfl(k, 16);
        ss = __shfl(seg_size, 16);
        if(lane_id+0  <ss) valB[kk+lane_id+0  ] = val[kk+rg_v1 ];
        if(lane_id+32 <ss) valB[kk+lane_id+32 ] = val[kk+rg_v3 ];
        kk = __shfl(k, 20);
        ss = __shfl(seg_size, 20);
        if(lane_id+0  <ss) valB[kk+lane_id+0  ] = val[kk+rg_v5 ];
        if(lane_id+32 <ss) valB[kk+lane_id+32 ] = val[kk+rg_v7 ];
        kk = __shfl(k, 24);
        ss = __shfl(seg_size, 24);
        if(lane_id+0  <ss) valB[kk+lane_id+0  ] = val[kk+rg_v9 ];
        if(lane_id+32 <ss) valB[kk+lane_id+32 ] = val[kk+rg_v11];
        kk = __shfl(k, 28);
        ss = __shfl(seg_size, 28);
        if(lane_id+0  <ss) valB[kk+lane_id+0  ] = val[kk+rg_v13];
        if(lane_id+32 <ss) valB[kk+lane_id+32 ] = val[kk+rg_v15];
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
/*   512  16       8      true  true      true */
template<class T>
__global__
void gen_bk512_wp8_tc16_r65_r128_strd( 
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
/*   256  16      16      true  true      true */
template<class T>
__global__
void gen_bk256_wp16_tc16_r129_r256_strd( 
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
        rg_k4  = (tid+64  <seg_size)?key[k+tid+64  ]:INT_MAX;
        rg_k5  = (tid+80  <seg_size)?key[k+tid+80  ]:INT_MAX;
        rg_k6  = (tid+96  <seg_size)?key[k+tid+96  ]:INT_MAX;
        rg_k7  = (tid+112 <seg_size)?key[k+tid+112 ]:INT_MAX;
        rg_k8  = (tid+128 <seg_size)?key[k+tid+128 ]:INT_MAX;
        rg_k9  = (tid+144 <seg_size)?key[k+tid+144 ]:INT_MAX;
        rg_k10 = (tid+160 <seg_size)?key[k+tid+160 ]:INT_MAX;
        rg_k11 = (tid+176 <seg_size)?key[k+tid+176 ]:INT_MAX;
        rg_k12 = (tid+192 <seg_size)?key[k+tid+192 ]:INT_MAX;
        rg_k13 = (tid+208 <seg_size)?key[k+tid+208 ]:INT_MAX;
        rg_k14 = (tid+224 <seg_size)?key[k+tid+224 ]:INT_MAX;
        rg_k15 = (tid+240 <seg_size)?key[k+tid+240 ]:INT_MAX;
        if(tid+0   <seg_size) rg_v0  = tid+0   ;
        if(tid+16  <seg_size) rg_v1  = tid+16  ;
        if(tid+32  <seg_size) rg_v2  = tid+32  ;
        if(tid+48  <seg_size) rg_v3  = tid+48  ;
        if(tid+64  <seg_size) rg_v4  = tid+64  ;
        if(tid+80  <seg_size) rg_v5  = tid+80  ;
        if(tid+96  <seg_size) rg_v6  = tid+96  ;
        if(tid+112 <seg_size) rg_v7  = tid+112 ;
        if(tid+128 <seg_size) rg_v8  = tid+128 ;
        if(tid+144 <seg_size) rg_v9  = tid+144 ;
        if(tid+160 <seg_size) rg_v10 = tid+160 ;
        if(tid+176 <seg_size) rg_v11 = tid+176 ;
        if(tid+192 <seg_size) rg_v12 = tid+192 ;
        if(tid+208 <seg_size) rg_v13 = tid+208 ;
        if(tid+224 <seg_size) rg_v14 = tid+224 ;
        if(tid+240 <seg_size) rg_v15 = tid+240 ;
        // sort 256 elements
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
        // exch_intxn: generate exch_intxn()
        exch_intxn(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,rg_k4 ,rg_k5 ,rg_k6 ,rg_k7 ,
                   rg_k8 ,rg_k9 ,rg_k10,rg_k11,rg_k12,rg_k13,rg_k14,rg_k15,
                   rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,rg_v4 ,rg_v5 ,rg_v6 ,rg_v7 ,
                   rg_v8 ,rg_v9 ,rg_v10,rg_v11,rg_v12,rg_v13,rg_v14,rg_v15,
                   0xf,bit4);
        // exch_paral: generate exch_paral()
        exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,rg_k4 ,rg_k5 ,rg_k6 ,rg_k7 ,
                   rg_k8 ,rg_k9 ,rg_k10,rg_k11,rg_k12,rg_k13,rg_k14,rg_k15,
                   rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,rg_v4 ,rg_v5 ,rg_v6 ,rg_v7 ,
                   rg_v8 ,rg_v9 ,rg_v10,rg_v11,rg_v12,rg_v13,rg_v14,rg_v15,
                   0x4,bit3);
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
        if(lane_id+128<ss) keyB[kk+lane_id+128] = rg_k8 ;
        if(lane_id+160<ss) keyB[kk+lane_id+160] = rg_k10;
        if(lane_id+192<ss) keyB[kk+lane_id+192] = rg_k12;
        if(lane_id+224<ss) keyB[kk+lane_id+224] = rg_k14;
        kk = __shfl(k, 16);
        ss = __shfl(seg_size, 16);
        if(lane_id+0  <ss) keyB[kk+lane_id+0  ] = rg_k1 ;
        if(lane_id+32 <ss) keyB[kk+lane_id+32 ] = rg_k3 ;
        if(lane_id+64 <ss) keyB[kk+lane_id+64 ] = rg_k5 ;
        if(lane_id+96 <ss) keyB[kk+lane_id+96 ] = rg_k7 ;
        if(lane_id+128<ss) keyB[kk+lane_id+128] = rg_k9 ;
        if(lane_id+160<ss) keyB[kk+lane_id+160] = rg_k11;
        if(lane_id+192<ss) keyB[kk+lane_id+192] = rg_k13;
        if(lane_id+224<ss) keyB[kk+lane_id+224] = rg_k15;
        kk = __shfl(k, 0 );
        ss = __shfl(seg_size, 0 );
        if(lane_id+0  <ss) valB[kk+lane_id+0  ] = val[kk+rg_v0 ];
        if(lane_id+32 <ss) valB[kk+lane_id+32 ] = val[kk+rg_v2 ];
        if(lane_id+64 <ss) valB[kk+lane_id+64 ] = val[kk+rg_v4 ];
        if(lane_id+96 <ss) valB[kk+lane_id+96 ] = val[kk+rg_v6 ];
        if(lane_id+128<ss) valB[kk+lane_id+128] = val[kk+rg_v8 ];
        if(lane_id+160<ss) valB[kk+lane_id+160] = val[kk+rg_v10];
        if(lane_id+192<ss) valB[kk+lane_id+192] = val[kk+rg_v12];
        if(lane_id+224<ss) valB[kk+lane_id+224] = val[kk+rg_v14];
        kk = __shfl(k, 16);
        ss = __shfl(seg_size, 16);
        if(lane_id+0  <ss) valB[kk+lane_id+0  ] = val[kk+rg_v1 ];
        if(lane_id+32 <ss) valB[kk+lane_id+32 ] = val[kk+rg_v3 ];
        if(lane_id+64 <ss) valB[kk+lane_id+64 ] = val[kk+rg_v5 ];
        if(lane_id+96 <ss) valB[kk+lane_id+96 ] = val[kk+rg_v7 ];
        if(lane_id+128<ss) valB[kk+lane_id+128] = val[kk+rg_v9 ];
        if(lane_id+160<ss) valB[kk+lane_id+160] = val[kk+rg_v11];
        if(lane_id+192<ss) valB[kk+lane_id+192] = val[kk+rg_v13];
        if(lane_id+224<ss) valB[kk+lane_id+224] = val[kk+rg_v15];
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
/*   256  16      32      true  true      true */
template<class T>
__global__
void gen_bk256_wp32_tc16_r257_r512_strd( 
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
        rg_k8  = (tid+256 <seg_size)?key[k+tid+256 ]:INT_MAX;
        rg_k9  = (tid+288 <seg_size)?key[k+tid+288 ]:INT_MAX;
        rg_k10 = (tid+320 <seg_size)?key[k+tid+320 ]:INT_MAX;
        rg_k11 = (tid+352 <seg_size)?key[k+tid+352 ]:INT_MAX;
        rg_k12 = (tid+384 <seg_size)?key[k+tid+384 ]:INT_MAX;
        rg_k13 = (tid+416 <seg_size)?key[k+tid+416 ]:INT_MAX;
        rg_k14 = (tid+448 <seg_size)?key[k+tid+448 ]:INT_MAX;
        rg_k15 = (tid+480 <seg_size)?key[k+tid+480 ]:INT_MAX;
        if(tid+0   <seg_size) rg_v0  = tid+0   ;
        if(tid+32  <seg_size) rg_v1  = tid+32  ;
        if(tid+64  <seg_size) rg_v2  = tid+64  ;
        if(tid+96  <seg_size) rg_v3  = tid+96  ;
        if(tid+128 <seg_size) rg_v4  = tid+128 ;
        if(tid+160 <seg_size) rg_v5  = tid+160 ;
        if(tid+192 <seg_size) rg_v6  = tid+192 ;
        if(tid+224 <seg_size) rg_v7  = tid+224 ;
        if(tid+256 <seg_size) rg_v8  = tid+256 ;
        if(tid+288 <seg_size) rg_v9  = tid+288 ;
        if(tid+320 <seg_size) rg_v10 = tid+320 ;
        if(tid+352 <seg_size) rg_v11 = tid+352 ;
        if(tid+384 <seg_size) rg_v12 = tid+384 ;
        if(tid+416 <seg_size) rg_v13 = tid+416 ;
        if(tid+448 <seg_size) rg_v14 = tid+448 ;
        if(tid+480 <seg_size) rg_v15 = tid+480 ;
        // sort 512 elements
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
        // exch_intxn: generate exch_intxn()
        exch_intxn(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,rg_k4 ,rg_k5 ,rg_k6 ,rg_k7 ,
                   rg_k8 ,rg_k9 ,rg_k10,rg_k11,rg_k12,rg_k13,rg_k14,rg_k15,
                   rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,rg_v4 ,rg_v5 ,rg_v6 ,rg_v7 ,
                   rg_v8 ,rg_v9 ,rg_v10,rg_v11,rg_v12,rg_v13,rg_v14,rg_v15,
                   0xf,bit4);
        // exch_paral: generate exch_paral()
        exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,rg_k4 ,rg_k5 ,rg_k6 ,rg_k7 ,
                   rg_k8 ,rg_k9 ,rg_k10,rg_k11,rg_k12,rg_k13,rg_k14,rg_k15,
                   rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,rg_v4 ,rg_v5 ,rg_v6 ,rg_v7 ,
                   rg_v8 ,rg_v9 ,rg_v10,rg_v11,rg_v12,rg_v13,rg_v14,rg_v15,
                   0x4,bit3);
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
        // exch_intxn: generate exch_intxn()
        exch_intxn(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,rg_k4 ,rg_k5 ,rg_k6 ,rg_k7 ,
                   rg_k8 ,rg_k9 ,rg_k10,rg_k11,rg_k12,rg_k13,rg_k14,rg_k15,
                   rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,rg_v4 ,rg_v5 ,rg_v6 ,rg_v7 ,
                   rg_v8 ,rg_v9 ,rg_v10,rg_v11,rg_v12,rg_v13,rg_v14,rg_v15,
                   0x1f,bit5);
        // exch_paral: generate exch_paral()
        exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,rg_k4 ,rg_k5 ,rg_k6 ,rg_k7 ,
                   rg_k8 ,rg_k9 ,rg_k10,rg_k11,rg_k12,rg_k13,rg_k14,rg_k15,
                   rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,rg_v4 ,rg_v5 ,rg_v6 ,rg_v7 ,
                   rg_v8 ,rg_v9 ,rg_v10,rg_v11,rg_v12,rg_v13,rg_v14,rg_v15,
                   0x8,bit4);
        // exch_paral: generate exch_paral()
        exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,rg_k4 ,rg_k5 ,rg_k6 ,rg_k7 ,
                   rg_k8 ,rg_k9 ,rg_k10,rg_k11,rg_k12,rg_k13,rg_k14,rg_k15,
                   rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,rg_v4 ,rg_v5 ,rg_v6 ,rg_v7 ,
                   rg_v8 ,rg_v9 ,rg_v10,rg_v11,rg_v12,rg_v13,rg_v14,rg_v15,
                   0x4,bit3);
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
        if(lane_id+128<ss) keyB[kk+lane_id+128] = rg_k8 ;
        if(lane_id+160<ss) keyB[kk+lane_id+160] = rg_k10;
        if(lane_id+192<ss) keyB[kk+lane_id+192] = rg_k12;
        if(lane_id+224<ss) keyB[kk+lane_id+224] = rg_k14;
        if(lane_id+256<ss) keyB[kk+lane_id+256] = rg_k1 ;
        if(lane_id+288<ss) keyB[kk+lane_id+288] = rg_k3 ;
        if(lane_id+320<ss) keyB[kk+lane_id+320] = rg_k5 ;
        if(lane_id+352<ss) keyB[kk+lane_id+352] = rg_k7 ;
        if(lane_id+384<ss) keyB[kk+lane_id+384] = rg_k9 ;
        if(lane_id+416<ss) keyB[kk+lane_id+416] = rg_k11;
        if(lane_id+448<ss) keyB[kk+lane_id+448] = rg_k13;
        if(lane_id+480<ss) keyB[kk+lane_id+480] = rg_k15;
        kk = __shfl(k, 0 );
        ss = __shfl(seg_size, 0 );
        if(lane_id+0  <ss) valB[kk+lane_id+0  ] = val[kk+rg_v0 ];
        if(lane_id+32 <ss) valB[kk+lane_id+32 ] = val[kk+rg_v2 ];
        if(lane_id+64 <ss) valB[kk+lane_id+64 ] = val[kk+rg_v4 ];
        if(lane_id+96 <ss) valB[kk+lane_id+96 ] = val[kk+rg_v6 ];
        if(lane_id+128<ss) valB[kk+lane_id+128] = val[kk+rg_v8 ];
        if(lane_id+160<ss) valB[kk+lane_id+160] = val[kk+rg_v10];
        if(lane_id+192<ss) valB[kk+lane_id+192] = val[kk+rg_v12];
        if(lane_id+224<ss) valB[kk+lane_id+224] = val[kk+rg_v14];
        if(lane_id+256<ss) valB[kk+lane_id+256] = val[kk+rg_v1 ];
        if(lane_id+288<ss) valB[kk+lane_id+288] = val[kk+rg_v3 ];
        if(lane_id+320<ss) valB[kk+lane_id+320] = val[kk+rg_v5 ];
        if(lane_id+352<ss) valB[kk+lane_id+352] = val[kk+rg_v7 ];
        if(lane_id+384<ss) valB[kk+lane_id+384] = val[kk+rg_v9 ];
        if(lane_id+416<ss) valB[kk+lane_id+416] = val[kk+rg_v11];
        if(lane_id+448<ss) valB[kk+lane_id+448] = val[kk+rg_v13];
        if(lane_id+480<ss) valB[kk+lane_id+480] = val[kk+rg_v15];
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
/* block tcf1 tcf2 quiet real_kern */
/*   128    4    8  true      true */
template<class T>
__global__
void gen_bk128_tc8_r513_r1024_orig(
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
    int k;
    int seg_size;
    int ext_seg_size;
    if(bin_it < bin_size) {
        k = segs[bin[bin_it]];
        seg_size = ((bin[bin_it]==length-1)?n:segs[bin[bin_it]+1])-segs[bin[bin_it]];
        ext_seg_size = ((seg_size + 127) / 128) * 128;
        int big_wp = (ext_seg_size - blockDim.x * 4) / 128;
        int sml_wp = blockDim.x / 32 - big_wp;
        int sml_len = sml_wp * 128;
        const int big_warp_id = (warp_id - sml_wp < 0)? 0: warp_id - sml_wp;
        bool sml_warp = warp_id < sml_wp;
        if(sml_warp) {
            rg_k0 = key[k+(warp_id<<7)+tid1+0   ];
            rg_k1 = key[k+(warp_id<<7)+tid1+32  ];
            rg_k2 = key[k+(warp_id<<7)+tid1+64  ];
            rg_k3 = key[k+(warp_id<<7)+tid1+96  ];
            rg_v0 = (warp_id<<7)+tid1+0   ;
            rg_v1 = (warp_id<<7)+tid1+32  ;
            rg_v2 = (warp_id<<7)+tid1+64  ;
            rg_v3 = (warp_id<<7)+tid1+96  ;
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
        } else {
            rg_k0  = (sml_len+tid1+(big_warp_id<<8)+0   <seg_size)?key[k+sml_len+tid1+(big_warp_id<<8)+0   ]:INT_MAX;
            rg_k1  = (sml_len+tid1+(big_warp_id<<8)+32  <seg_size)?key[k+sml_len+tid1+(big_warp_id<<8)+32  ]:INT_MAX;
            rg_k2  = (sml_len+tid1+(big_warp_id<<8)+64  <seg_size)?key[k+sml_len+tid1+(big_warp_id<<8)+64  ]:INT_MAX;
            rg_k3  = (sml_len+tid1+(big_warp_id<<8)+96  <seg_size)?key[k+sml_len+tid1+(big_warp_id<<8)+96  ]:INT_MAX;
            rg_k4  = (sml_len+tid1+(big_warp_id<<8)+128 <seg_size)?key[k+sml_len+tid1+(big_warp_id<<8)+128 ]:INT_MAX;
            rg_k5  = (sml_len+tid1+(big_warp_id<<8)+160 <seg_size)?key[k+sml_len+tid1+(big_warp_id<<8)+160 ]:INT_MAX;
            rg_k6  = (sml_len+tid1+(big_warp_id<<8)+192 <seg_size)?key[k+sml_len+tid1+(big_warp_id<<8)+192 ]:INT_MAX;
            rg_k7  = (sml_len+tid1+(big_warp_id<<8)+224 <seg_size)?key[k+sml_len+tid1+(big_warp_id<<8)+224 ]:INT_MAX;
            if(sml_len+tid1+(big_warp_id<<8)+0   <seg_size) rg_v0  = sml_len+tid1+(big_warp_id<<8)+0   ;
            if(sml_len+tid1+(big_warp_id<<8)+32  <seg_size) rg_v1  = sml_len+tid1+(big_warp_id<<8)+32  ;
            if(sml_len+tid1+(big_warp_id<<8)+64  <seg_size) rg_v2  = sml_len+tid1+(big_warp_id<<8)+64  ;
            if(sml_len+tid1+(big_warp_id<<8)+96  <seg_size) rg_v3  = sml_len+tid1+(big_warp_id<<8)+96  ;
            if(sml_len+tid1+(big_warp_id<<8)+128 <seg_size) rg_v4  = sml_len+tid1+(big_warp_id<<8)+128 ;
            if(sml_len+tid1+(big_warp_id<<8)+160 <seg_size) rg_v5  = sml_len+tid1+(big_warp_id<<8)+160 ;
            if(sml_len+tid1+(big_warp_id<<8)+192 <seg_size) rg_v6  = sml_len+tid1+(big_warp_id<<8)+192 ;
            if(sml_len+tid1+(big_warp_id<<8)+224 <seg_size) rg_v7  = sml_len+tid1+(big_warp_id<<8)+224 ;
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
        // Store register results to shared memory
        if(sml_warp) {
            smem[(warp_id<<7)+(tid1<<2)+0 ] = rg_k0 ;
            smem[(warp_id<<7)+(tid1<<2)+1 ] = rg_k1 ;
            smem[(warp_id<<7)+(tid1<<2)+2 ] = rg_k2 ;
            smem[(warp_id<<7)+(tid1<<2)+3 ] = rg_k3 ;
            tmem[(warp_id<<7)+(tid1<<2)+0 ] = rg_v0 ;
            tmem[(warp_id<<7)+(tid1<<2)+1 ] = rg_v1 ;
            tmem[(warp_id<<7)+(tid1<<2)+2 ] = rg_v2 ;
            tmem[(warp_id<<7)+(tid1<<2)+3 ] = rg_v3 ;
        } else {
            smem[sml_len+(big_warp_id<<8)+(tid1<<3)+0 ] = rg_k0 ;
            smem[sml_len+(big_warp_id<<8)+(tid1<<3)+1 ] = rg_k1 ;
            smem[sml_len+(big_warp_id<<8)+(tid1<<3)+2 ] = rg_k2 ;
            smem[sml_len+(big_warp_id<<8)+(tid1<<3)+3 ] = rg_k3 ;
            smem[sml_len+(big_warp_id<<8)+(tid1<<3)+4 ] = rg_k4 ;
            smem[sml_len+(big_warp_id<<8)+(tid1<<3)+5 ] = rg_k5 ;
            smem[sml_len+(big_warp_id<<8)+(tid1<<3)+6 ] = rg_k6 ;
            smem[sml_len+(big_warp_id<<8)+(tid1<<3)+7 ] = rg_k7 ;
            tmem[sml_len+(big_warp_id<<8)+(tid1<<3)+0 ] = rg_v0 ;
            tmem[sml_len+(big_warp_id<<8)+(tid1<<3)+1 ] = rg_v1 ;
            tmem[sml_len+(big_warp_id<<8)+(tid1<<3)+2 ] = rg_v2 ;
            tmem[sml_len+(big_warp_id<<8)+(tid1<<3)+3 ] = rg_v3 ;
            tmem[sml_len+(big_warp_id<<8)+(tid1<<3)+4 ] = rg_v4 ;
            tmem[sml_len+(big_warp_id<<8)+(tid1<<3)+5 ] = rg_v5 ;
            tmem[sml_len+(big_warp_id<<8)+(tid1<<3)+6 ] = rg_v6 ;
            tmem[sml_len+(big_warp_id<<8)+(tid1<<3)+7 ] = rg_v7 ;
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
        grp_start_off = (grp_start_wp_id<sml_wp)?grp_start_wp_id*128:sml_len+(grp_start_wp_id-sml_wp)*256;
        tmp_wp_id = grp_start_wp_id;
        lhs_len = ((tmp_wp_id+0<sml_wp)?128 :256 );
        rhs_len = ((tmp_wp_id+1<sml_wp)?128 :256 );
        gran = (warp_id<sml_wp)?(tid1<<2): (tid1<<3);
        if((warp_id&1)==0){
            gran += 0;
        }
        if((warp_id&1)==1){
            gran += ((tmp_wp_id+0<sml_wp)?128 :256 );
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
            rg_k4 = p ? tmp_k0 : tmp_k1;
            rg_v4 = p ? tmp_v0 : tmp_v1;
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
            rg_k5 = p ? tmp_k0 : tmp_k1;
            rg_v5 = p ? tmp_v0 : tmp_v1;
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
            rg_k6 = p ? tmp_k0 : tmp_k1;
            rg_v6 = p ? tmp_v0 : tmp_v1;
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
            rg_k7 = p ? tmp_k0 : tmp_k1;
            rg_v7 = p ? tmp_v0 : tmp_v1;
        }
        __syncthreads();
        // Store merged results back to shared memory
        if(sml_warp){
            smem[grp_start_off+gran+0 ] = rg_k0 ;
            smem[grp_start_off+gran+1 ] = rg_k1 ;
            smem[grp_start_off+gran+2 ] = rg_k2 ;
            smem[grp_start_off+gran+3 ] = rg_k3 ;
            tmem[grp_start_off+gran+0 ] = rg_v0 ;
            tmem[grp_start_off+gran+1 ] = rg_v1 ;
            tmem[grp_start_off+gran+2 ] = rg_v2 ;
            tmem[grp_start_off+gran+3 ] = rg_v3 ;
        } else {
            smem[grp_start_off+gran+0 ] = rg_k0 ;
            smem[grp_start_off+gran+1 ] = rg_k1 ;
            smem[grp_start_off+gran+2 ] = rg_k2 ;
            smem[grp_start_off+gran+3 ] = rg_k3 ;
            smem[grp_start_off+gran+4 ] = rg_k4 ;
            smem[grp_start_off+gran+5 ] = rg_k5 ;
            smem[grp_start_off+gran+6 ] = rg_k6 ;
            smem[grp_start_off+gran+7 ] = rg_k7 ;
            tmem[grp_start_off+gran+0 ] = rg_v0 ;
            tmem[grp_start_off+gran+1 ] = rg_v1 ;
            tmem[grp_start_off+gran+2 ] = rg_v2 ;
            tmem[grp_start_off+gran+3 ] = rg_v3 ;
            tmem[grp_start_off+gran+4 ] = rg_v4 ;
            tmem[grp_start_off+gran+5 ] = rg_v5 ;
            tmem[grp_start_off+gran+6 ] = rg_v6 ;
            tmem[grp_start_off+gran+7 ] = rg_v7 ;
        }
        __syncthreads();
        // Step 1
        grp_start_wp_id = ((warp_id>>2)<<2);
        grp_start_off = (grp_start_wp_id<sml_wp)?grp_start_wp_id*128:sml_len+(grp_start_wp_id-sml_wp)*256;
        tmp_wp_id = grp_start_wp_id;
        lhs_len = ((tmp_wp_id+0<sml_wp)?128 :256 )+
                  ((tmp_wp_id+1<sml_wp)?128 :256 );
        rhs_len = ((tmp_wp_id+2<sml_wp)?128 :256 )+
                  ((tmp_wp_id+3<sml_wp)?128 :256 );
        gran = (warp_id<sml_wp)?(tid1<<2): (tid1<<3);
        if((warp_id&3)==0){
            gran += 0;
        }
        if((warp_id&3)==1){
            gran += ((tmp_wp_id+0<sml_wp)?128 :256 );
        }
        if((warp_id&3)==2){
            gran += ((tmp_wp_id+0<sml_wp)?128 :256 )+
                    ((tmp_wp_id+1<sml_wp)?128 :256 );
        }
        if((warp_id&3)==3){
            gran += ((tmp_wp_id+0<sml_wp)?128 :256 )+
                    ((tmp_wp_id+1<sml_wp)?128 :256 )+
                    ((tmp_wp_id+2<sml_wp)?128 :256 );
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
            rg_k4 = p ? tmp_k0 : tmp_k1;
            rg_v4 = p ? tmp_v0 : tmp_v1;
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
            rg_k5 = p ? tmp_k0 : tmp_k1;
            rg_v5 = p ? tmp_v0 : tmp_v1;
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
            rg_k6 = p ? tmp_k0 : tmp_k1;
            rg_v6 = p ? tmp_v0 : tmp_v1;
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
            rg_k7 = p ? tmp_k0 : tmp_k1;
            rg_v7 = p ? tmp_v0 : tmp_v1;
        }
        if(sml_warp){
        } else {
        }
        if(sml_warp){
            if((tid<<2)+0 <seg_size) keyB[k+(tid<<2)+0 ] = rg_k0 ;
            if((tid<<2)+1 <seg_size) keyB[k+(tid<<2)+1 ] = rg_k1 ;
            if((tid<<2)+2 <seg_size) keyB[k+(tid<<2)+2 ] = rg_k2 ;
            if((tid<<2)+3 <seg_size) keyB[k+(tid<<2)+3 ] = rg_k3 ;
            if((tid<<2)+0 <seg_size) valB[k+(tid<<2)+0 ] = val[k+rg_v0 ];
            if((tid<<2)+1 <seg_size) valB[k+(tid<<2)+1 ] = val[k+rg_v1 ];
            if((tid<<2)+2 <seg_size) valB[k+(tid<<2)+2 ] = val[k+rg_v2 ];
            if((tid<<2)+3 <seg_size) valB[k+(tid<<2)+3 ] = val[k+rg_v3 ];
        } else {
            if((tid<<3)+0 -sml_len<seg_size) keyB[k+(tid<<3)+0 -sml_len] = rg_k0 ;
            if((tid<<3)+1 -sml_len<seg_size) keyB[k+(tid<<3)+1 -sml_len] = rg_k1 ;
            if((tid<<3)+2 -sml_len<seg_size) keyB[k+(tid<<3)+2 -sml_len] = rg_k2 ;
            if((tid<<3)+3 -sml_len<seg_size) keyB[k+(tid<<3)+3 -sml_len] = rg_k3 ;
            if((tid<<3)+4 -sml_len<seg_size) keyB[k+(tid<<3)+4 -sml_len] = rg_k4 ;
            if((tid<<3)+5 -sml_len<seg_size) keyB[k+(tid<<3)+5 -sml_len] = rg_k5 ;
            if((tid<<3)+6 -sml_len<seg_size) keyB[k+(tid<<3)+6 -sml_len] = rg_k6 ;
            if((tid<<3)+7 -sml_len<seg_size) keyB[k+(tid<<3)+7 -sml_len] = rg_k7 ;
            if((tid<<3)+0 -sml_len<seg_size) valB[k+(tid<<3)+0 -sml_len] = val[k+rg_v0 ];
            if((tid<<3)+1 -sml_len<seg_size) valB[k+(tid<<3)+1 -sml_len] = val[k+rg_v1 ];
            if((tid<<3)+2 -sml_len<seg_size) valB[k+(tid<<3)+2 -sml_len] = val[k+rg_v2 ];
            if((tid<<3)+3 -sml_len<seg_size) valB[k+(tid<<3)+3 -sml_len] = val[k+rg_v3 ];
            if((tid<<3)+4 -sml_len<seg_size) valB[k+(tid<<3)+4 -sml_len] = val[k+rg_v4 ];
            if((tid<<3)+5 -sml_len<seg_size) valB[k+(tid<<3)+5 -sml_len] = val[k+rg_v5 ];
            if((tid<<3)+6 -sml_len<seg_size) valB[k+(tid<<3)+6 -sml_len] = val[k+rg_v6 ];
            if((tid<<3)+7 -sml_len<seg_size) valB[k+(tid<<3)+7 -sml_len] = val[k+rg_v7 ];
        }
    }
}
/* block tcf1 tcf2 quiet real_kern */
/*   256    4    8  true      true */
template<class T>
__global__
void gen_bk256_tc8_r1025_r2048_strd(
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
    int k;
    int seg_size;
    int ext_seg_size;
    if(bin_it < bin_size) {
        k = segs[bin[bin_it]];
        seg_size = ((bin[bin_it]==length-1)?n:segs[bin[bin_it]+1])-segs[bin[bin_it]];
        ext_seg_size = ((seg_size + 127) / 128) * 128;
        int big_wp = (ext_seg_size - blockDim.x * 4) / 128;
        int sml_wp = blockDim.x / 32 - big_wp;
        int sml_len = sml_wp * 128;
        const int big_warp_id = (warp_id - sml_wp < 0)? 0: warp_id - sml_wp;
        bool sml_warp = warp_id < sml_wp;
        if(sml_warp) {
            rg_k0 = key[k+(warp_id<<7)+tid1+0   ];
            rg_k1 = key[k+(warp_id<<7)+tid1+32  ];
            rg_k2 = key[k+(warp_id<<7)+tid1+64  ];
            rg_k3 = key[k+(warp_id<<7)+tid1+96  ];
            rg_v0 = (warp_id<<7)+tid1+0   ;
            rg_v1 = (warp_id<<7)+tid1+32  ;
            rg_v2 = (warp_id<<7)+tid1+64  ;
            rg_v3 = (warp_id<<7)+tid1+96  ;
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
        } else {
            rg_k0  = (sml_len+tid1+(big_warp_id<<8)+0   <seg_size)?key[k+sml_len+tid1+(big_warp_id<<8)+0   ]:INT_MAX;
            rg_k1  = (sml_len+tid1+(big_warp_id<<8)+32  <seg_size)?key[k+sml_len+tid1+(big_warp_id<<8)+32  ]:INT_MAX;
            rg_k2  = (sml_len+tid1+(big_warp_id<<8)+64  <seg_size)?key[k+sml_len+tid1+(big_warp_id<<8)+64  ]:INT_MAX;
            rg_k3  = (sml_len+tid1+(big_warp_id<<8)+96  <seg_size)?key[k+sml_len+tid1+(big_warp_id<<8)+96  ]:INT_MAX;
            rg_k4  = (sml_len+tid1+(big_warp_id<<8)+128 <seg_size)?key[k+sml_len+tid1+(big_warp_id<<8)+128 ]:INT_MAX;
            rg_k5  = (sml_len+tid1+(big_warp_id<<8)+160 <seg_size)?key[k+sml_len+tid1+(big_warp_id<<8)+160 ]:INT_MAX;
            rg_k6  = (sml_len+tid1+(big_warp_id<<8)+192 <seg_size)?key[k+sml_len+tid1+(big_warp_id<<8)+192 ]:INT_MAX;
            rg_k7  = (sml_len+tid1+(big_warp_id<<8)+224 <seg_size)?key[k+sml_len+tid1+(big_warp_id<<8)+224 ]:INT_MAX;
            if(sml_len+tid1+(big_warp_id<<8)+0   <seg_size) rg_v0  = sml_len+tid1+(big_warp_id<<8)+0   ;
            if(sml_len+tid1+(big_warp_id<<8)+32  <seg_size) rg_v1  = sml_len+tid1+(big_warp_id<<8)+32  ;
            if(sml_len+tid1+(big_warp_id<<8)+64  <seg_size) rg_v2  = sml_len+tid1+(big_warp_id<<8)+64  ;
            if(sml_len+tid1+(big_warp_id<<8)+96  <seg_size) rg_v3  = sml_len+tid1+(big_warp_id<<8)+96  ;
            if(sml_len+tid1+(big_warp_id<<8)+128 <seg_size) rg_v4  = sml_len+tid1+(big_warp_id<<8)+128 ;
            if(sml_len+tid1+(big_warp_id<<8)+160 <seg_size) rg_v5  = sml_len+tid1+(big_warp_id<<8)+160 ;
            if(sml_len+tid1+(big_warp_id<<8)+192 <seg_size) rg_v6  = sml_len+tid1+(big_warp_id<<8)+192 ;
            if(sml_len+tid1+(big_warp_id<<8)+224 <seg_size) rg_v7  = sml_len+tid1+(big_warp_id<<8)+224 ;
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
        // Store register results to shared memory
        if(sml_warp) {
            smem[(warp_id<<7)+(tid1<<2)+0 ] = rg_k0 ;
            smem[(warp_id<<7)+(tid1<<2)+1 ] = rg_k1 ;
            smem[(warp_id<<7)+(tid1<<2)+2 ] = rg_k2 ;
            smem[(warp_id<<7)+(tid1<<2)+3 ] = rg_k3 ;
            tmem[(warp_id<<7)+(tid1<<2)+0 ] = rg_v0 ;
            tmem[(warp_id<<7)+(tid1<<2)+1 ] = rg_v1 ;
            tmem[(warp_id<<7)+(tid1<<2)+2 ] = rg_v2 ;
            tmem[(warp_id<<7)+(tid1<<2)+3 ] = rg_v3 ;
        } else {
            smem[sml_len+(big_warp_id<<8)+(tid1<<3)+0 ] = rg_k0 ;
            smem[sml_len+(big_warp_id<<8)+(tid1<<3)+1 ] = rg_k1 ;
            smem[sml_len+(big_warp_id<<8)+(tid1<<3)+2 ] = rg_k2 ;
            smem[sml_len+(big_warp_id<<8)+(tid1<<3)+3 ] = rg_k3 ;
            smem[sml_len+(big_warp_id<<8)+(tid1<<3)+4 ] = rg_k4 ;
            smem[sml_len+(big_warp_id<<8)+(tid1<<3)+5 ] = rg_k5 ;
            smem[sml_len+(big_warp_id<<8)+(tid1<<3)+6 ] = rg_k6 ;
            smem[sml_len+(big_warp_id<<8)+(tid1<<3)+7 ] = rg_k7 ;
            tmem[sml_len+(big_warp_id<<8)+(tid1<<3)+0 ] = rg_v0 ;
            tmem[sml_len+(big_warp_id<<8)+(tid1<<3)+1 ] = rg_v1 ;
            tmem[sml_len+(big_warp_id<<8)+(tid1<<3)+2 ] = rg_v2 ;
            tmem[sml_len+(big_warp_id<<8)+(tid1<<3)+3 ] = rg_v3 ;
            tmem[sml_len+(big_warp_id<<8)+(tid1<<3)+4 ] = rg_v4 ;
            tmem[sml_len+(big_warp_id<<8)+(tid1<<3)+5 ] = rg_v5 ;
            tmem[sml_len+(big_warp_id<<8)+(tid1<<3)+6 ] = rg_v6 ;
            tmem[sml_len+(big_warp_id<<8)+(tid1<<3)+7 ] = rg_v7 ;
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
        grp_start_off = (grp_start_wp_id<sml_wp)?grp_start_wp_id*128:sml_len+(grp_start_wp_id-sml_wp)*256;
        tmp_wp_id = grp_start_wp_id;
        lhs_len = ((tmp_wp_id+0<sml_wp)?128 :256 );
        rhs_len = ((tmp_wp_id+1<sml_wp)?128 :256 );
        gran = (warp_id<sml_wp)?(tid1<<2): (tid1<<3);
        if((warp_id&1)==0){
            gran += 0;
        }
        if((warp_id&1)==1){
            gran += ((tmp_wp_id+0<sml_wp)?128 :256 );
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
            rg_k4 = p ? tmp_k0 : tmp_k1;
            rg_v4 = p ? tmp_v0 : tmp_v1;
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
            rg_k5 = p ? tmp_k0 : tmp_k1;
            rg_v5 = p ? tmp_v0 : tmp_v1;
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
            rg_k6 = p ? tmp_k0 : tmp_k1;
            rg_v6 = p ? tmp_v0 : tmp_v1;
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
            rg_k7 = p ? tmp_k0 : tmp_k1;
            rg_v7 = p ? tmp_v0 : tmp_v1;
        }
        __syncthreads();
        // Store merged results back to shared memory
        if(sml_warp){
            smem[grp_start_off+gran+0 ] = rg_k0 ;
            smem[grp_start_off+gran+1 ] = rg_k1 ;
            smem[grp_start_off+gran+2 ] = rg_k2 ;
            smem[grp_start_off+gran+3 ] = rg_k3 ;
            tmem[grp_start_off+gran+0 ] = rg_v0 ;
            tmem[grp_start_off+gran+1 ] = rg_v1 ;
            tmem[grp_start_off+gran+2 ] = rg_v2 ;
            tmem[grp_start_off+gran+3 ] = rg_v3 ;
        } else {
            smem[grp_start_off+gran+0 ] = rg_k0 ;
            smem[grp_start_off+gran+1 ] = rg_k1 ;
            smem[grp_start_off+gran+2 ] = rg_k2 ;
            smem[grp_start_off+gran+3 ] = rg_k3 ;
            smem[grp_start_off+gran+4 ] = rg_k4 ;
            smem[grp_start_off+gran+5 ] = rg_k5 ;
            smem[grp_start_off+gran+6 ] = rg_k6 ;
            smem[grp_start_off+gran+7 ] = rg_k7 ;
            tmem[grp_start_off+gran+0 ] = rg_v0 ;
            tmem[grp_start_off+gran+1 ] = rg_v1 ;
            tmem[grp_start_off+gran+2 ] = rg_v2 ;
            tmem[grp_start_off+gran+3 ] = rg_v3 ;
            tmem[grp_start_off+gran+4 ] = rg_v4 ;
            tmem[grp_start_off+gran+5 ] = rg_v5 ;
            tmem[grp_start_off+gran+6 ] = rg_v6 ;
            tmem[grp_start_off+gran+7 ] = rg_v7 ;
        }
        __syncthreads();
        // Step 1
        grp_start_wp_id = ((warp_id>>2)<<2);
        grp_start_off = (grp_start_wp_id<sml_wp)?grp_start_wp_id*128:sml_len+(grp_start_wp_id-sml_wp)*256;
        tmp_wp_id = grp_start_wp_id;
        lhs_len = ((tmp_wp_id+0<sml_wp)?128 :256 )+
                  ((tmp_wp_id+1<sml_wp)?128 :256 );
        rhs_len = ((tmp_wp_id+2<sml_wp)?128 :256 )+
                  ((tmp_wp_id+3<sml_wp)?128 :256 );
        gran = (warp_id<sml_wp)?(tid1<<2): (tid1<<3);
        if((warp_id&3)==0){
            gran += 0;
        }
        if((warp_id&3)==1){
            gran += ((tmp_wp_id+0<sml_wp)?128 :256 );
        }
        if((warp_id&3)==2){
            gran += ((tmp_wp_id+0<sml_wp)?128 :256 )+
                    ((tmp_wp_id+1<sml_wp)?128 :256 );
        }
        if((warp_id&3)==3){
            gran += ((tmp_wp_id+0<sml_wp)?128 :256 )+
                    ((tmp_wp_id+1<sml_wp)?128 :256 )+
                    ((tmp_wp_id+2<sml_wp)?128 :256 );
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
            rg_k4 = p ? tmp_k0 : tmp_k1;
            rg_v4 = p ? tmp_v0 : tmp_v1;
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
            rg_k5 = p ? tmp_k0 : tmp_k1;
            rg_v5 = p ? tmp_v0 : tmp_v1;
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
            rg_k6 = p ? tmp_k0 : tmp_k1;
            rg_v6 = p ? tmp_v0 : tmp_v1;
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
            rg_k7 = p ? tmp_k0 : tmp_k1;
            rg_v7 = p ? tmp_v0 : tmp_v1;
        }
        __syncthreads();
        // Store merged results back to shared memory
        if(sml_warp){
            smem[grp_start_off+gran+0 ] = rg_k0 ;
            smem[grp_start_off+gran+1 ] = rg_k1 ;
            smem[grp_start_off+gran+2 ] = rg_k2 ;
            smem[grp_start_off+gran+3 ] = rg_k3 ;
            tmem[grp_start_off+gran+0 ] = rg_v0 ;
            tmem[grp_start_off+gran+1 ] = rg_v1 ;
            tmem[grp_start_off+gran+2 ] = rg_v2 ;
            tmem[grp_start_off+gran+3 ] = rg_v3 ;
        } else {
            smem[grp_start_off+gran+0 ] = rg_k0 ;
            smem[grp_start_off+gran+1 ] = rg_k1 ;
            smem[grp_start_off+gran+2 ] = rg_k2 ;
            smem[grp_start_off+gran+3 ] = rg_k3 ;
            smem[grp_start_off+gran+4 ] = rg_k4 ;
            smem[grp_start_off+gran+5 ] = rg_k5 ;
            smem[grp_start_off+gran+6 ] = rg_k6 ;
            smem[grp_start_off+gran+7 ] = rg_k7 ;
            tmem[grp_start_off+gran+0 ] = rg_v0 ;
            tmem[grp_start_off+gran+1 ] = rg_v1 ;
            tmem[grp_start_off+gran+2 ] = rg_v2 ;
            tmem[grp_start_off+gran+3 ] = rg_v3 ;
            tmem[grp_start_off+gran+4 ] = rg_v4 ;
            tmem[grp_start_off+gran+5 ] = rg_v5 ;
            tmem[grp_start_off+gran+6 ] = rg_v6 ;
            tmem[grp_start_off+gran+7 ] = rg_v7 ;
        }
        __syncthreads();
        // Step 2
        grp_start_wp_id = ((warp_id>>3)<<3);
        grp_start_off = (grp_start_wp_id<sml_wp)?grp_start_wp_id*128:sml_len+(grp_start_wp_id-sml_wp)*256;
        tmp_wp_id = grp_start_wp_id;
        lhs_len = ((tmp_wp_id+0<sml_wp)?128 :256 )+
                  ((tmp_wp_id+1<sml_wp)?128 :256 )+
                  ((tmp_wp_id+2<sml_wp)?128 :256 )+
                  ((tmp_wp_id+3<sml_wp)?128 :256 );
        rhs_len = ((tmp_wp_id+4<sml_wp)?128 :256 )+
                  ((tmp_wp_id+5<sml_wp)?128 :256 )+
                  ((tmp_wp_id+6<sml_wp)?128 :256 )+
                  ((tmp_wp_id+7<sml_wp)?128 :256 );
        gran = (warp_id<sml_wp)?(tid1<<2): (tid1<<3);
        if((warp_id&7)==0){
            gran += 0;
        }
        if((warp_id&7)==1){
            gran += ((tmp_wp_id+0<sml_wp)?128 :256 );
        }
        if((warp_id&7)==2){
            gran += ((tmp_wp_id+0<sml_wp)?128 :256 )+
                    ((tmp_wp_id+1<sml_wp)?128 :256 );
        }
        if((warp_id&7)==3){
            gran += ((tmp_wp_id+0<sml_wp)?128 :256 )+
                    ((tmp_wp_id+1<sml_wp)?128 :256 )+
                    ((tmp_wp_id+2<sml_wp)?128 :256 );
        }
        if((warp_id&7)==4){
            gran += ((tmp_wp_id+0<sml_wp)?128 :256 )+
                    ((tmp_wp_id+1<sml_wp)?128 :256 )+
                    ((tmp_wp_id+2<sml_wp)?128 :256 )+
                    ((tmp_wp_id+3<sml_wp)?128 :256 );
        }
        if((warp_id&7)==5){
            gran += ((tmp_wp_id+0<sml_wp)?128 :256 )+
                    ((tmp_wp_id+1<sml_wp)?128 :256 )+
                    ((tmp_wp_id+2<sml_wp)?128 :256 )+
                    ((tmp_wp_id+3<sml_wp)?128 :256 )+
                    ((tmp_wp_id+4<sml_wp)?128 :256 );
        }
        if((warp_id&7)==6){
            gran += ((tmp_wp_id+0<sml_wp)?128 :256 )+
                    ((tmp_wp_id+1<sml_wp)?128 :256 )+
                    ((tmp_wp_id+2<sml_wp)?128 :256 )+
                    ((tmp_wp_id+3<sml_wp)?128 :256 )+
                    ((tmp_wp_id+4<sml_wp)?128 :256 )+
                    ((tmp_wp_id+5<sml_wp)?128 :256 );
        }
        if((warp_id&7)==7){
            gran += ((tmp_wp_id+0<sml_wp)?128 :256 )+
                    ((tmp_wp_id+1<sml_wp)?128 :256 )+
                    ((tmp_wp_id+2<sml_wp)?128 :256 )+
                    ((tmp_wp_id+3<sml_wp)?128 :256 )+
                    ((tmp_wp_id+4<sml_wp)?128 :256 )+
                    ((tmp_wp_id+5<sml_wp)?128 :256 )+
                    ((tmp_wp_id+6<sml_wp)?128 :256 );
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
            rg_k4 = p ? tmp_k0 : tmp_k1;
            rg_v4 = p ? tmp_v0 : tmp_v1;
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
            rg_k5 = p ? tmp_k0 : tmp_k1;
            rg_v5 = p ? tmp_v0 : tmp_v1;
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
            rg_k6 = p ? tmp_k0 : tmp_k1;
            rg_v6 = p ? tmp_v0 : tmp_v1;
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
            rg_k7 = p ? tmp_k0 : tmp_k1;
            rg_v7 = p ? tmp_v0 : tmp_v1;
        }
        if(sml_warp){
            rg_k1  = __shfl_xor(rg_k1 , 0x1 );
            rg_k3  = __shfl_xor(rg_k3 , 0x1 );
            rg_v1  = __shfl_xor(rg_v1 , 0x1 );
            rg_v3  = __shfl_xor(rg_v3 , 0x1 );
            if(tid1&0x1 ) SWP(uintT, rg_k0 , rg_k1 , int, rg_v0 , rg_v1 );
            if(tid1&0x1 ) SWP(uintT, rg_k2 , rg_k3 , int, rg_v2 , rg_v3 );
            rg_k1  = __shfl_xor(rg_k1 , 0x1 );
            rg_k3  = __shfl_xor(rg_k3 , 0x1 );
            rg_v1  = __shfl_xor(rg_v1 , 0x1 );
            rg_v3  = __shfl_xor(rg_v3 , 0x1 );
            rg_k2  = __shfl_xor(rg_k2 , 0x2 );
            rg_k3  = __shfl_xor(rg_k3 , 0x2 );
            rg_v2  = __shfl_xor(rg_v2 , 0x2 );
            rg_v3  = __shfl_xor(rg_v3 , 0x2 );
            if(tid1&0x2 ) SWP(uintT, rg_k0 , rg_k2 , int, rg_v0 , rg_v2 );
            if(tid1&0x2 ) SWP(uintT, rg_k1 , rg_k3 , int, rg_v1 , rg_v3 );
            rg_k2  = __shfl_xor(rg_k2 , 0x2 );
            rg_k3  = __shfl_xor(rg_k3 , 0x2 );
            rg_v2  = __shfl_xor(rg_v2 , 0x2 );
            rg_v3  = __shfl_xor(rg_v3 , 0x2 );
            rg_k1  = __shfl_xor(rg_k1 , 0x4 );
            rg_k3  = __shfl_xor(rg_k3 , 0x4 );
            rg_v1  = __shfl_xor(rg_v1 , 0x4 );
            rg_v3  = __shfl_xor(rg_v3 , 0x4 );
            if(tid1&0x4 ) SWP(uintT, rg_k0 , rg_k1 , int, rg_v0 , rg_v1 );
            if(tid1&0x4 ) SWP(uintT, rg_k2 , rg_k3 , int, rg_v2 , rg_v3 );
            rg_k1  = __shfl_xor(rg_k1 , 0x4 );
            rg_k3  = __shfl_xor(rg_k3 , 0x4 );
            rg_v1  = __shfl_xor(rg_v1 , 0x4 );
            rg_v3  = __shfl_xor(rg_v3 , 0x4 );
            rg_k2  = __shfl_xor(rg_k2 , 0x8 );
            rg_k3  = __shfl_xor(rg_k3 , 0x8 );
            rg_v2  = __shfl_xor(rg_v2 , 0x8 );
            rg_v3  = __shfl_xor(rg_v3 , 0x8 );
            if(tid1&0x8 ) SWP(uintT, rg_k0 , rg_k2 , int, rg_v0 , rg_v2 );
            if(tid1&0x8 ) SWP(uintT, rg_k1 , rg_k3 , int, rg_v1 , rg_v3 );
            rg_k2  = __shfl_xor(rg_k2 , 0x8 );
            rg_k3  = __shfl_xor(rg_k3 , 0x8 );
            rg_v2  = __shfl_xor(rg_v2 , 0x8 );
            rg_v3  = __shfl_xor(rg_v3 , 0x8 );
            rg_k1  = __shfl_xor(rg_k1 , 0x10);
            rg_k3  = __shfl_xor(rg_k3 , 0x10);
            rg_v1  = __shfl_xor(rg_v1 , 0x10);
            rg_v3  = __shfl_xor(rg_v3 , 0x10);
            if(tid1&0x10) SWP(uintT, rg_k0 , rg_k1 , int, rg_v0 , rg_v1 );
            if(tid1&0x10) SWP(uintT, rg_k2 , rg_k3 , int, rg_v2 , rg_v3 );
            rg_k1  = __shfl_xor(rg_k1 , 0x10);
            rg_k3  = __shfl_xor(rg_k3 , 0x10);
            rg_v1  = __shfl_xor(rg_v1 , 0x10);
            rg_v3  = __shfl_xor(rg_v3 , 0x10);
        } else {
            rg_k1  = __shfl_xor(rg_k1 , 0x1 );
            rg_k3  = __shfl_xor(rg_k3 , 0x1 );
            rg_k5  = __shfl_xor(rg_k5 , 0x1 );
            rg_k7  = __shfl_xor(rg_k7 , 0x1 );
            rg_v1  = __shfl_xor(rg_v1 , 0x1 );
            rg_v3  = __shfl_xor(rg_v3 , 0x1 );
            rg_v5  = __shfl_xor(rg_v5 , 0x1 );
            rg_v7  = __shfl_xor(rg_v7 , 0x1 );
            if(tid1&0x1 ) SWP(uintT, rg_k0 , rg_k1 , int, rg_v0 , rg_v1 );
            if(tid1&0x1 ) SWP(uintT, rg_k2 , rg_k3 , int, rg_v2 , rg_v3 );
            if(tid1&0x1 ) SWP(uintT, rg_k4 , rg_k5 , int, rg_v4 , rg_v5 );
            if(tid1&0x1 ) SWP(uintT, rg_k6 , rg_k7 , int, rg_v6 , rg_v7 );
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
            if(tid1&0x2 ) SWP(uintT, rg_k0 , rg_k2 , int, rg_v0 , rg_v2 );
            if(tid1&0x2 ) SWP(uintT, rg_k1 , rg_k3 , int, rg_v1 , rg_v3 );
            if(tid1&0x2 ) SWP(uintT, rg_k4 , rg_k6 , int, rg_v4 , rg_v6 );
            if(tid1&0x2 ) SWP(uintT, rg_k5 , rg_k7 , int, rg_v5 , rg_v7 );
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
            if(tid1&0x4 ) SWP(uintT, rg_k0 , rg_k4 , int, rg_v0 , rg_v4 );
            if(tid1&0x4 ) SWP(uintT, rg_k1 , rg_k5 , int, rg_v1 , rg_v5 );
            if(tid1&0x4 ) SWP(uintT, rg_k2 , rg_k6 , int, rg_v2 , rg_v6 );
            if(tid1&0x4 ) SWP(uintT, rg_k3 , rg_k7 , int, rg_v3 , rg_v7 );
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
            if(tid1&0x8 ) SWP(uintT, rg_k0 , rg_k1 , int, rg_v0 , rg_v1 );
            if(tid1&0x8 ) SWP(uintT, rg_k2 , rg_k3 , int, rg_v2 , rg_v3 );
            if(tid1&0x8 ) SWP(uintT, rg_k4 , rg_k5 , int, rg_v4 , rg_v5 );
            if(tid1&0x8 ) SWP(uintT, rg_k6 , rg_k7 , int, rg_v6 , rg_v7 );
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
            if(tid1&0x10) SWP(uintT, rg_k0 , rg_k2 , int, rg_v0 , rg_v2 );
            if(tid1&0x10) SWP(uintT, rg_k1 , rg_k3 , int, rg_v1 , rg_v3 );
            if(tid1&0x10) SWP(uintT, rg_k4 , rg_k6 , int, rg_v4 , rg_v6 );
            if(tid1&0x10) SWP(uintT, rg_k5 , rg_k7 , int, rg_v5 , rg_v7 );
            rg_k2  = __shfl_xor(rg_k2 , 0x10);
            rg_k3  = __shfl_xor(rg_k3 , 0x10);
            rg_k6  = __shfl_xor(rg_k6 , 0x10);
            rg_k7  = __shfl_xor(rg_k7 , 0x10);
            rg_v2  = __shfl_xor(rg_v2 , 0x10);
            rg_v3  = __shfl_xor(rg_v3 , 0x10);
            rg_v6  = __shfl_xor(rg_v6 , 0x10);
            rg_v7  = __shfl_xor(rg_v7 , 0x10);
        }
        if(sml_warp){
            if(tid1+(warp_id<<7)+0   <seg_size) keyB[k+tid1+(warp_id<<7)+0   ] = rg_k0 ;
            if(tid1+(warp_id<<7)+32  <seg_size) keyB[k+tid1+(warp_id<<7)+32  ] = rg_k2 ;
            if(tid1+(warp_id<<7)+64  <seg_size) keyB[k+tid1+(warp_id<<7)+64  ] = rg_k1 ;
            if(tid1+(warp_id<<7)+96  <seg_size) keyB[k+tid1+(warp_id<<7)+96  ] = rg_k3 ;
            if(tid1+(warp_id<<7)+0   <seg_size) valB[k+tid1+(warp_id<<7)+0   ] = val[k+rg_v0 ];
            if(tid1+(warp_id<<7)+32  <seg_size) valB[k+tid1+(warp_id<<7)+32  ] = val[k+rg_v2 ];
            if(tid1+(warp_id<<7)+64  <seg_size) valB[k+tid1+(warp_id<<7)+64  ] = val[k+rg_v1 ];
            if(tid1+(warp_id<<7)+96  <seg_size) valB[k+tid1+(warp_id<<7)+96  ] = val[k+rg_v3 ];
        } else {
            if(sml_len+tid1+(big_warp_id<<8)+0   <seg_size) keyB[k+sml_len+tid1+(big_warp_id<<8)+0   ] = rg_k0 ;
            if(sml_len+tid1+(big_warp_id<<8)+32  <seg_size) keyB[k+sml_len+tid1+(big_warp_id<<8)+32  ] = rg_k4 ;
            if(sml_len+tid1+(big_warp_id<<8)+64  <seg_size) keyB[k+sml_len+tid1+(big_warp_id<<8)+64  ] = rg_k1 ;
            if(sml_len+tid1+(big_warp_id<<8)+96  <seg_size) keyB[k+sml_len+tid1+(big_warp_id<<8)+96  ] = rg_k5 ;
            if(sml_len+tid1+(big_warp_id<<8)+128 <seg_size) keyB[k+sml_len+tid1+(big_warp_id<<8)+128 ] = rg_k2 ;
            if(sml_len+tid1+(big_warp_id<<8)+160 <seg_size) keyB[k+sml_len+tid1+(big_warp_id<<8)+160 ] = rg_k6 ;
            if(sml_len+tid1+(big_warp_id<<8)+192 <seg_size) keyB[k+sml_len+tid1+(big_warp_id<<8)+192 ] = rg_k3 ;
            if(sml_len+tid1+(big_warp_id<<8)+224 <seg_size) keyB[k+sml_len+tid1+(big_warp_id<<8)+224 ] = rg_k7 ;
            if(sml_len+tid1+(big_warp_id<<8)+0   <seg_size) valB[k+sml_len+tid1+(big_warp_id<<8)+0   ] = val[k+rg_v0 ];
            if(sml_len+tid1+(big_warp_id<<8)+32  <seg_size) valB[k+sml_len+tid1+(big_warp_id<<8)+32  ] = val[k+rg_v4 ];
            if(sml_len+tid1+(big_warp_id<<8)+64  <seg_size) valB[k+sml_len+tid1+(big_warp_id<<8)+64  ] = val[k+rg_v1 ];
            if(sml_len+tid1+(big_warp_id<<8)+96  <seg_size) valB[k+sml_len+tid1+(big_warp_id<<8)+96  ] = val[k+rg_v5 ];
            if(sml_len+tid1+(big_warp_id<<8)+128 <seg_size) valB[k+sml_len+tid1+(big_warp_id<<8)+128 ] = val[k+rg_v2 ];
            if(sml_len+tid1+(big_warp_id<<8)+160 <seg_size) valB[k+sml_len+tid1+(big_warp_id<<8)+160 ] = val[k+rg_v6 ];
            if(sml_len+tid1+(big_warp_id<<8)+192 <seg_size) valB[k+sml_len+tid1+(big_warp_id<<8)+192 ] = val[k+rg_v3 ];
            if(sml_len+tid1+(big_warp_id<<8)+224 <seg_size) valB[k+sml_len+tid1+(big_warp_id<<8)+224 ] = val[k+rg_v7 ];
        }
    }
}
#endif
