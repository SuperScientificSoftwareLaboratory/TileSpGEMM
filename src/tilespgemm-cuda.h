#include "common.h"
#include "utils.h"

__forceinline__ __device__ int sum_32_shfl(int sum)
{
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1)
        sum += __shfl_xor_sync(0xffffffff, sum, mask);

    return sum;
}

__forceinline__ __device__ int sum_16_shfl(int sum)
{
#pragma unroll
    for (int mask = 1; mask < HALFWARP_SIZE; mask <<= 1)
        sum += __shfl_xor_sync(-1, sum, mask);

    return sum;
}

__forceinline__ __device__ int binary_search_exact_kernel(const int *d_array, int l, int r, int key)
{
    while (l <= r)
    {
        int m = l + (r - l) / 2;
        int elem = d_array[m];
        // Check if x is present at mid
        if (elem == key)
            return m;

        // If x greater, ignore left half
        if (elem < key)
            l = m + 1;

        // If x is smaller, ignore right half
        else
            r = m - 1;
    }

    // if we reach here, then element was
    // not present
    return -1;
}

__forceinline__ __device__ int binary_search_exact_kernel_v2(const int *s_array, const int *d_array, int splitter,
                                                             int l, int r, int key)
{
    while (l <= r)
    {
        int m = l + (r - l) / 2;
        int elem = m < splitter ? s_array[m] : d_array[m];
        // Check if x is present at mid
        if (elem == key)
            return m;

        // If x greater, ignore left half
        if (elem < key)
            l = m + 1;

        // If x is smaller, ignore right half
        else
            r = m - 1;
    }

    // if we reach here, then element was
    // not present
    return -1;
}

__forceinline__ __device__ int binary_search_exact_uchar_kernel(const unsigned char *__restrict__ d_array, int l, int r, unsigned char key)
{
    while (l <= r)
    {
        int m = l + (r - l) / 2;
        unsigned char elem = d_array[m];
        // Check if x is present at mid
        if (elem == key)
            return m;

        // If x greater, ignore left half
        if (elem < key)
            l = m + 1;

        // If x is smaller, ignore right half
        else
            r = m - 1;
    }

    // if we reach here, then element was
    // not present
    return -1;
}

__forceinline__ __device__ int binary_search_right_boundary_kernel(const int *__restrict__ d_row_pointer,
                                                                   const int key_input,
                                                                   const int size)
{
    int start = 0;
    int stop = size - 1;
    int median;
    int key_median;

    while (stop >= start)
    {
        median = (stop + start) / 2;

        key_median = ld_gbl_int32(d_row_pointer + median);

        if (key_input >= key_median)
            start = median + 1;
        else
            stop = median - 1;
    }

    return start - 1;
}

int binary_search_right_boundary_kernel_cpu(const int *d_row_pointer,
                                            const int key_input,
                                            const int size)
{
    int start = 0;
    int stop = size - 1;
    int median;
    int key_median;

    while (stop >= start)
    {
        median = (stop + start) / 2;

        key_median = d_row_pointer[median];

        if (key_input >= key_median)
            start = median + 1;
        else
            stop = median - 1;
    }

    return start - 1;
}

__device__ __forceinline__ int intersection_binarysearch_kernel(const int *d_arraya, int abase, int astop, int lena,
                                                                const int *d_arrayb, int bbase, int bstop, int lenb,
                                                                int *d_posa, int *d_posb, int lenpos, int *d_cnt,
                                                                int lane_id, int warpsize)
{
    if (lena == 0 || lenb == 0)
    {
    }
    else if (lena < lenb)
    {
        for (int i = lane_id; i < lena; i += warpsize)
        {
            int idxa = d_arraya[abase + i];
            int res = binary_search_exact_kernel(d_arrayb + bbase, 0, lenb - 1, idxa);
            if (res != -1)
            {
                int pos = atomicAdd(d_cnt, 1);
                if (pos < lenpos)
                {
                    d_posa[pos] = i;
                    d_posb[pos] = res;
                }
            }
        }
    }
    else
    {
        for (int i = lane_id; i < lenb; i += warpsize)
        {
            int idxb = d_arrayb[bbase + i];
            int res = binary_search_exact_kernel(d_arraya + abase, 0, lena - 1, idxb);
            if (res != -1)
            {
                int pos = atomicAdd(d_cnt, 1);
                if (pos < lenpos)
                {
                    d_posa[pos] = res;
                    d_posb[pos] = i;
                }
            }
        }
    }

    return 0;
}

__device__ __forceinline__ int intersection_binarysearch_smem_kernel(const int *d_arraya, int abase, int astop, int lena,
                                                                     const int *d_arrayb, int bbase, int bstop, int lenb,
                                                                     int *s_intersection,
                                                                     int *d_posa, int *d_posb, int lenpos, int *d_cnt,
                                                                     int lane_id, int warpsize)
{
    if (lena == 0 || lenb == 0)
    {
    }
    else if (lena < lenb)
    {
        // optimize abase and lena, by search bstart and bstop in a
        const int bendidx = d_arrayb[bstop - 1];

        int use_smem = lenb <= SMEM_INTERSECTION_LEN && lena > SMEM_INTERSECTION_TH;
        if (use_smem)
        {
            for (int i = lane_id; i < lenb; i += warpsize)
                s_intersection[i] = d_arrayb[bbase + i];
        }

        for (int i = lane_id; i < lena; i += warpsize)
        {
            int idxa = d_arraya[abase + i];
            const int *searchspace = use_smem ? s_intersection : &d_arrayb[bbase];
            int res = binary_search_exact_kernel(searchspace, 0, lenb - 1, idxa);
            if (res != -1)
            {
                int pos = atomicAdd(d_cnt, 1);
                if (pos < lenpos)
                {
                    d_posa[pos] = i;
                    d_posb[pos] = res;
                }
            }
        }
    }
    else
    {
        // optimize abase and lena, by search bstart and bstop in a
        int use_smem = lena <= SMEM_INTERSECTION_LEN && lenb > SMEM_INTERSECTION_TH;
        if (use_smem)
        {
            for (int i = lane_id; i < lena; i += warpsize)
                s_intersection[i] = d_arraya[abase + i];
        }

        for (int i = lane_id; i < lenb; i += warpsize)
        {
            int idxb = d_arrayb[bbase + i];
            const int *searchspace = use_smem ? s_intersection : &d_arraya[abase];
            int res = binary_search_exact_kernel(searchspace, 0, lena - 1, idxb);
            if (res != -1)
            {
                int pos = atomicAdd(d_cnt, 1);
                if (pos < lenpos)
                {
                    d_posa[pos] = res;
                    d_posb[pos] = i;
                }
            }
        }
    }
    return 0;
}

__global__ void tile_spgemm_step1_cuda_spa_kernel(int *d_blkrowptrA, int *d_blkcolidxA, int blkmA,
                                                  int *d_blkrowptrB, int *d_blkcolidxB, int blknB,
                                                  int *d_blkrowptrC)
{
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_warp_id = global_id >> 5; //global_id / WARP_SIZE;
    __shared__ unsigned int bitmask[WARP_PER_BLOCK * SPA_INT_PER_WARP];

    if (global_warp_id >= blkmA)
        return;

    const int nmasks = ceil((float)blknB / (float)32);
    const int local_warp_id = threadIdx.x >> 5; //global_id / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    unsigned int *bitmask_local = &bitmask[local_warp_id * SPA_INT_PER_WARP];

    for (int i = lane_id; i < nmasks; i += WARP_SIZE)
        bitmask_local[i] = 0;

    int astart = d_blkrowptrA[global_warp_id];
    int astop = d_blkrowptrA[global_warp_id + 1];
    for (int i = astart; i < astop; i++)
    {
        int rowidx = d_blkcolidxA[i];
        int bstart = d_blkrowptrB[rowidx];
        int bstop = d_blkrowptrB[rowidx + 1];
        for (int j = bstart + lane_id; j < bstop; j += WARP_SIZE)
        {
            int colidx = d_blkcolidxB[j];
            unsigned int mask = 1 << (31 - colidx % 32);
            atomicOr(&bitmask_local[colidx / 32], mask);
        }
    }
    //__syncthreads();

    int cnt = 0;
    for (int i = lane_id; i < nmasks; i += WARP_SIZE)
        cnt += __popc(bitmask_local[i]);
    cnt = sum_32_shfl(cnt);

    if (!lane_id)
        d_blkrowptrC[global_warp_id] = cnt;
}

__global__ void tile_spgemm_step1_numeric_cuda_spa_kernel(int *d_blkrowptrA, int *d_blkcolidxA, int blkmA,
                                                          int *d_blkrowptrB, int *d_blkcolidxB, int blknB,
                                                          int *d_blkrowptrC, int *d_blkrowidxC, int *d_blkcolidxC,
                                                          int *d_spec_intersection_cnt, int *d_spec_intersection_posa, int *d_spec_intersection_posb)
{
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_warp_id = global_id >> 5; //global_id / WARP_SIZE;
    __shared__ unsigned int bitmask[WARP_PER_BLOCK * SPA_INT_PER_WARP];

    if (global_warp_id >= blkmA)
        return;

    const int nmasks = ceil((float)blknB / (float)32);
    const int nmasks_warpwise = ceil((float)nmasks / (float)WARP_SIZE) * WARP_SIZE; // make sure shfl func works
    const int local_warp_id = threadIdx.x >> 5;                                     //global_id / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    unsigned int *bitmask_local = &bitmask[local_warp_id * SPA_INT_PER_WARP];

    for (int i = lane_id; i < nmasks_warpwise; i += WARP_SIZE)
        bitmask_local[i] = 0;

    int cbase = d_blkrowptrC[global_warp_id];

    int astart = d_blkrowptrA[global_warp_id];
    int astop = d_blkrowptrA[global_warp_id + 1];
    for (int i = astart; i < astop; i++)
    {
        int rowidx = d_blkcolidxA[i];
        int bstart = d_blkrowptrB[rowidx];
        int bstop = d_blkrowptrB[rowidx + 1];
        for (int j = bstart + lane_id; j < bstop; j += WARP_SIZE)
        {
            int colidx = d_blkcolidxB[j];
            unsigned int mask = 1 << (31 - colidx % 32);
            atomicOr(&bitmask_local[colidx / 32], mask);
        }
    }

    int cnt = 0;
    int offset = 0;
    for (int i = lane_id; i < nmasks_warpwise; i += WARP_SIZE)
    {
        unsigned int maski = bitmask_local[i];
        int cnt = __popc(maski);

        // inclusive scan
        int cnt_scan = scan_32_shfl(cnt, lane_id);
        cnt_scan += offset;

        // sum
        offset = __shfl_sync(0xffffffff, cnt_scan, 31);

        // to exclusive scan
        cnt_scan -= cnt;

        // write to gmem
        int localoff = 0;
#pragma unroll
        for (int biti = 0; biti < 32; biti++)
        {
            if ((maski >> (31 - biti)) & 0x1)
            {
                d_blkrowidxC[cbase + cnt_scan + localoff] = global_warp_id;
                d_blkcolidxC[cbase + cnt_scan + localoff] = i * 32 + biti;
                localoff++;
            }
        }
    }
}

__global__ void tile_spgemm_step3_cuda_kernel_2level(const int *d_blkrowptrA,
                                                     const int *__restrict__ d_blkcolidxA,
                                                     const int *d_nnzb_A,
                                                     MAT_VAL_TYPE *d_blkcsr_Val_A,
                                                     unsigned char *d_blkcsr_Col_A,
                                                     unsigned char *d_blkcsr_Ptr_A,
                                                     unsigned short *d_blkmaskA,
                                                     int blkmA, int blknA, int numblkA, int nnzA,
                                                     const int *__restrict__ d_blkcolptrB,
                                                     const int *__restrict__ d_blkrowidxB,
                                                     const int *__restrict__ d_nnzb_B,
                                                     const MAT_VAL_TYPE *__restrict__ d_blkcsr_Val_B,
                                                     const unsigned char *__restrict__ d_blkcsr_Col_B,
                                                     const unsigned char *__restrict__ d_blkcsr_Ptr_B,
                                                     const unsigned short *__restrict__ d_blkmaskB,
                                                     int blkmB, int blknB, int numblkB, int nnzB,
                                                     int *d_blkrowidxC,
                                                     int *d_blkcolidxC,
                                                     unsigned char *d_blkcsr_Ptr_C,
                                                     int *d_nnzb_C,
                                                     unsigned short *d_blkmaskC,
                                                     int *d_blksmem_tny_cnt,
                                                     int *d_blksmem_sml_cnt,
                                                     int *d_blksmem_lrg_cnt,
                                                     int *d_blksmem_dns_cnt,
                                                     int *d_blksmem_ful_cnt,
                                                     int *d_blkid_smem_tny,
                                                     int *d_blkid_smem_sml,
                                                     int *d_blkid_smem_lrg,
                                                     int *d_blkid_smem_dns,
                                                     int *d_blkid_smem_ful,
                                                     int numblkC)
{
    const int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_warp_id = global_id >> 5; //global_id / WARP_SIZE;

    __shared__ unsigned int s_maskc[WARP_PER_BLOCK * BLOCK_SIZE];
    __shared__ unsigned short s_blkmaskB[WARP_PER_BLOCK * BLOCK_SIZE];

    __shared__ int s_matched_posa[WARP_PER_BLOCK * SPECULATIVE_INTERSECTION];
    __shared__ int s_matched_posb[WARP_PER_BLOCK * SPECULATIVE_INTERSECTION];
    __shared__ int s_matchedcnt[WARP_PER_BLOCK];

    __shared__ int s_blksmem_tny_cnt[WARP_PER_BLOCK];
    __shared__ int s_blksmem_sml_cnt[WARP_PER_BLOCK];
    __shared__ int s_blksmem_lrg_cnt[WARP_PER_BLOCK];
    __shared__ int s_blksmem_dns_cnt[WARP_PER_BLOCK];
    __shared__ int s_blksmem_ful_cnt[WARP_PER_BLOCK];

    __shared__ int s_blkid_smem_tny[WARP_PER_BLOCK * TILE_PER_WARP];
    __shared__ int s_blkid_smem_sml[WARP_PER_BLOCK * TILE_PER_WARP];
    __shared__ int s_blkid_smem_lrg[WARP_PER_BLOCK * TILE_PER_WARP];
    __shared__ int s_blkid_smem_dns[WARP_PER_BLOCK * TILE_PER_WARP];
    __shared__ int s_blkid_smem_ful[WARP_PER_BLOCK * TILE_PER_WARP];

    const int lane_id = (WARP_SIZE - 1) & threadIdx.x;

    const int local_warp_id = threadIdx.x >> 5; //threadIdx.x / WARP_SIZE;

    unsigned int *s_maskc_local = &s_maskc[local_warp_id * BLOCK_SIZE];
    unsigned short *s_blkmaskB_local = &s_blkmaskB[local_warp_id * BLOCK_SIZE];

    int *s_matched_posa_local = &s_matched_posa[local_warp_id * SPECULATIVE_INTERSECTION];
    int *s_matched_posb_local = &s_matched_posb[local_warp_id * SPECULATIVE_INTERSECTION];
    int *s_matchedcnt_local = &s_matchedcnt[local_warp_id];

    int *s_blksmem_tny_cnt_local = &s_blksmem_tny_cnt[local_warp_id];
    int *s_blksmem_sml_cnt_local = &s_blksmem_sml_cnt[local_warp_id];
    int *s_blksmem_lrg_cnt_local = &s_blksmem_lrg_cnt[local_warp_id];
    int *s_blksmem_dns_cnt_local = &s_blksmem_dns_cnt[local_warp_id];
    int *s_blksmem_ful_cnt_local = &s_blksmem_ful_cnt[local_warp_id];

    int *s_blkid_smem_tny_local = &s_blkid_smem_tny[local_warp_id * TILE_PER_WARP];
    int *s_blkid_smem_sml_local = &s_blkid_smem_sml[local_warp_id * TILE_PER_WARP];
    int *s_blkid_smem_lrg_local = &s_blkid_smem_lrg[local_warp_id * TILE_PER_WARP];
    int *s_blkid_smem_dns_local = &s_blkid_smem_dns[local_warp_id * TILE_PER_WARP];
    int *s_blkid_smem_ful_local = &s_blkid_smem_ful[local_warp_id * TILE_PER_WARP];

    int tile_start = global_warp_id * TILE_PER_WARP;
    if (tile_start >= numblkC)
        return;

    int tile_end = tile_start + TILE_PER_WARP; //(global_warp_id + 1) * TPW;
    tile_end = tile_end >= numblkC ? numblkC : tile_end;

    if (!lane_id)
    {
        s_blksmem_tny_cnt_local[0] = 0;
        s_blksmem_sml_cnt_local[0] = 0;
        s_blksmem_lrg_cnt_local[0] = 0;
        s_blksmem_dns_cnt_local[0] = 0;
        s_blksmem_ful_cnt_local[0] = 0;
    }

    for (int tilei = tile_start; tilei < tile_end; tilei++)
    {
        if (lane_id < BLOCK_SIZE)
            s_maskc_local[lane_id] = 0;
        if (!lane_id)
            s_matchedcnt_local[0] = 0;

        const int blki = d_blkrowidxC[tilei];
        const int blkj = d_blkcolidxC[tilei];

        const int abase = d_blkrowptrA[blki];
        const int astop = d_blkrowptrA[blki + 1];
        int lena = astop - abase;

        const int bbase = ld_gbl_int32(d_blkcolptrB + blkj);
        const int bstop = ld_gbl_int32(d_blkcolptrB + blkj + 1);
        int lenb = bstop - bbase;

        // deal with some special cases first
        if (lena == blknA && lenb == blkmB) // if both full
        {
            for (int i = 0; i < lena; i++)
            {
                int posa = i;
                int posb = i;

                if (lane_id < BLOCK_SIZE)
                {
                    s_blkmaskB_local[lane_id] = ld_gbl_ushort(&d_blkmaskB[(bbase + posb) * BLOCK_SIZE + lane_id]);
                }

                const int nnzastart = d_nnzb_A[(abase + posa)];
                int nnztotala = d_nnzb_A[(abase + posa) + 1] - nnzastart;

                for (int i = lane_id; i < nnztotala; i += WARP_SIZE)
                {
                    unsigned char rowcolidx = d_blkcsr_Col_A[nnzastart + i];
                    unsigned int maskb = s_blkmaskB_local[rowcolidx & 0xf];
                    atomicOr(&s_maskc_local[rowcolidx >> 4], maskb);
                }
            }
        }
        else if (lena == blknA && lenb != blkmB)
        {
            for (int i = 0; i < lenb; i++)
            {
                int posb = d_blkrowidxB[bbase + i];
                int posa = posb;

                if (lane_id < BLOCK_SIZE)
                {
                    s_blkmaskB_local[lane_id] = ld_gbl_ushort(&d_blkmaskB[(bbase + posb) * BLOCK_SIZE + lane_id]);
                }

                const int nnzastart = d_nnzb_A[(abase + posa)];
                int nnztotala = d_nnzb_A[(abase + posa) + 1] - nnzastart;

                for (int i = lane_id; i < nnztotala; i += WARP_SIZE)
                {
                    unsigned char rowcolidx = d_blkcsr_Col_A[nnzastart + i];
                    unsigned int maskb = s_blkmaskB_local[rowcolidx & 0xf];
                    atomicOr(&s_maskc_local[rowcolidx >> 4], maskb);
                }
            }
        }
        else if (lena != blknA && lenb == blkmB)
        {
            for (int i = 0; i < lenb; i++)
            {
                int posa = d_blkcolidxA[abase + i];
                int posb = posa;

                if (lane_id < BLOCK_SIZE)
                {
                    s_blkmaskB_local[lane_id] = ld_gbl_ushort(&d_blkmaskB[(bbase + posb) * BLOCK_SIZE + lane_id]);
                }

                const int nnzastart = d_nnzb_A[(abase + posa)];
                int nnztotala = d_nnzb_A[(abase + posa) + 1] - nnzastart;

                for (int i = lane_id; i < nnztotala; i += WARP_SIZE)
                {
                    unsigned char rowcolidx = d_blkcsr_Col_A[nnzastart + i];
                    unsigned int maskb = s_blkmaskB_local[rowcolidx & 0xf];
                    atomicOr(&s_maskc_local[rowcolidx >> 4], maskb);
                }
            }
        }
        else // the rest general case
        {
            int specres = intersection_binarysearch_kernel(d_blkcolidxA, abase, astop, lena,
                                                           d_blkrowidxB, bbase, bstop, lenb,
                                                           s_matched_posa_local, s_matched_posb_local,
                                                           SPECULATIVE_INTERSECTION, s_matchedcnt_local,
                                                           lane_id, WARP_SIZE);

            int matchedcnt = s_matchedcnt_local[0];

            if (matchedcnt <= SPECULATIVE_INTERSECTION && specres == 0)
            {
                for (int i = 0; i < matchedcnt; i++)
                {
                    int posa = s_matched_posa_local[i];
                    int posb = s_matched_posb_local[i];

                    if (lane_id < BLOCK_SIZE)
                    {
                        s_blkmaskB_local[lane_id] = ld_gbl_ushort(&d_blkmaskB[(bbase + posb) * BLOCK_SIZE + lane_id]);
                    }

                    const int nnzastart = d_nnzb_A[(abase + posa)];
                    int nnztotala = d_nnzb_A[(abase + posa) + 1] - nnzastart;

                    for (int li = lane_id; li < nnztotala; li += WARP_SIZE)
                    {
                        unsigned char rowcolidx = d_blkcsr_Col_A[nnzastart + li];
                        unsigned int maskb = s_blkmaskB_local[rowcolidx & 0xf];
                        atomicOr(&s_maskc_local[rowcolidx >> 4], maskb);
                    }
                }
            }
            else
            {

                const int astart = d_blkcolidxA[abase];
                const int aend = d_blkcolidxA[astop - 1];
                const int bstart = ld_gbl_int32(d_blkrowidxB + bbase);
                const int bend = ld_gbl_int32(d_blkrowidxB + bstop - 1);

                int posa_real = 0;
                int posb_real = 0;

                if (bstart > astart)
                {
                    int posa_real_new = binary_search_right_boundary_kernel(d_blkcolidxA + abase, bstart, lena);
                    posa_real = posa_real_new < 0 ? 0 : posa_real_new;
                }
                else if (bstart < astart)
                {
                    int posb_real_new = binary_search_right_boundary_kernel(d_blkrowidxB + bbase, astart, lenb);
                    posb_real = posb_real_new < 0 ? 0 : posb_real_new;
                }

                if (bstop < astop)
                {
                    int lena_new = binary_search_right_boundary_kernel(d_blkcolidxA + abase, bend, lena) + 1;
                    lena = lena_new > lena ? lena : lena_new;
                }
                else if (bstop > astop)
                {
                    int lenb_new = binary_search_right_boundary_kernel(d_blkrowidxB + bbase, aend, lenb) + 1;
                    lenb = lenb_new > lenb ? lenb : lenb_new;
                }

                for (int posa = 0; posa < lena; posa++)
                {
                    int idxa = d_blkcolidxA[abase + posa];
                    int posb = binary_search_right_boundary_kernel(d_blkrowidxB + bbase + posb_real, idxa, lenb - posb_real);
                    if (posb < 0)
                        continue;
                    if (posb > lenb - posb_real)
                        break;
                    int idxb = ld_gbl_int32(d_blkrowidxB + bbase + posb_real + posb);

                    if (idxa == idxb)
                    {
                        posb_real = posb_real + posb;
                        if (lane_id < BLOCK_SIZE)
                        {
                            s_blkmaskB_local[lane_id] = ld_gbl_ushort(d_blkmaskB + (bbase + posb_real) * BLOCK_SIZE + lane_id);
                        }

                        const int nnzastart = d_nnzb_A[(abase + posa)];
                        int nnztotala = d_nnzb_A[(abase + posa) + 1] - nnzastart;

                        for (int i = lane_id; i < nnztotala; i += WARP_SIZE)
                        {
                            unsigned char rowcolidx = d_blkcsr_Col_A[nnzastart + i];
                            unsigned int maskb = s_blkmaskB_local[rowcolidx & 0xf];
                            atomicOr(&s_maskc_local[rowcolidx >> 4], maskb);
                        }
                    }
                }
            }
        }

        unsigned int maskc = lane_id < BLOCK_SIZE ? s_maskc_local[lane_id] : 0;
        int nnzcnt = __popc(maskc); //lane_id < BLOCK_SIZE ? __popc(maskc) : 0;

        int nnzcnt_sum = sum_32_shfl(nnzcnt);

        if (nnzcnt_sum == 0)
        {
            if (!lane_id)
                d_nnzb_C[tilei] = 0;
        }
        else
        {
            int nnzcnt_scan = scan_32_shfl(nnzcnt, lane_id);
            if (lane_id < BLOCK_SIZE){
                long long int pos_c= (long long int)(tilei) * BLOCK_SIZE + lane_id;
                d_blkcsr_Ptr_C[pos_c] = nnzcnt_scan - nnzcnt;

            }

            if (lane_id < BLOCK_SIZE){
                long long int pos_c= (long long int)(tilei) * BLOCK_SIZE + lane_id;
                d_blkmaskC[tilei * BLOCK_SIZE + lane_id] = s_maskc_local[lane_id];
            }

            if (!lane_id)
            {
                d_nnzb_C[tilei] = nnzcnt_sum;

                if (nnzcnt_sum <= SMEM_TNY_TH)
                {
                    int pos = atomicAdd(s_blksmem_tny_cnt_local, 1);
                    s_blkid_smem_tny_local[pos] = tilei;
                }
                else if (SMEM_TNY_TH < nnzcnt_sum && nnzcnt_sum <= SMEM_SML_TH)
                {
                    int pos = atomicAdd(s_blksmem_sml_cnt_local, 1);
                    s_blkid_smem_sml_local[pos] = tilei;
                }
                else if (SMEM_SML_TH < nnzcnt_sum && nnzcnt_sum <= SMEM_LRG_TH)
                {
                    int pos = atomicAdd(s_blksmem_lrg_cnt_local, 1);
                    s_blkid_smem_lrg_local[pos] = tilei;
                }
                else if (SMEM_LRG_TH < nnzcnt_sum && nnzcnt_sum < SMEM_DNS_TH)
                {
                    int pos = atomicAdd(s_blksmem_dns_cnt_local, 1);
                    s_blkid_smem_dns_local[pos] = tilei;
                }
                else if (nnzcnt_sum == SMEM_DNS_TH)
                {
                    int pos = atomicAdd(s_blksmem_ful_cnt_local, 1);
                    s_blkid_smem_ful_local[pos] = tilei;
                }
            }
        }
    }

    int len = s_blksmem_tny_cnt_local[0];
    int pos = 0;
    if (len)
    {
        pos = lane_id == 0 ? atomicAdd(d_blksmem_tny_cnt, len) : 0;
        pos = __shfl_sync(0xffffffff, pos, 0);
        if (lane_id < len)
            d_blkid_smem_tny[pos + lane_id] = s_blkid_smem_tny_local[lane_id];
    }

    len = s_blksmem_sml_cnt_local[0];
    if (len)
    {
        pos = lane_id == 0 ? atomicAdd(d_blksmem_sml_cnt, len) : 0;
        pos = __shfl_sync(0xffffffff, pos, 0);
        if (lane_id < len)
            d_blkid_smem_sml[pos + lane_id] = s_blkid_smem_sml_local[lane_id];
    }

    len = s_blksmem_lrg_cnt_local[0];
    if (len)
    {
        pos = lane_id == 0 ? atomicAdd(d_blksmem_lrg_cnt, len) : 0;
        pos = __shfl_sync(0xffffffff, pos, 0);
        if (lane_id < len)
            d_blkid_smem_lrg[pos + lane_id] = s_blkid_smem_lrg_local[lane_id];
    }

    len = s_blksmem_dns_cnt_local[0];
    if (len)
    {
        pos = lane_id == 0 ? atomicAdd(d_blksmem_dns_cnt, len) : 0;
        pos = __shfl_sync(0xffffffff, pos, 0);
        if (lane_id < len)
            d_blkid_smem_dns[pos + lane_id] = s_blkid_smem_dns_local[lane_id];
    }

    len = s_blksmem_ful_cnt_local[0];
    if (len)
    {
        pos = lane_id == 0 ? atomicAdd(d_blksmem_ful_cnt, len) : 0;
        pos = __shfl_sync(0xffffffff, pos, 0);
        if (lane_id < len)
            d_blkid_smem_ful[pos + lane_id] = s_blkid_smem_ful_local[lane_id];
    }
}

__global__ void tile_spgemm_step3_cuda_kernel_2level_halfwarp(const int *d_blkrowptrA,
                                                              const int *__restrict__ d_blkcolidxA,
                                                              const int *d_nnzb_A,
                                                              MAT_VAL_TYPE *d_blkcsr_Val_A,
                                                              unsigned char *d_blkcsr_Col_A,
                                                              unsigned char *d_blkcsr_Ptr_A,
                                                              unsigned short *d_blkmaskA,
                                                              int blkmA, int blknA, int numblkA, int nnzA,
                                                              const int *__restrict__ d_blkcolptrB,
                                                              const int *__restrict__ d_blkrowidxB,
                                                              const int *__restrict__ d_nnzb_B,
                                                              const MAT_VAL_TYPE *__restrict__ d_blkcsr_Val_B,
                                                              const unsigned char *__restrict__ d_blkcsr_Col_B,
                                                              const unsigned char *__restrict__ d_blkcsr_Ptr_B,
                                                              const unsigned short *__restrict__ d_blkmaskB,
                                                              int blkmB, int blknB, int numblkB, int nnzB,
                                                              unsigned int *d_blk_intersec_bitmask_A,
                                                              unsigned int *d_blk_intersec_bitmask_B,
                                                              int blk_intersec_bitmask_len,
                                                              int *d_blkrowidxC,
                                                              int *d_blkcolidxC,
                                                              unsigned char *d_blkcsr_Ptr_C,
                                                              int *d_nnzb_C,
                                                              unsigned short *d_blkmaskC,
                                                              int *d_blksmem_tny_cnt,
                                                              int *d_blksmem_sml_cnt,
                                                              int *d_blksmem_lrg_cnt,
                                                              int *d_blksmem_dns_cnt,
                                                              int *d_blksmem_ful_cnt,
                                                              int *d_blkid_smem_tny,
                                                              int *d_blkid_smem_sml,
                                                              int *d_blkid_smem_lrg,
                                                              int *d_blkid_smem_dns,
                                                              int *d_blkid_smem_ful,
                                                              int *d_spec_intersection_cnt,
                                                              int *d_spec_intersection_posa,
                                                              int *d_spec_intersection_posb,
                                                              int numblkC)
{
    const int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_halfwarp_id = global_id >> 4; //global_id / HALFWARP_SIZE;

    __shared__ unsigned int s_maskc[HALFWARP_PER_BLOCK * BLOCK_SIZE];
    __shared__ unsigned short s_blkmaskB[HALFWARP_PER_BLOCK * BLOCK_SIZE];

    __shared__ int s_matched_posa[HALFWARP_PER_BLOCK * SPECULATIVE_INTERSECTION];
    __shared__ int s_matched_posb[HALFWARP_PER_BLOCK * SPECULATIVE_INTERSECTION];
    __shared__ int s_matchedcnt[HALFWARP_PER_BLOCK];

    __shared__ int s_blksmem_tny_cnt[HALFWARP_PER_BLOCK];
    __shared__ int s_blksmem_sml_cnt[HALFWARP_PER_BLOCK];
    __shared__ int s_blksmem_lrg_cnt[HALFWARP_PER_BLOCK];
    __shared__ int s_blksmem_dns_cnt[HALFWARP_PER_BLOCK];
    __shared__ int s_blksmem_ful_cnt[HALFWARP_PER_BLOCK];

    __shared__ int s_blkid_smem_tny[HALFWARP_PER_BLOCK * TILE_PER_HALFWARP];
    __shared__ int s_blkid_smem_sml[HALFWARP_PER_BLOCK * TILE_PER_HALFWARP];
    __shared__ int s_blkid_smem_lrg[HALFWARP_PER_BLOCK * TILE_PER_HALFWARP];
    __shared__ int s_blkid_smem_dns[HALFWARP_PER_BLOCK * TILE_PER_HALFWARP];
    __shared__ int s_blkid_smem_ful[HALFWARP_PER_BLOCK * TILE_PER_HALFWARP];

    const int lane_id = (WARP_SIZE - 1) & threadIdx.x;
    const int halfwarp_lane_id = (HALFWARP_SIZE - 1) & threadIdx.x;

    const int local_halfwarp_id = threadIdx.x >> 4; //threadIdx.x / HALFWARP_SIZE;

    unsigned int *s_maskc_local = &s_maskc[local_halfwarp_id * BLOCK_SIZE];
    unsigned short *s_blkmaskB_local = &s_blkmaskB[local_halfwarp_id * BLOCK_SIZE];

    int *s_matched_posa_local = &s_matched_posa[local_halfwarp_id * SPECULATIVE_INTERSECTION];
    int *s_matched_posb_local = &s_matched_posb[local_halfwarp_id * SPECULATIVE_INTERSECTION];
    int *s_matchedcnt_local = &s_matchedcnt[local_halfwarp_id];
    int *s_blksmem_tny_cnt_local = &s_blksmem_tny_cnt[local_halfwarp_id];
    int *s_blksmem_sml_cnt_local = &s_blksmem_sml_cnt[local_halfwarp_id];
    int *s_blksmem_lrg_cnt_local = &s_blksmem_lrg_cnt[local_halfwarp_id];
    int *s_blksmem_dns_cnt_local = &s_blksmem_dns_cnt[local_halfwarp_id];
    int *s_blksmem_ful_cnt_local = &s_blksmem_ful_cnt[local_halfwarp_id];

    int *s_blkid_smem_tny_local = &s_blkid_smem_tny[local_halfwarp_id * TILE_PER_HALFWARP];
    int *s_blkid_smem_sml_local = &s_blkid_smem_sml[local_halfwarp_id * TILE_PER_HALFWARP];
    int *s_blkid_smem_lrg_local = &s_blkid_smem_lrg[local_halfwarp_id * TILE_PER_HALFWARP];
    int *s_blkid_smem_dns_local = &s_blkid_smem_dns[local_halfwarp_id * TILE_PER_HALFWARP];
    int *s_blkid_smem_ful_local = &s_blkid_smem_ful[local_halfwarp_id * TILE_PER_HALFWARP];

    int tile_start = global_halfwarp_id * TILE_PER_HALFWARP;

    int tile_end = tile_start + TILE_PER_HALFWARP; //(global_warp_id + 1) * TPW;

    if (!halfwarp_lane_id)
    {
        s_blksmem_tny_cnt_local[0] = 0;
        s_blksmem_sml_cnt_local[0] = 0;
        s_blksmem_lrg_cnt_local[0] = 0;
        s_blksmem_dns_cnt_local[0] = 0;
        s_blksmem_ful_cnt_local[0] = 0;
    }

    for (int tilei = tile_start; tilei < tile_end; tilei++)
    {
        s_maskc_local[halfwarp_lane_id] = 0;
        if (!halfwarp_lane_id)
        {
            s_matchedcnt_local[0] = 0;
        }

        unsigned int maskc = 0;
        int nnzcnt = 0;
        int matchedcnt = 0;
        int lena = 0;
        int lenb = 0;

        if (tilei < numblkC)
        {
            const int blki = d_blkrowidxC[tilei];
            const int blkj = d_blkcolidxC[tilei];

            const int abase = d_blkrowptrA[blki];
            const int astop = d_blkrowptrA[blki + 1];
            lena = astop - abase;

            const int bbase = ld_gbl_int32(d_blkcolptrB + blkj);
            const int bstop = ld_gbl_int32(d_blkcolptrB + blkj + 1);
            lenb = bstop - bbase;
            {
                int specres = 0;
                intersection_binarysearch_kernel(d_blkcolidxA, abase, astop, lena,
                                                 d_blkrowidxB, bbase, bstop, lenb,
                                                 s_matched_posa_local, s_matched_posb_local,
                                                 SPECULATIVE_INTERSECTION, s_matchedcnt_local,
                                                 halfwarp_lane_id, HALFWARP_SIZE);
                matchedcnt = s_matchedcnt_local[0];

                if (matchedcnt == 0)
                {
                }
                else if (matchedcnt <= SPECULATIVE_INTERSECTION && specres == 0)
                {
                    // save speculative posa and posb for step 4
                    if (USE_GMEM_SPECULATIVE_INTERSECTION && matchedcnt <= GMEM_SPECULATIVE_INTERSECTION)
                    {
                        if (!halfwarp_lane_id)
                            d_spec_intersection_cnt[tilei] = matchedcnt;
                        for (int si = halfwarp_lane_id; si < matchedcnt; si += HALFWARP_SIZE)
                        {
                            d_spec_intersection_posa[tilei * GMEM_SPECULATIVE_INTERSECTION + si] = s_matched_posa_local[si];
                            d_spec_intersection_posb[tilei * GMEM_SPECULATIVE_INTERSECTION + si] = s_matched_posb_local[si];
                        }
                    }

                    for (int i = 0; i < matchedcnt; i++)
                    {
                        int posa = s_matched_posa_local[i];
                        int posb = s_matched_posb_local[i];

                        const int nnzastart = d_nnzb_A[(abase + posa)];
                        int nnztotala = d_nnzb_A[(abase + posa) + 1] - nnzastart;

                        s_blkmaskB_local[halfwarp_lane_id] =
                            ld_gbl_ushort(&d_blkmaskB[(bbase + posb) * BLOCK_SIZE + halfwarp_lane_id]);
                        {
                            for (int i = halfwarp_lane_id; i < nnztotala; i += HALFWARP_SIZE)
                            {
                                unsigned char rowcolidx = d_blkcsr_Col_A[nnzastart + i];
                                unsigned int maskb = s_blkmaskB_local[rowcolidx & 0xf];
                                if (maskb)
                                    atomicOr(&s_maskc_local[rowcolidx >> 4], maskb);
                            }
                        }
                    }
                }
                else
                {
                    const int astart = d_blkcolidxA[abase];
                    const int aend = d_blkcolidxA[astop - 1];
                    const int bstart = ld_gbl_int32(d_blkrowidxB + bbase);
                    const int bend = ld_gbl_int32(d_blkrowidxB + bstop - 1);

                    int posa_real = 0;
                    int posb_real = 0;
                    if (bstart > astart)
                    {
                        int posa_real_new = binary_search_right_boundary_kernel(d_blkcolidxA + abase, bstart, lena);
                        posa_real = posa_real_new < 0 ? 0 : posa_real_new;
                    }
                    else if (bstart < astart)
                    {
                        int posb_real_new = binary_search_right_boundary_kernel(d_blkrowidxB + bbase, astart, lenb);
                        posb_real = posb_real_new < 0 ? 0 : posb_real_new;
                    }

                    if (bstop < astop)
                    {
                        int lena_new = binary_search_right_boundary_kernel(d_blkcolidxA + abase, bend, lena) + 1;
                        lena = lena_new > lena ? lena : lena_new;
                    }
                    else if (bstop > astop)
                    {
                        int lenb_new = binary_search_right_boundary_kernel(d_blkrowidxB + bbase, aend, lenb) + 1;
                        lenb = lenb_new > lenb ? lenb : lenb_new;
                    }

                    int posa = posa_real;
                    int posb = posb_real;
                    int idxa = 0;
                    int idxb = 0;
                    int posa_updated = 1;
                    int posb_updated = 1;

                    while (posa < lena && posb < lenb)
                    {
                        idxa = posa_updated ? d_blkcolidxA[abase + posa] : idxa; //a[posa] : idxa;
                        idxb = posb_updated ? d_blkrowidxB[bbase + posb] : idxb; //b[posb] : idxb;

                        if (idxa == idxb)
                        {
                            const int nnzastart = d_nnzb_A[(abase + posa)];
                            int nnztotala = d_nnzb_A[(abase + posa) + 1] - nnzastart;

                            s_blkmaskB_local[halfwarp_lane_id] =
                                ld_gbl_ushort(d_blkmaskB + (bbase + posb) * BLOCK_SIZE + halfwarp_lane_id);

                            for (int i = halfwarp_lane_id; i < nnztotala; i += HALFWARP_SIZE)
                            {
                                unsigned char rowcolidx = d_blkcsr_Col_A[nnzastart + i];
                                unsigned int maskb = s_blkmaskB_local[rowcolidx & 0xf];
                                atomicOr(&s_maskc_local[rowcolidx >> 4], maskb);
                            }

                            posa++;
                            posa_updated = 1;
                            posb++;
                            posb_updated = 1;
                        }
                        else
                        {
                            // the smaller index goes forward
                            posa_updated = idxa < idxb ? 1 : 0;
                            posa += posa_updated;
                            posb_updated = idxa > idxb ? 1 : 0;
                            posb += posb_updated;
                        }
                    }
                }
            }
            maskc = s_maskc_local[halfwarp_lane_id];
            nnzcnt = __popc(maskc); //lane_id < BLOCK_SIZE ? __popc(maskc) : 0;
        }

        int nnzcnt_sum = sum_16_shfl(nnzcnt);

        int nnzcnt_scan = scan_32_shfl(nnzcnt, lane_id);

        nnzcnt_scan -= nnzcnt;
        nnzcnt_scan -= __shfl_sync(0xffffffff, nnzcnt_scan, (lane_id >> 4) << 4);

        if (tilei < numblkC && nnzcnt_sum)
        {
            long long int pos_c = (long long int)(tilei) * BLOCK_SIZE + halfwarp_lane_id;
//            printf("posc = %lli\n", pos_c);
            d_blkcsr_Ptr_C[pos_c] = nnzcnt_scan; // - nnzcnt;
            d_blkmaskC[pos_c] = maskc; //s_maskc_local[halfwarp_lane_id];

            if (!halfwarp_lane_id)
            {
                d_nnzb_C[tilei] = nnzcnt_sum;

                if (nnzcnt_sum <= SMEM_TNY_TH && nnzcnt_sum != 0)
                {
                    int pos = atomicAdd(s_blksmem_tny_cnt_local, 1);
                    s_blkid_smem_tny_local[pos] = tilei;
                }
                else if (SMEM_TNY_TH < nnzcnt_sum && nnzcnt_sum <= SMEM_SML_TH)
                {
                    int pos = atomicAdd(s_blksmem_sml_cnt_local, 1);
                    s_blkid_smem_sml_local[pos] = tilei;
                }
                else if (SMEM_SML_TH < nnzcnt_sum && nnzcnt_sum <= SMEM_LRG_TH)
                {
                    int pos = atomicAdd(s_blksmem_lrg_cnt_local, 1);
                    s_blkid_smem_lrg_local[pos] = tilei;
                }
                else if (SMEM_LRG_TH < nnzcnt_sum && nnzcnt_sum < SMEM_DNS_TH)
                {
                    int pos = atomicAdd(s_blksmem_dns_cnt_local, 1);
                    s_blkid_smem_dns_local[pos] = tilei;
                }
                else if (nnzcnt_sum == SMEM_DNS_TH)
                {
                    int pos = atomicAdd(s_blksmem_ful_cnt_local, 1);
                    s_blkid_smem_ful_local[pos] = tilei;
                }
            }
        }
    }

    int len = s_blksmem_tny_cnt_local[0];
    int pos = 0;
    pos = halfwarp_lane_id == 0 ? atomicAdd(d_blksmem_tny_cnt, len) : 0;
    pos = __shfl_sync(0xffffffff, pos, (lane_id >> 4) << 4);

    if (halfwarp_lane_id < len)
        d_blkid_smem_tny[pos + halfwarp_lane_id] = s_blkid_smem_tny_local[halfwarp_lane_id];

    len = s_blksmem_sml_cnt_local[0];
    pos = halfwarp_lane_id == 0 ? atomicAdd(d_blksmem_sml_cnt, len) : 0;
    pos = __shfl_sync(0xffffffff, pos, (lane_id >> 4) << 4);
    if (halfwarp_lane_id < len)
        d_blkid_smem_sml[pos + halfwarp_lane_id] = s_blkid_smem_sml_local[halfwarp_lane_id];

    len = s_blksmem_lrg_cnt_local[0];
    pos = halfwarp_lane_id == 0 ? atomicAdd(d_blksmem_lrg_cnt, len) : 0;
    pos = __shfl_sync(0xffffffff, pos, (lane_id >> 4) << 4);
    if (halfwarp_lane_id < len)
        d_blkid_smem_lrg[pos + halfwarp_lane_id] = s_blkid_smem_lrg_local[halfwarp_lane_id];

    len = s_blksmem_dns_cnt_local[0];
    pos = halfwarp_lane_id == 0 ? atomicAdd(d_blksmem_dns_cnt, len) : 0;
    pos = __shfl_sync(0xffffffff, pos, (lane_id >> 4) << 4);

    if (halfwarp_lane_id < len)
        d_blkid_smem_dns[pos + halfwarp_lane_id] = s_blkid_smem_dns_local[halfwarp_lane_id];

    len = s_blksmem_ful_cnt_local[0];
    pos = halfwarp_lane_id == 0 ? atomicAdd(d_blksmem_ful_cnt, len) : 0;
    pos = __shfl_sync(0xffffffff, pos, (lane_id >> 4) << 4);
    if (halfwarp_lane_id < len)
        d_blkid_smem_ful[pos + halfwarp_lane_id] = s_blkid_smem_ful_local[halfwarp_lane_id];
}

__global__ void tile_spgemm_step3_cuda_kernel_dns_thread(const int *d_blkrowptrA,
                                                         const int *__restrict__ d_blkcolidxA,
                                                         const int *__restrict__ d_nnzb_A,
                                                         MAT_VAL_TYPE *d_blkcsr_Val_A,
                                                         unsigned char *__restrict__ d_blkcsr_Col_A,
                                                         unsigned char *d_blkcsr_Ptr_A,
                                                         unsigned short *d_blkmaskA,
                                                         int blkmA, int blknA, int numblkA, int nnzA,
                                                         const int *__restrict__ d_blkcolptrB,
                                                         const int *__restrict__ d_blkrowidxB,
                                                         const int *__restrict__ d_nnzb_B,
                                                         const MAT_VAL_TYPE *__restrict__ d_blkcsr_Val_B,
                                                         const unsigned char *__restrict__ d_blkcsr_Col_B,
                                                         const unsigned char *__restrict__ d_blkcsr_Ptr_B,
                                                         const unsigned short *__restrict__ d_blkmaskB,
                                                         int blkmB, int blknB, int numblkB, int nnzB,
                                                         unsigned int *d_blk_intersec_bitmask_A,
                                                         unsigned int *d_blk_intersec_bitmask_B,
                                                         int blk_intersec_bitmask_len,
                                                         int *d_blkrowidxC,
                                                         int *d_blkcolidxC,
                                                         unsigned char *d_blkcsr_Ptr_C,
                                                         int *d_nnzb_C,
                                                         unsigned short *d_blkmaskC,
                                                         int *d_blksmem_tny_cnt,
                                                         int *d_blksmem_sml_cnt,
                                                         int *d_blksmem_lrg_cnt,
                                                         int *d_blksmem_dns_cnt,
                                                         int *d_blksmem_ful_cnt,
                                                         int *d_blkid_smem_tny,
                                                         int *d_blkid_smem_sml,
                                                         int *d_blkid_smem_lrg,
                                                         int *d_blkid_smem_dns,
                                                         int *d_blkid_smem_ful,
                                                         int *d_spec_intersection_cnt,
                                                         int *d_spec_intersection_posa,
                                                         int *d_spec_intersection_posb,
                                                         int numblkC)
{
    const int tilei = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned short maskb[BLOCK_SIZE];

    unsigned short maskc[BLOCK_SIZE];
#pragma unroll
    for (int ci = 0; ci < BLOCK_SIZE; ci++)
        maskc[ci] = 0;

    if (tilei < numblkC)
    {
        int matchedcnt = 0;

        const int blki = d_blkrowidxC[tilei];
        const int blkj = d_blkcolidxC[tilei];

        const int abase = d_blkrowptrA[blki];

        const int bbase = ld_gbl_int32(d_blkcolptrB + blkj);

        int offseta = 0;
        int offsetb = 0;

        for (int di = 0; di < blk_intersec_bitmask_len; di++)
        {
            unsigned int bma = d_blk_intersec_bitmask_A[blki * blk_intersec_bitmask_len + di];
            unsigned int bmb = d_blk_intersec_bitmask_B[blkj * blk_intersec_bitmask_len + di];

            int posa = offseta;
            int posb = offsetb;

            if (__popc(bma & bmb))
            {
                for (int ii = 31; ii >= 0; ii--)
                {
                    unsigned int bita = (bma >> ii) & 0x1;
                    unsigned int bitb = (bmb >> ii) & 0x1;

                    if (bita && bitb)
                    {
                        if (USE_GMEM_SPECULATIVE_INTERSECTION && matchedcnt < GMEM_SPECULATIVE_INTERSECTION)
                        {
                            d_spec_intersection_cnt[tilei] = matchedcnt;
                            d_spec_intersection_posa[tilei * GMEM_SPECULATIVE_INTERSECTION + matchedcnt] = posa;
                            d_spec_intersection_posb[tilei * GMEM_SPECULATIVE_INTERSECTION + matchedcnt] = posb;
                        }
                        matchedcnt++;

                        const int nnzastart = ld_gbl_int32(d_nnzb_A + abase + posa);
                        const int nnztotala = ld_gbl_int32(d_nnzb_A + abase + posa + 1) - nnzastart;

#pragma unroll
                        for (int ci = 0; ci < BLOCK_SIZE; ci++)
                            maskb[ci] = ld_gbl_ushort(&d_blkmaskB[(bbase + posb) * BLOCK_SIZE + ci]);

                        for (int ni = 0; ni < nnztotala; ni++)
                        {
                            unsigned char rowcolidx = ld_gbl_uchar(d_blkcsr_Col_A + nnzastart + ni);
                            maskc[rowcolidx >> 4] |= maskb[rowcolidx & 0xf];
                        }
                    }

                    posa += bita;
                    posb += bitb;
                }
            }

            offseta += __popc(bma);
            offsetb += __popc(bmb);
        }

        int nnzcnt_sum = 0;
        long long int pos_c = (long long int)tilei * BLOCK_SIZE + 0;
        d_blkcsr_Ptr_C[pos_c] = 0;
        d_blkmaskC[pos_c] = maskc[0];
#pragma unroll
        for (int ci = 1; ci < BLOCK_SIZE; ci++)
        {
            nnzcnt_sum += __popc(maskc[ci - 1]);
            pos_c = (long long int)tilei * BLOCK_SIZE + ci;
            d_blkcsr_Ptr_C[pos_c] = nnzcnt_sum;
            d_blkmaskC[pos_c] = maskc[ci];
        }
        nnzcnt_sum += __popc(maskc[BLOCK_SIZE - 1]);

        if (nnzcnt_sum)
            d_nnzb_C[tilei] = nnzcnt_sum;

        if (nnzcnt_sum <= SMEM_TNY_TH && nnzcnt_sum != 0)
        {
            int pos = atomicAdd(d_blksmem_tny_cnt, 1);
            d_blkid_smem_tny[pos] = tilei;
        }
        else if (SMEM_TNY_TH < nnzcnt_sum && nnzcnt_sum <= SMEM_SML_TH)
        {
            int pos = atomicAdd(d_blksmem_sml_cnt, 1);
            d_blkid_smem_sml[pos] = tilei;
        }
        else if (SMEM_SML_TH < nnzcnt_sum && nnzcnt_sum <= SMEM_LRG_TH)
        {
            int pos = atomicAdd(d_blksmem_lrg_cnt, 1);
            d_blkid_smem_lrg[pos] = tilei;
        }
        else if (SMEM_LRG_TH < nnzcnt_sum && nnzcnt_sum < SMEM_DNS_TH)
        {
            int pos = atomicAdd(d_blksmem_dns_cnt, 1);
            d_blkid_smem_dns[pos] = tilei;
        }
        else if (nnzcnt_sum == SMEM_DNS_TH)
        {
            int pos = atomicAdd(d_blksmem_ful_cnt, 1);
            d_blkid_smem_ful[pos] = tilei;
        }
    }
}

__global__ void tile_spgemm_step3_cuda_kernel_dns_halfwarp(const int *d_blkrowptrA,
                                                           const int *__restrict__ d_blkcolidxA,
                                                           const int *__restrict__ d_nnzb_A,
                                                           MAT_VAL_TYPE *d_blkcsr_Val_A,
                                                           unsigned char *__restrict__ d_blkcsr_Col_A,
                                                           unsigned char *d_blkcsr_Ptr_A,
                                                           unsigned short *d_blkmaskA,
                                                           int blkmA, int blknA, int numblkA, int nnzA,
                                                           const int *__restrict__ d_blkcolptrB,
                                                           const int *__restrict__ d_blkrowidxB,
                                                           const int *__restrict__ d_nnzb_B,
                                                           const MAT_VAL_TYPE *__restrict__ d_blkcsr_Val_B,
                                                           const unsigned char *__restrict__ d_blkcsr_Col_B,
                                                           const unsigned char *__restrict__ d_blkcsr_Ptr_B,
                                                           const unsigned short *__restrict__ d_blkmaskB,
                                                           int blkmB, int blknB, int numblkB, int nnzB,
                                                           unsigned int *d_blk_intersec_bitmask_A,
                                                           unsigned int *d_blk_intersec_bitmask_B,
                                                           int blk_intersec_bitmask_len,
                                                           int *d_blkrowidxC,
                                                           int *d_blkcolidxC,
                                                           unsigned char *d_blkcsr_Ptr_C,
                                                           int *d_nnzb_C,
                                                           unsigned short *d_blkmaskC,
                                                           int *d_blksmem_tny_cnt,
                                                           int *d_blksmem_sml_cnt,
                                                           int *d_blksmem_lrg_cnt,
                                                           int *d_blksmem_dns_cnt,
                                                           int *d_blksmem_ful_cnt,
                                                           int *d_blkid_smem_tny,
                                                           int *d_blkid_smem_sml,
                                                           int *d_blkid_smem_lrg,
                                                           int *d_blkid_smem_dns,
                                                           int *d_blkid_smem_ful,
                                                           int *d_spec_intersection_cnt,
                                                           int *d_spec_intersection_posa,
                                                           int *d_spec_intersection_posb,
                                                           int numblkC)
{
    const int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_halfwarp_id = global_id >> 4; //global_id / HALFWARP_SIZE;

    __shared__ unsigned int s_maskc[HALFWARP_PER_BLOCK * BLOCK_SIZE];
    __shared__ unsigned short s_blkmaskB[HALFWARP_PER_BLOCK * BLOCK_SIZE];

    __shared__ int s_blksmem_tny_cnt[HALFWARP_PER_BLOCK];
    __shared__ int s_blksmem_sml_cnt[HALFWARP_PER_BLOCK];
    __shared__ int s_blksmem_lrg_cnt[HALFWARP_PER_BLOCK];
    __shared__ int s_blksmem_dns_cnt[HALFWARP_PER_BLOCK];
    __shared__ int s_blksmem_ful_cnt[HALFWARP_PER_BLOCK];

    __shared__ int s_blkid_smem_tny[HALFWARP_PER_BLOCK * TILE_PER_HALFWARP];
    __shared__ int s_blkid_smem_sml[HALFWARP_PER_BLOCK * TILE_PER_HALFWARP];
    __shared__ int s_blkid_smem_lrg[HALFWARP_PER_BLOCK * TILE_PER_HALFWARP];
    __shared__ int s_blkid_smem_dns[HALFWARP_PER_BLOCK * TILE_PER_HALFWARP];
    __shared__ int s_blkid_smem_ful[HALFWARP_PER_BLOCK * TILE_PER_HALFWARP];

    const int lane_id = (WARP_SIZE - 1) & threadIdx.x;
    const int halfwarp_lane_id = (HALFWARP_SIZE - 1) & threadIdx.x;

    const int local_halfwarp_id = threadIdx.x >> 4; //threadIdx.x / HALFWARP_SIZE;

    unsigned int *s_maskc_local = &s_maskc[local_halfwarp_id * BLOCK_SIZE];
    unsigned short *s_blkmaskB_local = &s_blkmaskB[local_halfwarp_id * BLOCK_SIZE];
    int *s_blksmem_tny_cnt_local = &s_blksmem_tny_cnt[local_halfwarp_id];
    int *s_blksmem_sml_cnt_local = &s_blksmem_sml_cnt[local_halfwarp_id];
    int *s_blksmem_lrg_cnt_local = &s_blksmem_lrg_cnt[local_halfwarp_id];
    int *s_blksmem_dns_cnt_local = &s_blksmem_dns_cnt[local_halfwarp_id];
    int *s_blksmem_ful_cnt_local = &s_blksmem_ful_cnt[local_halfwarp_id];

    int *s_blkid_smem_tny_local = &s_blkid_smem_tny[local_halfwarp_id * TILE_PER_HALFWARP];
    int *s_blkid_smem_sml_local = &s_blkid_smem_sml[local_halfwarp_id * TILE_PER_HALFWARP];
    int *s_blkid_smem_lrg_local = &s_blkid_smem_lrg[local_halfwarp_id * TILE_PER_HALFWARP];
    int *s_blkid_smem_dns_local = &s_blkid_smem_dns[local_halfwarp_id * TILE_PER_HALFWARP];
    int *s_blkid_smem_ful_local = &s_blkid_smem_ful[local_halfwarp_id * TILE_PER_HALFWARP];

    int tile_start = global_halfwarp_id * TILE_PER_HALFWARP;
    int tile_end = tile_start + TILE_PER_HALFWARP; //(global_warp_id + 1) * TPW;

    if (!halfwarp_lane_id)
    {
        s_blksmem_tny_cnt_local[0] = 0;
        s_blksmem_sml_cnt_local[0] = 0;
        s_blksmem_lrg_cnt_local[0] = 0;
        s_blksmem_dns_cnt_local[0] = 0;
        s_blksmem_ful_cnt_local[0] = 0;
    }

    for (int tilei = tile_start; tilei < tile_end; tilei++)
    {
        s_maskc_local[halfwarp_lane_id] = 0;

        unsigned int maskc = 0;
        int nnzcnt = 0;
        int matchedcnt = 0;

        if (tilei < numblkC)
        {
            const int blki = d_blkrowidxC[tilei];
            const int blkj = d_blkcolidxC[tilei];

            const int abase = d_blkrowptrA[blki];

            const int bbase = ld_gbl_int32(d_blkcolptrB + blkj);

            int offseta = 0;
            int offsetb = 0;

            for (int di = 0; di < blk_intersec_bitmask_len; di++)
            {
                unsigned int bma = d_blk_intersec_bitmask_A[blki * blk_intersec_bitmask_len + di];
                unsigned int bmb = d_blk_intersec_bitmask_B[blkj * blk_intersec_bitmask_len + di];

                int posa = offseta;
                int posb = offsetb;

                if (__popc(bma & bmb))
                {
                    for (int ii = 31; ii >= 0; ii--)
                    {
                        unsigned int bita = (bma >> ii) & 0x1;
                        unsigned int bitb = (bmb >> ii) & 0x1;

                        if (bita && bitb)
                        {
                            if (USE_GMEM_SPECULATIVE_INTERSECTION && matchedcnt < GMEM_SPECULATIVE_INTERSECTION)
                            {
                                if (!halfwarp_lane_id)
                                {
                                    d_spec_intersection_cnt[tilei] = matchedcnt;
                                    d_spec_intersection_posa[tilei * GMEM_SPECULATIVE_INTERSECTION + matchedcnt] = posa;
                                    d_spec_intersection_posb[tilei * GMEM_SPECULATIVE_INTERSECTION + matchedcnt] = posb;
                                }
                            }

                            matchedcnt++;

                            const int nnzastart = ld_gbl_int32(d_nnzb_A + abase + posa);
                            const int nnztotala = ld_gbl_int32(d_nnzb_A + abase + posa + 1) - nnzastart;

                            s_blkmaskB_local[halfwarp_lane_id] =
                                ld_gbl_ushort(&d_blkmaskB[(bbase + posb) * BLOCK_SIZE + halfwarp_lane_id]);

                            if (nnztotala >= NNZTOTALA_FAST_TRACK_TH2)
                            {
                                int astart = d_blkcsr_Ptr_A[(abase + posa) * BLOCK_SIZE + halfwarp_lane_id];
                                int astop = halfwarp_lane_id == HALFWARP_SIZE - 1 ? nnztotala : d_blkcsr_Ptr_A[(abase + posa) * BLOCK_SIZE + halfwarp_lane_id + 1];

                                for (int aci = astart; aci < astop; aci++)
                                {
                                    unsigned char rowcolidx = d_blkcsr_Col_A[nnzastart + aci];
                                    maskc |= s_blkmaskB_local[rowcolidx & 0xf];
                                }
                                s_maskc_local[halfwarp_lane_id] |= maskc;
                            }
                            else
                            {
                                for (int i = halfwarp_lane_id; i < nnztotala; i += HALFWARP_SIZE)
                                {
                                    unsigned char rowcolidx = ld_gbl_uchar(d_blkcsr_Col_A + nnzastart + i);
                                    unsigned int maskb = s_blkmaskB_local[rowcolidx & 0xf];
                                    atomicOr(&s_maskc_local[rowcolidx >> 4], maskb);
                                }
                            }
                        }

                        posa += bita;
                        posb += bitb;
                    }
                }

                offseta += __popc(bma);
                offsetb += __popc(bmb);
            }

            maskc = s_maskc_local[halfwarp_lane_id];
            nnzcnt = __popc(maskc); //lane_id < BLOCK_SIZE ? __popc(maskc) : 0;
        }

        int nnzcnt_sum = sum_16_shfl(nnzcnt);

        int nnzcnt_scan = scan_32_shfl(nnzcnt, lane_id);
        nnzcnt_scan -= nnzcnt;
        nnzcnt_scan -= __shfl_sync(0xffffffff, nnzcnt_scan, (lane_id >> 4) << 4);

        if (tilei < numblkC && nnzcnt_sum)
        {
            long long int pos_c  = (long long int)(tilei) * BLOCK_SIZE + halfwarp_lane_id;
            d_blkcsr_Ptr_C[pos_c] = nnzcnt_scan; // - nnzcnt;
            d_blkmaskC[pos_c] = maskc;           //s_maskc_local[halfwarp_lane_id];

            if (!halfwarp_lane_id)
            {
                d_nnzb_C[tilei] = nnzcnt_sum;

                if (nnzcnt_sum <= SMEM_TNY_TH && nnzcnt_sum != 0)
                {
                    int pos = atomicAdd(s_blksmem_tny_cnt_local, 1);
                    s_blkid_smem_tny_local[pos] = tilei;
                }
                else if (SMEM_TNY_TH < nnzcnt_sum && nnzcnt_sum <= SMEM_SML_TH)
                {
                    int pos = atomicAdd(s_blksmem_sml_cnt_local, 1);
                    s_blkid_smem_sml_local[pos] = tilei;
                }
                else if (SMEM_SML_TH < nnzcnt_sum && nnzcnt_sum <= SMEM_LRG_TH)
                {
                    int pos = atomicAdd(s_blksmem_lrg_cnt_local, 1);
                    s_blkid_smem_lrg_local[pos] = tilei;
                }
                else if (SMEM_LRG_TH < nnzcnt_sum && nnzcnt_sum < SMEM_DNS_TH)
                {
                    int pos = atomicAdd(s_blksmem_dns_cnt_local, 1);
                    s_blkid_smem_dns_local[pos] = tilei;
                }
                else if (nnzcnt_sum == SMEM_DNS_TH)
                {
                    int pos = atomicAdd(s_blksmem_ful_cnt_local, 1);
                    s_blkid_smem_ful_local[pos] = tilei;
                }
            }
        }
    }

    int len = s_blksmem_tny_cnt_local[0];
    int pos = 0;
    pos = halfwarp_lane_id == 0 ? atomicAdd(d_blksmem_tny_cnt, len) : 0;
    pos = __shfl_sync(0xffffffff, pos, (lane_id >> 4) << 4);
    if (halfwarp_lane_id < len)
        d_blkid_smem_tny[pos + halfwarp_lane_id] = s_blkid_smem_tny_local[halfwarp_lane_id];

    len = s_blksmem_sml_cnt_local[0];
    pos = halfwarp_lane_id == 0 ? atomicAdd(d_blksmem_sml_cnt, len) : 0;
    pos = __shfl_sync(0xffffffff, pos, (lane_id >> 4) << 4);
    if (halfwarp_lane_id < len)
        d_blkid_smem_sml[pos + halfwarp_lane_id] = s_blkid_smem_sml_local[halfwarp_lane_id];

    len = s_blksmem_lrg_cnt_local[0];
    pos = halfwarp_lane_id == 0 ? atomicAdd(d_blksmem_lrg_cnt, len) : 0;
    pos = __shfl_sync(0xffffffff, pos, (lane_id >> 4) << 4);
    if (halfwarp_lane_id < len)
        d_blkid_smem_lrg[pos + halfwarp_lane_id] = s_blkid_smem_lrg_local[halfwarp_lane_id];

    len = s_blksmem_dns_cnt_local[0];
    pos = halfwarp_lane_id == 0 ? atomicAdd(d_blksmem_dns_cnt, len) : 0;
    pos = __shfl_sync(0xffffffff, pos, (lane_id >> 4) << 4);
    if (halfwarp_lane_id < len)
        d_blkid_smem_dns[pos + halfwarp_lane_id] = s_blkid_smem_dns_local[halfwarp_lane_id];

    len = s_blksmem_ful_cnt_local[0];
    pos = halfwarp_lane_id == 0 ? atomicAdd(d_blksmem_ful_cnt, len) : 0;
    pos = __shfl_sync(0xffffffff, pos, (lane_id >> 4) << 4);
    if (halfwarp_lane_id < len)
        d_blkid_smem_ful[pos + halfwarp_lane_id] = s_blkid_smem_ful_local[halfwarp_lane_id];
}

template <int SMEM_MATNNZ>
__global__ void tile_spgemm_step4_cuda_kernel_smem_v3(int *d_blkrowptrA,
                                                      const int *__restrict__ d_blkcolidxA,
                                                      int *d_nnzb_A,
                                                      MAT_VAL_TYPE *d_blkcsr_Val_A,
                                                      unsigned char *d_blkcsr_Col_A,
                                                      unsigned char *d_blkcsr_Ptr_A,
                                                      int blkmA, int blknA, int numblkA, int nnzA,
                                                      const int *__restrict__ d_blkcolptrB,
                                                      const int *__restrict__ d_blkrowidxB,
                                                      const int *__restrict__ d_nnzb_B,
                                                      const MAT_VAL_TYPE *__restrict__ d_blkcsr_Val_B,
                                                      const unsigned char *__restrict__ d_blkcsr_Col_B,
                                                      const unsigned char *__restrict__ d_blkcsr_Ptr_B,
                                                      int blkmB, int blknB, int numblkB, int nnzB,
                                                      int *d_blkrowidxC,
                                                      int *d_blkcolidxC,
                                                      unsigned char *d_blkcsr_Ptr_C,
                                                      unsigned char *d_blkcsr_Col_C,
                                                      MAT_VAL_TYPE *d_blkcsr_Val_C,
                                                      int *d_nnzb_C,
                                                      unsigned short *d_blkmaskC,
                                                      int numblkC,
                                                      int *d_blkid,
                                                      int *d_spec_intersection_cnt,
                                                      int *d_spec_intersection_posa,
                                                      int *d_spec_intersection_posb)
{
    const int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int global_warp_id = global_id >> 5; //global_id / WARP_SIZE;

    if (global_warp_id >= numblkC)
        return;
    int tilei = d_blkid[global_warp_id];

    const int nnzcstart = d_nnzb_C[tilei];
    const int blknnzctotal = d_nnzb_C[tilei + 1] - nnzcstart;
    if (!blknnzctotal)
        return;

    const int local_warp_id = threadIdx.x >> 5; //threadIdx.x / WARP_SIZE;
    __shared__ unsigned char s_blkcsr_Idx_C[WARP_PER_BLOCK * SMEM_MATNNZ];
    __shared__ unsigned char s_blkcsr_Ptr_C[WARP_PER_BLOCK * BLOCK_SIZE];
    unsigned char *s_blkcsr_Idx_C_local = &s_blkcsr_Idx_C[local_warp_id * SMEM_MATNNZ];
    unsigned char *s_blkcsr_Ptr_C_local = &s_blkcsr_Ptr_C[local_warp_id * BLOCK_SIZE];

    __shared__ unsigned char s_csrRowPtrB[WARP_PER_BLOCK * BLOCK_SIZE];
    unsigned char *s_csrRowPtrB_local = &s_csrRowPtrB[local_warp_id * BLOCK_SIZE];

    __shared__ int s_matched_posa[WARP_PER_BLOCK * SPECULATIVE_INTERSECTION];
    __shared__ int s_matched_posb[WARP_PER_BLOCK * SPECULATIVE_INTERSECTION];
    __shared__ int s_matchedcnt[WARP_PER_BLOCK];

    int *s_matched_posa_local = &s_matched_posa[local_warp_id * SPECULATIVE_INTERSECTION];
    int *s_matched_posb_local = &s_matched_posb[local_warp_id * SPECULATIVE_INTERSECTION];
    int *s_matchedcnt_local = &s_matchedcnt[local_warp_id];

    const int lane_id = (WARP_SIZE - 1) & threadIdx.x;

    for (int i = lane_id; i < blknnzctotal; i += WARP_SIZE)
    {
        d_blkcsr_Val_C[nnzcstart + i] = 0.0;
    }

    if (lane_id < BLOCK_SIZE){
        long long int pos_c = (long long int)(tilei) * BLOCK_SIZE + lane_id;
        s_blkcsr_Ptr_C_local[lane_id] = d_blkcsr_Ptr_C[pos_c];
    }

    long long int pos_c = (long long int)tilei * BLOCK_SIZE + lane_id;
    unsigned int maskc = lane_id < BLOCK_SIZE ? d_blkmaskC[pos_c] : 0; //s_maskc_local[lane_id];
    unsigned char blknnzcstart = lane_id < BLOCK_SIZE ? s_blkcsr_Ptr_C_local[lane_id] : 0;
    if (!lane_id)
        s_matchedcnt_local[0] = 0;

    if (lane_id < BLOCK_SIZE)
    {
        int cnt = 0;
#pragma unroll
        for (int i = 0; i < BLOCK_SIZE; i++)
        {
            int idx = ((maskc >> BLOCK_SIZE - i - 1) & 0x1) == 1 ? i : -1;
            if (idx != -1)
            {
                s_blkcsr_Idx_C_local[blknnzcstart + cnt] = idx;
                cnt++;
            }
        }
    }

    const int blki = d_blkrowidxC[tilei];
    const int blkj = d_blkcolidxC[tilei];

    const int abase = ld_gbl_int32(d_blkrowptrA + blki);
    const int astop = ld_gbl_int32(d_blkrowptrA + blki + 1);
    int lena = astop - abase;

    const int bbase = ld_gbl_int32(d_blkcolptrB + blkj);
    const int bstop = ld_gbl_int32(d_blkcolptrB + blkj + 1);
    int lenb = bstop - bbase;

    int matchedcnt = 0;
    int specres = 0;

    if (USE_GMEM_SPECULATIVE_INTERSECTION)
        matchedcnt = d_spec_intersection_cnt[tilei];

    if (USE_GMEM_SPECULATIVE_INTERSECTION && matchedcnt > 0)
    {
        specres = 0;
        for (int si = lane_id; si < matchedcnt; si += WARP_SIZE)
        {
            s_matched_posa_local[si] = d_spec_intersection_posa[tilei * GMEM_SPECULATIVE_INTERSECTION + si];
            s_matched_posb_local[si] = d_spec_intersection_posb[tilei * GMEM_SPECULATIVE_INTERSECTION + si];
        }
    }
    else
    {
        specres = intersection_binarysearch_kernel(d_blkcolidxA, abase, astop, lena,
                                                   d_blkrowidxB, bbase, bstop, lenb,
                                                   s_matched_posa_local, s_matched_posb_local,
                                                   SPECULATIVE_INTERSECTION, s_matchedcnt_local,
                                                   lane_id, WARP_SIZE);

        matchedcnt = s_matchedcnt_local[0];
    }

    if (matchedcnt <= SPECULATIVE_INTERSECTION && specres == 0)
    {
        for (int posi = 0; posi < matchedcnt; posi++)
        {
            int posa = s_matched_posa_local[posi];
            int posb = s_matched_posb_local[posi];

            // atomic method
            const int nnzastart = d_nnzb_A[(abase + posa)];
            int nnztotala = d_nnzb_A[(abase + posa) + 1] - nnzastart;
            if (lane_id < BLOCK_SIZE)
                s_csrRowPtrB_local[lane_id] = ld_gbl_uchar(d_blkcsr_Ptr_B + (bbase + posb) * BLOCK_SIZE + lane_id);
            const int nnzbstart = ld_gbl_int32(d_nnzb_B + bbase + posb);
            int nnztotalb = ld_gbl_int32(d_nnzb_B + bbase + posb + 1) - nnzbstart;

            if (nnztotala > VECTORIZE_NNZA_OR_NNZB_TH)
            {
                for (int i = lane_id; i < nnztotala; i += WARP_SIZE)
                {
                    unsigned char rowcolidx = d_blkcsr_Col_A[nnzastart + i];
                    int rowidxa = rowcolidx >> 4;
                    int rowidxb = rowcolidx & 0xf;
                    MAT_VAL_TYPE val = d_blkcsr_Val_A[nnzastart + i];
                    int blkoffseta = s_blkcsr_Ptr_C_local[rowidxa];
                    int blkoffseta_stop = rowidxa == BLOCK_SIZE - 1 ? blknnzctotal : s_blkcsr_Ptr_C_local[rowidxa + 1];

                    const int startb = s_csrRowPtrB_local[rowidxb];                                            //d_csrRowPtrB[rowidxb];
                    const int stopb = rowidxb == BLOCK_SIZE - 1 ? nnztotalb : s_csrRowPtrB_local[rowidxb + 1]; //d_csrRowPtrB[rowidxb+1];
                    for (int k = startb; k < stopb; k++)
                    {
                        unsigned char colidx = ld_gbl_uchar(d_blkcsr_Col_B + nnzbstart + k);

                        MAT_VAL_TYPE valb = ld_gbl_real(d_blkcsr_Val_B + nnzbstart + k);

                        int cnt = binary_search_exact_uchar_kernel(s_blkcsr_Idx_C_local + blkoffseta, 0, blkoffseta_stop - blkoffseta - 1, colidx);
                        if (cnt != -1)
                            atomicAdd(&d_blkcsr_Val_C[nnzcstart + blkoffseta + cnt], val * valb);
                    }
                }
            }
            else
            {
                for (int i = 0; i < nnztotala; i++)
                {
                    unsigned char rowcolidx = d_blkcsr_Col_A[nnzastart + i];
                    int rowidxa = rowcolidx >> 4;
                    int rowidxb = rowcolidx & 0xf;
                    MAT_VAL_TYPE val = d_blkcsr_Val_A[nnzastart + i];
                    int blkoffseta = s_blkcsr_Ptr_C_local[rowidxa];
                    int blkoffseta_stop = rowidxa == BLOCK_SIZE - 1 ? blknnzctotal : s_blkcsr_Ptr_C_local[rowidxa + 1];

                    const int startb = s_csrRowPtrB_local[rowidxb];                                            //d_csrRowPtrB[rowidxb];
                    const int stopb = rowidxb == BLOCK_SIZE - 1 ? nnztotalb : s_csrRowPtrB_local[rowidxb + 1]; //d_csrRowPtrB[rowidxb+1];
                    int k = startb + lane_id;
                    if (k < stopb)
                    {
                        unsigned char colidx = ld_gbl_uchar(d_blkcsr_Col_B + nnzbstart + k);

                        MAT_VAL_TYPE valb = ld_gbl_real(d_blkcsr_Val_B + nnzbstart + k);

                        int cnt = binary_search_exact_uchar_kernel(s_blkcsr_Idx_C_local + blkoffseta, 0, blkoffseta_stop - blkoffseta - 1, colidx);
                        if (cnt != -1)
                            d_blkcsr_Val_C[nnzcstart + blkoffseta + cnt] += val * valb;
                    }
                }
            }
        }
    }
    else
    {
        const int astart = d_blkcolidxA[abase];
        const int aend = d_blkcolidxA[astop - 1];
        const int bstart = ld_gbl_int32(d_blkrowidxB + bbase);
        const int bend = ld_gbl_int32(d_blkrowidxB + bstop - 1);

        int posa_real = 0;
        int posb_real = 0;
        if (bstart > astart)
        {
            int posa_real_new = binary_search_right_boundary_kernel(d_blkcolidxA + abase, bstart, lena);
            posa_real = posa_real_new < 0 ? 0 : posa_real_new;
        }
        else if (bstart < astart)
        {
            int posb_real_new = binary_search_right_boundary_kernel(d_blkrowidxB + bbase, astart, lenb);
            posb_real = posb_real_new < 0 ? 0 : posb_real_new;
        }

        if (bstop < astop)
        {
            int lena_new = binary_search_right_boundary_kernel(d_blkcolidxA + abase, bend, lena) + 1;
            lena = lena_new > lena ? lena : lena_new;
        }
        else if (bstop > astop)
        {
            int lenb_new = binary_search_right_boundary_kernel(d_blkrowidxB + bbase, aend, lenb) + 1;
            lenb = lenb_new > lenb ? lenb : lenb_new;
        }

        int posa = posa_real;
        int posb = posb_real;
        int idxa = 0;
        int idxb = 0;
        int posa_updated = 1;
        int posb_updated = 1;

        while (posa < lena && posb < lenb)
        {
            idxa = posa_updated ? ld_gbl_int32(d_blkcolidxA + abase + posa) : idxa; //a[posa] : idxa;
            idxb = posb_updated ? ld_gbl_int32(d_blkrowidxB + bbase + posb) : idxb; //b[posb] : idxb;

            if (idxa == idxb)
            {
                const int nnzastart = d_nnzb_A[(abase + posa)];
                int nnztotala = d_nnzb_A[(abase + posa) + 1] - nnzastart;
                if (lane_id < BLOCK_SIZE)
                    s_csrRowPtrB_local[lane_id] = ld_gbl_uchar(d_blkcsr_Ptr_B + (bbase + posb) * BLOCK_SIZE + lane_id);
                const int nnzbstart = ld_gbl_int32(d_nnzb_B + bbase + posb);
                int nnztotalb = ld_gbl_int32(d_nnzb_B + bbase + posb + 1) - nnzbstart;

                if (nnztotala > VECTORIZE_NNZA_OR_NNZB_TH)
                {
                    for (int i = lane_id; i < nnztotala; i += WARP_SIZE)
                    {
                        unsigned char rowcolidx = d_blkcsr_Col_A[nnzastart + i];
                        int rowidxa = rowcolidx >> 4;
                        int rowidxb = rowcolidx & 0xf;
                        MAT_VAL_TYPE val = d_blkcsr_Val_A[nnzastart + i];
                        int blkoffseta = s_blkcsr_Ptr_C_local[rowidxa];
                        int blkoffseta_stop = rowidxa == BLOCK_SIZE - 1 ? blknnzctotal : s_blkcsr_Ptr_C_local[rowidxa + 1];

                        const int startb = s_csrRowPtrB_local[rowidxb];                                            //d_csrRowPtrB[rowidxb];
                        const int stopb = rowidxb == BLOCK_SIZE - 1 ? nnztotalb : s_csrRowPtrB_local[rowidxb + 1]; //d_csrRowPtrB[rowidxb+1];
                        for (int k = startb; k < stopb; k++)
                        {
                            unsigned char colidx = ld_gbl_uchar(d_blkcsr_Col_B + nnzbstart + k);

                            MAT_VAL_TYPE valb = ld_gbl_real(d_blkcsr_Val_B + nnzbstart + k);

                            int cnt = binary_search_exact_uchar_kernel(s_blkcsr_Idx_C_local + blkoffseta, 0, blkoffseta_stop - blkoffseta - 1, colidx);
                            if (cnt != -1)
                                atomicAdd(&d_blkcsr_Val_C[nnzcstart + blkoffseta + cnt], val * valb);
                        }
                    }
                }
                else
                {
                    for (int i = 0; i < nnztotala; i++)
                    {
                        unsigned char rowcolidx = d_blkcsr_Col_A[nnzastart + i];
                        int rowidxa = rowcolidx >> 4;
                        int rowidxb = rowcolidx & 0xf;
                        MAT_VAL_TYPE val = d_blkcsr_Val_A[nnzastart + i];
                        int blkoffseta = s_blkcsr_Ptr_C_local[rowidxa];
                        int blkoffseta_stop = rowidxa == BLOCK_SIZE - 1 ? blknnzctotal : s_blkcsr_Ptr_C_local[rowidxa + 1];

                        const int startb = s_csrRowPtrB_local[rowidxb];                                            //d_csrRowPtrB[rowidxb];
                        const int stopb = rowidxb == BLOCK_SIZE - 1 ? nnztotalb : s_csrRowPtrB_local[rowidxb + 1]; //d_csrRowPtrB[rowidxb+1];
                        int k = startb + lane_id;
                        if (k < stopb)
                        {
                            unsigned char colidx = ld_gbl_uchar(d_blkcsr_Col_B + nnzbstart + k);
                            MAT_VAL_TYPE valb = ld_gbl_real(d_blkcsr_Val_B + nnzbstart + k);

                            int cnt = binary_search_exact_uchar_kernel(s_blkcsr_Idx_C_local + blkoffseta, 0, blkoffseta_stop - blkoffseta - 1, colidx);
                            if (cnt != -1)
                                d_blkcsr_Val_C[nnzcstart + blkoffseta + cnt] += val * valb;
                        }
                    }
                }
                // do spgemm of this pair
                posa++;
                posa_updated = 1;
                posb++;
                posb_updated = 1;
            }
            else
            {
                // the smaller index goes forward
                posa_updated = idxa < idxb ? 1 : 0;
                posa += posa_updated;
                posb_updated = idxa > idxb ? 1 : 0;
                posb += posb_updated;
            }
        }
    }

    for (int i = lane_id; i < blknnzctotal; i += WARP_SIZE)
    {
        d_blkcsr_Col_C[nnzcstart + i] = s_blkcsr_Idx_C_local[i];
    }
}

template <int SMEM_MATNNZ>
__global__ void tile_spgemm_step4_cuda_kernel_smem_v3_halfwarp(int *d_blkrowptrA,
                                                               const int *__restrict__ d_blkcolidxA,
                                                               int *d_nnzb_A,
                                                               MAT_VAL_TYPE *d_blkcsr_Val_A,
                                                               unsigned char *d_blkcsr_Col_A,
                                                               unsigned char *d_blkcsr_Ptr_A,
                                                               int blkmA, int blknA, int numblkA, int nnzA,
                                                               const int *__restrict__ d_blkcolptrB,
                                                               const int *__restrict__ d_blkrowidxB,
                                                               const int *__restrict__ d_nnzb_B,
                                                               const MAT_VAL_TYPE *__restrict__ d_blkcsr_Val_B,
                                                               const unsigned char *__restrict__ d_blkcsr_Col_B,
                                                               const unsigned char *__restrict__ d_blkcsr_Ptr_B,
                                                               int blkmB, int blknB, int numblkB, int nnzB,
                                                               int *d_blkrowidxC,
                                                               int *d_blkcolidxC,
                                                               unsigned char *d_blkcsr_Ptr_C,
                                                               unsigned char *d_blkcsr_Col_C,
                                                               MAT_VAL_TYPE *d_blkcsr_Val_C,
                                                               int *d_nnzb_C,
                                                               unsigned short *d_blkmaskC,
                                                               int numblkC,
                                                               int *d_blkid,
                                                               int *d_spec_intersection_cnt,
                                                               int *d_spec_intersection_posa,
                                                               int *d_spec_intersection_posb)
{
    const int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int global_warp_id = global_id >> 4; //global_id / HALFWARP_SIZE;

    if (global_warp_id >= numblkC)
        return;
    int tilei = d_blkid[global_warp_id];

    const int nnzcstart = d_nnzb_C[tilei];
    const int blknnzctotal = d_nnzb_C[tilei + 1] - nnzcstart;
    if (!blknnzctotal)
        return;

    const int local_warp_id = threadIdx.x >> 4; //threadIdx.x / HALFWARP_SIZE;
    __shared__ unsigned char s_blkcsr_Idx_C[HALFWARP_PER_BLOCK * SMEM_MATNNZ];
    __shared__ unsigned char s_blkcsr_Ptr_C[HALFWARP_PER_BLOCK * BLOCK_SIZE];
    unsigned char *s_blkcsr_Idx_C_local = &s_blkcsr_Idx_C[local_warp_id * SMEM_MATNNZ];
    unsigned char *s_blkcsr_Ptr_C_local = &s_blkcsr_Ptr_C[local_warp_id * BLOCK_SIZE];

    __shared__ unsigned char s_csrRowPtrB[HALFWARP_PER_BLOCK * BLOCK_SIZE];
    unsigned char *s_csrRowPtrB_local = &s_csrRowPtrB[local_warp_id * BLOCK_SIZE];

    __shared__ int s_matched_posa[HALFWARP_PER_BLOCK * SPECULATIVE_INTERSECTION];
    __shared__ int s_matched_posb[HALFWARP_PER_BLOCK * SPECULATIVE_INTERSECTION];
    __shared__ int s_matchedcnt[HALFWARP_PER_BLOCK];

    int *s_matched_posa_local = &s_matched_posa[local_warp_id * SPECULATIVE_INTERSECTION];
    int *s_matched_posb_local = &s_matched_posb[local_warp_id * SPECULATIVE_INTERSECTION];
    int *s_matchedcnt_local = &s_matchedcnt[local_warp_id];

    const int lane_id = (HALFWARP_SIZE - 1) & threadIdx.x;

    for (int i = lane_id; i < blknnzctotal; i += HALFWARP_SIZE)
        d_blkcsr_Val_C[nnzcstart + i] = 0.0;

    if (lane_id < BLOCK_SIZE){
        long long int pos_c = (long long int)(tilei) * BLOCK_SIZE + lane_id;
        s_blkcsr_Ptr_C_local[lane_id] = d_blkcsr_Ptr_C[pos_c];}

    long long int pos_c = (long long int)(tilei) * BLOCK_SIZE + lane_id;
    unsigned int maskc = lane_id < BLOCK_SIZE ? d_blkmaskC[pos_c] : 0; //s_maskc_local[lane_id];
    unsigned char blknnzcstart = lane_id < BLOCK_SIZE ? s_blkcsr_Ptr_C_local[lane_id] : 0;
    if (!lane_id)
        s_matchedcnt_local[0] = 0;

    if (lane_id < BLOCK_SIZE)
    {
        int cnt = 0;
#pragma unroll
        for (int i = 0; i < BLOCK_SIZE; i++)
        {
            int idx = ((maskc >> BLOCK_SIZE - i - 1) & 0x1) == 1 ? i : -1;
            if (idx != -1)
            {
                s_blkcsr_Idx_C_local[blknnzcstart + cnt] = idx;
                cnt++;
            }
        }
    }

    const int blki = d_blkrowidxC[tilei];
    const int blkj = d_blkcolidxC[tilei];

    const int abase = ld_gbl_int32(d_blkrowptrA + blki);
    const int astop = ld_gbl_int32(d_blkrowptrA + blki + 1);
    int lena = astop - abase;

    const int bbase = ld_gbl_int32(d_blkcolptrB + blkj);
    const int bstop = ld_gbl_int32(d_blkcolptrB + blkj + 1);
    int lenb = bstop - bbase;

    int matchedcnt = 0;
    int specres = 0;

    if (USE_GMEM_SPECULATIVE_INTERSECTION)
        matchedcnt = d_spec_intersection_cnt[tilei];

    if (USE_GMEM_SPECULATIVE_INTERSECTION && matchedcnt > 0)
    {
        specres = 0;
        for (int si = lane_id; si < matchedcnt; si += HALFWARP_SIZE)
        {
            s_matched_posa_local[si] = d_spec_intersection_posa[tilei * GMEM_SPECULATIVE_INTERSECTION + si];
            s_matched_posb_local[si] = d_spec_intersection_posb[tilei * GMEM_SPECULATIVE_INTERSECTION + si];
        }
    }
    else
    {
        specres = intersection_binarysearch_kernel(d_blkcolidxA, abase, astop, lena,
                                                   d_blkrowidxB, bbase, bstop, lenb,
                                                   s_matched_posa_local, s_matched_posb_local,
                                                   SPECULATIVE_INTERSECTION, s_matchedcnt_local,
                                                   lane_id, HALFWARP_SIZE);

        matchedcnt = s_matchedcnt_local[0];
    }

    if (matchedcnt <= SPECULATIVE_INTERSECTION && specres == 0)
    {
        for (int posi = 0; posi < matchedcnt; posi++)
        {
            int posa = s_matched_posa_local[posi];
            int posb = s_matched_posb_local[posi];

            // atomic method
            const int nnzastart = d_nnzb_A[(abase + posa)];
            int nnztotala = d_nnzb_A[(abase + posa) + 1] - nnzastart;
            if (lane_id < BLOCK_SIZE)
                s_csrRowPtrB_local[lane_id] = ld_gbl_uchar(d_blkcsr_Ptr_B + (bbase + posb) * BLOCK_SIZE + lane_id);
            const int nnzbstart = ld_gbl_int32(d_nnzb_B + bbase + posb);
            int nnztotalb = ld_gbl_int32(d_nnzb_B + bbase + posb + 1) - nnzbstart;

            if (nnztotala > VECTORIZE_NNZA_OR_NNZB_TH)
            {
                for (int i = lane_id; i < nnztotala; i += HALFWARP_SIZE)
                {
                    unsigned char rowcolidx = d_blkcsr_Col_A[nnzastart + i];
                    int rowidxa = rowcolidx >> 4;
                    int rowidxb = rowcolidx & 0xf;
                    MAT_VAL_TYPE val = d_blkcsr_Val_A[nnzastart + i];
                    int blkoffseta = s_blkcsr_Ptr_C_local[rowidxa];
                    int blkoffseta_stop = rowidxa == BLOCK_SIZE - 1 ? blknnzctotal : s_blkcsr_Ptr_C_local[rowidxa + 1];

                    const int startb = s_csrRowPtrB_local[rowidxb];                                            //d_csrRowPtrB[rowidxb];
                    const int stopb = rowidxb == BLOCK_SIZE - 1 ? nnztotalb : s_csrRowPtrB_local[rowidxb + 1]; //d_csrRowPtrB[rowidxb+1];
                    for (int k = startb; k < stopb; k++)
                    {
                        unsigned char colidx = ld_gbl_uchar(d_blkcsr_Col_B + nnzbstart + k);
                        MAT_VAL_TYPE valb = ld_gbl_real(d_blkcsr_Val_B + nnzbstart + k);
                        int cnt = binary_search_exact_uchar_kernel(s_blkcsr_Idx_C_local + blkoffseta, 0, blkoffseta_stop - blkoffseta - 1, colidx);
                        if (cnt != -1)
                            atomicAdd(&d_blkcsr_Val_C[nnzcstart + blkoffseta + cnt], val * valb);
                    }
                }
            }
            else
            {
                for (int i = 0; i < nnztotala; i++)
                {
                    unsigned char rowcolidx = d_blkcsr_Col_A[nnzastart + i];
                    int rowidxa = rowcolidx >> 4;
                    int rowidxb = rowcolidx & 0xf;
                    MAT_VAL_TYPE val = d_blkcsr_Val_A[nnzastart + i];
                    int blkoffseta = s_blkcsr_Ptr_C_local[rowidxa];
                    int blkoffseta_stop = rowidxa == BLOCK_SIZE - 1 ? blknnzctotal : s_blkcsr_Ptr_C_local[rowidxa + 1];

                    const int startb = s_csrRowPtrB_local[rowidxb];                                            //d_csrRowPtrB[rowidxb];
                    const int stopb = rowidxb == BLOCK_SIZE - 1 ? nnztotalb : s_csrRowPtrB_local[rowidxb + 1]; //d_csrRowPtrB[rowidxb+1];
                    int k = startb + lane_id;
                    if (k < stopb)
                    {
                        unsigned char colidx = ld_gbl_uchar(d_blkcsr_Col_B + nnzbstart + k);
                        MAT_VAL_TYPE valb = ld_gbl_real(d_blkcsr_Val_B + nnzbstart + k);
                        int cnt = binary_search_exact_uchar_kernel(s_blkcsr_Idx_C_local + blkoffseta, 0, blkoffseta_stop - blkoffseta - 1, colidx);
                        if (cnt != -1)
                            d_blkcsr_Val_C[nnzcstart + blkoffseta + cnt] += val * valb;
                    }
                }
            }
        }
    }
    else
    {
        const int astart = d_blkcolidxA[abase];
        const int aend = d_blkcolidxA[astop - 1];
        const int bstart = ld_gbl_int32(d_blkrowidxB + bbase);
        const int bend = ld_gbl_int32(d_blkrowidxB + bstop - 1);

        int posa_real = 0;
        int posb_real = 0;
        if (bstart > astart)
        {
            int posa_real_new = binary_search_right_boundary_kernel(d_blkcolidxA + abase, bstart, lena);
            posa_real = posa_real_new < 0 ? 0 : posa_real_new;
        }
        else if (bstart < astart)
        {
            int posb_real_new = binary_search_right_boundary_kernel(d_blkrowidxB + bbase, astart, lenb);
            posb_real = posb_real_new < 0 ? 0 : posb_real_new;
        }

        if (bstop < astop)
        {
            int lena_new = binary_search_right_boundary_kernel(d_blkcolidxA + abase, bend, lena) + 1;
            lena = lena_new > lena ? lena : lena_new;
        }
        else if (bstop > astop)
        {
            int lenb_new = binary_search_right_boundary_kernel(d_blkrowidxB + bbase, aend, lenb) + 1;
            lenb = lenb_new > lenb ? lenb : lenb_new;
        }

        int posa = posa_real;
        int posb = posb_real;
        int idxa = 0;
        int idxb = 0;
        int posa_updated = 1;
        int posb_updated = 1;

        while (posa < lena && posb < lenb)
        {
            idxa = posa_updated ? ld_gbl_int32(d_blkcolidxA + abase + posa) : idxa; //a[posa] : idxa;
            idxb = posb_updated ? ld_gbl_int32(d_blkrowidxB + bbase + posb) : idxb; //b[posb] : idxb;

            if (idxa == idxb)
            {
                const int nnzastart = d_nnzb_A[(abase + posa)];
                int nnztotala = d_nnzb_A[(abase + posa) + 1] - nnzastart;
                if (lane_id < BLOCK_SIZE)
                    s_csrRowPtrB_local[lane_id] = ld_gbl_uchar(d_blkcsr_Ptr_B + (bbase + posb) * BLOCK_SIZE + lane_id);
                const int nnzbstart = ld_gbl_int32(d_nnzb_B + bbase + posb);
                int nnztotalb = ld_gbl_int32(d_nnzb_B + bbase + posb + 1) - nnzbstart;

                if (nnztotala > VECTORIZE_NNZA_OR_NNZB_TH)
                {
                    for (int i = lane_id; i < nnztotala; i += HALFWARP_SIZE)
                    {
                        unsigned char rowcolidx = d_blkcsr_Col_A[nnzastart + i];
                        int rowidxa = rowcolidx >> 4;
                        int rowidxb = rowcolidx & 0xf;
                        MAT_VAL_TYPE val = d_blkcsr_Val_A[nnzastart + i];
                        int blkoffseta = s_blkcsr_Ptr_C_local[rowidxa];
                        int blkoffseta_stop = rowidxa == BLOCK_SIZE - 1 ? blknnzctotal : s_blkcsr_Ptr_C_local[rowidxa + 1];

                        const int startb = s_csrRowPtrB_local[rowidxb];                                            //d_csrRowPtrB[rowidxb];
                        const int stopb = rowidxb == BLOCK_SIZE - 1 ? nnztotalb : s_csrRowPtrB_local[rowidxb + 1]; //d_csrRowPtrB[rowidxb+1];
                        for (int k = startb; k < stopb; k++)
                        {
                            unsigned char colidx = ld_gbl_uchar(d_blkcsr_Col_B + nnzbstart + k);
                            MAT_VAL_TYPE valb = ld_gbl_real(d_blkcsr_Val_B + nnzbstart + k);
                            int cnt = binary_search_exact_uchar_kernel(s_blkcsr_Idx_C_local + blkoffseta, 0, blkoffseta_stop - blkoffseta - 1, colidx);
                            if (cnt != -1)
                                atomicAdd(&d_blkcsr_Val_C[nnzcstart + blkoffseta + cnt], val * valb);
                        }
                    }
                }
                else
                {
                    for (int i = 0; i < nnztotala; i++)
                    {
                        unsigned char rowcolidx = d_blkcsr_Col_A[nnzastart + i];
                        int rowidxa = rowcolidx >> 4;
                        int rowidxb = rowcolidx & 0xf;
                        MAT_VAL_TYPE val = d_blkcsr_Val_A[nnzastart + i];
                        int blkoffseta = s_blkcsr_Ptr_C_local[rowidxa];
                        int blkoffseta_stop = rowidxa == BLOCK_SIZE - 1 ? blknnzctotal : s_blkcsr_Ptr_C_local[rowidxa + 1];

                        const int startb = s_csrRowPtrB_local[rowidxb];                                            //d_csrRowPtrB[rowidxb];
                        const int stopb = rowidxb == BLOCK_SIZE - 1 ? nnztotalb : s_csrRowPtrB_local[rowidxb + 1]; //d_csrRowPtrB[rowidxb+1];
                        int k = startb + lane_id;
                        if (k < stopb)
                        {
                            unsigned char colidx = ld_gbl_uchar(d_blkcsr_Col_B + nnzbstart + k);
                            MAT_VAL_TYPE valb = ld_gbl_real(d_blkcsr_Val_B + nnzbstart + k);
                            int cnt = binary_search_exact_uchar_kernel(s_blkcsr_Idx_C_local + blkoffseta, 0, blkoffseta_stop - blkoffseta - 1, colidx);
                            if (cnt != -1)
                                d_blkcsr_Val_C[nnzcstart + blkoffseta + cnt] += val * valb;
                        }
                    }
                }

                // do spgemm of this pair
                posa++;
                posa_updated = 1;
                posb++;
                posb_updated = 1;
            }
            else
            {
                // the smaller index goes forward
                posa_updated = idxa < idxb ? 1 : 0;
                posa += posa_updated;
                posb_updated = idxa > idxb ? 1 : 0;
                posb += posb_updated;
            }
        }
    }

    for (int i = lane_id; i < blknnzctotal; i += HALFWARP_SIZE)
    {
        d_blkcsr_Col_C[nnzcstart + i] = s_blkcsr_Idx_C_local[i];
    }
}

__global__ void tile_spgemm_step4_cuda_kernel_dns_noatomic_halfwarp(int *d_blkrowptrA,
                                                                    const int *__restrict__ d_blkcolidxA,
                                                                    int *d_nnzb_A,
                                                                    MAT_VAL_TYPE *d_blkcsr_Val_A,
                                                                    unsigned char *d_blkcsr_Col_A,
                                                                    unsigned char *d_blkcsr_Ptr_A,
                                                                    int blkmA, int blknA, int numblkA, int nnzA,
                                                                    const int *__restrict__ d_blkcolptrB,
                                                                    const int *__restrict__ d_blkrowidxB,
                                                                    const int *__restrict__ d_nnzb_B,
                                                                    const MAT_VAL_TYPE *__restrict__ d_blkcsr_Val_B,
                                                                    const unsigned char *__restrict__ d_blkcsr_Col_B,
                                                                    const unsigned char *__restrict__ d_blkcsr_Ptr_B,
                                                                    int blkmB, int blknB, int numblkB, int nnzB,
                                                                    int *d_blkrowidxC,
                                                                    int *d_blkcolidxC,
                                                                    unsigned char *d_blkcsr_Ptr_C,
                                                                    unsigned char *d_blkcsr_Col_C,
                                                                    MAT_VAL_TYPE *d_blkcsr_Val_C,
                                                                    int *d_nnzb_C,
                                                                    unsigned short *d_blkmaskC,
                                                                    int numblkC,
                                                                    int *d_blkid,
                                                                    int *d_spec_intersection_cnt,
                                                                    int *d_spec_intersection_posa,
                                                                    int *d_spec_intersection_posb)
{
    const int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int global_warp_id = global_id >> 4; //global_id / HALFWARP_SIZE;

    if (global_warp_id >= numblkC)
        return;
    int tilei = d_blkid[global_warp_id];

    const int nnzcstart = d_nnzb_C[tilei];
    const int blknnzctotal = d_nnzb_C[tilei + 1] - nnzcstart;
    if (!blknnzctotal)
        return;

    const int local_warp_id = threadIdx.x >> 4; //threadIdx.x / HALFWARP_SIZE;
    __shared__ MAT_VAL_TYPE s_blkcsr_Val_C[HALFWARP_PER_BLOCK * BLOCK_SIZE * BLOCK_SIZE];
    MAT_VAL_TYPE *s_blkcsr_Val_C_local = &s_blkcsr_Val_C[local_warp_id * BLOCK_SIZE * BLOCK_SIZE];

    __shared__ int s_matched_posa[HALFWARP_PER_BLOCK * SPECULATIVE_INTERSECTION];
    __shared__ int s_matched_posb[HALFWARP_PER_BLOCK * SPECULATIVE_INTERSECTION];
    __shared__ int s_matchedcnt[HALFWARP_PER_BLOCK];

    const int lane_id = (HALFWARP_SIZE - 1) & threadIdx.x;

    int *s_matched_posa_local = &s_matched_posa[local_warp_id * SPECULATIVE_INTERSECTION];
    int *s_matched_posb_local = &s_matched_posb[local_warp_id * SPECULATIVE_INTERSECTION];
    int *s_matchedcnt_local = &s_matchedcnt[local_warp_id];

#pragma unroll
    for (int i = 0; i < BLOCK_SIZE; i++)
        s_blkcsr_Val_C_local[i * BLOCK_SIZE + lane_id] = 0.0;

    if (!lane_id)
        s_matchedcnt_local[0] = 0;

    const int blki = d_blkrowidxC[tilei];
    const int blkj = d_blkcolidxC[tilei];

    const int abase = d_blkrowptrA[blki];
    const int astop = d_blkrowptrA[blki + 1];
    int lena = astop - abase;

    const int bbase = ld_gbl_int32(d_blkcolptrB + blkj);
    const int bstop = ld_gbl_int32(d_blkcolptrB + blkj + 1);
    int lenb = bstop - bbase;

    int matchedcnt = 0;
    int specres = 0;

    if (USE_GMEM_SPECULATIVE_INTERSECTION)
        matchedcnt = d_spec_intersection_cnt[tilei];

    if (USE_GMEM_SPECULATIVE_INTERSECTION && matchedcnt > 0)
    {
        specres = 0;
        for (int si = lane_id; si < matchedcnt; si += HALFWARP_SIZE)
        {
            s_matched_posa_local[si] = d_spec_intersection_posa[tilei * GMEM_SPECULATIVE_INTERSECTION + si];
            s_matched_posb_local[si] = d_spec_intersection_posb[tilei * GMEM_SPECULATIVE_INTERSECTION + si];
        }
    }
    else
    {
        specres = intersection_binarysearch_kernel(d_blkcolidxA, abase, astop, lena,
                                                   d_blkrowidxB, bbase, bstop, lenb,
                                                   s_matched_posa_local, s_matched_posb_local,
                                                   SPECULATIVE_INTERSECTION, s_matchedcnt_local,
                                                   lane_id, HALFWARP_SIZE);

        matchedcnt = s_matchedcnt_local[0];
    }

    if (matchedcnt <= SPECULATIVE_INTERSECTION && specres == 0)
    {
        for (int i = 0; i < matchedcnt; i++)
        {
            int posa = s_matched_posa_local[i];
            int posb = s_matched_posb_local[i];

            const int nnzastart = d_nnzb_A[(abase + posa)];
            int nnztotala = d_nnzb_A[(abase + posa) + 1] - nnzastart;
            const unsigned char *__restrict__ d_csrRowPtrB = &d_blkcsr_Ptr_B[(bbase + posb) * BLOCK_SIZE];
            const int nnzbstart = ld_gbl_int32(d_nnzb_B + bbase + posb);
            int nnztotalb = ld_gbl_int32(d_nnzb_B + bbase + posb + 1) - nnzbstart;

            for (int i = 0; i < nnztotala; i++)
            {
                unsigned char rowcolidx = d_blkcsr_Col_A[nnzastart + i];
                int rowidxa = rowcolidx >> 4;
                int rowidxb = rowcolidx & 0xf;
                MAT_VAL_TYPE val = d_blkcsr_Val_A[nnzastart + i];

                const int startb = ld_gbl_uchar(d_csrRowPtrB + rowidxb);
                const int stopb = rowidxb == BLOCK_SIZE - 1 ? nnztotalb : ld_gbl_uchar(d_csrRowPtrB + rowidxb + 1);
                int k = startb + lane_id;
                if (k < stopb)
                {
                    unsigned char colidx = ld_gbl_uchar(d_blkcsr_Col_B + nnzbstart + k);
                    MAT_VAL_TYPE valb = ld_gbl_real(d_blkcsr_Val_B + nnzbstart + k);
                    s_blkcsr_Val_C_local[rowidxa * BLOCK_SIZE + colidx] += val * valb;
                }
            }
        }
    }
    else
    {
        const int astart = d_blkcolidxA[abase];
        const int aend = d_blkcolidxA[astop - 1];
        const int bstart = ld_gbl_int32(d_blkrowidxB + bbase);
        const int bend = ld_gbl_int32(d_blkrowidxB + bstop - 1);

        int posa_real = 0;
        int posb_real = 0;
        if (bstart > astart)
        {
            int posa_real_new = binary_search_right_boundary_kernel(d_blkcolidxA + abase, bstart, lena);
            posa_real = posa_real_new < 0 ? 0 : posa_real_new;
        }
        else if (bstart < astart)
        {
            int posb_real_new = binary_search_right_boundary_kernel(d_blkrowidxB + bbase, astart, lenb);
            posb_real = posb_real_new < 0 ? 0 : posb_real_new;
        }

        if (bstop < astop)
        {
            int lena_new = binary_search_right_boundary_kernel(d_blkcolidxA + abase, bend, lena) + 1;
            lena = lena_new > lena ? lena : lena_new;
        }
        else if (bstop > astop)
        {
            int lenb_new = binary_search_right_boundary_kernel(d_blkrowidxB + bbase, aend, lenb) + 1;
            lenb = lenb_new > lenb ? lenb : lenb_new;
        }

        int posa = posa_real;
        int posb = posb_real;
        int idxa = 0;
        int idxb = 0;
        int posa_updated = 1;
        int posb_updated = 1;

        while (posa < lena && posb < lenb)
        {
            idxa = posa_updated ? ld_gbl_int32(d_blkcolidxA + abase + posa) : idxa; //a[posa] : idxa;
            idxb = posb_updated ? ld_gbl_int32(d_blkrowidxB + bbase + posb) : idxb; //b[posb] : idxb;

            if (idxa == idxb)
            {
                const int nnzastart = d_nnzb_A[(abase + posa)];
                int nnztotala = d_nnzb_A[(abase + posa) + 1] - nnzastart;
                const unsigned char *__restrict__ d_csrRowPtrB = &d_blkcsr_Ptr_B[(bbase + posb) * BLOCK_SIZE];
                const int nnzbstart = ld_gbl_int32(d_nnzb_B + bbase + posb);
                int nnztotalb = ld_gbl_int32(d_nnzb_B + bbase + posb + 1) - nnzbstart;
                if (lane_id < BLOCK_SIZE)
                {
                    unsigned char offseta_start = d_blkcsr_Ptr_A[(abase + posa) * BLOCK_SIZE + lane_id];
                    unsigned char offseta_end = lane_id == BLOCK_SIZE - 1 ? nnztotala : d_blkcsr_Ptr_A[(abase + posa) * BLOCK_SIZE + lane_id + 1];

                    for (int i = offseta_start; i < offseta_end; i++)
                    {
                        unsigned char rowcolidx = d_blkcsr_Col_A[nnzastart + i];
                        int rowidxa = rowcolidx >> 4;
                        int rowidxb = rowcolidx & 0xf;
                        MAT_VAL_TYPE val = d_blkcsr_Val_A[nnzastart + i];

                        const int startb = ld_gbl_uchar(d_csrRowPtrB + rowidxb);
                        const int stopb = rowidxb == BLOCK_SIZE - 1 ? nnztotalb : ld_gbl_uchar(d_csrRowPtrB + rowidxb + 1);
                        for (int k = startb; k < stopb; k++)
                        {
                            unsigned char colidx = ld_gbl_uchar(d_blkcsr_Col_B + nnzbstart + k);

                            MAT_VAL_TYPE valb = ld_gbl_real(d_blkcsr_Val_B + nnzbstart + k);
                            s_blkcsr_Val_C_local[rowidxa * BLOCK_SIZE + colidx] += val * valb;
                        }
                    }
                }

                posa++;
                posa_updated = 1;
                posb++;
                posb_updated = 1;
            }
            else
            {
                // the smaller index goes forward
                posa_updated = idxa < idxb ? 1 : 0;
                posa += posa_updated;
                posb_updated = idxa > idxb ? 1 : 0;
                posb += posb_updated;
            }
        }
    }

    if (blknnzctotal == 256)
    {
        for (int i = lane_id; i < BLOCK_SIZE * BLOCK_SIZE; i += HALFWARP_SIZE)
        {
            d_blkcsr_Col_C[nnzcstart + i] = i % BLOCK_SIZE;
            d_blkcsr_Val_C[nnzcstart + i] = s_blkcsr_Val_C_local[i];
        }
    }
    if (blknnzctotal != 256 && lane_id < BLOCK_SIZE)
    {
        long long int pos_c = (long long int)(tilei) * BLOCK_SIZE + lane_id;
        unsigned short maskc = d_blkmaskC[pos_c]; //s_maskc_local[lane_id];
        int cnt = 0;
        int blknnzcstart = d_blkcsr_Ptr_C[pos_c];
#pragma unroll 16
        for (int i = 0; i < BLOCK_SIZE; i++)
        {
            int idx = ((maskc >> BLOCK_SIZE - i - 1) & 0x1) == 1 ? i : -1;
            if (idx != -1)
            {
                d_blkcsr_Col_C[nnzcstart + blknnzcstart + cnt] = idx;
                d_blkcsr_Val_C[nnzcstart + blknnzcstart + cnt] = s_blkcsr_Val_C_local[lane_id * BLOCK_SIZE + idx];
                cnt++;
            }
        }
    }
}

__global__ void tile_spgemm_step4_cuda_kernel_ful_noatomic_halfwarp(int *d_blkrowptrA,
                                                                    const int *__restrict__ d_blkcolidxA,
                                                                    int *d_nnzb_A,
                                                                    MAT_VAL_TYPE *d_blkcsr_Val_A,
                                                                    unsigned char *d_blkcsr_Col_A,
                                                                    unsigned char *d_blkcsr_Ptr_A,
                                                                    int blkmA, int blknA, int numblkA, int nnzA,
                                                                    const int *__restrict__ d_blkcolptrB,
                                                                    const int *__restrict__ d_blkrowidxB,
                                                                    const int *__restrict__ d_nnzb_B,
                                                                    const MAT_VAL_TYPE *__restrict__ d_blkcsr_Val_B,
                                                                    const unsigned char *__restrict__ d_blkcsr_Col_B,
                                                                    const unsigned char *__restrict__ d_blkcsr_Ptr_B,
                                                                    int blkmB, int blknB, int numblkB, int nnzB,
                                                                    int *d_blkrowidxC,
                                                                    int *d_blkcolidxC,
                                                                    unsigned char *d_blkcsr_Ptr_C,
                                                                    unsigned char *d_blkcsr_Col_C,
                                                                    MAT_VAL_TYPE *d_blkcsr_Val_C,
                                                                    int *d_nnzb_C,
                                                                    unsigned short *d_blkmaskC,
                                                                    int numblkC,
                                                                    int *d_blkid,
                                                                    int *d_spec_intersection_cnt,
                                                                    int *d_spec_intersection_posa,
                                                                    int *d_spec_intersection_posb)
{
    const int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int global_warp_id = global_id >> 4; //global_id / HALFWARP_SIZE;

    if (global_warp_id >= numblkC)
        return;
    int tilei = d_blkid[global_warp_id];

    const int nnzcstart = d_nnzb_C[tilei];

    const int local_warp_id = threadIdx.x >> 4; //threadIdx.x / HALFWARP_SIZE;
    __shared__ MAT_VAL_TYPE s_blkcsr_Val_C[HALFWARP_PER_BLOCK * BLOCK_SIZE * BLOCK_SIZE];
    MAT_VAL_TYPE *s_blkcsr_Val_C_local = &s_blkcsr_Val_C[local_warp_id * BLOCK_SIZE * BLOCK_SIZE];

    __shared__ int s_matched_posa[HALFWARP_PER_BLOCK * SPECULATIVE_INTERSECTION];
    __shared__ int s_matched_posb[HALFWARP_PER_BLOCK * SPECULATIVE_INTERSECTION];
    __shared__ int s_matchedcnt[HALFWARP_PER_BLOCK];

    const int lane_id = (HALFWARP_SIZE - 1) & threadIdx.x;

    int *s_matched_posa_local = &s_matched_posa[local_warp_id * SPECULATIVE_INTERSECTION];
    int *s_matched_posb_local = &s_matched_posb[local_warp_id * SPECULATIVE_INTERSECTION];
    int *s_matchedcnt_local = &s_matchedcnt[local_warp_id];

#pragma unroll
    for (int i = 0; i < BLOCK_SIZE; i++)
        s_blkcsr_Val_C_local[i * BLOCK_SIZE + lane_id] = 0.0;

    if (!lane_id)
        s_matchedcnt_local[0] = 0;

    const int blki = d_blkrowidxC[tilei];
    const int blkj = d_blkcolidxC[tilei];

    const int abase = d_blkrowptrA[blki];
    const int astop = d_blkrowptrA[blki + 1];
    int lena = astop - abase;

    const int bbase = ld_gbl_int32(d_blkcolptrB + blkj);
    const int bstop = ld_gbl_int32(d_blkcolptrB + blkj + 1);
    int lenb = bstop - bbase;

    int matchedcnt = 0;
    int specres = 0;

    if (USE_GMEM_SPECULATIVE_INTERSECTION)
        matchedcnt = d_spec_intersection_cnt[tilei];

    if (USE_GMEM_SPECULATIVE_INTERSECTION && matchedcnt > 0)
    {
        specres = 0;
        for (int si = lane_id; si < matchedcnt; si += HALFWARP_SIZE)
        {
            s_matched_posa_local[si] = d_spec_intersection_posa[tilei * GMEM_SPECULATIVE_INTERSECTION + si];
            s_matched_posb_local[si] = d_spec_intersection_posb[tilei * GMEM_SPECULATIVE_INTERSECTION + si];
        }
    }
    else
    {
        specres = intersection_binarysearch_kernel(d_blkcolidxA, abase, astop, lena,
                                                   d_blkrowidxB, bbase, bstop, lenb,
                                                   s_matched_posa_local, s_matched_posb_local,
                                                   SPECULATIVE_INTERSECTION, s_matchedcnt_local,
                                                   lane_id, HALFWARP_SIZE);

        matchedcnt = s_matchedcnt_local[0];
    }

    if (matchedcnt <= SPECULATIVE_INTERSECTION && specres == 0)
    {
        for (int i = 0; i < matchedcnt; i++)
        {
            int posa = s_matched_posa_local[i];
            int posb = s_matched_posb_local[i];

            const int nnzastart = d_nnzb_A[(abase + posa)];
            int nnztotala = d_nnzb_A[(abase + posa) + 1] - nnzastart;
            const unsigned char *__restrict__ d_csrRowPtrB = &d_blkcsr_Ptr_B[(bbase + posb) * BLOCK_SIZE];
            const int nnzbstart = ld_gbl_int32(d_nnzb_B + bbase + posb);
            int nnztotalb = ld_gbl_int32(d_nnzb_B + bbase + posb + 1) - nnzbstart;

            for (int i = 0; i < nnztotala; i++)
            {
                unsigned char rowcolidx = d_blkcsr_Col_A[nnzastart + i];
                int rowidxa = rowcolidx >> 4;
                int rowidxb = rowcolidx & 0xf;
                MAT_VAL_TYPE val = d_blkcsr_Val_A[nnzastart + i];

                const int startb = ld_gbl_uchar(d_csrRowPtrB + rowidxb);
                const int stopb = rowidxb == BLOCK_SIZE - 1 ? nnztotalb : ld_gbl_uchar(d_csrRowPtrB + rowidxb + 1);
                int k = startb + lane_id;
                if (k < stopb)
                {
                    unsigned char colidx = ld_gbl_uchar(d_blkcsr_Col_B + nnzbstart + k);
                    MAT_VAL_TYPE valb = ld_gbl_real(d_blkcsr_Val_B + nnzbstart + k);
                    s_blkcsr_Val_C_local[rowidxa * BLOCK_SIZE + colidx] += val * valb;
                }
            }
        }
    }
    else
    {
        const int astart = d_blkcolidxA[abase];
        const int aend = d_blkcolidxA[astop - 1];
        const int bstart = ld_gbl_int32(d_blkrowidxB + bbase);
        const int bend = ld_gbl_int32(d_blkrowidxB + bstop - 1);

        int posa_real = 0;
        int posb_real = 0;
        if (bstart > astart)
        {
            int posa_real_new = binary_search_right_boundary_kernel(d_blkcolidxA + abase, bstart, lena);
            posa_real = posa_real_new < 0 ? 0 : posa_real_new;
        }
        else if (bstart < astart)
        {
            int posb_real_new = binary_search_right_boundary_kernel(d_blkrowidxB + bbase, astart, lenb);
            posb_real = posb_real_new < 0 ? 0 : posb_real_new;
        }

        if (bstop < astop)
        {
            int lena_new = binary_search_right_boundary_kernel(d_blkcolidxA + abase, bend, lena) + 1;
            lena = lena_new > lena ? lena : lena_new;
        }
        else if (bstop > astop)
        {
            int lenb_new = binary_search_right_boundary_kernel(d_blkrowidxB + bbase, aend, lenb) + 1;
            lenb = lenb_new > lenb ? lenb : lenb_new;
        }

        int posa = posa_real;
        int posb = posb_real;
        int idxa = 0;
        int idxb = 0;
        int posa_updated = 1;
        int posb_updated = 1;

        while (posa < lena && posb < lenb)
        {
            idxa = posa_updated ? ld_gbl_int32(d_blkcolidxA + abase + posa) : idxa; //a[posa] : idxa;
            idxb = posb_updated ? ld_gbl_int32(d_blkrowidxB + bbase + posb) : idxb; //b[posb] : idxb;

            if (idxa == idxb)
            {
                const int nnzastart = d_nnzb_A[(abase + posa)];
                int nnztotala = d_nnzb_A[(abase + posa) + 1] - nnzastart;
                const unsigned char *__restrict__ d_csrRowPtrB = &d_blkcsr_Ptr_B[(bbase + posb) * BLOCK_SIZE];
                const int nnzbstart = ld_gbl_int32(d_nnzb_B + bbase + posb);
                int nnztotalb = ld_gbl_int32(d_nnzb_B + bbase + posb + 1) - nnzbstart;

                if (lane_id < BLOCK_SIZE)
                {
                    unsigned char offseta_start = d_blkcsr_Ptr_A[(abase + posa) * BLOCK_SIZE + lane_id];
                    unsigned char offseta_end = lane_id == BLOCK_SIZE - 1 ? nnztotala : d_blkcsr_Ptr_A[(abase + posa) * BLOCK_SIZE + lane_id + 1];

                    for (int i = offseta_start; i < offseta_end; i++)
                    {
                        unsigned char rowcolidx = d_blkcsr_Col_A[nnzastart + i];
                        int rowidxa = rowcolidx >> 4;
                        int rowidxb = rowcolidx & 0xf;
                        MAT_VAL_TYPE val = d_blkcsr_Val_A[nnzastart + i];

                        const int startb = ld_gbl_uchar(d_csrRowPtrB + rowidxb);
                        const int stopb = rowidxb == BLOCK_SIZE - 1 ? nnztotalb : ld_gbl_uchar(d_csrRowPtrB + rowidxb + 1);
                        for (int k = startb; k < stopb; k++)
                        {
                            unsigned char colidx = ld_gbl_uchar(d_blkcsr_Col_B + nnzbstart + k);

                            MAT_VAL_TYPE valb = ld_gbl_real(d_blkcsr_Val_B + nnzbstart + k);
                            s_blkcsr_Val_C_local[rowidxa * BLOCK_SIZE + colidx] += val * valb;
                        }
                    }
                }

                posa++;
                posa_updated = 1;
                posb++;
                posb_updated = 1;
            }
            else
            {
                posa_updated = idxa < idxb ? 1 : 0;
                posa += posa_updated;
                posb_updated = idxa > idxb ? 1 : 0;
                posb += posb_updated;
            }
        }
    }

#pragma unroll
    for (int i = 0; i < BLOCK_SIZE; i++)
    {
        int offset_local = i * BLOCK_SIZE + lane_id;
        d_blkcsr_Col_C[nnzcstart + offset_local] = lane_id;
        d_blkcsr_Val_C[nnzcstart + offset_local] = s_blkcsr_Val_C_local[offset_local];
    }
}

void tilespgemm(SMatrix *matrixA,
                SMatrix *matrixB,
                SMatrix *matrixC,
                unsigned int *blk_intersec_bitmask_A,
                unsigned int *blk_intersec_bitmask_B,
                int blk_intersec_bitmask_len,
                double densityA,
                double densityB,
                unsigned long long int nnzCub,
                unsigned long long int *nnzC_computed,
                double *compression_rate,
                double *time_tile,
                double *gflops_tile,
                char *filename,
                double *time_step1, double *time_step2, double *time_step3, double *time_malloc)
{


    int *d_blkrowptrA;
    int *d_blkcolidxA;
    int *d_nnzb_A;
    MAT_VAL_TYPE *d_blkcsr_Val_A;
    unsigned char *d_blkcsr_Col_A;
    unsigned char *d_blkcsr_Ptr_A;
    int blkmA = matrixA->tilem;
    int blknA = matrixA->tilen;
    int nnzA = matrixA->nnz;
    int numblkA = matrixA->numtile;
    int *blkrowptrA = matrixA->tile_ptr;
    int *blkcolidxA = matrixA->tile_columnidx;
    int *nnzb_A = matrixA->tile_nnz;
    MAT_VAL_TYPE *blkcsr_Val_A = matrixA->tile_csr_Value;
    unsigned char *blkcsr_Col_A = matrixA->tile_csr_Col;
    unsigned char *blkcsr_Ptr_A = matrixA->tile_csr_Ptr;
    unsigned short *blkmaskA = matrixA->mask;

    cudaMalloc((void **)&d_blkrowptrA, (blkmA + 1) * sizeof(int));
    cudaMalloc((void **)&d_blkcolidxA, numblkA * sizeof(int));
    cudaMalloc((void **)&d_nnzb_A, (numblkA + 1) * sizeof(int));
    cudaMalloc((void **)&d_blkcsr_Val_A, nnzA * sizeof(MAT_VAL_TYPE));
    cudaMalloc((void **)&d_blkcsr_Col_A, nnzA * sizeof(unsigned char));
    cudaMalloc((void **)&d_blkcsr_Ptr_A, numblkA * BLOCK_SIZE * sizeof(unsigned char));

    cudaMemcpy(d_blkrowptrA, blkrowptrA, (blkmA + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_blkcolidxA, blkcolidxA, numblkA * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nnzb_A, nnzb_A, (numblkA + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_blkcsr_Val_A, blkcsr_Val_A, nnzA * sizeof(MAT_VAL_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(d_blkcsr_Col_A, blkcsr_Col_A, nnzA * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_blkcsr_Ptr_A, blkcsr_Ptr_A, numblkA * BLOCK_SIZE * sizeof(unsigned char), cudaMemcpyHostToDevice);

    int *d_blkcolptrB;
    int *d_blkrowidxB;
    int *d_blkrowptrB;
    int *d_blkcolidxB;
    int *d_nnzb_B;
    MAT_VAL_TYPE *d_blkcsr_Val_B;
    unsigned char *d_blkcsr_Col_B;
    unsigned char *d_blkcsr_Ptr_B;
    int blknB = matrixB->tilen;
    int blkmB = matrixB->tilem;
    int numblkB = matrixB->numtile;
    int nnzB = matrixB->nnz;
    unsigned int *d_blk_intersec_bitmask_A;
    unsigned int *d_blk_intersec_bitmask_B;
    int *blkcolptrB = matrixB->csc_tile_ptr;
    int *blkrowidxB = matrixB->csc_tile_rowidx;
    int *blkrowptrB = matrixB->tile_ptr;
    int *blkcolidxB = matrixB->tile_columnidx;
    int *nnzb_B = matrixB->tile_nnz;
    MAT_VAL_TYPE *blkcsr_Val_B = matrixB->tile_csr_Value;
    unsigned char *blkcsr_Col_B = matrixB->tile_csr_Col;
    unsigned char *blkcsr_Ptr_B = matrixB->tile_csr_Ptr;
    unsigned short *blkmaskB = matrixB->mask;

    cudaMalloc((void **)&d_blkcolptrB, (blknB + 1) * sizeof(int));
    cudaMalloc((void **)&d_blkrowidxB, numblkB * sizeof(int));
    cudaMalloc((void **)&d_blkrowptrB, (blkmB + 1) * sizeof(int));
    cudaMalloc((void **)&d_blkcolidxB, numblkB * sizeof(int));
    cudaMalloc((void **)&d_nnzb_B, (numblkB + 1) * sizeof(int));
    cudaMalloc((void **)&d_blkcsr_Val_B, nnzB * sizeof(MAT_VAL_TYPE));
    cudaMalloc((void **)&d_blkcsr_Col_B, nnzB * sizeof(unsigned char));
    cudaMalloc((void **)&d_blkcsr_Ptr_B, numblkB * BLOCK_SIZE * sizeof(unsigned char));
    cudaMalloc((void **)&d_blk_intersec_bitmask_A, blkmA * blk_intersec_bitmask_len * sizeof(unsigned int));
    cudaMalloc((void **)&d_blk_intersec_bitmask_B, blknB * blk_intersec_bitmask_len * sizeof(unsigned int));

    cudaMemcpy(d_blkcolptrB, blkcolptrB, (blknB + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_blkrowidxB, blkrowidxB, numblkB * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_blkrowptrB, blkrowptrB, (blkmB + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_blkcolidxB, blkcolidxB, numblkB * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nnzb_B, nnzb_B, (numblkB + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_blkcsr_Val_B, blkcsr_Val_B, nnzB * sizeof(MAT_VAL_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(d_blkcsr_Col_B, blkcsr_Col_B, nnzB * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_blkcsr_Ptr_B, blkcsr_Ptr_B, numblkB * BLOCK_SIZE * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_blk_intersec_bitmask_A, blk_intersec_bitmask_A, blkmA * blk_intersec_bitmask_len * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_blk_intersec_bitmask_B, blk_intersec_bitmask_B, blknB * blk_intersec_bitmask_len * sizeof(unsigned int), cudaMemcpyHostToDevice);

    unsigned short *d_blkmaskB;
    cudaMalloc((void **)&d_blkmaskB, numblkB * BLOCK_SIZE * sizeof(unsigned short));
    cudaMemcpy(d_blkmaskB, blkmaskB, numblkB * BLOCK_SIZE * sizeof(unsigned short), cudaMemcpyHostToDevice);

    unsigned short *d_blkmaskA;
    cudaMalloc((void **)&d_blkmaskA, numblkA * BLOCK_SIZE * sizeof(unsigned short));
    cudaMemcpy(d_blkmaskA, blkmaskA, numblkA * BLOCK_SIZE * sizeof(unsigned short), cudaMemcpyHostToDevice);

    int numblkC = 0;
    int nnzC = 0;
    double tile_spgemm_time = 0;

    int nstreams = 5;

    cudaStream_t *streams = (cudaStream_t *)malloc(sizeof(cudaStream_t) * nstreams);
    for (int i = 0; i < nstreams; i++)
    {
        cudaStreamCreate(&(streams[i]));
    }

    double time_all[REPEAT_NUM];

    int *d_blksmem_tny_cnt;
    int *d_blksmem_sml_cnt;
    int *d_blksmem_lrg_cnt;
    int *d_blksmem_dns_cnt;
    int *d_blksmem_ful_cnt;

    cudaMalloc((void **)&d_blksmem_tny_cnt, 1 * sizeof(int));
    cudaMalloc((void **)&d_blksmem_sml_cnt, 1 * sizeof(int));
    cudaMalloc((void **)&d_blksmem_lrg_cnt, 1 * sizeof(int));
    cudaMalloc((void **)&d_blksmem_dns_cnt, 1 * sizeof(int));
    cudaMalloc((void **)&d_blksmem_ful_cnt, 1 * sizeof(int));

    for (int ri = 0; ri < REPEAT_NUM; ri++)
    {
        // call cuda kernel
        struct timeval tstart, tend;
        struct timeval t1, t2;
        cudaDeviceSynchronize();
        gettimeofday(&tstart, NULL);

#if TIMING
        gettimeofday(&t1, NULL);
#endif

        int *d_blkrowptrC;
        cudaMalloc((void **)&d_blkrowptrC, (blkmA + 1) * sizeof(int));

#if TIMING
        *time_malloc = 0;
        gettimeofday(&t2, NULL);
        *time_malloc += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
#endif

        numblkC = 0;
        sfBIN bin;
#if TIMING
        gettimeofday(&t1, NULL);
#endif
        if (blknB > NUMCOLC_SPA_OR_HASH_TH)
        {
            /* Initialize bin */
            init_bin(&bin, blkmA);

            /* Set max bin */
            set_max_bin(d_blkrowptrA, d_blkcolidxA, d_blkrowptrB, &bin, blkmA);
            /* Count nz of C */
            set_row_nnz(d_blkrowptrA, d_blkcolidxA,
                        d_blkrowptrB, d_blkcolidxB,
                        d_blkrowptrC,
                        &bin,
                        blkmA,
                        &numblkC);
            /* Set bin */
            set_min_bin(&bin, blkmA);
        }
        else
        {
            int num_threads = WARP_SIZE * WARP_PER_BLOCK;
            int num_blocks = ceil((double)blkmA / (double)(WARP_PER_BLOCK));
            tile_spgemm_step1_cuda_spa_kernel<<<num_blocks, num_threads>>>(d_blkrowptrA, d_blkcolidxA, blkmA,
                                                                           d_blkrowptrB, d_blkcolidxB, blknB,
                                                                           d_blkrowptrC);
            exclusive_scan_device_cuda_thrust<int>(d_blkrowptrC, blkmA + 1);
            cudaMemcpy(&numblkC, &d_blkrowptrC[blkmA], sizeof(int), cudaMemcpyDeviceToHost);
        }

#if TIMING
        cudaDeviceSynchronize();
        gettimeofday(&t2, NULL);
        *time_step1 = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
#endif


#if TIMING
    gettimeofday(&t1, NULL);
#endif
        int *d_blkrowidxC;
        cudaMalloc((void **)&d_blkrowidxC, numblkC * sizeof(int));
        int *d_blkcolidxC;
        cudaMalloc((void **)&d_blkcolidxC, numblkC * sizeof(int));

#if TIMING

        gettimeofday(&t2, NULL);
        *time_malloc += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
#endif

#if TIMING
        gettimeofday(&t1, NULL);
#endif

        int *d_spec_intersection_cnt;
        int *d_spec_intersection_posa;
        int *d_spec_intersection_posb;

        if (USE_GMEM_SPECULATIVE_INTERSECTION)
        {
            cudaMalloc((void **)&d_spec_intersection_cnt, numblkC * sizeof(int));
            cudaMemset(d_spec_intersection_cnt, 0, numblkC * sizeof(int));
            cudaMalloc((void **)&d_spec_intersection_posa, numblkC * GMEM_SPECULATIVE_INTERSECTION * sizeof(int));
            cudaMalloc((void **)&d_spec_intersection_posb, numblkC * GMEM_SPECULATIVE_INTERSECTION * sizeof(int));
        }

        if (blknB > NUMCOLC_SPA_OR_HASH_TH)
        {
            /* Calculating value of C */
            calculate_value_col_bin(d_blkrowptrA, d_blkcolidxA, NULL,
                                    d_blkrowptrB, d_blkcolidxB, NULL,
                                    d_blkrowptrC, d_blkrowidxC, d_blkcolidxC, NULL,
                                    &bin, blkmA, blkmB);
            release_bin(bin);
        }
        else
        {
            int num_threads = WARP_SIZE * WARP_PER_BLOCK;
            int num_blocks = ceil((double)blkmA / (double)(WARP_PER_BLOCK));
            tile_spgemm_step1_numeric_cuda_spa_kernel<<<num_blocks, num_threads>>>(d_blkrowptrA, d_blkcolidxA, blkmA,
                                                                                   d_blkrowptrB, d_blkcolidxB, blknB,
                                                                                   d_blkrowptrC, d_blkrowidxC, d_blkcolidxC,
                                                                                   d_spec_intersection_cnt, d_spec_intersection_posa, d_spec_intersection_posb);
        }

#if TIMING
        cudaDeviceSynchronize();
        gettimeofday(&t2, NULL);
        *time_step1 += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
        if (ri == 0)
        {
            printf("step1 ----Calculate the number and tile-column index of tiles of matrixC---\n");
            printf("step1 ---------------------- Runtime is  %.2f ms-------------------------\n", *time_step1);
        }

#endif

#if TIMING
        gettimeofday(&t1, NULL);
#endif

        long long int lengthC =  (long long int)numblkC * BLOCK_SIZE;

        unsigned char *d_blkcsr_Ptr_C;
        cudaMalloc((void **)&d_blkcsr_Ptr_C, lengthC * sizeof(unsigned char));

        if (d_blkcsr_Ptr_C == NULL)
        {
            printf("d_blkcsr_Ptr_C failed\n");
        }

        int *d_nnzb_C;
        cudaMalloc((void **)&d_nnzb_C, (numblkC + 1) * sizeof(int));
        cudaMemset(d_nnzb_C, 0, (numblkC + 1) * sizeof(int));

        unsigned short *d_blkmaskC;
        cudaMalloc((void **)&d_blkmaskC, lengthC * sizeof(unsigned short));
        if (d_blkmaskC == NULL)
        {
            printf("d_blkmaskC failed\n");
        }


        int *d_blkid_smem_tny;
        int *d_blkid_smem_sml;
        int *d_blkid_smem_lrg;
        int *d_blkid_smem_dns;
        int *d_blkid_smem_ful;

        cudaMalloc((void **)&d_blkid_smem_tny, numblkC * sizeof(int));
        cudaMalloc((void **)&d_blkid_smem_sml, numblkC * sizeof(int));
        cudaMalloc((void **)&d_blkid_smem_lrg, numblkC * sizeof(int));
        cudaMalloc((void **)&d_blkid_smem_dns, numblkC * sizeof(int));
        cudaMalloc((void **)&d_blkid_smem_ful, numblkC * sizeof(int));

        cudaMemset(d_blksmem_tny_cnt, 0, 1 * sizeof(int));
        cudaMemset(d_blksmem_sml_cnt, 0, 1 * sizeof(int));
        cudaMemset(d_blksmem_lrg_cnt, 0, 1 * sizeof(int));
        cudaMemset(d_blksmem_dns_cnt, 0, 1 * sizeof(int));
        cudaMemset(d_blksmem_ful_cnt, 0, 1 * sizeof(int));

        int num_threads, num_blocks;

#if TIMING
        gettimeofday(&t2, NULL);
        *time_malloc += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
#endif

#if TIMING
        gettimeofday(&t1, NULL);
#endif

        if (densityA > INTERSECTION_SPARSE_OR_DNS_TH && densityB > INTERSECTION_SPARSE_OR_DNS_TH)
        {
            if (USE_DNS_THREAD)
            {
                int num_threads = WARP_SIZE * WARP_PER_BLOCK;
                int num_blocks = ceil((double)numblkC / (double)num_threads);
                tile_spgemm_step3_cuda_kernel_dns_thread<<<num_blocks, num_threads>>>(d_blkrowptrA, d_blkcolidxA, d_nnzb_A, d_blkcsr_Val_A, d_blkcsr_Col_A, d_blkcsr_Ptr_A, d_blkmaskA,
                                                                                      blkmA, blknA, numblkA, nnzA,
                                                                                      d_blkcolptrB, d_blkrowidxB, d_nnzb_B, d_blkcsr_Val_B, d_blkcsr_Col_B, d_blkcsr_Ptr_B, d_blkmaskB,
                                                                                      blkmB, blknB, numblkB, nnzB,
                                                                                      d_blk_intersec_bitmask_A, d_blk_intersec_bitmask_B, blk_intersec_bitmask_len,
                                                                                      d_blkrowidxC, d_blkcolidxC, d_blkcsr_Ptr_C, d_nnzb_C, d_blkmaskC,
                                                                                      d_blksmem_tny_cnt, d_blksmem_sml_cnt, d_blksmem_lrg_cnt, d_blksmem_dns_cnt, d_blksmem_ful_cnt,
                                                                                      d_blkid_smem_tny, d_blkid_smem_sml, d_blkid_smem_lrg, d_blkid_smem_dns, d_blkid_smem_ful,
                                                                                      d_spec_intersection_cnt, d_spec_intersection_posa, d_spec_intersection_posb,
                                                                                      numblkC);
            }
            else
            {
                num_threads = HALFWARP_SIZE * HALFWARP_PER_BLOCK;
                num_blocks = ceil((double)numblkC / (double)(HALFWARP_PER_BLOCK * TILE_PER_HALFWARP));
                tile_spgemm_step3_cuda_kernel_dns_halfwarp<<<num_blocks, num_threads>>>(d_blkrowptrA, d_blkcolidxA, d_nnzb_A, d_blkcsr_Val_A, d_blkcsr_Col_A, d_blkcsr_Ptr_A, d_blkmaskA,
                                                                                        blkmA, blknA, numblkA, nnzA,
                                                                                        d_blkcolptrB, d_blkrowidxB, d_nnzb_B, d_blkcsr_Val_B, d_blkcsr_Col_B, d_blkcsr_Ptr_B, d_blkmaskB,
                                                                                        blkmB, blknB, numblkB, nnzB,
                                                                                        d_blk_intersec_bitmask_A, d_blk_intersec_bitmask_B, blk_intersec_bitmask_len,
                                                                                        d_blkrowidxC, d_blkcolidxC, d_blkcsr_Ptr_C, d_nnzb_C, d_blkmaskC,
                                                                                        d_blksmem_tny_cnt, d_blksmem_sml_cnt, d_blksmem_lrg_cnt, d_blksmem_dns_cnt, d_blksmem_ful_cnt,
                                                                                        d_blkid_smem_tny, d_blkid_smem_sml, d_blkid_smem_lrg, d_blkid_smem_dns, d_blkid_smem_ful,
                                                                                        d_spec_intersection_cnt, d_spec_intersection_posa, d_spec_intersection_posb,
                                                                                        numblkC);
            }
        }
        else
        {


            if (USE_HALFWARP)
            {
                num_threads = HALFWARP_SIZE * HALFWARP_PER_BLOCK;
                num_blocks = ceil((double)numblkC / (double)(HALFWARP_PER_BLOCK * TILE_PER_HALFWARP));
                tile_spgemm_step3_cuda_kernel_2level_halfwarp<<<num_blocks, num_threads>>>(d_blkrowptrA, d_blkcolidxA, d_nnzb_A, d_blkcsr_Val_A, d_blkcsr_Col_A, d_blkcsr_Ptr_A, d_blkmaskA,
                                                                                           blkmA, blknA, numblkA, nnzA,
                                                                                           d_blkcolptrB, d_blkrowidxB, d_nnzb_B, d_blkcsr_Val_B, d_blkcsr_Col_B, d_blkcsr_Ptr_B, d_blkmaskB,
                                                                                           blkmB, blknB, numblkB, nnzB,
                                                                                           d_blk_intersec_bitmask_A, d_blk_intersec_bitmask_B, blk_intersec_bitmask_len,
                                                                                           d_blkrowidxC, d_blkcolidxC, d_blkcsr_Ptr_C, d_nnzb_C, d_blkmaskC,
                                                                                           d_blksmem_tny_cnt, d_blksmem_sml_cnt, d_blksmem_lrg_cnt, d_blksmem_dns_cnt, d_blksmem_ful_cnt,
                                                                                           d_blkid_smem_tny, d_blkid_smem_sml, d_blkid_smem_lrg, d_blkid_smem_dns, d_blkid_smem_ful,
                                                                                           d_spec_intersection_cnt, d_spec_intersection_posa, d_spec_intersection_posb,
                                                                                           numblkC);
            }
            else
            {
                num_threads = WARP_SIZE * WARP_PER_BLOCK;
                num_blocks = ceil((double)numblkC / (double)(WARP_PER_BLOCK * TILE_PER_WARP));
                tile_spgemm_step3_cuda_kernel_2level<<<num_blocks, num_threads>>>(d_blkrowptrA, d_blkcolidxA, d_nnzb_A, d_blkcsr_Val_A, d_blkcsr_Col_A, d_blkcsr_Ptr_A, d_blkmaskA,
                                                                                  blkmA, blknA, numblkA, nnzA,
                                                                                  d_blkcolptrB, d_blkrowidxB, d_nnzb_B, d_blkcsr_Val_B, d_blkcsr_Col_B, d_blkcsr_Ptr_B, d_blkmaskB,
                                                                                  blkmB, blknB, numblkB, nnzB,
                                                                                  d_blkrowidxC, d_blkcolidxC, d_blkcsr_Ptr_C, d_nnzb_C, d_blkmaskC,
                                                                                  d_blksmem_tny_cnt, d_blksmem_sml_cnt, d_blksmem_lrg_cnt, d_blksmem_dns_cnt, d_blksmem_ful_cnt,
                                                                                  d_blkid_smem_tny, d_blkid_smem_sml, d_blkid_smem_lrg, d_blkid_smem_dns, d_blkid_smem_ful,
                                                                                  numblkC);
            }
        }
            int *h_nnzb_C = (int *)malloc((numblkC + 1) * sizeof(int));
            memset(h_nnzb_C, 0, (numblkC + 1) * sizeof(int));
            cudaMemcpy(h_nnzb_C, d_nnzb_C, (numblkC + 1)* sizeof(int), cudaMemcpyDeviceToHost);

        exclusive_scan_device_cuda_thrust<int>(d_nnzb_C, numblkC + 1);
        nnzC = 0;
        cudaMemcpy(&nnzC, &d_nnzb_C[numblkC], sizeof(int), cudaMemcpyDeviceToHost);

#if TIMING
        cudaDeviceSynchronize();
        gettimeofday(&t2, NULL);
        *time_step2 = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
        if (ri == 0)
        {
            printf("\nstep2 --------Calculate the number of nonzeros of each tile of matrixC-----\n");
            printf("step2 ---------------------- Runtime is  %.2f ms-------------------------\n", *time_step2);
        }
#endif

#if TIMING
        gettimeofday(&t1, NULL);
#endif

        unsigned char *d_blkcsr_Col_C;
        cudaMalloc((void **)&d_blkcsr_Col_C, nnzC * sizeof(unsigned char));
        MAT_VAL_TYPE *d_blkcsr_Val_C;
        cudaMalloc((void **)&d_blkcsr_Val_C, nnzC * sizeof(MAT_VAL_TYPE));

        int blksmem_tny_cnt = 0;
        int blksmem_sml_cnt = 0;
        int blksmem_lrg_cnt = 0;
        int blksmem_dns_cnt = 0;
        int blksmem_ful_cnt = 0;

        cudaMemcpy(&blksmem_tny_cnt, d_blksmem_tny_cnt, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&blksmem_sml_cnt, d_blksmem_sml_cnt, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&blksmem_lrg_cnt, d_blksmem_lrg_cnt, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&blksmem_dns_cnt, d_blksmem_dns_cnt, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&blksmem_ful_cnt, d_blksmem_ful_cnt, sizeof(int), cudaMemcpyDeviceToHost);

#if TIMING
        gettimeofday(&t2, NULL);
        *time_malloc += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
#endif

#if TIMING
        gettimeofday(&t1, NULL);
#endif

        // tiny : 1 - 32
        if (blksmem_tny_cnt)
        {
            num_threads = HALFWARP_SIZE * HALFWARP_PER_BLOCK;
            num_blocks = ceil((double)blksmem_tny_cnt / (double)HALFWARP_PER_BLOCK);
            tile_spgemm_step4_cuda_kernel_smem_v3_halfwarp<SMEM_TNY_TH><<<num_blocks, num_threads, 0, streams[0]>>>(d_blkrowptrA, d_blkcolidxA, d_nnzb_A, d_blkcsr_Val_A, d_blkcsr_Col_A, d_blkcsr_Ptr_A,
                                                                                                                    blkmA, blknA, numblkA, nnzA,
                                                                                                                    d_blkcolptrB, d_blkrowidxB, d_nnzb_B, d_blkcsr_Val_B, d_blkcsr_Col_B, d_blkcsr_Ptr_B,
                                                                                                                    blkmB, blknB, numblkB, nnzB,
                                                                                                                    d_blkrowidxC, d_blkcolidxC, d_blkcsr_Ptr_C,
                                                                                                                    d_blkcsr_Col_C, d_blkcsr_Val_C,
                                                                                                                    d_nnzb_C, d_blkmaskC, blksmem_tny_cnt, d_blkid_smem_tny,
                                                                                                                    d_spec_intersection_cnt, d_spec_intersection_posa, d_spec_intersection_posb);
        }

        // small : 33 - 64
        if (blksmem_sml_cnt)
        {
            num_threads = WARP_SIZE * WARP_PER_BLOCK;
            num_blocks = ceil((double)blksmem_sml_cnt / (double)WARP_PER_BLOCK);
            tile_spgemm_step4_cuda_kernel_smem_v3<SMEM_SML_TH><<<num_blocks, num_threads, 0, streams[1]>>>(d_blkrowptrA, d_blkcolidxA, d_nnzb_A, d_blkcsr_Val_A, d_blkcsr_Col_A, d_blkcsr_Ptr_A,
                                                                                                           blkmA, blknA, numblkA, nnzA,
                                                                                                           d_blkcolptrB, d_blkrowidxB, d_nnzb_B, d_blkcsr_Val_B, d_blkcsr_Col_B, d_blkcsr_Ptr_B,
                                                                                                           blkmB, blknB, numblkB, nnzB,
                                                                                                           d_blkrowidxC, d_blkcolidxC, d_blkcsr_Ptr_C,
                                                                                                           d_blkcsr_Col_C, d_blkcsr_Val_C,
                                                                                                           d_nnzb_C, d_blkmaskC, blksmem_sml_cnt, d_blkid_smem_sml,
                                                                                                           d_spec_intersection_cnt, d_spec_intersection_posa, d_spec_intersection_posb);
        }

        // large : 65 - 128
        if (blksmem_lrg_cnt)
        {
            num_threads = WARP_SIZE * WARP_PER_BLOCK;
            num_blocks = ceil((double)blksmem_lrg_cnt / (double)WARP_PER_BLOCK);
            tile_spgemm_step4_cuda_kernel_smem_v3<SMEM_LRG_TH><<<num_blocks, num_threads, 0, streams[2]>>>(d_blkrowptrA, d_blkcolidxA, d_nnzb_A, d_blkcsr_Val_A, d_blkcsr_Col_A, d_blkcsr_Ptr_A,
                                                                                                           blkmA, blknA, numblkA, nnzA,
                                                                                                           d_blkcolptrB, d_blkrowidxB, d_nnzb_B, d_blkcsr_Val_B, d_blkcsr_Col_B, d_blkcsr_Ptr_B,
                                                                                                           blkmB, blknB, numblkB, nnzB,
                                                                                                           d_blkrowidxC, d_blkcolidxC, d_blkcsr_Ptr_C,
                                                                                                           d_blkcsr_Col_C, d_blkcsr_Val_C,
                                                                                                           d_nnzb_C, d_blkmaskC, blksmem_lrg_cnt, d_blkid_smem_lrg,
                                                                                                           d_spec_intersection_cnt, d_spec_intersection_posa, d_spec_intersection_posb);
        }

        // dns : 129 - dns
        if (blksmem_dns_cnt)
        {
            num_threads = HALFWARP_SIZE * HALFWARP_PER_BLOCK;
            num_blocks = ceil((double)blksmem_dns_cnt / (double)HALFWARP_PER_BLOCK);
            tile_spgemm_step4_cuda_kernel_dns_noatomic_halfwarp<<<num_blocks, num_threads, 0, streams[3]>>>(d_blkrowptrA, d_blkcolidxA, d_nnzb_A, d_blkcsr_Val_A, d_blkcsr_Col_A, d_blkcsr_Ptr_A,
                                                                                                            blkmA, blknA, numblkA, nnzA,
                                                                                                            d_blkcolptrB, d_blkrowidxB, d_nnzb_B, d_blkcsr_Val_B, d_blkcsr_Col_B, d_blkcsr_Ptr_B,
                                                                                                            blkmB, blknB, numblkB, nnzB,
                                                                                                            d_blkrowidxC, d_blkcolidxC, d_blkcsr_Ptr_C,
                                                                                                            d_blkcsr_Col_C, d_blkcsr_Val_C,
                                                                                                            d_nnzb_C, d_blkmaskC, blksmem_dns_cnt, d_blkid_smem_dns,
                                                                                                            d_spec_intersection_cnt, d_spec_intersection_posa, d_spec_intersection_posb);
        }

        // ful : 256
        if (blksmem_ful_cnt)
        {
            num_threads = HALFWARP_SIZE * HALFWARP_PER_BLOCK;
            num_blocks = ceil((double)blksmem_ful_cnt / (double)HALFWARP_PER_BLOCK);
            tile_spgemm_step4_cuda_kernel_ful_noatomic_halfwarp<<<num_blocks, num_threads, 0, streams[4]>>>(d_blkrowptrA, d_blkcolidxA, d_nnzb_A, d_blkcsr_Val_A, d_blkcsr_Col_A, d_blkcsr_Ptr_A,
                                                                                                            blkmA, blknA, numblkA, nnzA,
                                                                                                            d_blkcolptrB, d_blkrowidxB, d_nnzb_B, d_blkcsr_Val_B, d_blkcsr_Col_B, d_blkcsr_Ptr_B,
                                                                                                            blkmB, blknB, numblkB, nnzB,
                                                                                                            d_blkrowidxC, d_blkcolidxC, d_blkcsr_Ptr_C,
                                                                                                            d_blkcsr_Col_C, d_blkcsr_Val_C,
                                                                                                            d_nnzb_C, d_blkmaskC, blksmem_ful_cnt, d_blkid_smem_ful,
                                                                                                            d_spec_intersection_cnt, d_spec_intersection_posa, d_spec_intersection_posb);
        }

#if TIMING
        cudaDeviceSynchronize();
        gettimeofday(&t2, NULL);
        *time_step3 = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
        if (ri == 0)
        {
            printf("\nstep3 ---------Calculate the val&col of nonzeros of matrixC-------------\n");
            printf("step3 ---------------------- Runtime is  %.2f ms------------------------\n", *time_step3);
            printf("\n-----------------------Malloc uses %.2f ms-------------------------------\n", *time_malloc);
        }

#endif

        cudaDeviceSynchronize();
        gettimeofday(&tend, NULL);
        double time = (tend.tv_sec - tstart.tv_sec) * 1000.0 + (tend.tv_usec - tstart.tv_usec) / 1000.0;
        time_all[ri] = time;
        tile_spgemm_time += time;

#if CHECK_RESULT
        int *h_tile_nnz_C = (int *)malloc((numblkC + 1) * sizeof(int));
        int *h_tile_ptr_C = (int *)malloc((blkmA + 1) * sizeof(int));
        int *h_tile_columnidx_C = (int *)malloc(numblkC * sizeof(int));
        MAT_VAL_TYPE *h_tile_csr_Value_C = (MAT_VAL_TYPE *)malloc(nnzC * sizeof(MAT_VAL_TYPE));
        unsigned char *h_tile_csr_Col_C = (unsigned char *)malloc(nnzC * sizeof(unsigned char));
        unsigned char *h_tile_csr_Ptr_C = (unsigned char *)malloc(numblkC * BLOCK_SIZE * sizeof(unsigned char));

        cudaMemcpy(h_tile_nnz_C, d_nnzb_C, (numblkC + 1) * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_tile_ptr_C, d_blkrowptrC, (blkmA + 1) * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_tile_columnidx_C, d_blkcolidxC, numblkC * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_tile_csr_Value_C, d_blkcsr_Val_C, nnzC * sizeof(MAT_VAL_TYPE), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_tile_csr_Col_C, d_blkcsr_Col_C, nnzC * sizeof(unsigned char), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_tile_csr_Ptr_C, d_blkcsr_Ptr_C, numblkC * BLOCK_SIZE * sizeof(unsigned char), cudaMemcpyDeviceToHost);

        matrixC->tile_ptr = h_tile_ptr_C;
        matrixC->tile_columnidx = h_tile_columnidx_C;
        matrixC->tile_nnz = h_tile_nnz_C;
        matrixC->numtile = numblkC;
        matrixC->nnz = nnzC;
        matrixC->m = matrixA->m;
        matrixC->n = matrixB->n;
        matrixC->tilem = matrixA->tilem;
        matrixC->tilen = matrixB->tilen;
        matrixC->tile_csr_Value = h_tile_csr_Value_C;
        matrixC->tile_csr_Col = h_tile_csr_Col_C;
        matrixC->tile_csr_Ptr = h_tile_csr_Ptr_C;

#endif

        cudaFree(d_blkrowptrC);
        cudaFree(d_blkrowidxC);
        cudaFree(d_blkcolidxC);
        cudaFree(d_blkmaskC);
        cudaFree(d_nnzb_C);
        cudaFree(d_blkcsr_Ptr_C);
        cudaFree(d_blkcsr_Col_C);
        cudaFree(d_blkcsr_Val_C);
        cudaFree(d_blkid_smem_tny);
        cudaFree(d_blkid_smem_sml);
        cudaFree(d_blkid_smem_lrg);
        cudaFree(d_blkid_smem_dns);
        cudaFree(d_blkid_smem_ful);
        if (USE_GMEM_SPECULATIVE_INTERSECTION)
        {
            cudaFree(d_spec_intersection_cnt);
            cudaFree(d_spec_intersection_posa);
            cudaFree(d_spec_intersection_posb);
        }
    }

    double time_min = time_all[0];
    for (int ri = 1; ri < REPEAT_NUM; ri++)
        time_min = time_min > time_all[ri] ? time_all[ri] : time_min;

    *nnzC_computed = nnzC;
    *compression_rate = (double)nnzCub / (double)(*nnzC_computed);
    tile_spgemm_time = time_min;
    *time_tile = tile_spgemm_time;
    *gflops_tile = 2.0 * (double)nnzCub / (tile_spgemm_time * 1e6);

    printf("Non-empty tiles of C = %i\n", numblkC);
    printf("nnzC = %i\n", nnzC);
    printf("CUDA  TileSpGEMM runtime is %4.2f ms, gflops = %4.2f\n", tile_spgemm_time, *gflops_tile);

    cudaFree(d_blksmem_tny_cnt);
    cudaFree(d_blksmem_sml_cnt);
    cudaFree(d_blksmem_lrg_cnt);
    cudaFree(d_blksmem_dns_cnt);
    cudaFree(d_blksmem_ful_cnt);

    cudaFree(d_blkrowptrA);
    cudaFree(d_blkcolidxA);
    cudaFree(d_nnzb_A);
    cudaFree(d_blkcsr_Val_A);
    cudaFree(d_blkcsr_Col_A);
    cudaFree(d_blkcsr_Ptr_A);
    cudaFree(d_blkcolptrB);
    cudaFree(d_blkrowidxB);
    cudaFree(d_blkrowptrB);
    cudaFree(d_blkcolidxB);
    cudaFree(d_nnzb_B);
    cudaFree(d_blkcsr_Val_B);
    cudaFree(d_blkcsr_Col_B);
    cudaFree(d_blkcsr_Ptr_B);
    cudaFree(d_blkmaskB);
    cudaFree(d_blkmaskA);
    cudaFree(d_blk_intersec_bitmask_A);
    cudaFree(d_blk_intersec_bitmask_B);

    for (int i = 0; i < nstreams; i++)
    {
        cudaStreamDestroy(streams[i]);
    }
    free(streams);
}
