__device__ int find_kth3(uintT* a,
                          int aCount,
                          uintT* b,
                          int bCount,
                          int diag)
{
    int begin = max(0, diag - bCount);
    int end = min(diag, aCount);
 
    while(begin < end) {
        int mid = (begin + end)>> 1;
        uintT aKey = a[mid];
        uintT bKey = b[diag - 1 - mid];
        bool pred = aKey <= bKey;
        if(pred) begin = mid + 1;
        else end = mid;
    }
    return begin;
}
