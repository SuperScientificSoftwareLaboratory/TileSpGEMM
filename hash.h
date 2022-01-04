#include"common.h"
#include <stdio.h>
#include <stdlib.h>


void block_mul( const int *flag, const int *mA, SMatrix *submatrixA,SMatrix *submatrixB,SMatrix *submatrixC,
                int blockCid,int *nnzAnum,int *nnzBnum)
{
    int *num;
    num=(int*)malloc(((*mA)*SubNum)*sizeof(int));
	memset(num,0,((*mA)*SubNum)*sizeof(int));
	for (int colid=0;colid<SubNum;colid++)
	{
		int A=(blockCid/SubNum)*SubNum+colid;
		int B=blockCid%SubNum+colid*SubNum;
		//printf("A=%d,nnz[A]=%d\n",A,nnzAnum[A]);
		//printf("B=%d,nnz[B]=%d\n",B,nnzAnum[B]);
		int j=0;
		if (nnzAnum[A]!=0&&nnzBnum[B]!=0){
			for (int i=0;i<(*mA);i++){
				while(j<submatrixA[A].rowpointer[i+1])
				{	
					num[colid*(*mA)+i]+=submatrixB[B].rowpointer[submatrixA[A].columnindex[j]+1]-submatrixB[B].rowpointer[submatrixA[A].columnindex[j]];
					j++;
				}
			//	sum+=num[q*row+i];
			}
		}
	}
    int *rowCub; //calculate tasks for each row in subC
	rowCub=(int *)malloc((*mA)*sizeof(int));
	memset(rowCub,0,(*mA));
	for (int i=0;i<(*mA);i++)
	{
		for (int k=0;k<SubNum;k++)
			rowCub[i]+=num[k*(*mA)+i];
	}

	
	for (int iid=0;iid<(*mA);iid++)
	{
		hashsize_full_reg=rowCub[iid];
		int *tmpIdx2D0 = (int *)malloc(hashsize_full_reg* sizeof(int));  //index 
		
	}
                    

}