// c Zeros( M );
// A Ones ( M, N );
// b Ones ( N );


// thread m is responsible to calculate one inner product of row a and b.
// thread m, thread m+1 address contiguous column elements of A -> coalesced memory access
template<class Type>
__kernel void matvec1_cmajor__old(__global Type *c, __global Type *A, __global Type *b)
{
	int m = get_global_id(0); 

	if(m >= M) return;
	
	__global Type* a = A + m;

	Type s = 0;
	for(size_t n = 0; n < N; ++n)
		s += a[n*M] * b[n];
		
	c[m] = s;
}

// thread m is responsible to calculate one inner product of row a and b.
// thread m, thread m+1 address not contiguous column elements of A
template<class Type>
__kernel void matvec1_rmajor__old(__global Type *c, __global Type *A, __global Type *b)
{
	int m = get_global_id(0);

	if(m >= M) return;
	
	__global Type* a = A + m*N;

	Type s = 0;
	for(size_t n = 0; n < N; ++n)
		s += a[n] * b[n];
		
	c[m] = s;
}

template<class Type>

__kernel void matmat_rmajor( __global Type *dst, __global Type *src1 , __global Type *src2 , 
	int N_scr1, int K_dst, int M_scr2)
{
	int col = get_global_id(0); //x
	int row = get_global_id(1); //y
	int Kvar = K_dst;

	int sol =0;	
	
	int index = col + row*M_scr2;

	for(int x= 0; x < Kvar; x++){
		sol += src1[row  * K_dst + x] * src2[x * M_scr2 + col];
	}
	dst[index]= sol;

}

template<class Type>

__kernel void matmat_cmajor( __global Type *dst, __global Type *src1 , __global Type *src2, 
	int N_scr1, int K_dst, int M_scr2)
{
	int col = get_global_id(0); //x
	int row = get_global_id(1); //y
	int Kvar = K_dst;

	int sol =0;	
	
	int index = row + col*N_scr1;  

	for(int x= 0; x < Kvar; x++){
		sol += src1[row + N_scr1 * x]* src2[col * K_dst + x];
	}
	dst[index]= sol;

}



template<class Type>

#define BLOCK_SIZE 8
__kernel void matmat_cmajor_f4( __global Type *dst, __global Type *src1 , __global Type *src2, 
	int N_scr1, int K_dst, int M_scr2)
{

	//Identification of this workgroup
    int i = get_group_id(0);
    int j = get_group_id(1);
    //Identification of work-item
    int idX = get_local_id(0);
    int idY = get_local_id(1);
    //matrixes dimensions
    int N_dim = N_scr1;		//N*K x K*M
    int M_dim = M_scr2;
    int Kvar = K_dst;

	int col = get_global_id(0); //x
	int row = get_global_id(1); //y
	//int BLOCK_SIZE = 4;
	int numSubmatrices = Kvar / BLOCK_SIZE;

	float4 sol = (float4)(0,0,0,0);
	__local float A[BLOCK_SIZE][BLOCK_SIZE];
    __local float B[BLOCK_SIZE][BLOCK_SIZE];

    for (int k=0; k<numSubmatrices; k++)
    {
        //Copy submatrixes to local memory. Each worker copies one element
        //Notice that A[i,k] accesses elements starting from Scr[BLOCK_SIZE*i, BLOCK_SIZE*j]

        A[idX][idY] = src1[BLOCK_SIZE*i + idX + N_dim*(BLOCK_SIZE*k+idY)];
        B[idX][idY] = src2[BLOCK_SIZE*k + idX + Kvar*(BLOCK_SIZE*j+idY)];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int k2 = 0; k2 < BLOCK_SIZE; k2+=4)
        {
            float4 temp1=(float4)(A[idX][k2],A[idX][k2+1],A[idX][k2+2],A[idX][k2+3]);
            float4 temp2=(float4)(B[k2][idY],B[k2+1][idY],B[k2+2][idY],B[k2+3][idY]);
            sol += temp1 * temp2;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    dst[BLOCK_SIZE*i + idX + N_dim*(BLOCK_SIZE*j+idY)] = sol.x + sol.y + sol.z + sol.w;


}