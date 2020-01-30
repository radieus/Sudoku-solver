#include <cmath>
#include <cstdio>
#include <stdio.h>

#define N 9
#define n 3



typedef struct params{
    int row;
    int col;
}params_t;


__device__ __host__ void setbit(int val, int *data, int nshift) { 
	//set bit to data using val as sourse and nshift place where to set
	*data = *data | (val << nshift);
}

__device__ __host__ int getbit(int input, int nshift) {
	//return value in a certain position 0 or 1
	return (input >> nshift) & 1;
}

__device__ __host__ void copy_bits(int src, int *dst, int src_offset, int dst_offset, int len) {
	/*
	src - source of bits
	dst - destanation of bits
	src_offset - starting position from which take bit
	dst_offset - starting position where to set the bit
	len - how many bits
	*/
	for (int i = 0; i < len; i++) {
		setbit(getbit(src, src_offset + i), dst, i + dst_offset);
    }
    

}

__device__ __host__ void setup_board(int *src, int *board){
    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++){
            copy_bits(board[i*N+j], &src[i],0,j*4,4);
        }
    }
}

__device__ __host__ void print_sudoku_from_b64(int *val) {

    for (int i = 0; i < N; i++) {
        if (i % n == 0) {
            printf("-----------------------\n");
        }

        for (int j = 0; j < N; j++) {
            if (j % n == 0) {
            printf("| ");
            }
            int tmp=0;
            copy_bits(val[i],&tmp,j*4,0,4);
            printf("%li ", tmp);
        }

        printf("|\n");
    }
    printf("-----------------------\n");
}

__device__ __host__ params_t find_epmty_index(int *val, int row, int col){
   
    for(int i=row;i<N;i++){
        for(int j=0;j<N;j++){
            int tmp=0;
            copy_bits(val[i],&tmp,j*4,0,4);
            if(tmp==0){
                params_t temp={i,j};
                return temp;
            }
        }
    }
    params_t temp={0,0};
    return temp;
}

__device__ __host__ bool check_row(int *val, int row, int value){
    for(int i=0;i<N;i++){
        int tmp=0;
        copy_bits(val[row],&tmp,i*4,0,4);
        if(tmp==value)
            return false;
        }
    return true;
}

__device__ __host__ bool check_col(int *val, int column, int value){
    for(int i=0;i<N;i++){
        int tmp=0;
        copy_bits(val[i],&tmp,column*4,0,4);
        if(tmp==value)
            return false;
        }
    return true;
}

__device__ __host__ bool check_box(int *val, int row,int column, int value){
    for(int i=row*n;i<row*n+n;i++){
        for(int j=column*n; j<column*n+n;j++){
            int tmp=0;
            copy_bits(val[i],&tmp,j*4,0,4);
            if(tmp==value)
                return false;
            }
    }
    return true;
}

__device__ __host__ int count_zeros(int *val){
    int count=0;
    for(int i=0;i<N;i++){
        for(int j=0; j<N;j++){
            int tmp=0;
            copy_bits(val[i],&tmp,j*4,0,4);
            if(tmp==0)
                count++;
            }
    }
    return count;
}

