#include <cmath>
#include <cstdio>
#include <stdio.h>
#include <cuda_runtime.h>

#define N 9
#define n 3

typedef struct params{
    int row;
    int col;
}params_t;

__global__ void cudaBFSSudoku(uint64_t *old_boards,
        uint64_t *new_boards,
        int total_boards,
        int *board_index,
        int empty_row,
        int empty_col) {
    
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    unsigned int index=tid/N;

    int attempt = tid - N*index + 1;

    while (index < total_boards) {

        int next_board_index=0;           
        bool works = true;

        int box_row = empty_row/n;
        int box_col = empty_col/n;

        if (!check_row(old_boards+index*N,empty_row,attempt)) 
            works = false;
        else if (!check_col(old_boards+index*N,empty_col,attempt)) 
            works = false;
        else if (!check_box(old_boards+index*N,box_row,box_col,attempt))
            works = false;

        if (works) {
            next_board_index = atomicAdd(board_index, 1);
            for (int i = 0; i < N; i++) {
                new_boards[next_board_index*N+i]=old_boards[index*N+i];
            }
            copy_bits(attempt, &(new_boards+next_board_index*N)[empty_row],0,empty_col*4,4);
            }

        break; 
    }  
}

__device__ __host__ bool findEmptySpot(uint64_t *board, int *row, int *col) {
    for (int r = 0; r < N; r++) {
        for (int c = 0; c < N; c++) {
            if (board[r * N + c] == 0) {
                *row = r;
                *col = c;
                return true;
            }
        }
    }

    return false;
}

__device__ __host__ void print_sudoku(uint64_t *val) {

    for (int i = 0; i < N; i++) {
        if (i % n == 0) {
            printf("------------------------\n");
        }

        for (int j = 0; j < N; j++) {
            if (j % n == 0) {
            printf("| ");
            }
            uint64_t tmp=0;
            copy_bits(val[i],&tmp,j*4,0,4);
            printf("%li ", tmp);
        }

        printf("|\n");
    }
    printf("------------------------\n");
}


__device__ __host__ void setbit(uint64_t val, uint64_t *data, int nshift) { 
	//set bit to data using val as sourse and nshift place where to set
	*data = *data | (val << nshift);
}

__device__ __host__ int getbit(uint64_t input, int nshift) {
	//return value in a certain position 0 or 1
	return (input >> nshift) & 1;
}

__device__ __host__ void copy_bits(uint64_t src, uint64_t *dst, int src_offset, int dst_offset, int len) {
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

__device__ __host__ void setup_board(uint64_t *src, int *board){
    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++){
            copy_bits(board[i*N+j], &src[i],0,j*4,4);
        }
    }
}


__device__ __host__ params_t find_epmty_index(uint64_t *val, int row, int col){
   
    for(int i=row;i<N;i++){
        for(int j=0;j<N;j++){
            uint64_t tmp=0;
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

__device__ __host__ bool check_row(uint64_t *val, int row, int value){
    for(int i=0;i<N;i++){
        uint64_t tmp=0;
        copy_bits(val[row],&tmp,i*4,0,4);
        if(tmp==value)
            return false;
        }
    return true;
}

__device__ __host__ bool check_col(uint64_t *val, int column, int value){
    for(int i=0;i<N;i++){
        uint64_t tmp=0;
        copy_bits(val[i],&tmp,column*4,0,4);
        if(tmp==value)
            return false;
        }
    return true;
}

__device__ __host__ bool check_box(uint64_t *val, int row,int column, int value){
    for(int i=row*n;i<row*n+n;i++){
        for(int j=column*n; j<column*n+n;j++){
            uint64_t tmp=0;
            copy_bits(val[i],&tmp,j*4,0,4);
            if(tmp==value)
                return false;
            }
    }
    return true;
}

__device__ __host__ int count_zeros(uint64_t *val){
    int count=0;
    for(int i=0;i<N;i++){
        for(int j=0; j<N;j++){
            uint64_t tmp=0;
            copy_bits(val[i],&tmp,j*4,0,4);
            if(tmp==0)
                count++;
            }
    }
    return count;
}

