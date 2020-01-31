#include <cmath>
#include <cstdio>

#include <cuda_runtime.h>
#include "BitOperations.cu"

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

