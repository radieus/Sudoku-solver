#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <stdbool.h>  
#include <string.h>
#include <time.h>
#include <math.h>
#include <inttypes.h>
#include <chrono>

#include "CudaSudoku.cu"
#include "samples.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

int main(int argc, char* argv[]) {
    
    uint64_t test[N];
    memset(test,0,N*sizeof(uint64_t));

    uint64_t *new_boards;
    uint64_t *old_boards;
    int *board_index;
    const int sk = pow(2,27);

    gpuErrchk(cudaMallocManaged(&new_boards,sk*sizeof(uint64_t)));
    gpuErrchk(cudaMallocManaged(&old_boards,sk*sizeof(uint64_t)));
    gpuErrchk(cudaMallocManaged(&board_index,sizeof(int)));

    int host_count;
    int maxBlocks;
    int zeros;
    params_t params;

    setup_board(test,sample_board);
    print_sudoku_from_b64(test);

    zeros=count_zeros(test);

    gpuErrchk(cudaMemcpy(new_boards,test,N*sizeof(uint64_t),cudaMemcpyHostToDevice));

    auto start = std::chrono::high_resolution_clock::now(); 
    params=find_epmty_index(test,0,0);

    printf("Empty index %i : %i\n", params.row, params.col);

    cudaBFSSudoku<<<1,N>>>(new_boards, old_boards, 1, board_index, params.row, params.col);

    params=find_epmty_index(old_boards, params.row, params.col);
    
    for (int i = 0; i<zeros; i++) {

        //gpuErrchk(cudaMemcpy(&host_count, board_index, sizeof(int), cudaMemcpyDeviceToHost));
        printf("total boards after an iteration %d: %d\n", i, board_index);

        board_index = 0;

        maxBlocks=(N*host_count+256-1)/256;

        if (i % 2 == 0) {
            cudaBFSSudoku<<<maxBlocks,256>>>(old_boards, new_boards, host_count, board_index, params.row, params.col);
            params=find_epmty_index(new_boards, params.row, params.col);
        }
        else {
            cudaBFSSudoku<<<maxBlocks,256>>>(new_boards, old_boards, host_count, board_index, params.row, params.col);
            params=find_epmty_index(old_boards, params.row, params.col);
        }
    }

    //gpuErrchk(cudaMemcpy(&host_count, board_index, sizeof(int), cudaMemcpyDeviceToHost));
    
    printf("new number of boards retrieved is %d\n", host_count);
    if(zeros % 2 == 0){ // if odd number of iterations run, then send it old boards not new boards;
        print_sudoku(new_boards);
    }
    else{
        print_sudoku(old_boards);
    }

    gpuErrchk(cudaDeviceSynchronize());
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start); 
    std::cout << duration.count() << std::endl; 
    gpuErrchk(cudaFree(new_boards));
    gpuErrchk(cudaFree(old_boards));
    gpuErrchk(cudaFree(board_index));

    return 0; 
}

