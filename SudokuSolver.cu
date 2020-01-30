#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>  
#include <string.h>
#include <time.h>
#include <math.h>
#include <inttypes.h>

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

#pragma region Delclaration
    cudaEvent_t event1,event2;
    
    uint64_t test[N];
    uint64_t check[N];
    uint64_t fun[N];

    uint64_t *test64_1;
    uint64_t *test64_2;
    int *board_index;

    const int sk = pow(2,27);
    int host_count;
    int threadsPerBlock = 256;
    int maxBlocks;
    int zeros;
    params_t params;
    float dt_ms;

    gpuErrchk(cudaMalloc(&test64_1,sk*sizeof(uint64_t)));
    gpuErrchk(cudaMalloc(&test64_2,sk*sizeof(uint64_t)));
    gpuErrchk(cudaMalloc(&board_index,sizeof(int)));

    memset(test,0,N*sizeof(uint64_t));
    memset(check,0,N*sizeof(uint64_t));
    memset(fun,0,N*sizeof(uint64_t));
    gpuErrchk(cudaMemset(board_index,0,sizeof(int)));
    gpuErrchk(cudaMemset(test64_1,0,sk*sizeof(uint64_t)));
    gpuErrchk(cudaMemset(test64_2,0,sk*sizeof(uint64_t)));



    gpuErrchk(cudaEventCreate(&event1));
    gpuErrchk(cudaEventCreate(&event2));


    setup_board(test,test9);
    
    printBoard(test);

    zeros=count_zeros(test);
    gpuErrchk(cudaMemcpy(test64_1,test,N*sizeof(uint64_t),cudaMemcpyHostToDevice));

    gpuErrchk(cudaEventRecord(event1));

    params=find_epmty_index(test,0,0);

    printf("Empty index %i : %i\n",params.row, params.col);

    cudaBFSSudoku<<<1,N>>>(test64_1, test64_2, 1, board_index,params.row,params.col);

    gpuErrchk(cudaMemcpy(&fun, test64_2, N*sizeof(uint64_t), cudaMemcpyDeviceToHost))
    params=find_epmty_index(fun,params.row,params.col);


    
    for (int i = 0; i<zeros; i++) {

        gpuErrchk(cudaMemcpy(&host_count, board_index, sizeof(int), cudaMemcpyDeviceToHost));

        printf("total boards after an iteration %d: %d\n", i, host_count);

        gpuErrchk(cudaMemset(board_index, 0, sizeof(int)));

        maxBlocks=(N*host_count+threadsPerBlock-1)/threadsPerBlock;

        if (i % 2 == 0) {
            cudaBFSSudoku<<<maxBlocks,threadsPerBlock>>>(test64_2, test64_1, host_count, board_index,params.row,params.col);
            gpuErrchk(cudaMemcpy(&fun, test64_1, N*sizeof(uint64_t), cudaMemcpyDeviceToHost));
            params=find_epmty_index(fun,params.row,params.col);
        }
        else {
            cudaBFSSudoku<<<maxBlocks,threadsPerBlock>>>(test64_1, test64_2, host_count, board_index,params.row,params.col);
            gpuErrchk(cudaMemcpy(&fun, test64_2, N*sizeof(uint64_t), cudaMemcpyDeviceToHost));
            params=find_epmty_index(fun,params.row,params.col);
        }
    }

    gpuErrchk(cudaMemcpy(&host_count, board_index, sizeof(int), cudaMemcpyDeviceToHost));
    
    if(zeros % 2 == 0){
        gpuErrchk(cudaMemcpy(&check, test64_1, N*sizeof(uint64_t), cudaMemcpyDeviceToHost));
    }
    else{
        gpuErrchk(cudaMemcpy(&check, test64_2, N*sizeof(uint64_t), cudaMemcpyDeviceToHost));
    }
    
    printf("new number of boards retrieved is %d\n", host_count);
    printBoard(check);


    gpuErrchk(cudaEventRecord(event2));
    gpuErrchk(cudaEventSynchronize(event2));
    gpuErrchk(cudaDeviceSynchronize());

    cudaEventElapsedTime(&dt_ms, event1,event2);
    printf("Time : %f",dt_ms);

    gpuErrchk(cudaFree(test64_1));
    gpuErrchk(cudaFree(test64_2));
    gpuErrchk(cudaFree(board_index));

    return 0; 
}

