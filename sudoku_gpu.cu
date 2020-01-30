#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <iostream>
#include <random>
#include <string>
#include <cmath>

#include <cuda_runtime.h>

#define N 9
#define n 3


__device__ void clearBitmap(bool *map, int size) {
    for (int i = 0; i < size; i++) {
        map[i] = false;
    }
}

__device__ __host__ bool validBoard(int *board) {
    bool seen[N];
    clearBitmap(seen, N);

    // check if rows are valid
    for (int i = 0; i < N; i++) {
        clearBitmap(seen, N);

        for (int j = 0; j < N; j++) {
            int val = board[i * N + j];

            if (val != 0) {
                // if was in row - not valid, else - mark VALUE as seen
                if (seen[val - 1]) {
                    return false;
                } else {
                    seen[val - 1] = true;
                }
            }
        }
    }

    // check if columns are valid
    for (int j = 0; j < N; j++) {
        clearBitmap(seen, N);

        for (int i = 0; i < N; i++) {
            int val = board[i * N + j];

            if (val != 0) {
                if (seen[val - 1]) {
                    return false;
                } else {
                    seen[val - 1] = true;
                }
            }
        }
    }

    // finally check if the sub-boards are valid
    for (int ridx = 0; ridx < n; ridx++) {
        for (int cidx = 0; cidx < n; cidx++) {
            clearBitmap(seen, N);

            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    int val = board[(ridx * n + i) * N + (cidx * n + j)];
                    if (val != 0) {
                        if (seen[val - 1]) {
                            return false;
                        } else {
                            seen[val-1] = true;
                        }
                    }
                }
            }
        }
    }

    return true;
}



__device__ bool validBoard(int *board, int r, int c) {

    // if r is less than 0, then just default case
    if (r < 0) {
        return validBoard(board);
    }

    bool seen[N];
    clearBitmap(seen, N);

    // check if row is valid
    for (int i = 0; i < N; i++) {
        int val = board[r * N + i];

        if (val != 0) {
            if (seen[val - 1]) {
                return false;
            } else {
                seen[val - 1] = true;
            }
        }
    }

    // check if column is valid
    clearBitmap(seen, N);
    for (int j = 0; j < N; j++) {
        int val = board[j * N + c];

        if (val != 0) {
            if (seen[val - 1]) {
                return false;
            } else {
                seen[val - 1] = true;
            }
        }
    }

    // finally check if the sub-board is valid
    int ridx = r / n;
    int cidx = c / n;

    clearBitmap(seen, N);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int val = board[(ridx * n + i) * N + (cidx * n + j)];

            if (val != 0) {
                if (seen[val - 1]) {
                    return false;
                } else {
                    seen[val - 1] = true;
                }
            }
        }
    }

    return true;
}

__device__ bool doneBoard(int *board) {
    for (int i = 0; i < N * N; i++) {
        if (board[i] == 0) {
            return false;
        }
    }

    return true;
}

__device__ bool findEmptySpot(int *board, int *row, int *col) {
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

__device__ bool solveHelper(int *board) {
    int row;
    int col;
    if (!findEmptySpot(board, &row, &col)) {
        return true;
    }

    for (int k = 1; k <= N; k++) {
        board[row * N + col] = k;
        if (validBoard(board, row, col) && solveHelper(board)) {
            return true;
        }
        board[row * N + col] = 0;
    }

    return false;
}

__globlal__ bool solve(int *board) {

    // initial board is invalid
    if (!validBoard(board, -1, -1)) {

        printf("solve: invalid board\n");
        return false;
    }

    // board is already solved
    if (doneBoard(board)) {

        printf("solve: done board\n");
        return true;
    }

    // otherwise, try to solve the board
    if (solveHelper(board)) {

        // solved
        printf("solve: solved board\n");
        return true;
    } else {

        // unsolvable
        printf("solve: unsolvable\n");
        return false;
    }
}

__device__ __host__ void printBoard(int *board) {
    for (int i = 0; i < N; i++) {
        if (i % n == 0) {
            printf("-----------------------\n");
        }

    for (int j = 0; j < N; j++) {
        if (j % n == 0) {
            printf("| ");
        }
        printf("%d ", board[i * N + j]);
    }

    printf("|\n");

    }

    printf("-----------------------\n");
}

__device__ __host__ void load(char *FileName, int *board) 
{
    FILE * a_file = fopen(FileName, "r");

    if (a_file == NULL) {
      printf("File load fail!\n"); return;
    }

    char temp;

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (!fscanf(a_file, "%c\n", &temp)) {
                printf("File loading error!\n");
                return;
            }

            if (temp >= '1' && temp <= '9') {
                board[i * N + j] = (int) (temp - '0');
            } else {
                board[i * N + j] = 0;
            }
        }
    }
}

__global__ void sudokuBacktrack(int *boards, const int numBoards, int *emptySpaces, int *numEmptySpaces, int *finished, int *solved) 
{

    int index = blockDim.x * blockIdx.x + threadIdx.x;

    int *currentBoard;
    int *currentEmptySpaces;
    int currentNumEmptySpaces;

    while ((*finished == 0) && (index < numBoards)) {
        int emptyIndex = 0;

        currentBoard = boards + index * 81;
        currentEmptySpaces = emptySpaces + index * 81;
        currentNumEmptySpaces = numEmptySpaces[index];

        while ((emptyIndex >= 0) && (emptyIndex < currentNumEmptySpaces)) {
            currentBoard[currentEmptySpaces[emptyIndex]]++;
            
            if (!validBoard(currentBoard, currentEmptySpaces[emptyIndex])) {

                // if the board is invalid and we tried all numbers here already, backtrack
                // otherwise continue (it will just try the next number in the next iteration)
                if (currentBoard[currentEmptySpaces[emptyIndex]] >= 9) {
                    currentBoard[currentEmptySpaces[emptyIndex]] = 0;
                    emptyIndex--;
                }
            }
            // if valid board, move forward in algorithm
            else {
                emptyIndex++;
            }

        }

        if (emptyIndex == currentNumEmptySpaces) {
            // solved board found
            *finished = 1;

            // copy board to output
            for (int i = 0; i < N * N; i++) {
                solved[i] = currentBoard[i];
            }
        }

        index += gridDim.x * blockDim.x;
    }
}

int main() 
{
    int *board = new int[N * N];
    load("puzzle.txt", board);

    if (solve(board)) {
        std::cout << "solved" << std::endl;
        printBoard(board);
    }

    return 0;
}