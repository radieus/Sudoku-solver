all: cuda run 

cuda: SudokuSolver.cu
	nvcc SudokuSolver.cu -o cuda

run: 
	./cuda

clean:
	rm cuda