# gnuplot
# set terminal png
# set output 'mpi-1.png'
# plot 'out\mpi3-1.out' matrix with image notitle

CC = mpic++
CFLAGS = --std=c++11 -fopenmp -pedantic -Wall -Werror -Wno-sign-compare -Wno-long-long -lm
SOURSES = lab1.cpp
BIN = lab1

all:
	$(CC) $(CFLAGS) -o $(BIN) $(SOURSES)

run:
	mpirun -np 1 ./$(BIN) < tests/test_11.t
	mpirun -np 2 ./$(BIN) < tests/test_12.t
	mpirun -np 3 ./$(BIN) < tests/test_13.t
	mpirun -np 4 ./$(BIN) < tests/test_14.t
	mpirun -np 5 ./$(BIN) < tests/test_15.t
	mpirun -np 6 ./$(BIN) < tests/test_16.t

clean:
	rm -f *.o $(BIN)