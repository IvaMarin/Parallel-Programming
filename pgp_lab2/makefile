CC = nvcc
CFLAGS = --std=c++11 -Werror cross-execution-space-call -lm
SOURSES = lab2.cu
BIN = lab2

all:
	$(CC) $(CFLAGS) -o $(BIN) $(SOURSES)

run:
	./$(BIN) < ./tests/test_4900.t  
	./$(BIN) < ./tests/test_49729.t 
	./$(BIN) < ./tests/test_499849.t 
	./$(BIN) < ./tests/test_4999696.t 
	./$(BIN) < ./tests/test_49999041.t

clean:
	rm -f *.o $(BIN)

# CXX = g++
# CXXFLAGS = -std=c++11 -O2 -Wextra -Wall -Wno-sign-compare -Wno-unused-result
# SOURSES = lab2.cpp
# BIN = lab2

# all: BIN

# BIN:
# 	$(CXX) $(CXXFLAGS) -o $(BIN) $(SOURSES)

# debug:
# 	$(CXX) $(CXXFLAGS) -o $(BIN) $(SOURSES) -g3

# run:
# 	./$(BIN) < ./tests/test_4900.t  
# 	./$(BIN) < ./tests/test_49729.t 
# 	./$(BIN) < ./tests/test_499849.t 
# 	./$(BIN) < ./tests/test_4999696.t 
# 	./$(BIN) < ./tests/test_49999041.t

# clean:
# 	rm -f *.o $(BIN)