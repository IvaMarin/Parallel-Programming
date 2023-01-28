# CC = nvcc
# CFLAGS = --std=c++11 -Werror cross-execution-space-call -lm
# SOURSES = lab1.cu
# BIN = lab1

# all:
# 	$(CC) $(CFLAGS) -o $(BIN) $(SOURSES)

# run:
# 	./$(BIN) < ./tests/test_335.t
# 	./$(BIN) < ./tests/test_3355.t
# 	./$(BIN) < ./tests/test_33554.t
# 	./$(BIN) < ./tests/test_335544.t
# 	./$(BIN) < ./tests/test_3355443.t
# 	./$(BIN) < ./tests/test_33554431.t

# clean:
# 	rm -f *.o $(BIN)

CXX = g++
CXXFLAGS = -std=c++11 -O2 -Wextra -Wall -Wno-sign-compare -Wno-unused-result
SOURSES = lab1.cpp
BIN = lab1

all: BIN

BIN:
	$(CXX) $(CXXFLAGS) -o $(BIN) $(SOURSES)

debug:
	$(CXX) $(CXXFLAGS) -o $(BIN) $(SOURSES) -g3

run:
	./$(BIN) < ./tests/test_335.t
	./$(BIN) < ./tests/test_3355.t
	./$(BIN) < ./tests/test_33554.t
	./$(BIN) < ./tests/test_335544.t
	./$(BIN) < ./tests/test_3355443.t
	./$(BIN) < ./tests/test_33554431.t

clean:
	rm -f *.o $(BIN)