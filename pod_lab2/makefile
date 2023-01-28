CC = nvcc
CFLAGS = --std=c++11 -Werror cross-execution-space-call -lm
SOURSES = lab4.cu
BIN = lab4

all:
	$(CC) $(CFLAGS) -o $(BIN) $(SOURSES)

nvprof:
	nvprof --log-file nvprof_result.txt -e divergent_branch,global_store_transaction,l1_shared_bank_conflict,l1_local_load_hit -m sm_efficiency ./$(BIN) < ./tests/test_1000000.t 

run:
	./$(BIN) < ./tests/test_100.t 
	./$(BIN) < ./tests/test_10000.t
	./$(BIN) < ./tests/test_1000000.t
	./$(BIN) < ./tests/test_9998244.t  

clean:
	rm -f *.o $(BIN)

# CXX = g++
# CXXFLAGS = -std=c++11 -O2 -Wextra -Wall -Wno-sign-compare -Wno-unused-result
# SOURSES = lab4.cpp
# BIN = lab4

# all: BIN

# BIN:
# 	$(CXX) $(CXXFLAGS) -o $(BIN) $(SOURSES)

# debug:
# 	$(CXX) $(CXXFLAGS) -o $(BIN) $(SOURSES) -g3

# run:
# 	./$(BIN) < ./tests/test_100.t 
# 	./$(BIN) < ./tests/test_10000.t
# 	./$(BIN) < ./tests/test_1000000.t
# 	./$(BIN) < ./tests/test_9998244.t
	         
# clean:
# 	rm -f *.o $(BIN)