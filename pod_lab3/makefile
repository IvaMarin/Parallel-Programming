CC = nvcc
CFLAGS = --std=c++11 -Werror cross-execution-space-call -lm
SOURSES = lab3.cu
BIN = lab3

all:
	$(CC) $(CFLAGS) -o $(BIN) $(SOURSES)

nvprof:
	nvprof --log-file nvprof_result.txt -e divergent_branch,global_store_transaction,l1_shared_bank_conflict,l1_local_load_hit -m sm_efficiency ./$(BIN) < ./tests/int16000

run:
	./$(BIN) < ./tests/int160
	./$(BIN) < ./tests/int1600
	./$(BIN) < ./tests/int16000
	./$(BIN) < ./tests/int160000
	./$(BIN) < ./tests/int1600000 
    
clean:
	rm -f *.o $(BIN)

# CXX = g++
# CXXFLAGS = -std=c++11 -O2 -Wextra -Wall -Wno-sign-compare -Wno-unused-result
# SOURSES = lab3.cpp
# BIN = lab3

# all: BIN

# BIN:
# 	$(CXX) $(CXXFLAGS) -o $(BIN) $(SOURSES)

# debug:
# 	$(CXX) $(CXXFLAGS) -o $(BIN) $(SOURSES) -g3

# run:
# 	./$(BIN) < ./tests/int160
# 	./$(BIN) < ./tests/int1600
# 	./$(BIN) < ./tests/int16000 
# 	./$(BIN) < ./tests/int160000
# 	./$(BIN) < ./tests/int1600000 
	                   
# clean:
# 	rm -f *.o $(BIN)