CPP = g++ -Wall -Ofast -ffast-math -funroll-loops -fopenmp
SRCS = -std=c++11 main.cpp fluid_solver.cpp EventManager.cpp

all: phase1 phase2

phase1:
	$(CPP) $(SRCS) -o fluid_sim_seq

phase2:
	$(CPP) $(SRCS) -o fluid_sim

clean:
	@echo Cleaning up...
	@rm -f fluid_sim fluid_sim_seq
	@echo Done.

runseq: 
	OMP_NUM_THREADS=1 ./fluid_sim_seq

runpar: 
	OMP_NUM_THREADS=20 ./fluid_sim
