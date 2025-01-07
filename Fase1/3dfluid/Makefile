CPP = g++ -Wall -O3 -ffast-math -funroll-loops
SRCS = -std=c++11 main.cpp fluid_solver.cpp EventManager.cpp
	
all:
	$(CPP) $(SRCS) -o fluid_sim
	
clean:
	@echo Cleaning up...
	@rm fluid
	@echo Done.
