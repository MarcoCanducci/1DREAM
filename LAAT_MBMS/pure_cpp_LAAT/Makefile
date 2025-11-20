# Build a executable called vaca

#export OMP_THREAD_LIMIT := 8

# the compiler: gcc for C program, define as g++ for C++
CC := g++


# compiler flags:
#  -g    adds debugging information to the executable file
#  -Wall turns on most, but not all, compiler warnings
CFLAGS := -g -Wall -fopenmp

#Optimization Flags
OPTIMIZATION := -O3

GPROF := -pg

# EXTERNAL_LIBRARIES directory
EXTERNAL_LIBRARIES_1 = ../ext
EXTERNAL_LIBRARIES_2 = ../ext/eigen
EXTERNAL_LIBRARIES_3 = ../src/lib
EXTERNAL_LIBRARIES_4 = ../src/lib/LAAT

#Source Directories
GDir := ../src/lib/LAAT
GDir_PURE_CPP := ../src/lib/LAAT/PURE_CPP
##
GSources_1 := antsearch.cpp antsearch_DynamicRadius.cpp computing_initial_ant_probabilities.cpp 
GSources_1 += computing_initial_particle_properties.cpp evaporatepheromone.cpp initializeants.cpp 
GSources_1 += LAAT.cpp  preprocess.cpp progressbar.cpp random_generator.cpp rangesearch.cpp

GSources_PURE_CPP := exporting_data.cpp main.cpp reading_data.cpp

GENERAL_1 = $(patsubst %,$(GDir)/%,$(GSources_1))
GENERAL_PURE_CPP = $(patsubst %,$(GDir_PURE_CPP)/%,$(GSources_PURE_CPP))

#All sources
SOURCES = $(GENERAL_1) $(GENERAL_PURE_CPP) 


#Objects
#ODIR = ob3

OBJ := antsearch.o antsearch_DynamicRadius.o computing_initial_ant_probabilities.o 
OBJ += computing_initial_particle_properties.o evaporatepheromone.o exporting_data.o 
OBJ += initializeants.o LAAT.o main.o preprocess.o progressbar.o random_generator.o 
OBJ += rangesearch.o reading_data.o

#OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

.PHONY: run

# the build target executable:

# Main.cpp Initialization.cpp Jacobi.cpp GS.cpp Fine2coarse.cpp Coarse2fine.cpp Vgrid.cpp Fgrid.cpp Wgrid.cpp Error.cpp

# the build target executable:
EXECUTABLE := LAAT.exe
	
all:
	make compile -s
	make link -s

compile:
	$(CC) $(CFLAGS) -I$(EXTERNAL_LIBRARIES_1) -I$(EXTERNAL_LIBRARIES_2) -I$(EXTERNAL_LIBRARIES_3) -I$(EXTERNAL_LIBRARIES_4) $(OPTIMIZATION) -c $(SOURCES)

link:
	$(CC)  $(OBJ) -o $(EXECUTABLE) -lm  $(CFLAGS)
	
gprof: *.cpp
	$(CC) $(CFLAGS) -I$(EXTERNAL_LIBRARIES_1) -I$(EXTERNAL_LIBRARIES_2) -I$(EXTERNAL_LIBRARIES_3) -I$(EXTERNAL_LIBRARIES_4) $(INICOND) $(OPTIMIZATION) $(GPROF) -o $(EXECUTABLE).gprof $(SOURCES) -lm
	
clean:
	$(RM) -f $(EXECUTABLE) *.o
	
run:
	./$(EXECUTABLE)	
	
test:
	make clean -s
	make respaldo -s
	make run -s
	#make clean -s
	
respaldo: 
	$(CC) $(CFLAGS) -I$(EXTERNAL_LIBRARIES_1) -I$(EXTERNAL_LIBRARIES_2) -I$(EXTERNAL_LIBRARIES_3) $(OPTIMIZATION) -o $(EXECUTABLE) $(SOURCES) -lm	
	
