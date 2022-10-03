HIP_PATH?= $(wildcard /opt/rocm/hip)
ifeq (,$(HIP_PATH))
	HIP_PATH=../../..
endif

HIPCC=$(HIP_PATH)/bin/hipcc

TARGET=hcc

SOURCES = zero_copy_cpu_busy_wait_nthreads.cpp
OBJECTS = $(SOURCES:.cpp=.o)

EXECUTABLE=./zero_copy_cpu_busy_wait_nthreads

.PHONY: test


all: $(EXECUTABLE) test

CXXFLAGS =-g
CXX=$(HIPCC)


$(EXECUTABLE): $(OBJECTS)
	$(HIPCC) $(OBJECTS) -o $@ -fopenmp


test: $(EXECUTABLE)
	GOMP_CPU_AFFINITY="0 1" $(EXECUTABLE) 2


clean:
	rm -f $(EXECUTABLE)
	rm -f $(OBJECTS)
	rm -f $(HIP_PATH)/src/*.o

