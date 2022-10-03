HIP_PATH?= $(wildcard /opt/rocm/hip)
ifeq (,$(HIP_PATH))
	HIP_PATH=../../..
endif

HIPCC=$(HIP_PATH)/bin/hipcc

TARGET=hcc

SOURCES = zero_copy_cpu_busy_wait_nthreads_1flag.cpp
#OBJECTS = $(SOURCES:.cpp)
TESTSRC  = zero_copy_cpu_busy_wait_nthreads_1flag.cpp

EXECUTABLE=./zero_copy_cpu_busy_wait_nthreads_1flag

.PHONY: test


all: $(EXECUTABLE) test

CXXFLAGS =-g
CXX=$(HIPCC)


$(EXECUTABLE): $(TESTSRC)
	$(HIPCC) -fopenmp $^ -o $@


test: $(EXECUTABLE)
	GOMP_CPU_AFFINITY="0 1" $(EXECUTABLE) 2


clean:
	rm -f $(EXECUTABLE)
	rm -f *.o
	rm -f $(HIP_PATH)/src/*.o

