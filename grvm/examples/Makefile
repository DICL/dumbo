.PHONY: all clean test


CUFLAGS = -gencode arch=compute_35,code=sm_35 \
  -rdc=true \

all: DMMSBSVM DMMCUDA DMMStreamInnerClass.class DMMStream.class

DMMSBSVM: DMMSBSVM.c DMMSBSVMKernel.ptx.c
	gcc -g -I../src/native -o $@ DMMSBSVM.c DMMSBSVMKernel.ptx.c -L../bin -lgrvm

DMMCUDA: DMMCUDA.cu
	nvcc $(CUFLAGS) -o $@ DMMCUDA.cu

DMMSBSVMKernel.ptx.c: DMMSBSVMKernel.ptx
	python3 ../util/ptx2carr.py \
	dmm_kernel_ptx \
	DMMSBSVMKernel.ptx \
	DMMSBSVMKernel.ptx.c \
	DMMSBSVMKernel.ptx.h

DMMSBSVMKernel.ptx: DMMSBSVMKernel.c Makefile
	nvcc --device-c --ptx -x cu $(CUFLAGS) -I../src/native -o $@ DMMSBSVMKernel.c

DMMStreamInnerClass.class: DMMStreamInnerClass.java
	javac DMMStreamInnerClass.java

DMMStream.class: DMMStream.java
	javac DMMStream.java

clean:
	rm -f DMMSBSVM
	rm -f *.class

clobber: clean
	rm -f DMMSBSVMKernel.ptx
	rm -f DMMSBSVMKernel.ptx.c
	rm -f DMMSBSVMKernel.ptx.h