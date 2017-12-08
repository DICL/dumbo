
OUT = libsbsvm.so

CXXFLAGS = -std=c++11 -O2 -I$(JAVA_HOME)/include -I$(JAVA_HOME)/include/linux 
CUFLAGS = --shared -Xcompiler -fPIC -gencode arch=compute_35,code=sm_35 -gencode arch=compute_52,code=sm_52 --relocatable-device-code=true -lcuda

CUSRC = cpupointer.cu memcpy.cu Sbsvm.cu example.cu
INCLUDES = memcpy.h cpupointer.h cptr.h org_sbsvm_Sbsvm.h example.h

.PHONY: all debug test clean

all : $(OUT)

debug : CXXFLAGS += -DDEBUG -g -lineinfo
debug : $(OUT)

org/sbsvm/Sbsvm.class : org/sbsvm/Sbsvm.java
	javac -source 6 -target 6 $<

org_sbsvm_Sbsvm.h : org/sbsvm/Sbsvm.class
	javah -jni org.sbsvm.Sbsvm

$(OUT) : $(CUSRC) $(INCLUDES) 
	nvcc $(CUFLAGS) $(CXXFLAGS) -o $@ $(CUSRC)

Test.class: Test.java
	javac -source 6 -target 6 $<

test : $(OUT) Test.class
	rvm Test

clean :
	rm -f Test.class $(OUT) org_sbsvm_Sbsvm.h org/sbsvm/Sbsvm.class
