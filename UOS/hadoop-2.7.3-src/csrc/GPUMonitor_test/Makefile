JAVA_PATH ?=$(JAVA_HOME)
CUDA_PATH ?=/usr/local/cuda-8.0

JNI_INCLUDE:= -I$(JAVA_PATH)/include/ 
JNI_INCLUDE+= -I$(JAVA_PATH)/include/linux

all: dirs lib
	./exe

dirs:
	mkdir -p obj

lib: obj/main.o
	g++ -o exe $< -L${CUDA_PATH}/lib64 -lnvidia-ml

obj/main.o: main.cpp
	g++ -c $< -o $@ -fPIC -I${CUDA_PATH}/include $(JNI_INCLUDE)

clean:
	rm -f *.o
	rm -f *.so
	rm -f obj/*
	rm -f exe
