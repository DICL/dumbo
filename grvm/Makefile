.PHONY: all grvm libs clean clobber examples

CUDA_HOME = /home/bclee/e/cuda/8.0
JAVA_HOME = /usr/lib/jvm/java-8-openjdk-amd64
JSRC = $(shell find src/java -name *.java)
CPS = lib/jksvm.jar:lib/rvmrt.jar

HSRC =  src/native/grvm.c \
	src/native/host.c \
	src/native/device.ptx.c

CUFLAGS = -gencode arch=compute_35,code=sm_35 -rdc=true
CFLAGS= -I$(JAVA_HOME)/include \
  -I$(JAVA_HOME)/include/linux \
  -I$(CUDA_HOME)/include

all: grvm libs

grvm: bin/grvm.jar bin/libgrvm.so

libs: lib/JikesRVM lib/jdk8rt.jar

bin/grvm.jar: $(JSRC)
	mkdir -p classes bin
	javac $(JFLAGS) -d classes -classpath $(CPS) -sourcepath src/src:examples $(JSRC)
	jar cf $@ -C classes .

bin/libgrvm.so: $(HSRC) src/native/*.h
	mkdir -p lib
	gcc $(CFLAGS)  -fPIC  -g -o $@ -shared $(HSRC) -lcuda

src/native/device.ptx.c: src/native/device.ptx
	python3 util/ptx2carr.py sbsvm_device_ptx \
	src/native/device.ptx \
	src/native/device.ptx.c \
	src/native/device.ptx.h

src/native/device.ptx: src/native/device.cu src/native/cpuptr.h src/native/rpc.h
	nvcc --device-c --ptx -x cu $(CUFLAGS) -Isrc/native -o $@ src/native/device.cu

lib/JikesRVM:
	mkdir -p lib
	make -C jvm rvm
	cp -f jvm/jikesrvm/dist/production_x86_64_m64-linux/JikesRVM lib/
	cp -f jvm/jikesrvm/dist/production_x86_64_m64-linux/lib*.so lib/
	cp -f jvm/jikesrvm/dist/production_x86_64_m64-linux/*.jar lib/
	cp -f jvm/jikesrvm/dist/production_x86_64_m64-linux/*.image lib/
	cp -f jvm/jikesrvm/dist/production_x86_64_m64-linux/*.map lib/

lib/jdk8rt.jar:
	mkdir -p lib
	make -C jvm jdk8rt.jar
	cp jvm/jdk8rt.jar $@

examples:
	make -C examples

clean:
	rm -rf bin/grvm.jar bin/libgrvm.so classes
	make -C examples clean
	make -C jvm clean

clobber: clean
	rm -f src/native/device.ptx
	rm -f src/native/device.ptx.c
	rm -f src/native/device.ptx.h 
	make -C jvm clobber 
