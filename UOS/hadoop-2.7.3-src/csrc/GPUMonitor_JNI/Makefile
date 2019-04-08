JAVA_PATH ?=/usr/java/latest
CUDA_PATH ?=/usr/local/cuda-8.0

JNI_INCLUDE:= -I$(JAVA_PATH)/include/ 
JNI_INCLUDE+= -I$(JAVA_PATH)/include/linux

SRC_FILE=../../hadoop-yarn-project/hadoop-yarn/hadoop-yarn-server/hadoop-yarn-server-nodemanager/src/main/java/org/apache/hadoop/yarn/server/nodemanager/containermanager/launcher/GPUMonitor.java

HEADER_FILE=org_apache_hadoop_yarn_server_nodemanager_containermanager_launcher_GPUMonitor.h
PACKAGE=org.apache.hadoop.yarn.server.nodemanager.containermanager.launcher.GPUMonitor


EXEC=nvcc
JAVA_HOME ?= /usr/lib/jvm/java
INC:= -I$(JAVA_HOME)/include
INC+= -I$(JAVA_HOME)/include/linux

all: dirs libJNI_GPUMonitor.so $(EXE_FILE)

dirs:
	mkdir -p obj

libJNI_GPUMonitor.so: obj/JNI_GPUMonitor.o obj/matmul.o
	$(EXEC) -o $@ $+ -shared -L${CUDA_PATH}/lib64 -lnvidia-ml

obj/matmul.o: matmul.cu
	$(EXEC) $(INC)  -o $@ -c $+ -O0 -Xcompiler -fPIC

obj/JNI_GPUMonitor.o: JNI_GPUMonitor.cpp $(HEADER_FILE)
	$(EXEC) $(INC) -o $@ -c $< -O0 -Xcompiler -fPIC -I${CUDA_PATH}/include

$(HEADER_FILE): $(SRC_FILE)
	javac $< -d ./
	javah $(PACKAGE)
	rm -f org_apache_hadoop_yarn_server_nodemanager_containermanager_launcher_GPUMonitor_STATE.h

clean:
	rm -f *.o
	rm -f *.so
	rm -rf obj
	rm -rf org
	rm -f $(HEADER_FILE)
	rm -f org_apache_hadoop_yarn_server_nodemanager_containermanager_launcher_GPUMonitor_STATE.h
