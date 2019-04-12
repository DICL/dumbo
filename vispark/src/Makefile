all:
	#g++ c_manager.cpp -std=c++1y
	#nvcc -o new_manager c_manager.cu -std=c++11 -O3 -arch=sm_52 -lcuda -Xcompiler -fopenmp
	#nvcc -o new_manager c_manager_rdma.cu -std=c++11 -O3 -lcuda -Xcompiler -fopenmp -lrdmacm 
	#nvcc -o new_manager c_manager.cu -std=c++11 -O3 -lcuda -Xcompiler -fopenmp  
	#nvcc -o ../bin/gpu_manager c_manager.cu -std=c++11 -O3 -lcuda -Xcompiler -fopenmp -lmpi -lmpi_cxx -L/home/sumin246/.local/lib -I/home/sumin246/.local/include -lEGL -lGL -lGLU -lGLEW
	nvcc -o ../bin/gpu_manager c_manager.cu -std=c++11 -O3 -lcuda -Xcompiler -fopenmp -lmpi -L/home/sumin246/.local/lib -I/home/sumin246/.local/include


