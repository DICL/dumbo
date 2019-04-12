//#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h> 
#include <time.h>
#include <math.h>
#include <assert.h>
#include <arpa/inet.h> //inet_addr
#include <netdb.h> //hostent
#include <omp.h>

#include <mpi.h>

#include <iostream>
#include <fstream>
#include <string>
#include <regex>
#include <sstream>
#include <vector>
#include <iterator>
#include <map>
#include <chrono>
#include <thread>

#include <cuda.h>
#include <cuda_runtime.h>
#include "builtin_types.h"
#include "device_launch_parameters.h"

#define EGL 0

#if EGL
#include <EGL/egl.h>

#define GL_GLEXT_PROTOTYPES 1
//#include <GL/gl.h>
//#include <GL/glext.h>
#include <GL/glew.h>
#include <cuda_gl_interop.h>
#endif

#define SOCK_REUSE 0
#define MSGSIZE (4096)
#define CHUNKSIZE (1024*1024*5)
#define PORT 4949


using namespace std;
using namespace chrono;

typedef unsigned int uint;
typedef unsigned char uchar;

template <typename T> class vispark_data; 
template <typename T> class halo_data; 

typedef pair<string,vispark_data<char>*> vpair;
typedef pair<int,halo_data<char>*> hpair;

char *host_name;
int world_rank;
int isEGLvalid = 0;
int numConnect = 0;
//GLuint vertexVBOID;
int debug_run_count =0;

#define RUN_LOG 0

#if defined(RUN_LOG) && RUN_LOG > 0
    #define log_print(fmt, args...) fprintf(stderr, "[%s] %s():%04d - " fmt, \
        host_name,__func__, __LINE__, ##args)
#else
    #define log_print(fmt, args...) /* Don't do anything in release builds */
#endif

#if defined(RUN_LOG) && RUN_LOG > 0
    #define CUDA_MEM_CHECK(fmt, args...) {fprintf(stderr, "[%s] %s():%d: " fmt ,\
        host_name,__func__, __LINE__, ##args);\
    size_t free = 0, total = 0;     \
    cudaMemGetInfo(&free,&total);   \
    free  /= 1024*1024;             \
    total /= 1024*1024;             \
    fprintf(stderr,"(Memory Usages : %zu MB / %zu MB [%4.2f%%])\n",total-free,total,100.0*(total-free)/total); \
}
#else
    #define CUDA_MEM_CHECK(fmt, args...) /* Don't do anything in release builds */
#endif

#if defined(RUN_LOG) && RUN_LOG > 0
    #define checkLastError() {                                                          \
    cudaError_t error = cudaGetLastError();                                             \
    int id; cudaGetDevice(&id);                                                         \
    if(error != cudaSuccess) {                                                          \
        fprintf(stderr,"[%s] cuda failure error in file '%s' in line %i: '%s' at device %d \n",      \
            host_name,__FILE__,__LINE__, cudaGetErrorString(error), id);                          \
        exit(EXIT_FAILURE);                                                             \
        }                                                                               \
    }   
#else
    #define checkLastError() 
#endif


#if EGL
void assertOpenGLError(const std::string& msg) {
    GLenum error = glGetError();

    if (error != GL_NO_ERROR) {
        stringstream s;
        s << "OpenGL error 0x" << std::hex << error << " at " << msg;
        throw runtime_error(s.str());
    }
}

void assertEGLError(const std::string& msg) {
    EGLint error = eglGetError();

    if (error != EGL_SUCCESS) {
        stringstream s;
        s << "EGL error 0x" << std::hex << error << " at " << msg;
        throw runtime_error(s.str());
    }
}

GLuint p[3];

static const int pbufferWidth = 10;
static const int pbufferHeight = 10;

int width;
int height;
int struct_size;

static const EGLint pbufferAttribs[] = {
    EGL_WIDTH, pbufferWidth,
    EGL_HEIGHT, pbufferHeight,
    EGL_NONE,
};


static const EGLint configAttribs[] = {
    EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
    EGL_BLUE_SIZE, 8,
    EGL_GREEN_SIZE, 8,
    EGL_RED_SIZE, 8,
    EGL_DEPTH_SIZE, 8,
    EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
    EGL_NONE
};    

GLuint VBO;
#endif

bool check_space(size_t obj)
{
    size_t free = 0, total = 0;
    cudaMemGetInfo(&free,&total);
   
    if (free - obj < 100*1024*1024)
        return false; 
    
    return true;     
}



#define NUM_THREADS 2

void omp_memcpy(char *dst, char *src, size_t len)
{
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int num_threads = omp_get_num_threads();

        size_t start = (len/num_threads)*tid;
        size_t end   = (len/num_threads)*(tid+1);

        if (tid == num_threads - 1)  
            end   = len;

        memcpy(dst + start , src + start , end-start);
    } 
} 


void call_memcpy(char *dst, char *src, int start, int end)
{
    memcpy(dst + start , src + start , end-start);
}

void mt_memcpy(char *dst, char *src, size_t len)
{
    auto th1 = thread(call_memcpy,dst,src,len*0.00,len*0.25);
    auto th2 = thread(call_memcpy,dst,src,len*0.25,len*0.50);
    auto th3 = thread(call_memcpy,dst,src,len*0.50,len*0.75);
    auto th4 = thread(call_memcpy,dst,src,len*0.75,len*1.00);

    th1.join();
    th2.join();
    th3.join();
    th4.join();
}

void error(const char *msg)
{
    perror(msg);
    exit(0);
}

void msg_print(vector<string> msg)
{
    for (auto n : msg)
        cout << n << " ";
    cout << endl;
}

vector<string> msg_parser(const char* msg_buffer)
{
    const string s(msg_buffer);
    istringstream ist(s);

    vector<string> tmp,ss;
    copy(istream_iterator<string>(ist), istream_iterator<string>(),
        back_inserter(tmp));

    for (auto n : tmp) {
        if (strcmp(n.c_str(),"END") == 0)
            break;
        ss.push_back(n);
    }
    
    return ss;
}

string RandomString(const char * prefix, int len, int type)
{
//    if (type > 0)
   string str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
   string newstr(prefix);
   int pos;
   while(newstr.size() != len) {
        pos = ((rand() % (str.size() - 1)));
        newstr += str.substr(pos,1);
   }
   //newstr += '\n';
   return newstr;
}

class msg_create
{
    char *msg;
    size_t msg_size;
    int lenn;

    public:
    msg_create(uint num_msg = 1){

        uint size = (num_msg + 1)*MSGSIZE;
        this->msg_size = sizeof(char)*size;

        msg = new char[size];
        bzero(msg,size);
    }

    void set_head(string s){

        memcpy(msg,s.c_str(),s.size());
    }

    void set_msg(string s){
        memcpy(msg + MSGSIZE, s.c_str(),s.size());
    }

    char *ptr(){
        return msg;
    }
    
    size_t size(){
        return msg_size;
    }

    void print(){
        for (int i =0 ; i < msg_size ; i++)
            printf("%c",msg[i]);
        printf("\n");
    }

};

template <typename T>
class halo_data 
{

    T* host_ptr=nullptr;

    T* host_ptr_up = nullptr;
    T* host_ptr_dn = nullptr;

    size_t malloc_size = 0;
    size_t data_size = 0;

    string data_type = "char";  
    string data_key = "";    

 
    public:
        halo_data(size_t data_size){
     //       printf("Constructor \n");
            this->malloc_size = 2*(data_size*sizeof(T) + MSGSIZE);
            this->data_size = data_size*sizeof(T);

            //cudaHostAlloc((void**)&host_ptr, malloc_size, cudaHostAllocDefault); 
            host_ptr = new T[malloc_size];

            host_ptr_up = host_ptr;
            host_ptr_dn = host_ptr + this->data_size;           
        }

        ~halo_data()
        {
            delete [] host_ptr;
            //printf("Distructor Halo\n");
            //cudaFreeHost(host_ptr);
            //cudaFreeHost(host_ptr);
        }

        void extract(vispark_data<T>* data_elem)
        {  
            size_t up_off= data_size;
            size_t dn_off= data_elem->getDataSize()-2*data_size;

            if (data_elem->inGPU() == true){
        
                auto data_ptr = data_elem->getDevPtr();
                                
                cudaMemcpy(host_ptr_up,data_ptr + up_off, data_size,cudaMemcpyDeviceToHost);
                cudaMemcpy(host_ptr_dn,data_ptr + dn_off, data_size,cudaMemcpyDeviceToHost);

            }
            else {
                auto data_ptr = data_elem->getHostPtr();

                memcpy(host_ptr_up,data_ptr + up_off, data_size);
                memcpy(host_ptr_dn,data_ptr + dn_off, data_size);

            }
            cudaDeviceSynchronize();
        }

        void append(vispark_data<T>* data_elem, int type)
        {
            T* halo_ptr;
            size_t offset;

            // Find from Up block
            if (type == -1){
                halo_ptr = host_ptr_dn;
                offset = 0;
            }
            // Find from Dn block
            else if (type == 1){
                halo_ptr = host_ptr_up;
                offset = data_elem->getDataSize()-data_size;
            }else {
                assert(0);
            }

            if (data_elem->inGPU() == true){
                auto data_ptr = data_elem->getDevPtr();
                cudaMemcpy(data_ptr + offset,halo_ptr, data_size,cudaMemcpyHostToDevice);
            }else {
                auto data_ptr = data_elem->getHostPtr();
                memcpy(data_ptr + offset,halo_ptr, data_size);
            }
            cudaDeviceSynchronize();
        }    

        size_t getDataSize(){
            return data_size;
        }

        T* getHostPtr(){
            return host_ptr;
        }
        
        void print(char* filename){

            FILE* fp;
            fp = fopen(filename,"w+");
            fwrite(host_ptr,sizeof(T),2*data_size,fp);
            fclose(fp);
        }


}; 


template <typename T>
class vispark_data 
{
    T* host_ptr = nullptr;
    T* dev_ptr = nullptr;
    size_t malloc_size = 0;
    size_t data_size = 0;
    string data_type = "char";  
    string data_key = "";    
    bool in_mem_flag = false;
    cudaStream_t *stream_list;
    int stream_num;
    bool is_persist = false;
    
    #if EGL
    GLuint vertexVBOID;
    cudaGraphicsResource_t vboRes;
    bool isVBOflush = false;
    #endif 


    public:
        vispark_data(size_t data_size){
     //       printf("Constructor \n");
            this->malloc_size = data_size*sizeof(T) + MSGSIZE;
            this->data_size = data_size*sizeof(T);
            //cudaHostAlloc((void**)&host_ptr, malloc_size, cudaHostAllocDefault); 
            host_ptr = new T[malloc_size];
            assert(check_space(data_size));

            if (isEGLvalid){
		#if EGL
                glGenBuffers(1, &vertexVBOID);
                glBindBuffer(GL_ARRAY_BUFFER, vertexVBOID);
                glBufferData(GL_ARRAY_BUFFER, data_size, NULL, GL_DYNAMIC_COPY);

                cudaGraphicsGLRegisterBuffer(&vboRes, vertexVBOID, cudaGraphicsRegisterFlagsNone);
                cudaGraphicsMapResources(1,&vboRes);
                cudaGraphicsResourceGetMappedPointer((void**)&dev_ptr,&data_size,vboRes);
		#endif
            }else {
                cudaMalloc((void**)&dev_ptr, data_size); 
                cudaMemset(dev_ptr,0,data_size);
            }
//            cudaMemset((void**)&dev_ptr,0,data_size);
            //log_print("[CUDA_MALLOC] %d MB \n",data_size/(1024*1024));
           // checkLastError();
/*
            int stream_num = data_size % CHUNKSIZE > 0 ? data_size/CHUNKSIZE + 1  : data_size/CHUNKSIZE;
            stream_list = new cudaStream_t[stream_num];

            for (int i = 0 ; i < stream_num ; i++)
                cudaStreamCreate(&(stream_list[i]));
*/
        }

        ~vispark_data()
        {
            #if 1
            //log_print("Distructor vispark %s \n",data_key.c_str());
            checkLastError();
            //cudaFreeHost(host_ptr);
            delete [] host_ptr;
            if (isEGLvalid){
		#if EGL
    		if(isVBOflush == false)
                    cudaGraphicsUnmapResources(1,&vboRes);
                cudaGraphicsUnregisterResource(vboRes);
                glDeleteBuffers(1,&vertexVBOID);
		#endif

            }
            else
                cudaFree(dev_ptr);
            checkLastError();
            #endif
        }


        vispark_data(const vispark_data &A)
        {
        //    printf("Copy \n");
/*
            this->malloc_size = A.malloc_size;
            this->data_size = A.data_size;
            cudaHostAlloc((void**)&host_ptr, malloc_size, cudaHostAllocDefault); 
    //        host_ptr = new T[malloc_size];
            cudaMalloc((void**)&dev_ptr, malloc_size); 
            

            memcpy(this->host_ptr,A.host_ptr,malloc_size);
*/
        }


        //void htod(cudaStream_t stream){
        void htod(){
            //cudaMemsetAsync(dev_ptr,0,malloc_size,stream);
            cudaMemcpy(dev_ptr,host_ptr,data_size,cudaMemcpyHostToDevice);
            in_mem_flag = true;
        } 
    
        void dtoh(){
            
            cudaMemcpy(host_ptr,dev_ptr,data_size,cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            //in_mem_flag = false;
            //cudaMemsetAsync(dev_ptr,0,malloc_size,stream);
        }

        vector<cudaStream_t*> dtoh_stream(){
            vector<cudaStream_t*> stream_list;
        
            int offset =0;
            while(offset < data_size){
                cudaStream_t *stream = new cudaStream_t;
                cudaStreamCreate(stream);

                cudaMemcpyAsync(host_ptr + offset,dev_ptr+offset,CHUNKSIZE,cudaMemcpyDeviceToHost,*stream);
               
                offset += CHUNKSIZE;
                stream_list.push_back(stream);
            }
         
            return stream_list;
        }


        size_t getMallocSize(){
            return malloc_size;
        }
        
        size_t getDataSize(){
            return data_size;
        }

        T* getHostPtr(){
            //log_print("Host call vispark %s \n",data_key.c_str());
            return host_ptr;
        }
        
        T* getDevPtr(){
            //log_print("Device call vispark %s \n",data_key.c_str());
            return dev_ptr;
        }

        void setDataKey(string data_key){
            this->data_key = data_key;
        }

        string getDataKey(){
            return data_key;
        }
        
        bool inGPU(){
            return in_mem_flag;
        }
        
        void setInGPU(bool flag){
            in_mem_flag = flag;
        }          
        
        bool isPersist(){
            return is_persist;
        } 
        
        void setPersist(){
            is_persist = true;
        }

        void print(char* filename){

            if (this->inGPU())
                this->dtoh();


            FILE* fp;
            fp = fopen(filename,"w+");
            fwrite(host_ptr,sizeof(T),data_size,fp);
            fclose(fp);
        }

	#if EGL
        GLuint getVBO(){
            //cudaGLUnmapBufferObject(vertexVBOID);
            if (isVBOflush == false){
                cudaGraphicsUnmapResources(1,&vboRes);
                isVBOflush = true;
            }
            return vertexVBOID;
        }
	#endif 
}; 


void getBlockDim(vector<int> kernel_info, int& gx, int& gy, int& gz, int& bx, int& by, int& bz)
{
    gx = gy = gz = bx = by = bz = 1;

    int dimx, dimy, dimz;

    if (kernel_info.size() == 1){
        dimx = kernel_info[0];

        if (dimx  < 512)
            bx = dimx;
        else{ 
            gx = dimx/512 + 1;
            bx = 512; 
        }
    }   
    else if (kernel_info.size() == 2){
        dimx = kernel_info[0];
        dimy = kernel_info[1];

        gx = dimx/16 + 1;
        bx = 16;

        gy = dimy/16 + 1;
        by = 16; 

    }else {

        dimx = kernel_info[0];
        dimy = kernel_info[1];
        dimz = kernel_info[2];

        gx = dimx/8 + 1;
        bx = 8;

        gy = dimy/8 + 1;
        by = 8; 

        gz = dimz/8 + 1;
        bz = 8; 
    }   
}

//   -*-   -*-   -*-
void renderScene(void);
void saveScene(char* filename);
void setVBO(char* cuda_ptr, size_t sturct_size,unsigned int struct_num);
void initVBO(long long MAXIMUM_NUM, size_t struct_size);
   




//   -*-   -*-   -*-

CUcontext context;
CUdevice device;
CUfunction kernelfunc;
CUmodule module;

//   -*-   -*-   -*-

unsigned int* h_numvert;
int h_name;

//   -*-   -*-   -*-

//kernel_copy<<<10,10>>>(dptr,&devOutArr1,h_numvert[0]);
__global__ void kernel_copy(float4 *dptr, float* vertex, unsigned int num_triangle)
{
    int tri_idx = threadIdx.x + blockDim.x * blockIdx.x; 

    if (tri_idx < num_triangle){
        dptr[tri_idx].x = vertex[tri_idx*3+0];        
        dptr[tri_idx].y = vertex[tri_idx*3+1];        
        dptr[tri_idx].z = vertex[tri_idx*3+2];        
        dptr[tri_idx].w = 0;

        //printf("%f %f %f \n",dptr[tri_idx].x,dptr[tri_idx].y,dptr[tri_idx].z);
    }
}


CUresult kernel_call(vispark_data<char>* out_data, vispark_data<char>* in_data, vector< tuple<string,int,char*> >* args,vector<int> kernel_info) {
    CUdeviceptr devInArr1, devOutArr1;
    CUresult err;

    devInArr1 = (CUdeviceptr) in_data->getDevPtr(); 
    devOutArr1 = (CUdeviceptr) out_data->getDevPtr(); 

    vector<void*> kernelParams;
    kernelParams.push_back(&devOutArr1);
    kernelParams.push_back(&devInArr1);

    for (auto n : *args){

        string type = get<0>(n);
        int    len  = get<1>(n);
        char * data = get<2>(n);

        if (strcmp(type.c_str(),"int") == 0){
            int *data_ptr = (int *) data;
            if (len == 1)  
                kernelParams.push_back(const_cast<int*>(data_ptr));
            else{

                CUdeviceptr* local_arr = new CUdeviceptr;
                cuMemAlloc(local_arr, sizeof(int) * len);
                cuMemcpyHtoD(*local_arr, data_ptr, sizeof(int) * len);
                kernelParams.push_back(local_arr);
            }
            //printf("[%s] args = %d\n",host_name,*data_ptr); 
        }

        if (strcmp(type.c_str(),"double") == 0){
            double *data_ptr = (double *) data;
            if (len == 1)  
                kernelParams.push_back(const_cast<double*>(data_ptr));
            else{

                CUdeviceptr* local_arr = new CUdeviceptr;
                cuMemAlloc(local_arr, sizeof(double) * len);
                cuMemcpyHtoD(*local_arr, data_ptr, sizeof(double) * len);
                kernelParams.push_back(local_arr);
            } 
                 
        }

        if (strcmp(type.c_str(),"float") == 0){
            float *data_ptr = (float *) data;
            if (len == 1)  
                kernelParams.push_back(const_cast<float*>(data_ptr));
            else{

                CUdeviceptr* local_arr = new CUdeviceptr;
                cuMemAlloc(local_arr, sizeof(float) * len);
                cuMemcpyHtoD(*local_arr, data_ptr, sizeof(float) * len);
                kernelParams.push_back(local_arr);
            } 
                 
        }

    }
 
   
    int gx,gy,gz,bx,by,bz;
 
    getBlockDim(kernel_info,gx,gy,gz,bx,by,bz);
    //log_print("%d %d %d %d %d %d \n",gx,gy,gz,bx,by,bz);

    err = cuLaunchKernel(kernelfunc, gx,gy,gz,  bx, by, bz,  0, 0, &kernelParams[0], 0);
    if (err != CUDA_SUCCESS) return err;
    checkLastError();

    if (isEGLvalid == true){


        //Marching Cube debug
        CUdeviceptr* d_numvert = (CUdeviceptr*) kernelParams[7]; 
        h_numvert = new unsigned int[2];
        cuMemcpyDtoH(h_numvert,*d_numvert,sizeof(unsigned int)*2);   
        checkLastError();

        int num_vert = h_numvert[0];
        h_name = h_numvert[1]; 
        //printf("NUM VERTEX = %u \n",num_vert);
    }

    for (int i = 2; i < kernelParams.size() ;i++)
        cuMemFree(*((CUdeviceptr*)kernelParams[i]));
    checkLastError();
           
    out_data->setInGPU(true);

    return err;
};

//    map<string,vispark_data<char>*> data_dict;
//    map<string,vector<tuple<string,int,char*>>*> args_dict;

int GPU_TEST(const char *ptxfile, const char* func_name, vispark_data<char>* out_data, vispark_data<char>* in_data, vector<tuple<string,int,char*>>* args, vector<int> kernel_info) {
    CUresult err;
    
    /*
    int deviceCount = 0;


    //err = cuInit(0);
    if (err != CUDA_SUCCESS) { printf("cuInit error... .\n"); return err; }
    err = cuDeviceGetCount(&deviceCount);
    if (err != CUDA_SUCCESS) { printf("cuDeviceGetCount error... .\n"); return err; }
    if (deviceCount == 0) { printf("No CUDA-capable devices... .\n"); return err; } 
    err = cuDeviceGet(&device, 0);
    if (err != CUDA_SUCCESS) { printf("cuDeviceGet error... .\n"); return err; }
    err = cuCtxCreate(&context, 0, device);
    if (err != CUDA_SUCCESS) { printf("cuCtxCreate error... .\n"); return err; }
    */
    err = cuModuleLoad(&module, ptxfile);
    if (err != CUDA_SUCCESS) { log_print("cuModuleLoad error... .\n"); return err; }
    err = cuModuleGetFunction(&kernelfunc, module, func_name);
    if (err != CUDA_SUCCESS) { log_print("cuModuleGetFunction error... .\n"); return err; }

    err = kernel_call(out_data,in_data,args,kernel_info);
    if (err != CUDA_SUCCESS) { log_print("Kernel invocation failed... .\n"); return err; }
  //  for (int i = 0; i < 10; ++i) printf("%d + %d = %d\n", inArr1[i], inArr2[i], outArr1[i]);
    //cuCtxSynchronize();
    //cuCtxDetach(context);
    return 0;
}


vector<string> workers;
int num_workers;
int w_idx;

int getWorkerRank(string w){
    
    int rank = 0;
    for (auto n : workers){
        if (strcmp(w.c_str(),n.c_str()) == 0)
            break;
       rank++;
    }

    return rank; 
}


#if EGL
static void initGL()
{
	//GLfloat fieldOfView = 90.0f;
	glViewport(0, 0, width, height);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(-1,1,-1,1,-20,20);	

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glShadeModel(GL_SMOOTH);

    //if (wireframe)
    //	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    //else
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_NORMALIZE);
    glEnable(GL_POLYGON_SMOOTH);
    //glEnable(GL_BLEND);
    //glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

}


void renderScene(void){

    glClearColor(1.0,1.0,1.0,1.0);
    glClearDepth(1.0);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);  


	//glPushMatrix();

    //glTranslatef(-0.5f, -0.5f, 0.0f);
    //glTranslatef(0.0f, 0.0f,-5.0f);
    //glRotatef(20.0f,0,1,0);
    //glRotatef(40.0f,1,0,0);

    //glPushMatrix();
}

void renderObject(GLuint vertexVBOID, unsigned int num_vert, float* projMat, float* modelMat, float* viewMat, vector<tuple<string,string,int,char*>> args_data){

    glBindBuffer(GL_ARRAY_BUFFER, vertexVBOID);
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_NORMAL_ARRAY);
    
    glVertexPointer(3, GL_FLOAT, struct_size, ((void*)(0)));
    glNormalPointer(GL_FLOAT, struct_size, ((void*)(12)));
    
    if (struct_size > 24){
        glEnableClientState(GL_COLOR_ARRAY);
        glColorPointer(3,GL_FLOAT,struct_size,((void*)(24)));
    }

    for(auto elem : args_data){
        auto elem_name = get<0>(elem);
        auto elem_type = get<1>(elem);
        auto elem_length = get<2>(elem);
        auto elem_data = get<3>(elem);

        GLint var;
        var = glGetUniformLocation(p[0],elem_name.c_str());

        if (elem_length == 4){
            float *data = (float*) elem_data;
            glUniform4f(var,data[0],data[1],data[2],data[3]);
        }

        if (elem_length == 3){
            float *data = (float*) elem_data;
            glUniform3f(var,data[0],data[1],data[2]);
        }

    } 

 
    GLint loc_projMat;
    GLint loc_modelMat;
    GLint loc_viewMat;

	glMatrixMode(GL_PROJECTION);
	glGetFloatv(GL_PROJECTION_MATRIX,(float*)projMat); 

	loc_projMat = glGetUniformLocation(p[0],"projMat");
	loc_modelMat = glGetUniformLocation(p[0],"modelMat");
	loc_viewMat = glGetUniformLocation(p[0],"viewMat");

    glUniformMatrix4fv(loc_projMat,1,false,(float*)projMat); // MARK
    glUniformMatrix4fv(loc_modelMat,1,false,(float*)modelMat); // MARK
    glUniformMatrix4fv(loc_viewMat,1,false,(float*)viewMat); // MARK

    glDrawArrays(GL_TRIANGLES, 0, num_vert);//, GL_UNSIGNED_INT, ((void*)(0))); 
 
    glFlush();
    assertOpenGLError("glDraws");
}

void saveScene(char* filename)
{
    /*
     * Read the framebuffer's color attachment and save it as a PNG file.
     */
    //cv::Mat image(500, 500, CV_8UC3);
    unsigned char *data2 = new unsigned char[width*height*3];
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    //glReadPixels(0, 0, width, height, GL_BGR, GL_UNSIGNED_BYTE, data2);
    glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, data2);
    assertOpenGLError("glReadPixels");

    //cv::imwrite("img.png", image);

    FILE *pFile;
    pFile = fopen(filename,"w");
    fwrite(data2,sizeof(unsigned char),width*height*3,pFile);
    fclose(pFile);
}


void readScene()
{

    unsigned char *data2 = new unsigned char[width*height*3];
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    //glReadPixels(0, 0, width, height, GL_BGR, GL_UNSIGNED_BYTE, data2);
    glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, data2);
    assertOpenGLError("glReadPixels");


    float* depth_buffer = new float[width*height];
    glReadPixels(0,0,width,height,GL_DEPTH_COMPONENT,GL_FLOAT,depth_buffer);
    assertOpenGLError("Read Depth");

}

char *textFileRead(char *fn) {


	FILE *fp;
	char *content = NULL;

	int count=0;

	if (fn != NULL) {
		fp = fopen(fn,"rt");

		if (fp != NULL) {
      
      fseek(fp, 0, SEEK_END);
      count = ftell(fp);
      rewind(fp);

			if (count > 0) {
				content = (char *)malloc(sizeof(char) * (count+1));
				count = fread(content,sizeof(char),count,fp);
				content[count] = '\0';
			}
			fclose(fp);
		}
	}
	return content;
}

//GLuint loadShader(GLenum shadertype, char *c)
GLuint loadShader(GLenum shadertype, char *ss)
{
	GLuint s = glCreateShader( shadertype );
	//char *ss = textFileRead( c );

    //printf("Length for %s = %d \n",c,strlen(ss));
    //printf("Length for  %d \n",strlen(ss));

    //printf("%s",ss);

	const char *css = ss;
	glShaderSource(s, 1, &css, NULL);
	//free( ss );
	glCompileShader( s );

	// validation
	int status, sizeLog;
	glGetShaderiv(s, GL_COMPILE_STATUS, &status);
	if(status == GL_FALSE)
	{
		glGetShaderiv(s, GL_INFO_LOG_LENGTH, &sizeLog);
		char *log = new char[sizeLog + 1];
		glGetShaderInfoLog(s, sizeLog, NULL, log);
		std::cout << "Shader Compilation Error: " << log << std::endl;
		delete [] log;
		assert( false );
	}

	return s;
}

GLuint createGLSLProgram(char *vs, char *gs, char *fs) 
{
    //printf("CreateGLSL %s,%s,%s\n",vs,gs,fs);
	GLuint v, g, f, p;
	
	if( !vs && !gs && !fs ) return 0;
		
	p = glCreateProgram();
	
	if( vs ) 
	{
		v = loadShader( GL_VERTEX_SHADER, vs );
		glAttachShader(p,v);
	}
	if( gs )
	{
		g = loadShader( GL_GEOMETRY_SHADER_EXT, gs );
		glAttachShader(p,g);
	}
	if( fs )
	{
		f = loadShader( GL_FRAGMENT_SHADER, fs );
		glAttachShader(p,f);
	}

	glLinkProgram(p);

	// validating program
	GLint status;
	glGetProgramiv(p, GL_LINK_STATUS, &status);
	if(status == GL_FALSE)
	{
		GLint sizeLog;
		glGetProgramiv(p, GL_INFO_LOG_LENGTH, &sizeLog);
		char *log = new char[sizeLog + 1];
		glGetProgramInfoLog(p, sizeLog, NULL, log);
		std::cout << "Program Link Error: " << log << std::endl;
		delete [] log;
		assert( false );
	}
	
	glValidateProgram(p);
	glGetProgramiv(p, GL_VALIDATE_STATUS, &status);

	if (status == GL_FALSE)
	{
		std::cerr << "Error validating program: "<< p << std::endl;
		assert( false );
	}

	// validation passed.. therefore, we will use this program
	glUseProgram(p);

	return p;
}


void EGLinit(){

    /*
     * EGL initialization and OpenGL context creation.
     */

    EGLDisplay display;
    EGLConfig config;
    EGLContext context;
    //EGLSurface surface;
    EGLint num_config;
    EGLint major, minor;


    display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    assertEGLError("eglGetDisplay");
    
    eglInitialize(display, &major, &minor);
    assertEGLError("eglInitialize");

    eglChooseConfig(display, configAttribs, &config, 1, &num_config);
    assertEGLError("eglChooseConfig");
    
    assertEGLError("eglCreatePbufferSurface");
    
    eglBindAPI(EGL_OPENGL_API);
    assertEGLError("eglBindAPI");
    
    context = eglCreateContext(display, config, EGL_NO_CONTEXT, nullptr);
    assertEGLError("eglCreateContext");

    eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, context);
    assertEGLError("eglMakeCurrent");

}

void GLEWinit(){

    /*
     * GLEW initialization.
     */

	GLenum err = glewInit();
    
   if (GLEW_OK != err)
    {
        /* Problem: glewInit failed, something is seriously wrong. */
        fprintf(stderr, "Error: %s\n", glewGetErrorString(err));

    }

	if (glewIsSupported("GL_VERSION_3_3"))
		log_print("Ready for OpenGL 3.3\n");
	else {
		log_print("OpenGL 3.3 is not supported\n");
		exit(1);
	}

}

void frameBufferinit(){

    /*
     * Create an OpenGL framebuffer as render target.
     */
    GLuint frameBuffer;
    glGenFramebuffers(1, &frameBuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, frameBuffer);
    assertOpenGLError("glBindFramebuffer");

    GLuint renderBuffer;
    GLuint renderBufferWidth = width;
    GLuint renderBufferHeight = height;

    glGenRenderbuffers(1, &renderBuffer);
    glBindRenderbuffer(GL_RENDERBUFFER, renderBuffer);
    glRenderbufferStorage(GL_RENDERBUFFER,
            GL_RGB565,
            renderBufferWidth,
            renderBufferHeight);
    assertOpenGLError("glBindrenderbuffer");

    /*
     * Attach the texture to the framebuffer.
     */

    glFramebufferRenderbuffer(GL_FRAMEBUFFER,
          GL_COLOR_ATTACHMENT0,
          GL_RENDERBUFFER,
          renderBuffer);
    assertOpenGLError("glFramebufferRenderbuffer");
    /*
     * Create an OpenGL depthbuffer as render target.
     */

    GLuint depthRenderbuffer;
    glGenRenderbuffers(1, &depthRenderbuffer);
    glBindRenderbuffer(GL_RENDERBUFFER, depthRenderbuffer);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT16,     renderBufferWidth, renderBufferHeight);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthRenderbuffer);

    /*
     * Check FBO status
     */

    GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if(status != GL_FRAMEBUFFER_COMPLETE) {
        log_print("Problem with OpenGL framebuffer after specifying color render buffer: \n%x\n", status);
    } else {
        log_print("FBO creation succedded\n");
    }
}
#endif



int main(int argc, char* argv[])
{
    srand(getpid());

    MPI_Init(NULL, NULL);

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Get the name of the processor
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

/*
    // Print off a hello world message
    printf("Hello world from processor %s, rank %d"
           " out of %d processors\n",
           processor_name, world_rank, world_size);

    // Finalize the MPI environment.
    MPI_Finalize();
*/
    //omp_set_num_threads(NUM_THREADS);

#if 0
    //Argument
    if (argc > 3){

        char *slave_name = argv[1];  
        host_name = argv[2];
        char *pid_name = argv[3];    

        string line; 
        ifstream slave_file(slave_name);
        while (getline(slave_file,line))
        {
            line.erase(std::remove(line.begin(), line.end(), ' '), line.end());
            line = line.substr(0,line.find("#"));

            if (line.size() > 0)
                workers.push_back(line);
        }

        num_workers = workers.size();    

        for (int i = 0 ; i < workers.size() ; i++){
            auto n = workers[i];
            if (strcmp(n.c_str(),host_name) == 0){
                w_idx = i;   
                break;
            }
        }

        ofstream pid_file(pid_name,ios::trunc);
                 pid_file << getpid();
                 pid_file.close();
    
        log_print("Launch Process among %d/%d (%d)\n",w_idx,num_workers,getpid());
    }
#else
    //Argument
    if (argc > 1){

        char *slave_name = argv[1];  

        char *pid_name = argv[2];    

        string line; 
        ifstream slave_file(slave_name);
        while (getline(slave_file,line))
        {
            line.erase(std::remove(line.begin(), line.end(), ' '), line.end());
            line = line.substr(0,line.find("#"));

            if (line.size() > 0)
                workers.push_back(line);
        }

        num_workers = workers.size();    

        assert(num_workers == world_size);
        
        auto worker_name = workers[world_rank];
            
        host_name = new char[worker_name.size()+1];

        // cppstyle   .
        //strcpy(host_name, worker_name.size()+1, worker_name.c_str());
        strcpy(host_name, worker_name.c_str());

        ofstream pid_file(pid_name,ios::trunc);
                 pid_file << getpid();
                 pid_file.close();
    }
#endif 
 
    //Dict  
    map<string,vispark_data<char>*> data_dict;
    map<int,halo_data<char>*> halo_dict;
    map<string,vector<tuple<string,int,char*>>*> args_dict;
    map<string,string> code_dict; 
//    map<int,string> actual_nodes;


    //Variable
    int sockfd, newsockfd, portno;
    socklen_t clilen;
    struct sockaddr_in serv_addr, cli_addr;
    int n;
    //int lenn;

    //Timer
    time_point<system_clock> start, end;
    time_point<system_clock> t1,t2;
    double etime;
    int throughput;

    
    //CUDA
    CUresult err;
    int deviceCount = 0;

    err = cuInit(0);
    if (err != CUDA_SUCCESS) { printf("cuInit error... .\n"); return err; }
    err = cuDeviceGetCount(&deviceCount);
    if (err != CUDA_SUCCESS) { printf("cuDeviceGetCount error... .\n"); return err; }
    if (deviceCount == 0) { printf("No CUDA-capable devices... .\n"); return err; } 
    err = cuDeviceGet(&device, 0);
    if (err != CUDA_SUCCESS) { printf("cuDeviceGet error... .\n"); return err; }
    err = cuCtxCreate(&context, 0, device);
    if (err != CUDA_SUCCESS) { printf("cuCtxCreate error... .\n"); return err; }

    //Socket Binding
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0)
        error("ERROR opening socket");


#if SOCK_REUSE 
    //add reuse
    int enable = 1;
    if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(int)) < 0)
    error("setsockopt(SO_REUSEADDR) failed");
#endif

    bzero((char *) &serv_addr, sizeof(serv_addr));
    portno = PORT;
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = INADDR_ANY;
    serv_addr.sin_port = htons(portno);
    if (bind(sockfd, (struct sockaddr *) &serv_addr,
                sizeof(serv_addr)) < 0)
        error("ERROR on binding");
    listen(sockfd,16);

    //printf("Port %d Open \n",portno);


   // const char* test = "SEND 8 9 uchar END 00000000000";
    //log_print("Launch Process among %d/%d (%d)\n",world_rank,world_size,getpid());
    printf("Launch Process among %d/%d (%d)\n",world_rank,world_size,getpid());

   
    char *buffer, *send_buf;
    buffer = new char[MSGSIZE];
    send_buf= new char[MSGSIZE];
    memset(buffer,0,sizeof(char)*MSGSIZE);
    memset(send_buf,0,sizeof(char)*MSGSIZE);

    /* width = 1000; */
    /* height = 1000; */
    /* struct_size=24; */
    /*  */
    /* EGLinit(); */
    /*  */
    /* GLEWinit(); */
    /*  */
    /* frameBufferinit(); */
    /* initGL(); */

                
    while(true)
    {
        vector<string> log_msg; 
        clilen = sizeof(cli_addr); 
        newsockfd = accept(sockfd, (struct sockaddr *) &cli_addr, &clilen);

        start = system_clock::now();

        if (newsockfd < 0) 
            error("ERROR on accept");
        //log_print("Accept %d\n",numConnect);


        string data_key = RandomString("Task_",12,0);
  

        int total_recv = 0;  
        while(true){
            if (total_recv >= MSGSIZE)
                break;
            n = read(newsockfd,buffer+total_recv,MSGSIZE-total_recv);
            if (n < 0) error("ERROR reading from socket");
            else if (n < MSGSIZE) log_print("Reading %d packet %d\n",n,total_recv);
            total_recv += n;
        } 

        //log_print("Head %d %d\n",numConnect,total_recv);
        assert(total_recv == MSGSIZE);

        auto msg = msg_parser(buffer);
        //msg_print(msg);
        vector<string>::iterator msg_iter= msg.begin();
        int total_lenn = stoi(msg_iter[1]);
 
        total_recv = 0;       
        //log_print("Start %s (%d) \n",data_key.c_str(),total_lenn);

        char *data_ptr = new char[total_lenn * MSGSIZE];

        while(true){
            if (total_recv == total_lenn*MSGSIZE)
                break;
    
            n = read(newsockfd,data_ptr+total_recv,MSGSIZE);
            
            //if (n < 0) error("ERROR reading from socket");
            //else if (n < MSGSIZE) printf("Reading %d packet \n",n);
//            if (n == 0) break;
            total_recv += n ;
        }

        if (total_lenn*MSGSIZE != total_recv){
            log_print("%d != %d \n",total_lenn*MSGSIZE,total_recv);
        }
        assert(total_lenn*MSGSIZE == total_recv);


        end = system_clock::now();

        duration<double> elapsed = end-start;
        throughput = (total_lenn*MSGSIZE)/(1024*1024);
        etime = elapsed.count();
        //log_print("[MSGRECV] %f MB/s (%d / %f)\n",throughput/etime,total_lenn*MSGSIZE,etime);
        //log_print("Recv message %d\n",numConnect);

        //while(false)
        //for (int read_ptr = 0 ; read_ptr < total_recv ; read_ptr += MSGSIZE)
        //start = system_clock::now();
        int read_ptr =0;

        while (read_ptr < total_recv) 
        {
            start = system_clock::now();
           
            memcpy(buffer,data_ptr+read_ptr,sizeof(char)*MSGSIZE);
            read_ptr += MSGSIZE;
            auto msg = msg_parser(buffer);

            vector<string>::iterator iter = msg.begin();
            auto cmd = *iter;
            
            //log_print("Command = %s \n",cmd.c_str());

            if (strcmp(cmd.c_str(),"SEND")==0){

                
                int lenn     = stoi(iter[1]);
                long long data_len = stoll(iter[2]);
                string data_type = iter[3];

                //cout << lenn<< " " << data_len << " "<<data_type <<endl;

                auto data = new vispark_data<char>(data_len);
                data->setDataKey(data_key);

                char* dest_ptr = data->getHostPtr();

                memcpy(dest_ptr,data_ptr + read_ptr,lenn*MSGSIZE*sizeof(char));  
                read_ptr += lenn*MSGSIZE;

                data->htod();
                data_dict.insert(vpair(data_key,data));
                
                checkLastError();
                CUDA_MEM_CHECK();

            }
            else if (strcmp(cmd.c_str(),"SEQ")==0){
                t1 = system_clock::now();

                int lenn     = stoi(iter[1]);
                long long data_len = stoll(iter[2]);
                string data_type = iter[3];

                vector<string> target_list;

                for (int i = 0 ; i < lenn; i++){
                    memcpy(buffer,data_ptr+read_ptr,sizeof(char)*MSGSIZE);
                    read_ptr += MSGSIZE;
                    auto local_msg= msg_parser(buffer);
                    auto local_iter = local_msg.begin();
                    target_list.push_back(local_iter[1]); 
                } 

                /* for ( auto n : target_list) */
                /*     printf("[%s] Combine: %s \n",host_name,n.c_str()); */
                
                data_len = 0;
                
                for ( auto target_key : target_list){
                    auto struct_ptr = data_dict.find(target_key)->second;
                    
                    data_len += struct_ptr->getDataSize();
                }

                auto data = new vispark_data<char>(data_len);
 //               printf("Data len = %d \n",data_len);
                CUDA_MEM_CHECK();

                //char* dest_ptr = data->getHostPtr();
                char* dest_ptr = data->getDevPtr();
                char* source_ptr;

                int copy_off = 0;
                for ( auto target_key : target_list){
                    auto struct_ptr = data_dict.find(target_key)->second;
                    auto source_size = struct_ptr->getDataSize();

                    //printf("%s is %d \n",target_key.c_str(),struct_ptr->inGPU());
                    if (struct_ptr->inGPU() == true){
                        source_ptr = struct_ptr->getDevPtr();
                        cudaMemcpy(dest_ptr + copy_off, source_ptr,source_size*sizeof(char),cudaMemcpyDeviceToDevice);                  
                    }else {
                        source_ptr = struct_ptr->getHostPtr();
                        cudaMemcpy(dest_ptr + copy_off, source_ptr,source_size*sizeof(char),cudaMemcpyHostToDevice);                 
                    }
                    copy_off += source_size;
                }

                string result_key = RandomString("Task_",12,0);
                data ->setDataKey(result_key);
                data ->setInGPU(true); 
                
                data_dict.insert(vpair(result_key,data));
                data_key = result_key;
                
                for ( auto target_key : target_list){
                    auto struct_ptr = data_dict.find(target_key)->second;
 
                    if(struct_ptr->isPersist() == false){
                        delete struct_ptr;
                        data_dict.erase(target_key);
                    }
                }
                //log_print("[SEQ] new key is %s \n",data_key.c_str());

                checkLastError();

 
                t2 = system_clock::now();
                elapsed = t2-t1;
                etime = elapsed.count();
                printf("[%s] SEQ : %f \n",host_name,etime);


                #if 0
                data->dtoh();
                char* host_ptr = data->getHostPtr();
                size_t host_len  = data->getDataSize();

                FILE *pFile;
                pFile = fopen(result_key.c_str(),"w");
                fwrite(host_ptr,sizeof(char),host_len,pFile);
                fclose(pFile);
                #endif
        

                //memcpy(dest_ptr,data_ptr + read_ptr,lenn*MSGSIZE*sizeof(char));  
                //read_ptr += lenn*MSGSIZE;
                /*
                for (int i = 0 ; i < lenn; i++){
                    n = read(newsockfd,data_ptr+i*MSGSIZE,MSGSIZE);
                    if (n < 0) error("ERROR reading from socket");
                    else if (n < MSGSIZE) printf("Reading %d packet \n",n);
                }
                */
                /*
                while (true){
                    n = read(newsockfd,data_ptr+i*MSGSIZE,MSGSIZE);
                    if (n < 0) error("ERROR reading from socket");
                    else if (n < MSGSIZE) printf("Reading %d packet \n",n);
                    if (n == 0) break;
                }   */



            }
            else if (strcmp(cmd.c_str(),"Save")==0){

                t1 = system_clock::now();
 
                string path= iter[1];
                int lenn     = stoi(iter[2]);

                auto struct_ptr = data_dict.find(data_key)->second;
                struct_ptr->dtoh();
                char* host_ptr = struct_ptr->getHostPtr();
                size_t host_len  = struct_ptr->getDataSize();
                if (lenn > 0) host_len = lenn;

                FILE *pFile;
                pFile = fopen(path.c_str(),"w");
                fwrite(host_ptr,sizeof(char),host_len,pFile);
                fclose(pFile);
 
                t2 = system_clock::now();
                elapsed = t2-t1;
                etime = elapsed.count();
                printf("[%s] Save : %f \n",host_name,etime);
                printf("------------------\n");

                if(struct_ptr->isPersist() == false){
                    delete struct_ptr;
                    data_dict.erase(data_key);
                }
            }
#if 0
            else if (strcmp(cmd.c_str(),"CHECK")==0){

                int lenn     = stoi(iter[1]);
                int data_len = stoi(iter[2]);
                string data_type = iter[3];

            
                vector<string> target_list;

                for (int i = 0 ; i < lenn; i++){
                    memcpy(buffer,data_ptr+read_ptr,sizeof(char)*MSGSIZE);
                    read_ptr += MSGSIZE;
                    auto local_msg= msg_parser(buffer);
                    auto local_iter = local_msg.begin();
                    target_list.push_back(local_iter[1]); 
                } 

                //cout<<"REQUIRED"<<endl;
                //for ( auto n : target_list)
                //    printf("[%s] REQURIED %s \n",host_name,n.c_str());


                int missing_cnt = 0;
                
                for ( auto target_key : target_list){
                    auto struct_iter = data_dict.find(target_key);
                    if (struct_iter == data_dict.end())
                        missing_cnt ++;
                }
               
                if (missing_cnt == 0)
                    n = write(newsockfd,"CHECKED",7);
                else 
                    n = write(newsockfd,"NOT",3);
                shutdown(newsockfd,SHUT_WR);
                
                //log_print("MISSING CNT %d \n",missing_cnt);

            }
            else if (strcmp(cmd.c_str(),"FILL")==0){

                int lenn     = stoi(iter[1]);
                int data_len = stoi(iter[2]);
                string data_type = iter[3];

            
                vector<string> target_list;

                for (int i = 0 ; i < lenn; i++){
                    memcpy(buffer,data_ptr+read_ptr,sizeof(char)*MSGSIZE);
                    read_ptr += MSGSIZE;
                    auto local_msg= msg_parser(buffer);
                    auto local_iter = local_msg.begin();
                    target_list.push_back(local_iter[1]); 
                } 

                //cout<<"REQUIRED"<<endl;
                //for ( auto n : target_list)
                //    printf("[%s] REQURIED %s \n",host_name,n.c_str());

                vector<string> missing_list;
                
                for ( auto target_key : target_list){
                    auto struct_iter = data_dict.find(target_key);
                    if (struct_iter == data_dict.end())
                        missing_list.push_back(target_key);  
                }

                //cout<<"MISSING"<<endl;
                for ( auto n : missing_list)
                    printf("[%s] MISSING %s \n",host_name,n.c_str());

    
                if (missing_list.size() > 0){
 
                    auto send_obj =msg_create();
                    string head = "Start 1 END ";
                    string cont = "REQUEST ";

                    for (auto n : missing_list)
                        cont = cont + n + " ";
        
                    cont += "END ";
           
                    //while (head.size() < MSGSIZE)
                    //    head += '0';
 
                    //while (cont.size() < MSGSIZE)
                     //   cont += '0';

   
                    //string send_obj = head+cont;

                    //cout<<send_obj<<endl;
                    //cout<<send_obj.size()<<endl;
                    send_obj.set_head(head);          
                    send_obj.set_msg(cont);          

                    //send_obj.print();

                    //cout<<send_obj.size()<<endl;
                    
                    auto send_ptr = send_obj.ptr();
                    auto send_len = send_obj.size();

                    for (auto address : workers){
    
                        if (strcmp(address.c_str(),host_name) == 0)
                            continue;

                        struct sockaddr_in other_addr;

                        //Create socket
                        auto send_sock = socket(AF_INET , SOCK_STREAM , 0);
                        bzero((char *) &other_addr, sizeof(other_addr));
                        
                        //setup address structure
                        if(inet_addr(address.c_str()) == -1)
                        {
                            struct hostent *he;
                            struct in_addr **addr_list;

                            //resolve the hostname, its not an ip address
                            if ( (he = gethostbyname( address.c_str() ) ) == NULL)
                            {
                                //gethostbyname failed
                                herror("gethostbyname");
                                //cout<<"Failed to resolve hostname\n";

                                return false;
                            }

                            //Cast the h_addr_list to in_addr , since h_addr_list also has the ip address in long format only
                            addr_list = (struct in_addr **) he->h_addr_list;

                            for(int i = 0; addr_list[i] != NULL; i++)
                            {
                                //strcpy(ip , inet_ntoa(*addr_list[i]) );
                                other_addr.sin_addr = *addr_list[i];

                                //cout<<address<<" resolved to "<<inet_ntoa(*addr_list[i])<<endl;
                                break;
                            }
                        }
                        //plain ip address
                        else
                        {
                            other_addr.sin_addr.s_addr = inet_addr( address.c_str() );
                        }

                        other_addr.sin_family = AF_INET;
                        other_addr.sin_port = htons( portno );
                        
                        //cout<<"TRY TO SEND REQUEST to "<<address<<endl;

                        //Connect to remote server
                        if (connect(send_sock , (struct sockaddr *)&other_addr , sizeof(other_addr)) >= 0)
                        {
                            //log_print("Connected %s and %s \n",host_name,address.c_str());
 
                            for (uint offset = 0 ; offset < send_len; offset += MSGSIZE)
                            {
                                n = write(send_sock,send_ptr + offset,MSGSIZE);
                                //if (n < MSGSIZE) error("ERROR reading from socket 1");
                            }
                            shutdown(send_sock,SHUT_WR);
                            close(send_sock);

                        }
                        else
                            ; 
                            //log_print("Fail to Connected %s and %s \n",host_name,address.c_str());
                    }

                }
            }
#endif
            else if (strcmp(cmd.c_str(),"RECV")==0){
                t1 = system_clock::now();

                //            string data_key = iter[1];
                //cout<<"RECV KEY "<< data_key <<endl;
                auto struct_ptr = data_dict.find(data_key)->second;
                if (struct_ptr->inGPU() == true)
                    struct_ptr->dtoh();
                
                checkLastError();

                char* host_ptr = struct_ptr->getHostPtr();
                //cout<<struct_ptr<<endl;
                //uint send_len  = struct_ptr->getMallocSize();
                size_t data_len  = struct_ptr->getDataSize();
               // uint lenn     = send_len / MSGSIZE;


                for (uint offset = 0 ; offset < data_len ; offset += MSGSIZE)
                {
                    uint send_size = min((long long) MSGSIZE,(long long) data_len-offset);
                    n = write(newsockfd,host_ptr + offset,send_size);
                    //if (n < 0) error("ERROR reading from socket 1");
                    //else if (n < MSGSIZE) printf("Sending %d packet \n",n);
                }
                shutdown(newsockfd,SHUT_WR);

                if(struct_ptr->isPersist() == false){
                    delete struct_ptr;
                    data_dict.erase(data_key);
                }
 
                t2 = system_clock::now();
                elapsed = t2-t1;
                etime = elapsed.count();

                printf("[%s] RECV : %f \n",host_name,etime);


                //cout<<"Finish Task : "<<data_key<<endl; 
            }
            else if (strcmp(cmd.c_str(),"VIEWER")==0){
                t1 = system_clock::now();

                //            string data_key = iter[1];
                //cout<<"RECV KEY "<< data_key <<endl;
                int  send_byte = stoi(iter[1]);      
                string location = iter[2];      
                int    loc_port = stoi(iter[3]);
                //cout<<"FILE Name "<< location<<endl;

                cout<<data_key<<" "<<location<<endl;
 
                auto struct_ptr = data_dict.find(data_key)->second;
                if (struct_ptr->inGPU() == true)
                    struct_ptr->dtoh();
                unsigned char* host_ptr = (unsigned char*) struct_ptr->getHostPtr();
                //uint send_len  = struct_ptr->getMallocSize();
                size_t data_len  = struct_ptr->getDataSize();
                data_len = send_byte;
               // uint lenn     = send_len / MSGSIZE;
                    
                checkLastError();

                struct sockaddr_in other_addr;
                auto send_sock = socket(AF_INET , SOCK_STREAM , 0);
                bzero((char *) &other_addr, sizeof(other_addr));

                other_addr.sin_addr.s_addr = inet_addr("192.168.1.11");
                other_addr.sin_family = AF_INET;
                other_addr.sin_port = htons( loc_port );

                if (connect(send_sock , (struct sockaddr *)&other_addr , sizeof(other_addr)) >= 0)
                {
                    for (uint offset = 0 ; offset < data_len; offset += MSGSIZE){
                        uint send_size = min((long long) MSGSIZE,(long long) data_len-offset);
                        n = write(send_sock,host_ptr + offset,send_size);
                    }
                    shutdown(send_sock,SHUT_WR);
                    close(send_sock);
                }
                else
                    ; 

               // n = write(newsockfd,data_key.c_str(),data_key.size());
               // shutdown(newsockfd,SHUT_WR);
        
                t2 = system_clock::now();
                elapsed = t2-t1;
                etime = elapsed.count();
                /*  */
                /* char imagename[100]; */
                /* sprintf(imagename,"/home/smhong/image_%d.raw",debug_run_count); */
                /* debug_run_count+=1; */
                /* printf("Write to %s\n",imagename); */
                /*  */
                /* FILE *pFile; */
                /* pFile = fopen(imagename,"w"); */
                /* fwrite(host_ptr,sizeof(char),send_byte,pFile); */
                /* fclose(pFile); */
                /*  */

                printf("[%s] Viewer : %f \n",host_name,etime);
                printf("------------------\n");
                if(struct_ptr->isPersist() == false){
                    delete struct_ptr;
                    data_dict.erase(data_key);
                }



                //cout<<"Finish Task : "<<data_key<<endl; 
            }

            else if (strcmp(cmd.c_str(),"RUN")==0){
                //log_print("Try RUN\n");
                string result_key = RandomString("Task_",12,0);
                
                t1 = system_clock::now();

                int lenn1     = stoi(iter[1]);
                int code_len  = stoi(iter[2]);
                int lenn2     = stoi(iter[3]);
                long long data_len  = stoll(iter[4]);
                string func_name = iter[5];      
                long long result_len = stoll(iter[6]);
                int kernel_dim = stoi(iter[7]);               
                int forced_free= stoi(iter[8]);               
                vector<int> kernel_info;
            
                for (int i = 0 ; i < kernel_dim ; i++)
                    kernel_info.push_back(stoi(iter[9+i]));
                
    
                int last_off = 8 + kernel_info.size();

                //for (auto i : kernel_info)
                //    cout<<i<<endl;
 
                //checkLastError();
                //cout<<func_name<<" "<<result_len<<endl;
 
                char* code_ptr;
                char* args_ptr;

                //cout<<lenn1<<" "<<code_len<<endl;
                //cout<<lenn2<<" "<<data_len<<endl;

                code_ptr = new char[(lenn1)*MSGSIZE];
                args_ptr = new char[(lenn2)*MSGSIZE];


                memcpy(code_ptr,data_ptr + read_ptr, lenn1*MSGSIZE);
                read_ptr += lenn1*MSGSIZE; 

                memcpy(args_ptr,data_ptr + read_ptr, lenn2*MSGSIZE);
                read_ptr += lenn2*MSGSIZE; 
                /*
                int total_recv = 0;
                for (int i = 0 ; i < lenn1; i++){
                    n = read(newsockfd,code_ptr+i*MSGSIZE,MSGSIZE);
                    total_recv += n ;
                    if (n < 0) error("ERROR reading from socket");
                    else if (n < MSGSIZE) printf("Reading %d packet \n",n);
                    //if (n == 0) break;
                }



                int total_recv1 = 0;
                for (int i = 0 ; i < lenn2; i++){
                    n = read(newsockfd,data_ptr+i*MSGSIZE,MSGSIZE);
                    total_recv1 += n ;
                    if (n < 0) error("ERROR reading from socket");
                    else if (n < MSGSIZE) printf("Reading %d packet \n",n);
                    //if (n == 0) break;
                }

                cout <<total_recv<<" "<<total_recv1<<endl;
                */
                string cuda_code(code_ptr,0,code_len);
                int offset = 0;

                auto args_data = new vector<tuple<string,int,char*>>;

                for (auto arg_iter = iter+last_off ; arg_iter != msg.end() ; arg_iter++)
                {
                    string elem_type = *arg_iter;

                    int elem_len = 1;
                    //cout<< elem_type<<endl;               
 
                    if (strcmp(elem_type.c_str(),"int") == 0)
                        elem_len = 4;
                    else if (strcmp(elem_type.c_str(),"double") == 0)
                        elem_len = 8;
                    else if (strcmp(elem_type.c_str(),"float") == 0)
                        elem_len = 4;
                    else
                        continue;

                    int    elem_num  = stoi(*(arg_iter+1));
                    arg_iter++;
                    //cout<<elem_type<<" "<<elem_len << " " <<elem_num<<" " <<offset<<endl;
                    
                    char* data_read = new char[elem_len * elem_num];
                    memcpy(data_read,args_ptr+offset,elem_num*elem_len*sizeof(char));
 
                    //args_data->push_back(make_pair(n.first,string(data_read)));
                    args_data->push_back(make_tuple(elem_type,elem_num,data_read));
                    offset += elem_num*elem_len;
       
                }
       
                args_dict.insert(make_pair(result_key,args_data));
                checkLastError();


                /***************************************/
                /* CUDA compile */
                /***************************************/
                string filename = RandomString("/tmp/cuda_",16,code_len);
                string cudafile = filename + ".ptx";
                //string ptxfile  = filename + ".ptx";
                //cout<<cudafile<<" "<<cuda_code.size()<<endl;


                ofstream file(cudafile.c_str(),ios::trunc);
                file << cuda_code;
                file.close();

                /*            
                if ( access(ptxfile.c_str(),F_OK) != 0){
                    string command = "nvcc -ptx " + cudafile + " -o " + ptxfile ;
                    ofstream file(cudafile.c_str());
                    file << cuda_code;
                    file.close();
               
                    n = system(command.c_str()); 
                }*/

                /***************************************/
                /* CUDA RUN */
                /***************************************/

                auto result_data = new vispark_data<char>(result_len);
                checkLastError();
                //CUDA_MEM_CHECK();
                result_data->setDataKey(result_key);
             
                //log_print("[RUN] %s -> %s \n",data_key.c_str(),result_key.c_str());
 
                //cout<<data_key<<endl;
                //cout<<result_key<<endl;
 
                auto data_elem = data_dict.find(data_key)->second;
                auto args_elem = args_dict.find(result_key)->second;


                n = GPU_TEST(cudafile.c_str(),func_name.c_str(),result_data,data_elem,args_elem,kernel_info); 
                //cuCtxSynchronize();
                checkLastError();
                
                          
                data_dict.insert(vpair(result_key,result_data));


                t2 = system_clock::now();
                elapsed = t2-t1;
                etime = elapsed.count();
                //printf("[%s] RUN %s -> %s %f\n",host_name,data_key.c_str(),result_key.c_str(),etime);
                //printf("[%s] RUN : %f \n",host_name,etime);

                #if 0
                {
                    if (debug_run_count % 10 ==0){
                        auto target = data_elem;
                        target->dtoh();
                        char* host_ptr = target->getHostPtr();
                        size_t host_len  = target->getDataSize();

                        char imagename[100];
                        sprintf(imagename,"debug/%s_%02d.raw",data_key.c_str(),debug_run_count);

                        FILE *pFile;
                        pFile = fopen(imagename,"w");
                        fwrite(host_ptr,sizeof(char),host_len,pFile);
                        fclose(pFile);
                    }
                    debug_run_count ++;
                }
                #endif

                if(forced_free == true || data_elem->isPersist() == false){
                    //log_print("[%s] Free %s \n",host_name,data_key.c_str());
                    delete data_elem;
                    data_dict.erase(data_key);
                }

                data_key = result_key;
                checkLastError();

            }
            else if (strcmp(cmd.c_str(),"HIT")==0){
                data_key = iter[1];      
                //cout<<"HIT KEY "<< data_key <<endl;
            }
            else if (strcmp(cmd.c_str(),"kill")==0){
                
                MPI_Finalize();
               return 0;
                //cout<<"HIT KEY "<< data_key <<endl;
            }
	    #if EGL
            else if (strcmp(cmd.c_str(),"EGLinit")==0){
                width= stoi(iter[1]);
                height  = stoi(iter[2]);
                long long max_len= stoll(iter[3]);
                struct_size= stoi(iter[4]);

                //if (isEGLvalid == false){
                if (true){
                    isEGLvalid = 1;
                    EGLinit();

                    GLEWinit();    

                    frameBufferinit();
                }
                initGL();
                //initVBO(max_len);
            }
            else if (strcmp(cmd.c_str(),"Render")==0){
                t1 = system_clock::now();
                
                auto struct_ptr = data_dict.find(data_key)->second;


                char *vert_str = NULL;
                char *geo_str = NULL;
                char *frag_str = NULL;
                
                int vert_len = 0;
                int geo_len = 0;
                int frag_len = 0;
                
                int vert_lenn = 0;
                int geo_lenn = 0;
                int frag_lenn = 0;

 
                auto arg_iter = iter+1; 
                while (arg_iter != msg.end())
                {
                    string elem_type = *arg_iter;

                    int elem_len = 1;
                    //cout<< elem_type<<endl;               
 
                    if (strcmp(elem_type.c_str(),"vert") == 0){
                        vert_len  = stoi(*(arg_iter+1));
                        vert_lenn  = stoi(*(arg_iter+2));
                        arg_iter += 2;    
                    }
                    else if (strcmp(elem_type.c_str(),"geo") == 0){
                        geo_len  = stoi(*(arg_iter+1));
                        geo_lenn  = stoi(*(arg_iter+2));
                        arg_iter += 2;    
                    }
                    else if (strcmp(elem_type.c_str(),"frag") == 0){
                        frag_len  = stoi(*(arg_iter+1));
                        frag_lenn  = stoi(*(arg_iter+2));
                        arg_iter += 2;    
                    }
                    else if (strcmp(elem_type.c_str(),"shader_end") == 0){
                        arg_iter +=1;
                        break;
                    }
                    arg_iter++;
                }

     
                int offset = 0;
                auto args_data = new vector<tuple<string,string,int,char*>>;
                char *args_ptr =  data_ptr + read_ptr;
                while ( arg_iter != msg.end()){

                    string elem_name = *(arg_iter+0);
                    if (strcmp(elem_name.c_str(),"args_end") == 0){
                        break;
                    }
                    
                    string elem_type = *(arg_iter+1);

                    int elem_len = 1;
 
                    if (strcmp(elem_type.c_str(),"int") == 0)
                        elem_len = 4;
                    else if (strcmp(elem_type.c_str(),"double") == 0)
                        elem_len = 8;
                    else if (strcmp(elem_type.c_str(),"float") == 0)
                        elem_len = 4;

                    int    elem_num  = stoi(*(arg_iter+2));
                    arg_iter+=3;
 
                    char* data_read = new char[elem_len * elem_num];
                    memcpy(data_read,args_ptr+offset,elem_num*elem_len*sizeof(char));
                    offset += elem_num*elem_len;
 
                    args_data->push_back(make_tuple(elem_name,elem_type,elem_num,data_read));
                    //printf("%s %s %d \n",elem_name.c_str(),elem_type.c_str(),elem_num);
                }
        
                if (args_data->size() > 0) 
                    read_ptr += MSGSIZE; 

                string *vert_code = NULL;
                string *geo_code = NULL;
                string *frag_code = NULL;
                
                if (vert_len > 0){
                    vert_str = new char[vert_lenn*MSGSIZE];
                    memcpy(vert_str,data_ptr + read_ptr, vert_len);
                    read_ptr += vert_lenn*MSGSIZE;

                    vert_code = new string(vert_str,0,vert_len);
                    vert_code->append("\0");
                }
                if (geo_len > 0){
                    geo_str = new char[geo_lenn*MSGSIZE];
                    memcpy(geo_str,data_ptr + read_ptr, geo_len);
                    read_ptr += geo_lenn*MSGSIZE;
                    
                    geo_code = new string(geo_str,0,geo_len);
                    geo_code->append("\0");
                }
                if (frag_len > 0){
                    frag_str = new char[frag_lenn*MSGSIZE];
                    memcpy(frag_str,data_ptr + read_ptr, frag_len);
                    read_ptr += frag_lenn*MSGSIZE;
                    
                    frag_code = new string(frag_str,0,frag_len);
                    frag_code->append("\0");
                }

                if (vert_code != NULL)
                    vert_str = (char*) vert_code->c_str();
                if (geo_code != NULL)
                    geo_str = (char*) geo_code->c_str();
                if (frag_code != NULL)
                    frag_str = (char*) frag_code->c_str();

                //float *projMat = (float*) *(iter+1).c_str();
                float* projMat = (float*) (data_ptr+read_ptr+sizeof(float)*16*0);
                float* modelMat= (float*) (data_ptr+read_ptr+sizeof(float)*16*1);
                float* viewMat = (float*) (data_ptr+read_ptr+sizeof(float)*16*2);
                
                read_ptr += MSGSIZE;
    
                //p[0] = createGLSLProgram( "phong.vert", NULL, "phong.frag" ); // Phong
                p[0] = createGLSLProgram(vert_str, geo_str, frag_str ); // Phong

                int num_vert = h_numvert[0];
                printf("[%s] NUM VERTEX = %u \n",host_name,num_vert);

                renderScene();
                //setVBO(struct_ptr->getDevPtr(),num_vert);
                auto vertexVBOID = struct_ptr->getVBO();
                //cudaGLUnmapBufferObject(vertexVBOID);
                renderObject(vertexVBOID,num_vert,projMat,modelMat,viewMat,*args_data); 

                 
                auto result_data = new vispark_data<char>(width*height*(3+sizeof(float)));
                unsigned char *data2 = (unsigned char*) result_data->getHostPtr();
                float* depth_buffer = (float*) (result_data->getHostPtr() + width*height*3);
                //unsigned char *data2 = new unsigned char[width*height*3];
                glReadBuffer(GL_COLOR_ATTACHMENT0);
                //glReadPixels(0, 0, width, height, GL_BGR, GL_UNSIGNED_BYTE, data2);
                glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, data2);
                assertOpenGLError("glReadPixels");

                //float* depth_buffer = new float[width*height];
                glReadPixels(0,0,width,height,GL_DEPTH_COMPONENT,GL_FLOAT,depth_buffer);
                assertOpenGLError("Read Depth");


                checkLastError();
                CUDA_MEM_CHECK();
                string result_key = RandomString("Task_",12,0);
                result_data->setDataKey(result_key);
             
                log_print("[RENDER] %s -> %s \n",data_key.c_str(),result_key.c_str());
 
                //cout<<data_key<<endl;
                //cout<<result_key<<endl;
 
                auto data_elem = data_dict.find(data_key)->second;

                checkLastError();
           
                data_dict.insert(vpair(result_key,result_data));

                if(data_elem->isPersist() == false){
                    delete data_elem;
                    data_dict.erase(data_key);
                }

                data_key = result_key;
                checkLastError();


                t2 = system_clock::now();
                elapsed = t2-t1;
                etime = elapsed.count();
                printf("[%s] RENDER : %f \n",host_name,etime);


            }
	    #endif
            else if (strcmp(cmd.c_str(),"init")==0){

                //cout<<"Worker Free "<<endl;
                //CUDA_MEM_CHECK();
                
                while (!data_dict.empty()){
                    auto it = data_dict.begin();
                    delete it->second;
                    data_dict.erase(it);
                }
    
                while (!halo_dict.empty()){
                    auto it = halo_dict.begin();
                    delete it->second;
                    halo_dict.erase(it);
                }


                //printf("[%s] halo dict size %d -> %d\n",host_name,val1,val2);

                //debug_run_count = 0;
                //cudaDeviceReset();
                //CUDA_MEM_CHECK();

            }
            else if (strcmp(cmd.c_str(),"ACT")==0){
                int cache_flag     = stoi(iter[1]);

                if (cache_flag > 0)
                    data_dict.find(data_key)->second->setPersist();
                else if (cache_flag < 0)
                {
                    auto it = data_dict.find(data_key);
                    delete it->second;
                    data_dict.erase(it);
                } 
                    
                n = write(newsockfd,data_key.c_str(),data_key.size());  
                //if (n < 0) error("ERROR reading from socket 1");
                //else if (n < MSGSIZE) printf("Sending %d packet \n",n);

                shutdown(newsockfd,SHUT_WR);
                //log_print("ACT End %s (%d) \n",data_key.c_str(),n);
                //cout<<"Finish Task : "<<data_key<<endl; 
            }
#if 0
            else if (strcmp(cmd.c_str(),"REQUEST")==0){

                vector<string> recv_list;

                for (auto n = iter+1; n != msg.end(); n++){

                    auto struct_iter = data_dict.find(*n);
                    if (struct_iter != data_dict.end())
                        recv_list.push_back(*n);
                }

                //cout<<inet_ntoa(cli_addr.sin_addr)<<endl;
                //cout<<cli_addr.sin_addr.s_addr<<endl;

                //for ( auto n : recv_list)
                //    printf("[%s] REQUESTED %s \n",host_name,n.c_str());
                
               

                for ( auto n : recv_list)
                {
                    auto struct_ptr = data_dict.find(n)->second;
                    //struct_ptr->dtoh();
                    //auto stream_iter = struct_ptr->dtoh_stream().begin();
                    auto stream_list = struct_ptr->dtoh_stream();
                    auto stream_iter = stream_list.begin();
                    //auto stream_end = struct_ptr->dtoh_stream().end();
                    int  proc_size = 0;
//                    cudaDeviceSynchronize();
                   
                    /* 
                    for (auto stream_iter : struct_ptr->dtoh_stream()){
                        auto err =  cudaStreamSynchronize(*(stream_iter));
                        if (err != CUDA_SUCCESS) { log_print("Stream error... .\n"); }
                        proc_size += CHUNKSIZE;
                        log_print("PROCESSED %d \n",proc_size);
                    }
                    */

                    //log_print("NUM STREAM %d \n",stream_list.size());
                    int host_len = struct_ptr->getDataSize();
                    char* host_ptr = struct_ptr->getHostPtr();
                    int lenn =  host_len%MSGSIZE == 0 ? host_len/MSGSIZE : host_len/MSGSIZE + 1;

//                    log_print("Send %d/%d data \n",lenn,host_len);

                    auto send_obj =msg_create();
                    string head = "Start "+ to_string(lenn+1) + " END";
                    string cont = "Transfer " + n + " " + to_string(lenn) + " " 
                                +  to_string(host_len) + " END";

                    send_obj.set_head(head);          
                    send_obj.set_msg(cont);         
 
                    struct sockaddr_in other_addr;
                    auto send_sock = socket(AF_INET , SOCK_STREAM , 0);
                    bzero((char *) &other_addr, sizeof(other_addr));
                    
                    other_addr.sin_addr.s_addr = cli_addr.sin_addr.s_addr;
                    other_addr.sin_family = AF_INET;
                    other_addr.sin_port = htons( portno );

                    if (connect(send_sock , (struct sockaddr *)&other_addr , sizeof(other_addr)) >= 0)
                    {
                        char *send_ptr = send_obj.ptr();
                        for (uint offset = 0 ; offset < 2*MSGSIZE; offset += MSGSIZE)
                            n = write(send_sock,send_ptr + offset,MSGSIZE);

                        for (uint offset = 0 ; offset < lenn*MSGSIZE; offset += MSGSIZE){
                            if (offset >= proc_size){
                                auto err =  cudaStreamSynchronize(*(*stream_iter));
                                if (err != CUDA_SUCCESS) { log_print("Stream error...[%d]\n",err); }
                                proc_size += CHUNKSIZE;
                                //log_print("PROCESSED %d \n",proc_size);
                                stream_iter++;
                            }
                            n = write(send_sock,host_ptr + offset,MSGSIZE);
                        }
                        shutdown(send_sock,SHUT_WR);
                        close(send_sock);
                    }
                    else
                        ; 
                       // log_print("Fail to Connected  \n");
                }

            }
            else if (strcmp(cmd.c_str(),"Transfer")==0){

                string recv_key = iter[1];
                int lenn     = stoi(iter[2]);
                int data_len = stoi(iter[3]);
                //string data_type = iter[3];

                //cout << lenn<< " " << data_len << " "<<data_type <<endl;

                auto data = new vispark_data<char>(data_len);

                char* dest_ptr = data->getHostPtr();

                //memcpy(dest_ptr,data_ptr + read_ptr,lenn*MSGSIZE*sizeof(char));  
                //adv_memcpy(dest_ptr,data_ptr + read_ptr,lenn*MSGSIZE*sizeof(char));  
                mt_memcpy(dest_ptr,data_ptr + read_ptr,lenn*MSGSIZE*sizeof(char));  
                //memmove(dest_ptr,data_ptr + read_ptr,lenn*MSGSIZE*sizeof(char));  
                read_ptr += lenn*MSGSIZE;

                //log_print("GET %s (%d) \n",recv_key.c_str(),lenn);
                /*
                for (int i = 0 ; i < lenn; i++){
                    n = read(newsockfd,data_ptr+i*MSGSIZE,MSGSIZE);
                    if (n < 0) error("ERROR reading from socket");
                    else if (n < MSGSIZE) printf("Reading %d packet \n",n);
                }
                */
                /*
                while (true){
                    n = read(newsockfd,data_ptr+i*MSGSIZE,MSGSIZE);
                    if (n < 0) error("ERROR reading from socket");
                    else if (n < MSGSIZE) printf("Reading %d packet \n",n);
                    if (n == 0) break;
                }   */


                data_dict.insert(vpair(recv_key,data));

            }
            else if (strcmp(cmd.c_str(),"TransferHalo")==0){

                int block_idx = stoi(iter[1]);
                int lenn      = stoi(iter[2]);
                int data_len  = stoi(iter[3])/2;
                int send_rank = stoi(iter[4]);
                //string data_type = iter[3];

                //cout << lenn<< " " << data_len << " "<<data_type <<endl;

                auto halo_ptr = new halo_data<char>(data_len);

                char* dest_ptr = halo_ptr->getHostPtr();

#if 0
                memcpy(dest_ptr,data_ptr + read_ptr,lenn*MSGSIZE*sizeof(char));  
                //adv_memcpy(dest_ptr,data_ptr + read_ptr,lenn*MSGSIZE*sizeof(char));  
                //mt_memcpy(dest_ptr,data_ptr + read_ptr,lenn*MSGSIZE*sizeof(char));  
                //memmove(dest_ptr,data_ptr + read_ptr,lenn*MSGSIZE*sizeof(char));  
                read_ptr += lenn*MSGSIZE;
#endif
                            
                MPI_Status status;
                MPI_Request request;
                log_print("[TransferHalo] Try to receive Block_%02d \n",block_idx);
                
                log_print("connected recv block %d from %d\n",block_idx,send_rank);

                MPI_Irecv(dest_ptr, MSGSIZE, MPI_CHAR, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD,&request);

                MPI_Wait(&request, &status);

                log_print("done recv block %d from %d\n",block_idx,send_rank);
                //log_print("GET %s (%d) \n",recv_key.c_str(),lenn);
                /*
                for (int i = 0 ; i < lenn; i++){
                    n = read(newsockfd,data_ptr+i*MSGSIZE,MSGSIZE);
                    if (n < 0) error("ERROR reading from socket");
                    else if (n < MSGSIZE) printf("Reading %d packet \n",n);
                */
                /*
                while (true){
                    n = read(newsockfd,data_ptr+i*MSGSIZE,MSGSIZE);
                    if (n < 0) error("ERROR reading from socket");
                    else if (n < MSGSIZE) printf("Reading %d packet \n",n);
                    if (n == 0) break;
                }   */


                //data_dict.insert(vpair(recv_key,data));
                halo_dict.insert(hpair(block_idx,halo_ptr));
            }
#endif
            else if (strcmp(cmd.c_str(),"Extract")==0){

                string block_name = iter[1];
                int block_idx = stoi(iter[1]);
                long long halo_size  = stoll(iter[2]);

                //log_print("[Extract] %s for Block_%02d [%d] \n",data_key.c_str(),block_idx,halo_size);
                //printf("[%s] Extract for Block_%02d [%d] \n",host_name,block_idx,halo_size);

                auto it = halo_dict.find(block_idx);
                halo_data<char>* halo_ptr;

                if (it != halo_dict.end())
                    halo_ptr = halo_dict.find(block_idx)->second;
                else{  
                    halo_ptr= new halo_data<char>(halo_size);
                    halo_dict.insert(hpair(block_idx,halo_ptr));
                }
                auto data_ptr = data_dict.find(data_key)->second;

                halo_ptr->extract(data_ptr);
                checkLastError();



                //printf("[%s] halo dict size %d -> %d\n",host_name,val1,val2);
                /*
                   char fp_name[50];
                sprintf(fp_name,"block_%d_before.raw",block_idx);
                data_ptr->print(fp_name);
                char fp_name2[50];
                sprintf(fp_name2,"debug/halo_%d_before.raw",block_idx);
                halo_ptr->print(fp_name2);
                */
            }
            else if (strcmp(cmd.c_str(),"Append")==0){

                int block_idx = stoi(iter[1]);
                long long halo_size  = stoll(iter[2]);

                //log_print("[Append] %s for Block_%02d \n",data_key.c_str(),block_idx);

                auto data_ptr = data_dict.find(data_key)->second;
                for (int offset = -1; offset <= 1; offset+=2)
                {
                    int target_idx = block_idx+offset;
                    auto struct_iter = halo_dict.find(target_idx);

                    if (struct_iter != halo_dict.end()){
                        auto halo_ptr = struct_iter->second;

                        halo_ptr->append(data_ptr,offset);
                        //delete halo_ptr;
                        //halo_dict.erase(target_idx);
                        log_print("Append  %d halo for block %d [%s]\n",target_idx,block_idx,data_key.c_str());
                    }else 
                        log_print("Missing %d halo for block %d [%s]\n",target_idx,block_idx,data_key.c_str());
                        //printf("[%s] Missing %d halo for block %d [%s]\n",host_name,target_idx,block_idx,data_key.c_str());
                } 
                checkLastError();
                /*
                char fp_name[50];
                sprintf(fp_name,"block_%d_after.raw",block_idx);
                data_ptr->print(fp_name);
                */
            }
            else if (strcmp(cmd.c_str(),"direct2")==0){
                //log_print("Try direct shuffle \n");
                if (world_rank == 0)
                    printf("\n");
                vector<tuple<string,string>> send_info;
                int block_num= stoi(iter[1]);

                if (block_num > 0){

                    ifstream inFile("/home/smhong/vispark.txt");

                    string elem, dest;
                    for (int i = 0 ; i < block_num ; i++){
                        getline(inFile,elem);
                        getline(inFile,dest);

                        //log_print("Direct send for %s to %s\n",elem.c_str(),dest.c_str());
                        send_info.push_back(make_tuple(elem,dest));
                    }

                    inFile.close();     
                }
                else{
                    for (auto arg_iter = iter+2 ; arg_iter != msg.end() ; arg_iter++)
                    {
                        //int    elem_idx  = stoi(*arg_iter);
                        string elem = *arg_iter;

                        if (strcmp(elem.c_str(),"END")==0)
                            break;

                        //string source = *(arg_iter+1);
                        string dest   = *(arg_iter+1);

                        arg_iter += 1;

                        //log_print("Direct send for %s to %s\n",elem.c_str(),dest.c_str());
                        send_info.push_back(make_tuple(elem,dest));
                    }
                }

                int id_count =0;
                int comm_count = 0;
                for (auto elem : send_info){
                    id_count += 1;

                    auto elem_key = get<0>(elem);
                    auto dest_str    = get<1>(elem);

                    bool isSource = false;
                    bool isDest   = false;
                    int dest = getWorkerRank(dest_str);

                    if (data_dict.find(elem_key) != data_dict.end() && dest != world_rank) isSource = true;
                    if (data_dict.find(elem_key) == data_dict.end() && dest == world_rank) isDest = true;

                    //log_print("Direct send for %s to %s\n",elem_key.c_str(),dest.c_str());

                    //log_print("Worker %d isSend %d , isDest %d for %s \n",world_rank,isSource,isDest,elem_key.c_str());
                    if (isSource || isDest ){

                        int send_amount, recv_amount;
                        MPI_Status status;

                        if (isSource){

                            auto struct_ptr = data_dict.find(elem_key)->second;
                            checkLastError();

                            if (struct_ptr->inGPU())
                                struct_ptr->dtoh();
                            
                             checkLastError();
                        
                            char* host_ptr = struct_ptr->getHostPtr();
                            send_amount = struct_ptr->getDataSize();
 
                            MPI_Send(&send_amount, 1, MPI_INT, dest, 1000 + id_count, MPI_COMM_WORLD);
                            MPI_Send(host_ptr, send_amount, MPI_CHAR, dest, 2000 + id_count, MPI_COMM_WORLD);
                            //log_print("Send %d data to %d \n",send_amount,dest);
                            //printf("[%s] Send %d data to %d \n",host_name,send_amount,dest+1);
            
   //                         comm_count++;
 //                           log_print("Comm count = %d\n",comm_count);
                             
                            checkLastError();

                            if(struct_ptr->isPersist() == false){
                                delete struct_ptr;
                                data_dict.erase(elem_key);
                            }

                            checkLastError();

                        }
                        if (isDest){

                            MPI_Recv(&recv_amount, 1, MPI_INT, MPI_ANY_SOURCE, 1000 + id_count, MPI_COMM_WORLD,&status);

                            auto data = new vispark_data<char>(recv_amount);
                            checkLastError();
                            //CUDA_MEM_CHECK();

                            char* dest_ptr = data->getHostPtr();

 
                            MPI_Recv(dest_ptr, recv_amount, MPI_CHAR, MPI_ANY_SOURCE,2000+ id_count, MPI_COMM_WORLD,&status);
                            //log_print("Recv %d data \n",recv_amount);

                            comm_count++;
                            //log_print("Comm count = %d\n",comm_count);

                            data->setDataKey(elem_key);
                            data->htod();
                            data->setInGPU(true);
                            data_dict.insert(vpair(elem_key,data));
                            checkLastError();
                        }


                    }
                
                }
            }
            else if (strcmp(cmd.c_str(),"direct")==0){
                log_print("Try direct shuffle \n");

                vector<tuple<string,string,string>> send_info;

                for (auto arg_iter = iter+1 ; arg_iter != msg.end() ; arg_iter++)
                {
                    //int    elem_idx  = stoi(*arg_iter);
                    string elem = *arg_iter;

                    if (strcmp(elem.c_str(),"END")==0)
                        break;

                    string source = *(arg_iter+1);
                    string dest   = *(arg_iter+2);
               
                    arg_iter += 2;

                    log_print("Direct send for %s from %s to %s\n",elem.c_str(),source.c_str(),dest.c_str());
                    send_info.push_back(make_tuple(elem,source,dest));
                }

                for (auto elem : send_info){

                    auto elem_key = get<0>(elem);
                    auto source_str  = get<1>(elem);
                    auto dest_str    = get<2>(elem);

                    int source = getWorkerRank(source_str); 
                    int dest   = getWorkerRank(dest_str);

                    if (world_rank == source || world_rank == dest){

                        int send_amount, recv_amount;
                        MPI_Status status;

                        if (world_rank == source){

                            auto struct_ptr = data_dict.find(elem_key)->second;

                            if (struct_ptr->inGPU())
                                struct_ptr->dtoh();
                            checkLastError();
                        
                            char* host_ptr = struct_ptr->getHostPtr();
                            send_amount = struct_ptr->getDataSize();
                    
    
                            MPI_Send(&send_amount, 1, MPI_INT, dest, 123, MPI_COMM_WORLD);
                            MPI_Send(host_ptr, send_amount, MPI_CHAR, dest, 124, MPI_COMM_WORLD);
                            log_print("Send %d data to %d \n",send_amount,dest);

                            if(struct_ptr->isPersist() == false){
                                delete struct_ptr;
                                data_dict.erase(elem_key);
                            }


                        }
                        if (world_rank == dest){

                            MPI_Recv(&recv_amount, 1, MPI_INT, source, 123, MPI_COMM_WORLD,&status);

                            auto data = new vispark_data<char>(recv_amount);
                            checkLastError();
                            CUDA_MEM_CHECK();

                            char* dest_ptr = data->getHostPtr();

 
                            MPI_Recv(dest_ptr, recv_amount, MPI_CHAR, source, 124, MPI_COMM_WORLD,&status);
                            log_print("Recv %d data from %d \n",recv_amount,source);

                            data->setDataKey(elem_key);
                            data->htod();
                            data->setInGPU(true);
                            data_dict.insert(vpair(elem_key,data));
         


                        }


                    }
                
                }
            }
            else if (strcmp(cmd.c_str(),"alive")==0){
                n = write(newsockfd,"DONE",4);
                shutdown(newsockfd,SHUT_WR);
 
            }
            else if (strcmp(cmd.c_str(),"Shuffle")==0){
                //log_print("Recv Shuffle Signal\n");

                checkLastError();
                vector<tuple<int,string,string>> send_info;

                for (auto arg_iter = iter+1 ; arg_iter != msg.end() ; arg_iter++)
                {
                    //int    elem_idx  = stoi(*arg_iter);
                    int elem = stoi(*arg_iter);

                    if (elem < 0) break;

                    string source = *(arg_iter+1);
                    string dest   = *(arg_iter+2);
               
                    arg_iter += 2;
                    
                    //log_print("Direct send for %d from %s to %s\n",elem,source.c_str(),dest.c_str());
                    send_info.push_back(make_tuple(elem,source,dest));
                }

                for (auto elem : send_info){

                    auto elem_key = get<0>(elem);
                    auto source_str  = get<1>(elem);
                    auto dest_str    = get<2>(elem);

                    if (strcmp(dest_str.c_str(),"all") == 0){


                    }else {
                
                        int source = getWorkerRank(source_str); 
                        int dest   = getWorkerRank(dest_str);

                        if (world_rank == source || world_rank == dest){

                            int send_amount, recv_amount;
                            MPI_Status status;

                            if (world_rank == source){
                                //log_print("Try Send %d data to %d \n",send_amount,dest);

                                //auto struct_ptr = data_dict.find(elem_key)->second;
                                auto struct_ptr = halo_dict.find(elem_key)->second;

                                //if (struct_ptr->inGPU())
                                //    struct_ptr->dtoh();

                                char* host_ptr = struct_ptr->getHostPtr();
                                send_amount = struct_ptr->getDataSize();


                                MPI_Send(&send_amount, 1, MPI_INT, dest, 123, MPI_COMM_WORLD);
                                MPI_Send(host_ptr, send_amount*2, MPI_CHAR, dest, 124, MPI_COMM_WORLD);
                                //log_print("Send %d data to %d \n",send_amount,dest);
                            }
                            if (world_rank == dest){

                                MPI_Recv(&recv_amount, 1, MPI_INT, source, 123, MPI_COMM_WORLD,&status);
                                //log_print("Try Recv %d data from %d \n",recv_amount,source+1);
                                //log_print("Try Transfer %d %d \n",source+1,world_rank+1);

                                //auto halo_ptr= new halo_data<char>(recv_amount);
                                halo_data<char>* halo_ptr;
                                auto it = halo_dict.find(elem_key);
                                //CUDA_MEM_CHECK();

                                if (it != halo_dict.end())
                                    halo_ptr = halo_dict.find(elem_key)->second;
                                else{  
                                    halo_ptr= new halo_data<char>(recv_amount);
                                    halo_dict.insert(hpair(elem_key,halo_ptr));
                                }

                                char* dest_ptr = halo_ptr->getHostPtr();


                                MPI_Recv(dest_ptr, recv_amount*2, MPI_CHAR, source, 124, MPI_COMM_WORLD,&status);
                                //log_print("Recv %d data from %d \n",recv_amount,source);
                                //log_print("Done Transfer %d %d \n",source+1,world_rank+1);

                                /* auto it = halo_dict.find(elem_key); */
                                /* if (it != halo_dict.end()){ */
                                /*     delete it->second; */
                                /*     halo_dict.erase(elem_key);   */
                                /* } */
                                /* //halo_dict.insert(vpair(elem_key,data)); */
                                /* halo_dict.insert(hpair(elem_key,halo_ptr)); */
                                /*  */
                            }
                        }
                        //MPI_Barrier(MPI_COMM_WORLD);
                    }

                }
                checkLastError();
            }

            
 
//                for (auto arg_iter = iter+1 ; arg_iter != msg.end() ; arg_iter++)
//                    cout<<*arg_iter<<endl;

#if 0
                for (auto arg_iter = iter+2 ; arg_iter != msg.end() ; arg_iter++)
                {
                    int    elem_idx  = stoi(*arg_iter);
                   
                    if (elem_idx < 0) 
                        break;
 
                    string elem_loc = *(arg_iter+1);
                    arg_iter++;

                    //log_print("Block %d in Node %s \n",elem_idx,elem_loc.c_str());
                    data_locate.insert(make_pair(elem_idx,elem_loc));
                }

                for (int target_off= -1; target_off <= 1; target_off+=2)
                {
                    int target_idx = block_idx+target_off;
                    auto find_iter = data_locate.find(target_idx);
                    if (find_iter == data_locate.end()) continue;

                    auto target_address = data_locate.find(target_idx)->second.c_str();
                    if (strcmp(target_address,host_name) != 0){

                        auto struct_ptr = halo_dict.find(block_idx)->second;

                        int host_len = struct_ptr->getDataSize()*2;
                        char* host_ptr = struct_ptr->getHostPtr();
                        int lenn =  host_len%MSGSIZE == 0 ? host_len/MSGSIZE : host_len/MSGSIZE + 1;
                        
                        log_print("will send block %d [%d] to %s\n",block_idx,lenn,target_address);
                        //lenn = 500;

                        //lenn = 1;
                        lenn = 0;
                        auto send_obj =msg_create();
                        string head = "Start "+ to_string(lenn + 1) + " END";
                        string cont = "TransferHalo " + to_string(block_idx) + " " + to_string(lenn) + " " 
                                    +  to_string(host_len) + " " + to_string(world_rank) + " END";

                        send_obj.set_head(head);          
                        send_obj.set_msg(cont);         
 
                        struct sockaddr_in other_addr;
                        auto send_sock = socket(AF_INET , SOCK_STREAM , 0);
                        bzero((char *) &other_addr, sizeof(other_addr));
 
                        //setup address structure
                        if(inet_addr(target_address) == -1)
                        {
                            struct hostent *he;
                            struct in_addr **addr_list;

                            //resolve the hostname, its not an ip address
                            if ( (he = gethostbyname( target_address ) ) == NULL)
                            {
                                //gethostbyname failed
                                herror("gethostbyname");
                                //cout<<"Failed to resolve hostname\n";

                                return false;
                            }

                            //Cast the h_addr_list to in_addr , since h_addr_list also has the ip address in long format only
                            addr_list = (struct in_addr **) he->h_addr_list;

                            for(int i = 0; addr_list[i] != NULL; i++)
                            {
                                //strcpy(ip , inet_ntoa(*addr_list[i]) );
                                other_addr.sin_addr = *addr_list[i];

                                //cout<<target_address<<" resolved to "<<inet_ntoa(*addr_list[i])<<endl;
                                break;
                            }
                        }
                        //plain ip address
                        else
                        {
                            other_addr.sin_addr.s_addr = inet_addr( target_address );
                        }
                   
                        //other_addr.sin_addr.s_addr = cli_addr.sin_addr.s_addr;
                        other_addr.sin_family = AF_INET;
                        other_addr.sin_port = htons( portno );


                        if (connect(send_sock , (struct sockaddr *)&other_addr , sizeof(other_addr)) >= 0)
                        {
                            log_print("connected send block %d to %s\n",block_idx,target_address);
                            char *send_ptr = send_obj.ptr();
                            for (uint offset = 0 ; offset < 2*MSGSIZE; offset += MSGSIZE)
                                n = write(send_sock,send_ptr + offset,MSGSIZE);

                            for (uint offset = 0 ; offset < lenn*MSGSIZE; offset += MSGSIZE){
                                n = write(send_sock,host_ptr + offset,MSGSIZE);
                            }
                            shutdown(send_sock,SHUT_WR);
                            close(send_sock);

                            MPI_Request request;
                            MPI_Status status;
                              
 
                            int dest_rank= 0;

                            for (auto n : workers){
                                if (strcmp(target_address,n.c_str()) == 0)
                                    break;
                                dest_rank++;
                            }

                            assert(dest_rank < world_size);

                            MPI_Isend(host_ptr, MSGSIZE, MPI_CHAR, dest_rank, 123, MPI_COMM_WORLD,&request);
                            MPI_Wait(&request, &status);

                            log_print("done send block %d to %s[%d]\n",block_idx,target_address,dest_rank);
                        }
                        else
                            log_print("Fail to Connected  \n");

                    }
                    
                }
#endif  
            else {
               log_print("ERROR %s \n",cmd.c_str()); 
            //    break;
            }

//            fflush(stdout); 
 //           cout.flush();
            //log_print("[%s] Recv : %f MB/s (%d / %f)\n",cmd.c_str(),throughput/etime,total_lenn*MSGSIZE,etime);
            end = system_clock::now();
            elapsed = end-start;
            etime = elapsed.count();
          //  log_print("[%s] %f MB/s (%d / %f)\n",cmd.c_str(),throughput/etime,total_lenn*MSGSIZE,etime);
        }
       
        delete[] data_ptr;
 
        close(newsockfd);
        //log_print("Close %d\n",numConnect);
        numConnect++;
        //log_print("EXEC Bandwidth : %f MB/s (%d / %f)\n",throughput/etime,total_lenn*MSGSIZE,etime);

                /*  
        GetTimeDiff(0);
        //Data transfer
        for (int i = 0 ; i < lenn ;i++){
            int offset = i*MSGSIZE;

            n = read(newsockfd,buffer + offset,MSGSIZE);
            if (n < 0) error("ERROR reading from socket");
        } 
        n = write(newsockfd,"I got your message",18);
        if (n < 0) error("ERROR writing to socket");

        clock_gettime(CLOCK_REALTIME, &spec);
        s  = spec.tv_sec;
        ms = round(spec.tv_nsec / 1.0e6); // Convert nanoseconds to milliseconds

        printf("%ld.%03ld\n", (intmax_t)s, ms);
        */
    }
    return 0;

}

