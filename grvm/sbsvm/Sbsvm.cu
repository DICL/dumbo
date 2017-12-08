#include "org_sbsvm_Sbsvm.h"

#include <cstdlib>
#include <iostream>
#include <vector>

#include <cuda.h>

#include "cptr.h"
#include "example.h"


#define BUFFER_SIZE 8192


static CUdevice device;
static CUcontext context;
static std::vector<CUmodule> module;
static std::vector<CUfunction> function;
static std::vector<CUstream> stream;

extern "C" {
/*
 * Class:     org_sbsvm_Sbsvm
 * Method:    initialize
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_org_sbsvm_Sbsvm_initialize
  (JNIEnv *env, jobject o)
{
  // Initialize
  cuInit(0);

  // Get number of devices supporting CUDA
  int deviceCount = 0;
  cuDeviceGetCount(&deviceCount);
  if (deviceCount == 0) {
    std::cerr << "There is no device supporting CUDA." << std::endl;
    std::exit(0);
  }

  // Get handle for device 0
  cuDeviceGet(&device, 0);

  // Create context
  cuCtxCreate(&context, 0, device);
    
  cpu_pointer::initialize();
  
  cuCtxPopCurrent(&context);
}

/*
 * Class:     org_sbsvm_Sbsvm
 * Method:    finalize
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_org_sbsvm_Sbsvm_finalize
  (JNIEnv *env, jobject o)
{
  cpu_pointer::finalize();
}


/*
 * Class:     org_sbsvm_Sbsvm
 * Method:    clear
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_org_sbsvm_Sbsvm_clear
  (JNIEnv *env, jobject o)
{
  cpu_pointer::clear_cache();
}

/*
 * Class:     org_sbsvm_Sbsvm
 * Method:    run
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_org_sbsvm_Sbsvm_run
  (JNIEnv *env, jobject o)
{
  cpu_pointer::run_handler();
}

/*
 * Class:     org_sbsvm_Sbsvm
 * Method:    loadModule
 * Signature: (Ljava/nio/ByteBuffer;)J
 */
JNIEXPORT jlong JNICALL Java_org_sbsvm_Sbsvm_loadModule
  (JNIEnv *env, jobject o, jobject image)
{
  const void *pImage = env->GetDirectBufferAddress(image);
  
  cuCtxPushCurrent(context);
  CUmodule m;
  CUjit_option options[3];
  void* values[3];
  char error_log[BUFFER_SIZE];
  options[0] = CU_JIT_ERROR_LOG_BUFFER;
  values[0]  = (void*)error_log;
  options[1] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
  values[1]  = (void*)BUFFER_SIZE;
  options[2] = CU_JIT_TARGET_FROM_CUCONTEXT;
  values[2]  = 0;
  int err = cuModuleLoadDataEx(&m, pImage, 3, options, values);
  if (err != CUDA_SUCCESS) {
    std::cerr << "Link error:" << std::endl << error_log << std::endl;
    std::exit(0);
  }
  cuCtxPopCurrent(&context);
  
  module.push_back(m);
  return module.size()-1;
}

/*
 * Class:     org_sbsvm_Sbsvm
 * Method:    getFunction
 * Signature: (JLjava/nio/ByteBuffer;)J
 */
JNIEXPORT jlong JNICALL Java_org_sbsvm_Sbsvm_getFunction
  (JNIEnv *env, jobject o, jlong moduleID, jobject name)
{
  char *pName = reinterpret_cast<char*>(env->GetDirectBufferAddress(name));
  CUfunction f;
  cuCtxPushCurrent(context);
  cuModuleGetFunction(&f, module[moduleID], pName);
  cuCtxPopCurrent(&context);
  function.push_back(f);
  return function.size()-1;
}

/*
 * Class:     org_sbsvm_Sbsvm
 * Method:    createStream
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_org_sbsvm_Sbsvm_createStream
  (JNIEnv *env, jobject o)
{
  CUstream s;
  cuCtxPushCurrent(context);
  cuStreamCreate(&s, CU_STREAM_NON_BLOCKING);
  cuCtxPopCurrent(&context);
  stream.push_back(s);
  return stream.size()-1;
}

/*
 * Class:     org_sbsvm_Sbsvm
 * Method:    launchKernel
 * Signature: (JJJJJJJJJ)V
 */
JNIEXPORT void JNICALL Java_org_sbsvm_Sbsvm_launchKernel
  (JNIEnv *env, jobject o, jlong functionID, jlong gridDimX, jlong gridDimY, jlong gridDimZ, jlong blockDimX, jlong blockDimY, jlong blockDimZ, jlong sharedMemBytes, jlong streamID)
{
  cuCtxPushCurrent(context);
  cuLaunchKernel(function[functionID], gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, stream[streamID], nullptr, nullptr);
  cuCtxPopCurrent(&context);
}

/*
 * Class:     org_sbsvm_Sbsvm
 * Method:    test
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_org_sbsvm_Sbsvm_test
  (JNIEnv *env, jobject o)
{
  float (*A)[MAT_SIZE] = new float[MAT_SIZE][MAT_SIZE];
  float (*B)[MAT_SIZE] = new float[MAT_SIZE][MAT_SIZE];
  float (*C)[MAT_SIZE] = new float[MAT_SIZE][MAT_SIZE];
  dim3 block(32, 8);
  dim3 grid(MAT_SIZE/block.x+(MAT_SIZE%block.x!=0), MAT_SIZE/block.y+(MAT_SIZE%block.y!=0));
  gpu_client<<<grid, block>>>(A, B, C);
}

} // extern "C"
