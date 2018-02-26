#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <jni.h>
#include "sbsvm.h"

#define DEBUG (0)

static jfieldID fid_integer_value = NULL;
static jfieldID fid_paramobject_value = NULL;

static jfieldID getInstanceField(
    JNIEnv *env, const char *fqcname,
    const char *fname, const char *fdesc) {
    jclass cls;
    jfieldID fid;

    cls  = (*env)->FindClass(env, fqcname);
    assert(cls != NULL);
    fid=(*env)->GetFieldID(env, cls, fname, fdesc);
    assert(fid != NULL);
    if (DEBUG) {
      printf("fid = %ld for %s.%s\n",
           (uintptr_t)fid, fqcname, fname);
    }
    return fid;
}

static void **kparam_create(JNIEnv *env, jbyteArray ptypes, jobjectArray pvalues) {
  void **kparams;
  jbyte *_ptypes;
  int i, n;

  n = (*env)->GetArrayLength(env, ptypes);  
  kparams = (void **) malloc(sizeof(void *) * (n + 1));
  kparams[n] = NULL;
  _ptypes = (*env)->GetByteArrayElements(env, ptypes, NULL);
  for(i = 0; i < n;i++) {
    jbyte tdesc;
    jobject value;

    tdesc = _ptypes[i];
    value = (*env)->GetObjectArrayElement(env, pvalues, i);
    switch (tdesc) {
    case 'I': {
      jint ivalue;

      ivalue = (*env)->GetIntField(env, value, fid_integer_value);
      kparams[i]=malloc(sizeof(int));
      *(int *)kparams[i] = ivalue;
      if (DEBUG) printf("kparam[%d]: %d\n", i, ivalue);
      break;
    }
    case 'A': {
      jlong jvalue;

      jvalue = (*env)->GetLongField(env, value, fid_paramobject_value);
      kparams[i]=malloc(sizeof(void *));
      *(void **)kparams[i] = (void *)jvalue;
      if (DEBUG) printf("kparam[%d]: 0x%012lx\n", i, (uintptr_t)jvalue);
      break;
    }
    default:
      assert(0);
      break;
    }
  }
  (*env)->ReleaseByteArrayElements(env, ptypes, _ptypes, JNI_ABORT);
  return kparams;
}

static void kparam_release(void **kparams) {
  int i;

  for(i = 0;kparams[i] != NULL;i++) {
    free(kparams[i]);
  }
  free(kparams);
}
static void copyJavaByteArray(
    JNIEnv *env, jbyteArray code,
    int *ptxcodelen, char **ptxcode) {
  int balen;
  jboolean isCopy;
  jbyte *barray;
  char *arrayCopy;

  balen = (*env)->GetArrayLength(env, code);
  arrayCopy = malloc(balen);
  barray = (*env)->GetByteArrayElements(env, code, &isCopy);
  assert(barray != NULL);
  memcpy(arrayCopy, barray, balen);
  (*env)->ReleaseByteArrayElements(env, code, barray, JNI_ABORT);
  *ptxcodelen = balen;
  *ptxcode = arrayCopy;
}

JNIEXPORT void JNICALL Java_org_grvm_dispatch_Dispatcher_init
(JNIEnv *env, jclass cls)  {
    fid_integer_value = getInstanceField(env,
        "java/lang/Integer", "value", "I");
    fid_paramobject_value =getInstanceField(env,
        "org/grvm/dispatch/ParamObject", "value", "J");
}

JNIEXPORT void JNICALL Java_org_grvm_dispatch_Dispatcher__1launch
  (JNIEnv *env, jclass cls, jbyteArray code,
   jbyteArray ptypes, jint from, jint to,
   jobjectArray pvalues) {
  sbsvmcontext ctx;
  char *ptxcode;
  int ptxcodelen;
  void **kparams;

  kparams = kparam_create(env, ptypes, pvalues);
  copyJavaByteArray(env, code, &ptxcodelen, &ptxcode);

  if (DEBUG) {
    int r = sbsvm_validate_code(ptxcode, ptxcodelen);
    assert(r == 0);
  }

  ctx = sbsvm_open(1024);
  sbsvm_execute(ctx, ptxcode, ptxcodelen, from, to, kparams);
  sbsvm_close(ctx);
  kparam_release(kparams);
  free(ptxcode);
}
