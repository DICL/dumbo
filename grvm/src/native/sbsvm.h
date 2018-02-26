#ifndef _SBSVM_H_
#define _SBSVM_H_

#include <stddef.h>

struct _sbsvmcontext;
typedef struct _sbsvmcontext *sbsvmcontext;

extern int sbsvm_validate_code(
    const char *ptxcode, int ptxlen);
extern sbsvmcontext sbsvm_open(size_t size_queue);
extern void sbsvm_close(sbsvmcontext cxt);
extern void sbsvm_execute(
    sbsvmcontext ctx, const char *ptxcode, int ptxclen,
    int from, int to, void **kparams);

#endif
