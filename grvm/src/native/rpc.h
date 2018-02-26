#ifndef _RPC_H_
#define _RPC_H_

#include <cuda.h>

#define PAGE_BIT (12u)
#define PAGE_SIZE (1u << PAGE_BIT)
#define PAGE_MASK (~((long)PAGE_SIZE - 1))

struct Tag {
	void *hpage;
	unsigned int ref_counts;
	int lock;
	int residence;
	char dirty;
};

enum operation {
  NONE,
  READ,
  WRITE,
};

struct Message {
	void *haddress;
	void *daddress;
	unsigned int size;
	volatile enum operation state;
};

struct _sbsvmcontext {
  struct {
    size_t size_queue;
    struct Message *messages;
    unsigned int next_msg_index;
    int *locks;
  } rpc;
  struct {
    int size;
    char (*pages)[PAGE_SIZE];
    struct Tag *tag;
  } pcache;
  CUcontext cucxt;
  CUevent end;
  CUstream stream;
};

#endif
