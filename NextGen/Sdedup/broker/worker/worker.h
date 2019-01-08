#ifndef WORKER_H_PDYIWSVK
#define WORKER_H_PDYIWSVK

#include <sys/types.h>
#include "lfq/lfq.h"

#define WORKER_QUIT ((void *)-1)

struct worker {
	pthread_t tid;
	struct lfq_ctx ctx;
	pthread_mutex_t mutex;
	pthread_cond_t  condition;
	void *private;
};

//struct worker *worker_start(void *(*func)(void *, void *), void *private);
struct worker *worker_start(void (*func)(void *, void *), void *private);
void worker_stop(struct worker *work);
void worker_put(struct worker *work, void *data);
void *worker_get(struct worker *work);
struct worker *__worker_start(void);
struct worker *worker_chain_start(int nr_func, void (*func[])(void *, void *), void *private);
void worker_chain_stop(int nr_func, struct worker *work);

#endif /* end of include guard: WORKER_H_PDYIWSVK */
