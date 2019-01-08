#define _GNU_SOURCE

#include <stdio.h>
#include <pthread.h>
#include <unistd.h>
#include <stdlib.h>

#include "worker.h"

struct thread_data {
	void (*process)(void *item, void *private);
	struct worker *work;
};

void *worker_get(struct worker *work)
{
	void *data = NULL;
	while ((data = lfq_dequeue(&work->ctx)) == NULL) {
		// 데이터 없음
		pthread_mutex_lock(&work->mutex);
		pthread_cond_wait(&work->condition, &work->mutex);
		pthread_mutex_unlock(&work->mutex);
	}

	return data;
}

void worker_put(struct worker *work, void *data)
{
	while (lfq_enqueue(&work->ctx, data) < 0) {
		usleep(1000);
		pthread_yield();
	}

	// 데이터 생김
	pthread_mutex_lock(&work->mutex);
	pthread_cond_signal(&work->condition);
	pthread_mutex_unlock(&work->mutex);
}

static void *while_routine_consumer(void *data)
{
	struct thread_data *dt = data;
	struct worker *work = dt->work;
	void *item = NULL;

	while(1) {
		item = worker_get(work);

		if (item == WORKER_QUIT) {
			break;
		}

		dt->process(item, work->private);
	}

	free(dt);

	return NULL;
}

struct worker *__worker_start(void)
{
	struct worker *worker = malloc(sizeof(struct worker));

	if (worker == NULL) {
		return NULL;
	}

	if (lfq_init(&worker->ctx) < 0) {
		free(worker);
		return NULL;
	}

	pthread_mutex_init(&worker->mutex, NULL);
	pthread_cond_init(&worker->condition, NULL);

	return worker;
}

struct worker *worker_start(void (*func)(void *, void *), void *private)
{
	struct worker *worker = NULL;
	struct thread_data *dt = NULL;

	worker = __worker_start();
	if (worker == NULL) {
		return NULL;
	}

	dt = malloc(sizeof(struct thread_data));
	if (dt == NULL) {
		lfq_clean(&worker->ctx);
		free(worker);
		return NULL;
	}

	worker->private = private;
	dt->process = func;
	dt->work = worker;

	if (pthread_create(&worker->tid, NULL, while_routine_consumer, dt) < 0) {
		free(dt);
		free(worker);
		return NULL;
	}

	return worker;
}

void worker_stop(struct worker *work)
{
	worker_put(work, WORKER_QUIT);

	pthread_join(work->tid, NULL);

	lfq_clean(&work->ctx);

	free(work);
}

struct worker *worker_chain_start(int nr_func, void (*func[])(void *, void *), void *private)
{
	struct worker *prev = private, *cur = NULL;
	int i = 0;

	for (i = 0; i < nr_func; ++i) {
		cur = worker_start(func[nr_func - i - 1], prev); // FIXME 할당 실패시 처리 필요
		prev = cur;
	}

	return cur;
}

void worker_chain_stop(int nr_func, struct worker *work)
{
	struct worker *cur = NULL, *tmp = NULL;
	int i = 0;

	tmp = work;

	for (i = 0; i < nr_func; ++i) {
		cur = tmp;
		tmp = tmp->private;
		worker_stop(cur);
	}
}
