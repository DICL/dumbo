#define _GNU_SOURCE

#include <stdio.h>
#include <pthread.h>
#include <unistd.h>
#include <stdlib.h>

#include "worker.h"
#include "elapse.h"

void test_work_consumer(void __attribute__((unused))*data, void __attribute__((unused))*private)
{
//	unsigned long a = (unsigned long)data;
//
//	printf("pr : %p, %lu\n", private, a);
}

void one(void __attribute__((unused))*data, void *private)
{
	worker_put(private, (void*)2);
}

void two(void __attribute__((unused))*private, void __attribute__((unused))*data)
{
}

void test1(void)
{
	unsigned long i = 0;
	struct worker *work1 = worker_start(test_work_consumer, NULL);
	
	for (i = 1; i < 1000; ++i) {
		worker_put(work1, (void *)i);
	}

	worker_stop(work1);
}

void test2(void)
{
	unsigned long i = 0;
	struct worker *work1 = worker_start(test_work_consumer, NULL);
	if (work1 == NULL) {
		printf("work1 init failed\n");
		return ;
	}
	
	for (i = 1; i < 210000000; ++i) {
		worker_put(work1, (void *)i);
	//	printf("%'lu\n", i);
	}

	worker_stop(work1);
}

void chain(void)
{
	unsigned long i = 0;

	void (*func[])(void *, void *) = {
		one,
		two
	};
 	struct worker *work = worker_chain_start(2, func, NULL);
	
	printf("start main\n");
	for (i = 1; i < 1000; ++i) {
		worker_put(work, (void *)i);
//		printf("%'lu\n", i);
	}

	worker_chain_stop(2, work);
}
#include <sys/resource.h>

int main(void)
{
	int mem = 1*1024*1024*1024;
	struct rlimit rl = {mem, mem}; 
	if (setrlimit(RLIMIT_AS, &rl) < 0) {
		printf("error\n");
		return -1;
	}

	struct elapse el;
	elapse_start(el);
	test1();
	elapse_end(el);
	elapse_start(el);
	test2();
	elapse_end(el);

	elapse_start(el);
	chain();
	elapse_end(el);
//
	return 0;
}
