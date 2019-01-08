#ifndef __LFQ_H__
#define __LFQ_H__

struct lfq_node{
	void * data;
	struct lfq_node * next;
};

struct lfq_ctx{
	struct lfq_node * head;
	struct lfq_node * tail;
	int count;
};

int lfq_init(struct lfq_ctx *ctx);
int lfq_clean(struct lfq_ctx *ctx);
int lfq_enqueue(struct lfq_ctx *ctx, void * data);
void * lfq_dequeue(struct lfq_ctx *ctx );
int lfq_cnt(struct lfq_ctx *ctx);
#endif
