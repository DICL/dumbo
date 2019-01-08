#include "lfq.h"
#include <stdlib.h> 
#include <string.h>
#include <errno.h>

int lfq_cnt(struct lfq_ctx *ctx)
{
	return __sync_add_and_fetch(&ctx->count, 0);
}

int lfq_init(struct lfq_ctx *ctx) {
	struct lfq_node * tmpnode = calloc(1,sizeof(struct lfq_node));
	if (!tmpnode) 
		return -errno;
	
	memset(ctx,0,sizeof(struct lfq_ctx));
	ctx->head=ctx->tail=tmpnode;
	return 0;
}

int lfq_clean(struct lfq_ctx *ctx){
	if ( ctx->tail && ctx->head ) { // if have data in queue
		struct lfq_node * walker = ctx->head, *tmp;
		while ( walker != ctx->tail ) { // while still have node
			tmp = walker->next;
			free(walker);
			walker=tmp;
		}
		free(ctx->head); // free the empty node
		memset(ctx,0,sizeof(struct lfq_ctx));
	}
	return 0;
}

int lfq_enqueue(struct lfq_ctx *ctx, void * data) {
	struct lfq_node * p;
	struct lfq_node * tmpnode = calloc(1,sizeof(struct lfq_node));
	if (!tmpnode)
		return -errno;
	
	tmpnode->data=data;
	do {
		p = ctx->tail;
		if ( __sync_bool_compare_and_swap(&ctx->tail,p,tmpnode)) {
			p->next=tmpnode;
			break;	
		}
	} while(1);
	__sync_add_and_fetch( &ctx->count, 1);
	return 0;
}

void * lfq_dequeue(struct lfq_ctx *ctx ) {
	void * ret=0;
	struct lfq_node * p;
	do {
		p = ctx->head;
	} while(p==0 || !__sync_bool_compare_and_swap(&ctx->head,p,0));
	
	if( p->next==0)	{
		ctx->head=p;
		return 0;
	}
	ret=p->next->data;
	ctx->head=p->next;
	__sync_sub_and_fetch( &ctx->count, 1);
	free(p);
	return ret;
}
