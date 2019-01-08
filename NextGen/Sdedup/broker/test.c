#include <stdbool.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h>
#include <pthread.h>



#include "zhelpers.h"
//#include "int_str_table.h"
#include "worker/worker.h"
#include "stdatomic.h"
#include "uthash.h"
#include "arg.h"

// BUFFER
struct buffer {
	char *data;
	uint64_t capacity; // const
	uint64_t usage;
	uint64_t offset;
	void *private;
};

void buffer_reset(struct buffer *buf)
{
	buf->offset = 0;
	buf->usage = 0;
}

int buffer_init(struct buffer *buffer, void *private, uint64_t capacity)
{
	buffer->data = malloc(capacity);
	if (buffer->data == NULL) {
		return -1;
	}
	buffer->private = private;
	buffer->capacity = capacity;

	buffer_reset(buffer);

	return 0;
}

void buffer_deinit(struct buffer *buffer)
{
	free(buffer->data);
}

int buffer_send(struct buffer *buf)
{
	printf("send\n");
	ssize_t size = 0;
	int i = 0;
	if (buf->usage == buf->offset) {
		return -1;
	}

	
	i = s_sendmore (buf->private, "foo");
	if ( i < 0 ) {
		perror("");
		fprintf(stderr, "sendmore data failed\n");
	}
//	size = s_send (private, data);
	size = zmq_send (buf->private, &buf->data[buf->offset], buf->usage - buf->offset, 0);
	if ( size < 0 ) {
		perror("");
		fprintf(stderr, "send data failed\n");
		return -1;
	}

	buf->offset += size;

	return 0;
}

int buffer_push(struct buffer *buf, void *data, uint64_t len)
{
	int r = 0;
	if (buf->usage + len > buf->capacity) {
		r = buffer_send(buf);
		if (r < 0) {
			return -1;
		}

		if (buf->usage == buf->offset) {
			buffer_reset(buf);
		} else { // unlink
			memmove(buf->data, &buf->data[buf->offset], buf->usage - buf->offset);
			buf->usage -= buf->offset;
			buf->offset = 0;
		}
	}

	memcpy(&buf->data[buf->usage], data, len);
	buf->usage += len;

	return 0;
}

struct buffer g_buffer;
// END BUFFER

struct hash {
	uint64_t key;
	UT_hash_handle hh;
};

#define HASH_FIND_ULL(head,findint,out)                                          \
	HASH_FIND(hh,head,findint,sizeof(uint64_t),out)

#define HASH_ADD_ULL(head,intfield,add)                                          \
	HASH_ADD(hh,head,intfield,sizeof(uint64_t),add)

enum {HASH_SUCCESS, HASH_COLLSION};

struct hash *hash;

pthread_rwlock_t lock;
atomic_ullong nr_collision;
struct arg arg;

int add_hash(uint64_t key)
{
	struct hash *new;

	if (pthread_rwlock_rdlock(&lock) != 0) {
		perror("");
		abort();
	}
	HASH_FIND_ULL(hash, &key, new);
	pthread_rwlock_unlock(&lock);
	if (new == NULL) {
		new = malloc(sizeof(struct hash));
		assert(new);
		new->key = key;

		if (pthread_rwlock_wrlock(&lock) != 0) {
			perror("");
			abort();
		}
		HASH_ADD_ULL(hash, key, new);
		pthread_rwlock_unlock(&lock);
		return HASH_SUCCESS;
	} else {
		if (new->key != key) {
			fprintf(stderr, "collision original %lu new %lu\n", new->key, key);
		}

		atomic_fetch_add(&nr_collision, 1);
		return HASH_COLLSION;
	}
}

void delete_all_hash(void)
{
	struct hash *cur, *tmp;
	HASH_ITER(hh, hash, cur, tmp) {
		HASH_DEL(hash, cur);  /* delete it (users advances to next) */
		free(cur);            /* free it */
	}
}


#define MAX_BUFFER_SIZE 1048576

int dcnt = 0;
int ucnt = 0;
int IDX = 0;
atomic_ullong nr_records; // have a key

atomic_ullong push; // have a key
atomic_ullong pop; // have a key
char *value_null;

int checkFlag(int flag, int bits) 
{ 
	return ((flag & bits) != 0); 
}

int isConcordant(int flag) 
{ 
	return checkFlag(flag, 0x2); 
}

int isDiscordant(int flag) 
{ 
	return !isConcordant(flag); 
}

int isUnmapped(int flag) 
{ 
	return checkFlag(flag, 0x4); 
}

char *parse_line(char *buf) {
	char *delim = NULL;
	char *next = NULL;
	char *nnext = NULL;
	char *flag_next = NULL;
	unsigned long long flag;
	int is_collision = 0;
	off_t ret_off = 0;
	off_t sp_ret = 0;
	uint64_t key;
	char *ret = NULL;
	int is_unmmaped = 1;
	int hash_ret = 0;
	//int is_unmmaped = 0;

	if (buf[0] == '\0') {
		return 0;
	}

	// filtering
	// 1. @로 시작
	if (buf[0] == '@') {
		return 0;
	}

	char *buff = strdup(buf);
	assert(buff);

	// 2. tab을 기준으로 3번째가 *인것
	// MIXING with processing
	// end filtering

	// processing
	delim = strstr(buff, "XXQQ");

	if (delim == NULL) {
		fprintf(stderr, "처리 못함 %s\n", buf);
		free(buff);
		return NULL;
	}
	atomic_fetch_add(&nr_records, 1);

	delim[0] = '\0';
	next = &delim[4]; // printing을 위한 fix

	{ // nnext의 수명
		nnext = next;

		while (*++nnext != '\t') {
			// tab 1
		}

		nnext[0] = '\0';

		++nnext;
		flag = strtoull(nnext, &flag_next, 10);
	}

	flag_next[0] = '\0';
	++flag_next;

	if (flag_next[0] == '*' && flag_next[1] == '\t') {
		// filtering
		// 2. tab을 기준으로 3번째가 *인것
		free(buff);
		return NULL;
	}

	key = strtoull(buff, NULL, 10);
	hash_ret = add_hash(key);
	if (hash_ret == HASH_SUCCESS) {
		free(buff);
		return NULL;
	} else {
		is_collision = 1;
		flag |= 0x400;
	}

	// 여기서부터는 일단 collision이 발생함
	ret = malloc(strlen(buf) + 10);
	assert(ret);

	sp_ret = sprintf(&ret[ret_off], "%sXXQQ", buff);
	ret_off += sp_ret;

	delim = strstr(&flag_next[3], "XXQQ");
	if (delim == NULL) {
		sp_ret = sprintf(&ret[ret_off], "%s\t%llu\t%s\n", next, flag, flag_next);
		if (is_unmmaped) {
			is_unmmaped = isUnmapped(flag);
		}

		free(buff);
		if (is_unmmaped) {
			free(ret);
			ret = NULL;
		} else {
			//atomic_fetch_add(&nr_collision, 1);
		}

		ret_off += sp_ret;
		return ret;
	}

	do {
		delim[0] = '\0';
		sp_ret = sprintf(&ret[ret_off], "%s\t%llu\t%sXXQQ", next, flag, flag_next);
		//is_unmmaped |= isUnmapped(flag);
		if (is_unmmaped) {
			is_unmmaped = isUnmapped(flag);
			//is_unmapped |= isUnmapped(flag)
		}
		ret_off += sp_ret;

		next = &delim[4];

		{ // nnext의 수명
			nnext = next;

			while (*++nnext != '\t') {
				// tab 1
			}

			nnext[0] = '\0';

			++nnext;
			flag = strtoull(nnext, &flag_next, 10);
			if (is_collision) {
				flag |= 0x400;
			}
		}

		flag_next[0] = '\0';
		++flag_next;
	} while ((delim = strstr(&flag_next[3], "XXQQ")) != NULL);

	sp_ret = sprintf(&ret[ret_off], "%s\t%llu\t%s\n", next, flag, flag_next);
	ret_off += sp_ret;

	if (is_unmmaped) {
		free(ret);
		ret = NULL;
	} else {
		//atomic_fetch_add(&nr_collision, 1);
	}
	free(buff);
	return ret;
}

void parse(void *data, void *private)
{
	char *str = data;
	char *ptr = NULL;
	size_t len = 0;
	while ((ptr = strsep(&str, "\n")) != NULL) {
		char * nl = parse_line(ptr);
		if (nl) {
		} else {
			if (ptr[0] == '\0') {
				//printf("h\n");
				continue;
			}
			len = strlen(ptr);
			nl = malloc(len+2);
			assert(nl);
			memcpy(nl, ptr, len);
			nl[len] = '\n';
			nl[len+1] = '\0';
		}
		atomic_fetch_add(&push, 1);
		worker_put(private, nl);
	}

	free(data);
}

void sender(void *data, void __attribute__((unused)) *private)
{
	//char *buf = data;
	atomic_fetch_add(&pop, 1);

	buffer_push(&g_buffer, data, strlen(data));
//	r = s_sendmore (private, "foo");
//	if ( r < 0 ) {
//		perror("");
//		fprintf(stderr, "sendmore data failed\n");
//	}
//	r = s_send (private, data);
//	if ( r < 0 ) {
//		perror("");
//		fprintf(stderr, "send data \"%s\" failed\n", (char *)data);
//	}

	if (arg.out_fd > 0) {
		write(arg.out_fd, data, strlen(data));
	}

	free(data);
}

#define NR_THREAD 18
#define RB_NEXT(x, max) (((x)+(1))%(max))

int main (int argc, char *argv[])
{
	arg_parse(argc, argv, &arg);
	atomic_store(&nr_records, 0);
	atomic_store(&nr_collision, 0);
	atomic_store(&push, 0);
	atomic_store(&pop, 0);

	if (pthread_rwlock_init(&lock,NULL) != 0) {
		perror("");
		abort();
	}

	char *buff = malloc(MAX_BUFFER_SIZE);
	struct worker* worker[NR_THREAD];
	struct worker* send_worker = NULL;
	char *ret = NULL;
	int i = 0;
	int rr = 0;
	int zmq_ret = 0;
	assert(buff);


	void *context = zmq_ctx_new ();
	void *responder = zmq_socket (context, ZMQ_PULL);
	int rc = zmq_bind (responder, arg.inbound);
	assert (rc == 0);
	void *publisher = NULL;
	int pc = 0;
	publisher = zmq_socket (context, ZMQ_PUB);
	pc = zmq_bind (publisher, arg.outbound);
	//assert (pc == 0);
	if (pc < 0) {
		perror("");
		return -1;
	}
	buffer_init(&g_buffer, publisher, 1024*1024);

	send_worker = worker_start(sender, publisher);

	for (i = 0; i < NR_THREAD; ++i) {
		worker[i] = worker_start(parse, send_worker);
	}

	while (1) {
		zmq_ret = zmq_recv (responder, buff, MAX_BUFFER_SIZE, 0);
		if (zmq_ret < 0) {
			continue;
		}
		buff[zmq_ret] = '\0';

		if (zmq_ret == 13) {
			if (!memcmp("##SDEDUP_END\n", buff, 13)) {
				break;
			}
		} 

		ret = strdup(buff);
		assert(ret);

		worker_put(worker[rr], ret);
		rr = RB_NEXT(rr, NR_THREAD);
	}

	for (i = 0; i < NR_THREAD; ++i) {
		worker_stop(worker[i]);
	}

	atomic_fetch_add(&push, 1);
	worker_put(send_worker, buff);
	worker_stop(send_worker);

	printf("last buffer send\n");
	buffer_send(&g_buffer);
	
	s_sendmore(publisher, "foo");
	s_send(publisher, "##SDEDUP_END\n");


	delete_all_hash();

	printf("collision %llu / record %llu\n", atomic_load(&nr_collision), atomic_load(&nr_records));
	printf("push %llu / pop %llu\n", atomic_load(&push), atomic_load(&pop));

	arg_close(&arg);

	buffer_deinit(&g_buffer);

	zmq_close(publisher);
	zmq_close(responder);
	return 0;
}
