#include<sys/socket.h>
#include<sys/stat.h>
#include<arpa/inet.h>
#include<netinet/in.h>
#include<sys/stat.h>
#include<fcntl.h>
#include<sys/types.h>
#include<sys/epoll.h>
#include<stdio.h>
#include<stdlib.h>
#include<stdint.h>
#include<unistd.h>
#include<strings.h>
#include<errno.h>

#define	BUF_LEN	0x10000

struct infd {
	int fd;
	struct infd *prev;
	struct infd *next;
};

char magic[5] = "S2PD";

ssize_t forceRead(int fd, void *buf, size_t len);

int main(int argc, char **argv) {
	unsigned short port = 1814;
	int qcount, i, j;

	char buf[BUF_LEN];
	ssize_t size;
	size_t remain, toread;

	if( argc < 2 ) {
		fprintf(stderr, "USAGE:./sink <number of sources> <port offset>\n");
		return -1;
	}

	qcount = atoi(argv[1]);

	if( argc > 2 ) {
		port = 1814 + atoi(argv[2]);
	}


	int sockfd, fd, clen = sizeof(struct sockaddr_in);

    struct sockaddr_in sa,ca;
	sockfd = socket(AF_INET, SOCK_STREAM, 0);

	bzero(&sa, sizeof(sa));
	sa.sin_family = AF_INET;
	sa.sin_addr.s_addr = htonl(INADDR_ANY);
	sa.sin_port = htons(port);
	char headerdone = 0;

	if( bind( sockfd, (struct sockaddr *)&sa, sizeof(sa) ) == -1 ) {
		perror("bind error:");
		exit(-1);
	}

	if( listen(sockfd, 32) == -1 ) {
		perror("listen error:");
		exit(-1);
	}

	fprintf(stderr, "A sink instance start. qcount = %d, port = %d\n",qcount, port);
	struct epoll_event ev; 
	ev.events = EPOLLIN;
	int epollfd = epoll_create1(0);

	for( i = 0 ; i < qcount ; i++) {
		fd = accept(sockfd, (struct sockaddr *)&ca, (socklen_t *)&clen);
		if( fd < 0 ) {
			fprintf(stderr,"Error on accepting connection\n");
			exit(-1);
		}

		// handle header
		size = forceRead(fd, &remain, sizeof(remain));
		if( headerdone == 0 ) fprintf(stderr, "Receiving header. header len = %lu (in %ld bytes)\n", fd, remain, size);

		if( size < sizeof(remain) )	{
			fprintf(stderr,"Error on reading header\n");
			exit(-1);
		}

		while( remain > 0 ) {
			toread = (remain < BUF_LEN) ? remain : BUF_LEN;
			size = forceRead(fd, buf, toread );
			if( headerdone == 0 ) {
				write(1, buf, size);
			}
			remain -= size;
		}
		headerdone = 1;

		ev.data.fd = fd;
		if( epoll_ctl( epollfd, EPOLL_CTL_ADD, fd, &ev) ) {
			perror("epoll_ctl");
			exit(-1);
		}
	}
	close(sockfd);

	fprintf(stderr, "Sink[%d] successfully received %d connections\n", port, qcount);

	int evt_cnt;
	struct epoll_event *events = calloc( qcount, sizeof(struct epoll_event)); 
	while( (evt_cnt = epoll_wait(epollfd, events,  qcount, -1) ) > 0 ) {
		for( j = 0 ; j < evt_cnt ; j++) {
			fd = events[j].data.fd;
			size = forceRead(fd, &remain, sizeof(remain));
			if( size == 0 ) {
				ev.data.fd = fd;
				if( epoll_ctl(epollfd, EPOLL_CTL_DEL, fd, &ev) < 0 ) {
					perror("epoll_clt for delete");
				}
				qcount--;
				close(fd);
				continue;
			}
			
			while( remain > 0 ) {
				toread = (remain < BUF_LEN) ? remain : BUF_LEN;
				size = forceRead(fd, buf, toread);
				if( write(1, buf, size) < size ) fprintf(stderr,"fail to write to standard output");
				remain -= size;
			}
		}
	}

	fprintf(stderr,"Sink on port %d complete\n", port);

	close(epollfd);
	return 0;
}

ssize_t forceRead(int fd, void *buf, size_t len)
{
	ssize_t ret = 0, size;
	while( len > 0 && (size = read(fd, buf, len)) != 0) {
		if( size < 0 ) {
			if( errno == EINTR )
				continue;
			perror("forceRead");
			exit(-1);
		}
		len -= size;
		buf += size;
		ret += size;
	}
	return ret;
}
