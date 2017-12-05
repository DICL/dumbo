#define _GNU_SOURCE
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <pthread.h>
#include <errno.h>
#include <sys/wait.h>
#include <signal.h>
#include <time.h>
#define RAND_BUFFER 1048576
#define IO_SIZE_MAX 1073741824
#define MAX_WAIT_TIME 10
#define NUM_THREADS 4
#ifndef POSIX_FADV_STREAMID
#define POSIX_FADV_STREAMID 8
#endif
struct op {
    long write;
    long offset;
    long size;
};
void zombie_handler()
{    int status;
    wait(&status);
}int main(int argc, char *argv[]) {
    int i, j, k;
    int pid;
    char filename[50];
    char path[50];
    char suffix[50];
    char urandom[RAND_BUFFER];
    int len;
    FILE *commands;
    struct op oparr[100];
    long o_time, num_op, io_start_time;
    long curr_time = 0, wait_time;
    long offset, size;
    char tmpchar;
    int fd;
    int myid;
    int m_factor = 1; //Multiply I/O
    off_t stream_id = -1; //disabled
    int ret;
    struct iovec iov[2048];
    int iov_count;
    struct timespec wait_timespec;
    signal(SIGCHLD, (void *)zombie_handler);
    if (argc > 2) {
        m_factor = atoi(argv[2]);
    }
    // Multi stream enabled
    if (argc > 3) { 
        stream_id = atoi(argv[3]);
    }
    srand(time(NULL));
    sprintf(suffix, "_%05d", rand() % 100000);
    for (i = 0; i < NUM_THREADS; i++){
        pid = fork();
        if (pid == 0) {
            myid = i;
            break;
        }
    }
    if (pid != 0) {
        sleep(3);
        printf("Start waiting\n");
        for(i = 0; i < 10; i++) {
            wait(NULL);
        }
        printf("All finished\n");
        return 0;
    }
    memset(filename, 0, 30);
    sprintf(filename, "%s%d", argv[1], i);
    len = strlen(filename);
    commands = fopen(filename, "r");
    if (commands == NULL) {
        return 0;
    }
    fd = open("/dev/urandom", O_RDONLY);
    read(fd, urandom, RAND_BUFFER);
    for (i = 0; i < RAND_BUFFER; i++) {
        if (urandom[i] == 0) {
            urandom[i] = 'a';
        }
    }
    for (i = 0; i < 1024; i++) {
        iov[i].iov_base = urandom;
        iov[i].iov_len = RAND_BUFFER;
    }
    for (i = 0; i < 200; i++) {
        ret = fscanf(commands, "%s %ld %ld %c %ld ", path, &o_time, &num_op, &tmpchar, &io_start_time);
        if (ret == EOF) break;
        strcat(path, suffix);
        for (j = 0; j < num_op; j++) {
            fscanf(commands, "(%c %ld %ld) ", &tmpchar, &offset, &size);
            oparr[j].write = (tmpchar == 'W')?1:0;
            oparr[j].offset = offset * (long)m_factor;
            oparr[j].size = size * (long)m_factor;
            //printf("myid %d currop %d\n", myid, j);
        }
        wait_time = o_time - curr_time;
        if (wait_time < 0)        printf("otime %ld currtime %ld waittime %ld\n", o_time, curr_time, o_time - curr_time);
        wait_timespec.tv_sec = wait_time / 1000000;
        if (wait_timespec.tv_sec > MAX_WAIT_TIME) {
            wait_timespec.tv_sec = MAX_WAIT_TIME;
        }
        wait_timespec.tv_nsec = (wait_time % 1000000) * 1000;
        nanosleep(&wait_timespec, NULL);
        wait(NULL);
        pid = fork();
        if (pid == 0) {
            fd = open(path, O_WRONLY | O_CREAT | O_TRUNC);
            printf("open %s\n", path);
            if (fd == -1) {
                printf("Bad file descriptor -1 %s\n", strerror(errno));
                return 0;
            }
            if (stream_id != -1) {
                posix_fadvise(fd, stream_id, 0, POSIX_FADV_STREAMID);
fsync(fd);
            }
            //printf("%d fd %d %s\n", myid, fd, strerror(errno));
            wait_time = io_start_time - o_time;
            wait_timespec.tv_sec = wait_time / 1000000;
            if (wait_timespec.tv_sec > MAX_WAIT_TIME) {
                wait_timespec.tv_sec = MAX_WAIT_TIME;
            }
            wait_timespec.tv_nsec = (wait_time % 1000000) * 1000;
            nanosleep(&wait_timespec, NULL);
            for (j = 0; j < num_op; j++) {
                if (wait_time < 0) printf("%d j %d size %d offset %d\n", myid, j, oparr[j].size, oparr[j].offset);
                while (oparr[j].size > IO_SIZE_MAX) {
                    printf("oparr[j].size > IO_SIZE_MAX size %ld\n", oparr[j].size);
                    for (k = 0; k < IO_SIZE_MAX / RAND_BUFFER; k++) {
                        iov[k].iov_len = RAND_BUFFER;
                    }
                    ret = pwritev(fd, iov, IO_SIZE_MAX / RAND_BUFFER, oparr[j].offset);
                    //ret = writev(fd, iov, IO_SIZE_MAX / RAND_BUFFER);
                    if (ret == -1) {
                        printf("%d fd %d %s\n", myid, fd, strerror(errno));
                    }
                    oparr[j].offset += IO_SIZE_MAX;
                    oparr[j].size -= IO_SIZE_MAX;
                }
                iov_count = oparr[j].size / RAND_BUFFER;
                for (k = 0; k < iov_count; k++) {
                    iov[k].iov_len = RAND_BUFFER;
                }
                if (oparr[j].size % RAND_BUFFER != 0) {
                    iov_count++;
                    iov[k].iov_len = oparr[j].size % RAND_BUFFER;
                }
                    
                ret = pwritev(fd, iov, iov_count, (int)oparr[j].offset);
                if (ret == -1) {
                    printf("%d fd %d size %ld offset %ld %s\n", myid, fd, oparr[j].size, oparr[j].offset, strerror(errno));
                    close(fd);
                    return 0;
                }
            }
            close(fd);
            return 0;
        }
        curr_time += wait_time;
    }
    wait(NULL);
    return 0;
}
