#ifndef ELAPSE_H_A2976XIL
#define ELAPSE_H_A2976XIL

#include <stdio.h>
#include <time.h>

struct elapse {
	struct timespec start, end, diff;
};

static void timespec_diff(struct timespec *start, struct timespec *stop,
		   struct timespec *result)
{
	if ((stop->tv_nsec - start->tv_nsec) < 0) {
		result->tv_sec = stop->tv_sec - start->tv_sec - 1;
		result->tv_nsec = stop->tv_nsec - start->tv_nsec + 1000000000;
	} else {
		result->tv_sec = stop->tv_sec - start->tv_sec;
		result->tv_nsec = stop->tv_nsec - start->tv_nsec;
	}
}

#define elapse_start(el) clock_gettime(CLOCK_REALTIME, &(el).start)

#define elapse_end(el) do { \
	clock_gettime(CLOCK_REALTIME, &(el).end); \
	timespec_diff(&(el).start, &(el).end, &(el).diff);\
	printf("%lld.%.9ld\n", (long long)(el).diff.tv_sec, (el).diff.tv_nsec);\
} while (0)

#define elapse_end_fmt(el, ...) do { \
	printf(__VA_ARGS__); \
	elapse_end(el); \
} while (0);

#endif /* end of include guard: ELAPSE_H_A2976XIL */