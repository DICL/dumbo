#ifndef ARG_H_VBLKTD9N
#define ARG_H_VBLKTD9N

struct arg {
	char *output_file;
	int out_fd;

	char *inbound;
	char *outbound;
};

void arg_parse(int argc, char **argv, struct arg *arg);
void arg_close(struct arg *arg);

#endif /* end of include guard: ARG_H_VBLKTD9N */
