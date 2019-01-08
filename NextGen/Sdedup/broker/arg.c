#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <argp.h>
#include <errno.h>
#include "arg.h"

void int_parse_or_fail(char key, char *value, struct argp_state *state, int *ret);

static void set_default(struct arg *arg)
{
	arg->output_file = NULL;
	arg->out_fd = 0;
	arg->inbound = NULL;
	arg->outbound = NULL;
}

void arg_close(struct arg *arg)
{
	if (arg->out_fd > 0) {
		close(arg->out_fd);
	}
}

#define NR_MANDATORY_ARG 2
static char args_doc[] = "INBOUND OUTBOUND (EX:> tcp://10.0.0.109:4466 tcp://10.0.0.109:4477";

static struct argp_option options[] = {
	{"output",   'o', "FILE", 0, "output file for zmq_send data", 0},
	{NULL	 ,     0,   NULL, 0, NULL			    , 0}
};

void mandatory_parse(int no, char *value, struct argp_state *state, struct arg* arg)
{       
	switch (no) {
	case 0: 
		arg->inbound = value;
		break;
	case 1: 
		arg->outbound = value;
		break;
	}
}

static error_t parse_opt (int key, char *value, struct argp_state *state)
{
	struct arg *arg = state->input;

	switch (key)
	{
	case 'o':
		arg->output_file = value;
		arg->out_fd = creat(value, 0644);
		if (arg->out_fd < 0) {
			argp_failure(state, -1, errno, "%s file open failed", value); 
		}
		break;

	case ARGP_KEY_ARG:
		if (state->arg_num >= NR_MANDATORY_ARG)
			/* Too many arguments. */
			argp_usage (state);

		mandatory_parse(state->arg_num, value, state, arg);
		break;

	case ARGP_KEY_END:
		if (state->arg_num < NR_MANDATORY_ARG)
			/* Not enough arguments. */
			argp_usage (state);
		break;

	default:
		return ARGP_ERR_UNKNOWN;
	}
	return 0;
}

//// Don't touch below code!!!

static struct argp argp = { options, parse_opt, args_doc, 0 , NULL, NULL, NULL};

void arg_parse(int argc, char **argv, struct arg *arg)
{
	set_default(arg);
	argp_parse (&argp, argc, argv, 0, 0, arg);
}

void int_parse_or_fail(char key, char *value, struct argp_state *state, int *ret)
{ 
	char *tmp = NULL; 

	*ret = strtol(value, &tmp, 10); 
	if (*tmp != 0) { 
		argp_error(state, "'-%c' option's value is not a number\n", key); 
	} 
}
