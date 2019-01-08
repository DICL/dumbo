#include <stdio.h>
#include <zmq.h>
#include "zhelpers.h"

int main(int argc, char * argv[]) {
	/*    void *context = zmq_init (1);
	      void *sender = zmq_socket (context, ZMQ_PUB);
	      zmq_connect (sender, "tcp://10.0.0.109:4466");
	      */
	void *context = zmq_init (1);
	void *sender = zmq_socket (context, ZMQ_PUSH);
	//int i = 0;
	//for(i=0; i<10; i++) {
	zmq_connect (sender, argv[0]);
	//}
	//s_send(sender, "@SDEDUP_END\n");

	char buf[] = "##SDEDUP_END\n";
	s_send(sender, buf);

	zmq_close (sender);
	zmq_term (context);
}
