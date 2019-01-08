if (NOT TARGET lfq)
	add_library(lfq lfq.c)
	target_compile_options(lfq PRIVATE -g -Wall -std=gnu99 -O0)
endif()
