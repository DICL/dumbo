cmake_minimum_required(VERSION 2.6)
project(main)

add_subdirectory(worker)

find_library(ZMQ
	NAMES zmq
	HINTS "/home/dblab/tools/libzmq/build/lib"
)

add_executable(main test.c arg.c)
target_compile_options(main PRIVATE -g -Wall -Wextra -std=gnu99)
target_link_libraries(main worker ${ZMQ})
