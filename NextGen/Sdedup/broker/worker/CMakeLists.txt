cmake_minimum_required(VERSION 2.6)
project(main)

#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})
add_subdirectory(lfq)

add_library(worker worker.c)
target_compile_options(worker PRIVATE -g -Wall -Wextra -std=gnu99)
target_link_libraries(worker lfq pthread)

add_executable(test_worker test.c)
target_compile_options(test_worker PRIVATE -g -Wall -Wextra -std=gnu99)
target_link_libraries(test_worker worker)
