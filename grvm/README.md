This is a repository for the project of offloading Java computation to GPU devices.

Instructions for Building and Running 
-------------------------------------

First, edit setup.sh to ensure all required software packages are accessible.
Second, execute "make" to build grvm, a Jikes RVM with GPU acceration
Finally, test the grvm with a few applications.

$ source setup.sh

$ make

$ make test