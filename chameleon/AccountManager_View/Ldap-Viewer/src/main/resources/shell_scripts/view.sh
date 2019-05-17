#!/bin/sh
sshpass -p $1 ssh -o StrictHostKeyChecking=no $2 ldapsearch -x $4 -b $3