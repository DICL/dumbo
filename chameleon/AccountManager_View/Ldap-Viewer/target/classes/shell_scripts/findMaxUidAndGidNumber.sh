#!/bin/sh

#변수선언
sshpassword=$1
domain=$2
dc=$3
ManagerPassword=$4
cn=$5

# uidNumber MAX값 출력
sshpass -p $sshpassword ssh -o StrictHostKeyChecking=no $domain ldapsearch -x -b $dc | awk '/uidNumber: / {print $2}' | sort | tail -n 1
# gidNumber MAX값 출력 -> hadoop user gid 값 추출
sshpass -p $sshpassword ssh -o StrictHostKeyChecking=no $domain ldapsearch -x cn=$cn -b $dc | awk '/gidNumber: / {print $2}' | sort | tail -n 1


#sshpass -p $sshpassword ssh -o StrictHostKeyChecking=no $domain ldapsearch -x -b $dc | awk '/gidNumber: / {print $2}' | sort | tail -n 1