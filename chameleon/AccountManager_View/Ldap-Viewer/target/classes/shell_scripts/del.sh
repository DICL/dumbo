#!/bin/sh

sshpassword=$1
domain=$2
dc=$3
ManagerPassword=$4
dn=$5
managerName=$6

sshpass -p $sshpassword ssh -o StrictHostKeyChecking=no $domain ldapdelete -x -w $ManagerPassword -D cn=$managerName,$dc $dn