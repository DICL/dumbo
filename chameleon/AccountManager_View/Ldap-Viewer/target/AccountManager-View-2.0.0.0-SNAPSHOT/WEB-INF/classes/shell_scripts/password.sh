#!/bin/sh

#변수선언
sshpassword=$1
domain=$2
password=$3

expect <<EOF
spawn sshpass -p $sshpassword ssh -o StrictHostKeyChecking=no $domain slappasswd
expect "New password:"
	send "$password\r"
expect "Re-enter new password:"
	send "$password\r"
expect eof
EOF