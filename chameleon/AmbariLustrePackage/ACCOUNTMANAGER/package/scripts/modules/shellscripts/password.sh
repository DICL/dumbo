#!/bin/sh

#변수선언
password=$1

expect <<EOF
spawn slappasswd
expect "New password:"
        send "$password\r"
expect "Re-enter new password:"
        send "$password\r"
expect eof
EOF