#!/bin/sh
#사용안함
#변수선언
cn=$1
password=$2

expect <<EOF
spawn ldapadd -x -D cn=$cn -W -f /tmp/openldap/basedomain.ldif
expect "Enter LDAP Password:"
        send "$password\r"
expect eof
EOF