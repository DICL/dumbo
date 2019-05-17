#!/bin/sh

#변수선언
sshpassword=$1
domain=$2
dc=$3
ManagerPassword=$4
uid=$5
cn=$6
sn=$7
userPassword=$8
loginShell=$9
uidNumber=${10}
gidNumber=${11}
homeDirectory=${12}
managerName=${13}

sshpass -p $sshpassword ssh -o StrictHostKeyChecking=no $domain ldapadd -x -D cn=$managerName,$dc -w $ManagerPassword << EOF
dn: uid=$uid,ou=People,$dc
objectClass: inetOrgPerson
objectClass: posixAccount
objectClass: shadowAccount
cn: $cn
sn: $sn
userPassword: $userPassword
loginShell: $loginShell
uidNumber: $uidNumber
gidNumber: $gidNumber
homeDirectory: $homeDirectory
EOF
# dn: cn=$uid,ou=Group,$dc
# objectClass: posixGroup
# cn: $cn
# gidNumber: $gidNumber
# memberUid: $uid
# EOF