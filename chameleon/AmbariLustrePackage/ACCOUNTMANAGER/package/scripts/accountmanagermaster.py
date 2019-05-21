import sys
import os
import subprocess
import ConfigParser
import time
import random
import commands
import socket
import json
from subtools import SubTools
from resource_management import *
from resource_management.core.resources.system import File, Execute, Directory
from resource_management.libraries.functions.format import format
from resource_management.core.resources.service import Service
from resource_management.core.exceptions import ComponentIsNotRunning
from resource_management.core import shell
from resource_management.libraries.script.script import Script
from ambari_agent import AmbariConfig

class ACCOUNTMANAGERMASTER(Script):
    def __init__(self):
        self.sbtls = SubTools()

    def install(self, env):
        Execute(format("yum -y install expect sshpass"))
        Execute(format("yum -y install openldap-servers openldap-clients"))
        Execute(format("cp /usr/share/openldap-servers/DB_CONFIG.example /var/lib/ldap/DB_CONFIG"))
        Execute(format("chown ldap. /var/lib/ldap/DB_CONFIG"))
        
        file_path = os.path.dirname(os.path.realpath(__file__))
        #Execute(format('cp '+file_path+'/modules/Ambari_View/AccountManager-View-2.0.0.0-SNAPSHOT.war /var/lib/ambari-server/resources/views/ '));
       
        
        self.configure(env);
        print 'Installation complete.'

    def stop(self, env):
        print 'Stop UserSyncServiceMaster'
        check = os.path.exists('/var/run/openldap/slapd.pid');
        if check:
            Execute(format("systemctl stop slapd"));
            Execute(format("systemctl disable slapd"));
        else:
            print "Open LDAP is Not Running"
        # self.sbtls.sp_open('python /var/lib/ambari-agent/cache/stacks/HDP/2.6/services/USERSYNC/package/scripts/daemon-usersync.py stop')
        # Stop your service

    def start(self, env):
        print 'Start UserSyncServiceMaster'
        check = os.path.exists('/var/run/openldap/slapd.pid');
        if check:
            print "Open LDAP is Running"
        else:
            Execute(format("systemctl start slapd"));
            Execute(format("systemctl enable slapd"));
        # self.sbtls.sp_open('python /var/lib/ambari-agent/cache/stacks/HDP/2.6/services/USERSYNC/package/scripts/daemon-usersync.py start')
        # Reconfigure all files
        # Start your service

    # def clientsync(self, env):
    #     self.sbtls.excuteDaemon('clientsync',5800)
    #     print("clientsync!!");

    def status(self, env):
        # check = self.sbtls.sp_open('python /var/lib/ambari-agent/cache/stacks/HDP/2.6/services/USERSYNC/package/scripts/daemon-usersync.py status')
        check = os.path.exists('/var/run/openldap/slapd.pid');
        print check
        if check:
            pass;
        else:
            raise ComponentIsNotRunning;

    def configure(self, env):
        import params;
        ldap_manager_pass = params.ldap_manager_pass;
        ldap_manager_name = params.ldap_manager_name;
        ldap_domain = params.ldap_domain;

        Execute(format("systemctl start slapd"));
        Execute(format("systemctl enable slapd"));
        file_path = os.path.dirname(os.path.realpath(__file__));
        check = self.sbtls.sp_open('sh '+file_path+'/modules/shellscripts/password.sh '+ ldap_manager_pass);
        ssha_pass = check[0].split('\r\n')[3];

        self.sbtls.sp_open('rm -rf /tmp/openldap');
        Execute(format("mkdir /tmp/openldap"));
        f = open("/tmp/openldap/chrootpw.ldif", 'w');
        data = """
# specify the password generated above for "olcRootPW" section
dn: olcDatabase={0}config,cn=config
changetype: modify
add: olcRootPW
olcRootPW: """ + ssha_pass;
        f.write(data);
        f.close();
        Execute(format("ldapadd -Y EXTERNAL -H ldapi:/// -f /tmp/openldap/chrootpw.ldif "));

        Execute(format("ldapadd -Y EXTERNAL -H ldapi:/// -f /etc/openldap/schema/cosine.ldif"));
        Execute(format("ldapadd -Y EXTERNAL -H ldapi:/// -f /etc/openldap/schema/nis.ldif"));
        Execute(format("ldapadd -Y EXTERNAL -H ldapi:/// -f /etc/openldap/schema/inetorgperson.ldif"));


        check = self.sbtls.sp_open('sh '+file_path+'/modules/shellscripts/password.sh '+ ldap_manager_pass);
        ssha_pass = check[0].split('\r\n')[3];
        f = open("/tmp/openldap/chdomain.ldif", 'w');
        data = """# replace to your own domain name for "dc=***,dc=***" section
# specify the password generated above for "olcRootPW" section
dn: olcDatabase={1}monitor,cn=config
changetype: modify
replace: olcAccess
olcAccess: {0}to * by dn.base="gidNumber=0+uidNumber=0,cn=peercred,cn=external,cn=auth"
    read by dn.base="cn="""+ldap_manager_name+""","""+ldap_domain+"""\" read by * none


dn: olcDatabase={2}hdb,cn=config
changetype: modify
replace: olcSuffix
olcSuffix: """+ldap_domain+"""

dn: olcDatabase={2}hdb,cn=config
changetype: modify
replace: olcRootDN
olcRootDN: cn="""+ldap_manager_name+""","""+ldap_domain+"""

dn: olcDatabase={2}hdb,cn=config
changetype: modify
add: olcRootPW
olcRootPW: """+ssha_pass+"""

dn: olcDatabase={2}hdb,cn=config
changetype: modify
add: olcAccess
olcAccess: {0}to attrs=userPassword,shadowLastChange by
    dn=\"cn="""+ldap_manager_name+""","""+ldap_domain+"""" write by anonymous auth by self write by * none
olcAccess: {1}to dn.base="" by * read
olcAccess: {2}to * by dn=\"cn="""+ldap_manager_name+""","""+ldap_domain+"""" write by * read
""";
        f.write(data);
        f.close();
        Execute(format("ldapmodify -Y EXTERNAL -H ldapi:/// -f /tmp/openldap/chdomain.ldif "));

        #ldap_name = ldap_domain.split("=")[1];
        ldap_name = ldap_domain.split('=')[1].split(',')[0];

        f = open("/tmp/openldap/basedomain.ldif", 'w');
        data = """dn: """+ldap_domain+"""
objectClass: top
objectClass: dcObject
objectclass: organization
o: LDAP server
dc: """+ldap_name+"""

dn: cn="""+ldap_manager_name+""","""+ldap_domain+"""
objectClass: organizationalRole
cn: Manager
description: Directory Manager

dn: ou=People,"""+ldap_domain+"""
objectClass: organizationalUnit
ou: People

dn: ou=Group,"""+ldap_domain+"""
objectClass: organizationalUnit
ou: Group""";
        f.write(data);
        f.close();

        Execute(format("ldapadd -x -D cn="+ldap_manager_name+","+ldap_domain+" -w "+ ldap_manager_pass +" -f /tmp/openldap/basedomain.ldif"));
        # self.sbtls.sp_open('sh '+file_path+'/modules/shellscripts/ldapadd.sh '+ ldap_manager_name+','+ldap_domain +' '+ldap_manager_pass);

        self.sbtls.sp_open('sh '+file_path+'/modules/shellscripts/ldapuser.sh '+ldap_domain);
        Execute(format("ldapadd -x -D cn="+ldap_manager_name+","+ldap_domain+" -w "+ldap_manager_pass+" -f /tmp/openldap/ldapuser.ldif"));

if __name__ == "__main__":
    ACCOUNTMANAGERMASTER().execute()
