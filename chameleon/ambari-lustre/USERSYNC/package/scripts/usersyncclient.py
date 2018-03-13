import sys
import os
import subprocess
import ConfigParser
import time
import random
import commands
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

try:
    import requests
except ImportError:
    file_path = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(file_path + '/modules')
    import requests
 
class UserSyncClient(Script):
  def __init__(self):
    self.sbtls = SubTools()

  def install(self, env):
    try:
      Execute(format("yum -y install openldap-clients nss-pam-ldapd "));
    except ExecutionFailed as e:
      print(e)
      pass
   
    self.configure(env);
    print 'install'

  def status(self, env):
    raise ClientComponentHasNoStatus();

  def configure(self, env):
    import params;
    ldapbasedn = params.ldap_domain;
    ldapserver_list = self.getMasterhost();
    ldapserver = ldapserver_list[0];
    Execute(format("authconfig --enableldap --enableldapauth --ldapserver="+ldapserver+" --ldapbasedn=\""+ldapbasedn+"\" --enablemkhomedir --update"));
  
  
  
  def getMasterhost(self):
        
    config = ConfigParser.ConfigParser()
    config.read("/etc/ambari-agent/conf/ambari-agent.ini")
    global ambari_url
    ambari_url = config.get('server','hostname')
    print "Ambari server Hostname : " + ambari_url
    headers = {
      'X-Requested-By': 'ambari',
    }
    r = requests.get('http://'+ambari_url+':8080/api/v1/clusters', headers=headers, auth=('admin', 'admin'))
    j = json.loads(r.text)
    items = j["items"][0]["Clusters"]
    print "Cluster Name : " + items["cluster_name"]
    global cluster_name
    cluster_name = items["cluster_name"]

    # LustreClient HostNames
    # curl -u admin:admin http://192.168.1.194:8080/api/v1/clusters/+cluster_name+/services/LUSTRE/components/LUSTRE_CLIENT
    r = requests.get('http://'+ambari_url+':8080/api/v1/clusters/'+cluster_name+'/services/USERSYNC/components/UserSyncMaster', headers=headers, auth=('admin', 'admin'))
    j = json.loads(r.text)
    result =[]
    for component in j["host_components"]:
      result.append(component["HostRoles"]["host_name"])
    
    return result





if __name__ == "__main__":
  UserSyncClient().execute()