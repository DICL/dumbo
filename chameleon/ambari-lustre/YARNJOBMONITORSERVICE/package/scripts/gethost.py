#!/usr/bin/env python
# -*- coding: utf-8 -*-
import commands
import sys, os, subprocess, ConfigParser, time, random, commands, socket, json, SocketServer
from socket import *
from select import *
from daemon import Daemon
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
    import requests;
except ImportError:
    file_path = os.path.dirname(os.path.realpath(__file__));
    sys.path.append(file_path+'/modules');
    import requests;


def sp_open(command):
    popen = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    (stdoutdata, stderrdata) = popen.communicate()
    return stdoutdata, stderrdata

# ambari server hostname
config = ConfigParser.ConfigParser()
config.read("/etc/ambari-agent/conf/ambari-agent.ini")
global ambari_url
ambari_url = config.get('server','hostname')
print "Ambari server Hostname : " + ambari_url
# cluster_name
headers = {
  'X-Requested-By': 'ambari',
}
r = requests.get('http://'+ambari_url+':8080/api/v1/clusters', headers=headers, auth=('admin', 'admin'))
j = json.loads(r.text)
items = j["items"][0]["Clusters"]
print "Cluster Name : " + items["cluster_name"]

# LustreClient HostNames
# curl -u admin:admin http://192.168.1.194:8080/api/v1/clusters/bigcluster/services/LUSTRE/components/LUSTRE_CLIENT
r = requests.get('http://'+ambari_url+':8080/api/v1/clusters/'+items["cluster_name"]+'/services/YARNJOBMONITORSERVICE/components/YARNJobMonitorClient', headers=headers, auth=('admin', 'admin'))
j = json.loads(r.text)
result =[]
for component in j["host_components"]:
  result.append(component["HostRoles"]["host_name"])


print result
#### umount ####
check = commands.getstatusoutput('mount | grep lustre')
if len(str(check[1])) == 0:
  print('not mount lustre')     

else:
  # hosts = self.getHostsInfo()
  # hosts = ['cn7','cn8','cn9','gpu']
  hosts = result
  try:
    for hostname in hosts:
      sp_open("ssh root@"+hostname+" -T \"nohup umount /lustre > /tmp/mounthistory \"")
      print('success lustre unmount !')

  except:
    try:
      for hostname in hosts:
        sp_open("ssh root@"+hostname+" -T \"nohup umount -l /lustre > /tmp/mounthistory \"")
        print('success lustre unmount by -l option')

    except Exception, e:
      print('not umount lustre!!')
      raise e


