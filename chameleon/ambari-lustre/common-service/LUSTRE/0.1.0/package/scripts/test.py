# -*- coding: utf-8 -*-
import sys
import os
import subprocess
import ConfigParser
import time
import random
import commands
import socket
import requests
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

def getHostsInfo():
  
  # local_hostname = socket.gethostname()
  # print "Local Hostname : " + local_hostname
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
  global cluster_name
  cluster_name = items["cluster_name"]

  # LustreClient HostNames
  # curl -u admin:admin http://192.168.1.194:8080/api/v1/clusters/+cluster_name+/services/LUSTRE/components/LUSTRE_CLIENT
  r = requests.get('http://'+ambari_url+':8080/api/v1/clusters/'+cluster_name+'/services/LUSTRE/components/LUSTRE_CLIENT', headers=headers, auth=('admin', 'admin'))
  j = json.loads(r.text)
  result =[]
  for component in j["host_components"]:
    result.append(component["HostRoles"]["host_name"])
  
  return result
def subprocess_open(command):
  popen = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
  (stdoutdata, stderrdata) = popen.communicate()
  return stdoutdata, stderrdata
#### mount ####
check = commands.getstatusoutput('mount | grep lustre')
if len(str(check[1])) == 0:      
  # for hostname in ['1.1.1.1','2.2.2.2']:
  hosts = getHostsInfo()
  # hosts = ['cn7','cn8','cn9','gpu']
  for hostname in hosts:
    subprocess_open("ssh root@"+hostname+" -T \"nohup mount -t lustre mds1i@o2ib:/lust_fs /lustre > /tmp/mounthistory \"")

    try:
      subprocess_open("ssh root@"+hostname+" -T \"nohup mkdir -p /lustre/temp_disk/"+hostname+"\"")
    except Exception, e:
      raise e
    subprocess_open("ssh root@"+hostname+" -T \"nohup mount -B /lustre/temp_disk/"+hostname+" /tmp2 > /tmp/mounthistory \"")

else:
  print('already mount')

