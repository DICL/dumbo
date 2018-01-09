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

class Master(Script):
  def __init__(self):
    self.sbtls = SubTools()

  def install(self, env):
    
    self.start(env)
    self.mount(env)

    ## init process
    Execute(format("yum install wget libselinux-devel nfs-utils-lib-devel -y"))
    Execute(format("yum groupinstall development tools -y"))
    Execute(format("wget -P /tmp http://scteam.ksc.re.kr/~jhkwak/lustre/client/lustre-client-2.7.0-2.6.32_504.8.1.el6.x86_64.src.rpm"))
    Execute(format("ssh root@localhost -T \"nohup rpmbuild --rebuild --define 'lustre_name lustre-client' --define 'configure_args --disable-server --enable-client' --without servers /tmp/lustre-client-2.7.0-2.6.32_504.8.1.el6.x86_64.src.rpm > /tmp/buildhistory\""))
    Execute(format("yum localinstall /root/rpmbuild/RPMS/x86_64/lustre-iokit-* -y"))
    Execute(format("yum localinstall /root/rpmbuild/RPMS/x86_64/lustre-client-* -y"))
    os.makedirs("/mnt/lustre/hadoop")
    os.system("ssh root@localhost -t \"mkdir -p /mnt/lustre/hadoop\"")
    Execute(format("wget http://repos.fedorapeople.org/repos/dchen/apache-maven/epel-apache-maven.repo -O /etc/yum.repos.d/epel-apache-maven.repo"))
    Execute(format("sed -i s/\$releasever/6/g /etc/yum.repos.d/epel-apache-maven.repo"))
    Execute(format("yum install -y apache-maven git"))
    Execute(format("git clone https://github.com/Seagate/lustrefs"))
    Execute(format("mvn -f lustrefs/ package"))
    Execute(format("/bin/mount -t lustre 192.168.1.196@tcp:/testfs  /mnt/lustre"))
    Execute(format("mkdir /lustre"))
    Execute(format("rm -rf /root/rpmbuild"))
    Execute(format("yes | cp  lustrefs/target/lustrefs-hadoop-0.9.1.jar /usr/hdp/current/hadoop-hdfs-namenode/lib/"))
    Execute(format("yes | cp  lustrefs/target/lustrefs-hadoop-0.9.1.jar /usr/hdp/current/hadoop-yarn-nodemanager/lib/"))
    Execute(format("yes | cp  lustrefs/target/lustrefs-hadoop-0.9.1.jar /usr/hdp/current/hadoop-mapreduce-client/lib/"))
    Execute(format("yes | cp  lustrefs/target/lustrefs-hadoop-0.9.1.jar /usr/hdp/current/hadoop-mapreduce-historyserver/lib/"))

    
    print('install is done.')
  def stop(self, env):
    self.sbtls.sp_open('python /var/lib/ambari-agent/cache/common-services/LUSTRE/0.1.0/package/scripts/daemon-lustre.py stop')

  def start(self, env):
    self.sbtls.sp_open('python /var/lib/ambari-agent/cache/common-services/LUSTRE/0.1.0/package/scripts/daemon-lustre.py start')

  def switchtohdfs(self, env):
    self.sbtls.excuteDaemon('tohdfs')

  def switchtolustrefs(self, env):
    self.sbtls.excuteDaemon('tolustrefs')
    
  def getHostsInfo(self):
    global local_hostname
    local_hostname = socket.gethostname()
    print "Local Hostname : " + local_hostname

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
    r = requests.get('http://'+ambari_url+':8080/api/v1/clusters/'+items["cluster_name"]+'/services/LUSTRE/components/LUSTRE_CLIENT', headers=headers, auth=('admin', 'admin'))
    j = json.loads(r.text)

    result =[]
    for component in j["host_components"]:
      result.append(component["HostRoles"]["host_name"])
    
    return result

  def mount(self, env):
    self.sbtls.excuteDaemon('mount')
    print("mount!!")

  def unmount(self, env):
    self.sbtls.excuteDaemon('umount')
    print("umount!!")

  def decommission(self, env):
    print('Decommission')

  def status(self, env):
    check = self.sbtls.sp_open('python /var/lib/ambari-agent/cache/common-services/LUSTRE/0.1.0/package/scripts/daemon-lustre.py status')
    print check
    if 'not' in str(check):
      raise ComponentIsNotRunning
    pass


if __name__ == "__main__":
  Master().execute()
