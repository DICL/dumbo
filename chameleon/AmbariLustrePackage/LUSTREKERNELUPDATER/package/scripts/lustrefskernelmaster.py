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

try:
    import requests
except ImportError:
    file_path = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(file_path + '/modules')
    import requests

class LustreFSKernelMaster(Script):
    def __init__(self):
        self.sbtls = SubTools()

    def install(self, env):
        print 'Installation complete.'

    def stop(self, env):
        print 'Stop LustreFSKernelMaster'
        # Stop your service
        self.sbtls.sp_open('python /var/lib/ambari-agent/cache/stacks/HDP/2.6/services/LUSTREKERNELUPDATER/package/scripts/daemon-lustrefs-kernel.py stop')

    def start(self, env):
        print 'Start LustreFSKernelMaster'
        # Reconfigure all files
        # Start your service
        self.sbtls.sp_open('python /var/lib/ambari-agent/cache/stacks/HDP/2.6/services/LUSTREKERNELUPDATER/package/scripts/daemon-lustrefs-kernel.py start')

    def reboot(self, env):
        # self.sbtls.excuteDaemon('reboot',5700)

        hosts = self.getHostsInfo()
        for hostname in hosts:
            try:
                result_uname = subprocess.check_output (format("ssh root@"+hostname+" -T \"uname -a\"") , shell=True)
                if 'lustre' in result_uname:
                    print("lustre", result_uname)
                else:
                    Execute(format("ssh root@"+hostname+" -T \"reboot \""))
                    
            except ExecutionFailed as e:
                print(e)
                


        for hostname in hosts:
            while True:
                response = os.system("ping -c 1 " + hostname);
                if response == 0:
                    print hostname, 'Reboot successful!'
                    break
                else:
                    print hostname, 'Rebooting..!'


            while True:
                try:
                    Execute(format("ssh root@"+hostname+" -T \"echo 'ssh run'\""))
                    break
                except ExecutionFailed as e:
                    print(e)
                    continue

            try:
                Execute(format("ssh root@"+hostname+" -T \"mkdir /etc/ambari-metrics-monitor/conf\""))
            except ExecutionFailed as e:
                print(e)
            
            try:
                Execute(format("ssh root@"+hostname+" -T \"mkdir -m 0755 /var/log/ambari-metrics-monitor\""))
            except ExecutionFailed as e:
                print(e)
            
            try:
                Execute(format("ssh root@"+hostname+" -T \"mkdir -m 0755 /var/run/ambari-metrics-monitor\""))
            except ExecutionFailed as e:
                print(e)


            Execute(format("ssh root@"+hostname+" -T \"chown ams:hadoop /var/log/ambari-metrics-monitor\""))
            Execute(format("ssh root@"+hostname+" -T \"chown ams:hadoop /etc/ambari-metrics-monitor/conf\""))
            Execute(format("ssh root@"+hostname+" -T \"/var/lib/ambari-agent/ambari-sudo.sh chown -R ams:hadoop /var/log/ambari-metrics-monitor\""))
            Execute(format("ssh root@"+hostname+" -T \"chown ams:hadoop /var/run/ambari-metrics-monitor\""))

                

            Execute(format("ssh root@"+hostname+" -T \"/usr/sbin/ambari-metrics-monitor --config /etc/ambari-metrics-monitor/conf start\""))

            result_uname = subprocess.check_output (format("ssh root@"+hostname+" -T \"uname -a\"") , shell=True)
            print result_uname

        print("LustreFS kernel reboot!!")

    def status(self, env):
        check = self.sbtls.sp_open('python /var/lib/ambari-agent/cache/stacks/HDP/2.6/services/LUSTREKERNELUPDATER/package/scripts/daemon-lustrefs-kernel.py status')
        print check
        if 'not' in str(check):
            raise ComponentIsNotRunning
        pass

    def getHostsInfo(self):    
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
        r = requests.get('http://'+ambari_url+':8080/api/v1/clusters/'+cluster_name+'/services/LUSTREKERNELUPDATER/components/LustreFSKernelClient', headers=headers, auth=('admin', 'admin'))
        j = json.loads(r.text)
        result =[]
        for component in j["host_components"]:
          result.append(component["HostRoles"]["host_name"])
        
        return result

    def check_version(self, env):
        hosts = self.getHostsInfo()
        for hostname in hosts:
            try:
                result_uname = subprocess.check_output (format("ssh root@"+hostname+" -T \"uname -a\"") , shell=True)
                print(hostname, result_uname)                
            except ExecutionFailed as e:
                print(e)


if __name__ == "__main__":
    LustreFSKernelMaster().execute()
