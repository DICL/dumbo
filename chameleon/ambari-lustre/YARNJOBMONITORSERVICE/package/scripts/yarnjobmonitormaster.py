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
    import requests;
except ImportError:
    file_path = os.path.dirname(os.path.realpath(__file__));
    sys.path.append(file_path+'/modules');
    import requests;

class YARNJobMonitorMaster(Script):
    def __init__(self):
        self.sbtls = SubTools()

    def install(self, env):
        print 'Install YARNJobMonitorMaster.'

        # Load the all configuration files
        config = Script.get_config()
        # Bind to a local variable
        # HadoopLustreAdapterMgmtService_user = config['configurations']['lustrefs-config-env']['HadoopLustreAdapterMgmtService_user']

        # Install packages
        self.install_packages(env)

        # Create a new user and group
        # Execute( format("groupadd -f {HadoopLustreAdapterMgmtService_user}") )
        # Execute( format("id -u {HadoopLustreAdapterMgmtService_user} &>/dev/null || useradd -s /bin/bash {HadoopLustreAdapterMgmtService_user} -g {HadoopLustreAdapterMgmtService_user}") )

        ### Continue installing and configuring your service

        print 'Installation complete.'

    def stop(self, env):
        print 'Stop YARNJobMonitorMaster'
        # Stop your service

        #Since we have not installed a real service, there is no PID file created by it.
        #Therefore we are going to artificially remove the PID.
        # Execute( "rm -f /tmp/HadoopLustreAdapterMgmtService.pid" )
        file_path = os.path.dirname(os.path.realpath(__file__))
        # self.sbtls.sp_open('python /var/lib/ambari-agent/cache/stacks/HDP/2.4/services/YARNJOBMONITORSERVICE/package/scripts/daemon-yarnjobmonitor.py stop')
        self.sbtls.sp_open('python '+file_path+'/daemon-yarnjobmonitor.py stop')

    def start(self, env):
        print 'Start YARNJobMonitorMaster'
        file_path = os.path.dirname(os.path.realpath(__file__))
        # Reconfigure all files
        # Start your service

        #Since we have not installed a real service, there is no PID file created by it.
        #Therefore we are going to artificially create the PID.
        # Execute( "touch /tmp/HadoopLustreAdapterMgmtService.pid" )
        # self.sbtls.sp_open('python /var/lib/ambari-agent/cache/stacks/HDP/2.4/services/YARNJOBMONITORSERVICE/package/scripts/daemon-yarnjobmonitor.py start')
        self.sbtls.sp_open('python '+file_path+'/daemon-yarnjobmonitor.py start')

    # def status(self, env):
    #     print 'Status of HadoopLustreAdapterMgmtService'
    #     HadoopLustreAdapterMgmtService_pid_file = "/tmp/HadoopLustreAdapterMgmtService.pid"
    #     #check_process_status(dummy_master_pid_file)
    #     Execute( format("cat {HadoopLustreAdapterMgmtService_pid_file}") )
    #     pass

    # def switchtohdfs(self, env):
    #     self.sbtls.excuteDaemon('sample02',5679)
    #     print("switchtohdfs!!")

    # def switchtolustrefs(self, env):
    #     print 'init'
    #     self.sbtls.excuteDaemon('sample-time',5679)
    #     print 'init-done'
    #     print("switchtolustrefs!!")
    
    # def getHostsInfo(self):
    #     global local_hostname
    #     local_hostname = socket.gethostname()
    #     print "Local Hostname : " + local_hostname

    # # ambari server hostname
    #     config = ConfigParser.ConfigParser()
    #     config.read("/etc/ambari-agent/conf/ambari-agent.ini")
    #     global ambari_url
    #     ambari_url = config.get('server','hostname')
    #     print "Ambari server Hostname : " + ambari_url

    #     # cluster_name
    #     headers = {
    #         'X-Requested-By': 'ambari',
    #     }
    #     r = requests.get('http://'+ambari_url+':8080/api/v1/clusters', headers=headers, auth=('admin', 'admin'))
    #     j = json.loads(r.text)
    #     items = j["items"][0]["Clusters"]
    #     print "Cluster Name : " + items["cluster_name"]

    #     # LustreClient HostNames
    #     # curl -u admin:admin http://192.168.1.194:8080/api/v1/clusters/bigcluster/services/LUSTRE/components/LUSTRE_CLIENT
    #     r = requests.get('http://'+ambari_url+':8080/api/v1/clusters/'+items["cluster_name"]+'/services/YARNJOBMONITORSERVICE/components/YARNJobMonitorMaster', headers=headers, auth=('admin', 'admin'))
    #     j = json.loads(r.text)
    
    #     result =[]
    #     for component in j["host_components"]:
    #       result.append(component["HostRoles"]["host_name"])
        
    #     return result

    # def mount(self, env):
    #     self.sbtls.excuteDaemon('mount',5679)
    #     print("mount!!")

    # def unmount(self, env):
    #     self.sbtls.excuteDaemon('umount',5679)
    #     print("umount!!")

    def status(self, env):
        file_path = os.path.dirname(os.path.realpath(__file__))
        check = self.sbtls.sp_open('python '+file_path+'/daemon-yarnjobmonitor.py status')
        # check = self.sbtls.sp_open('python /var/lib/ambari-agent/cache/stacks/YARNJOBMONITORSERVICE/0.1.0/package/scripts/daemon-yarnjobmonitor.py status')
        print check
        if 'not' in str(check):
            raise ComponentIsNotRunning
        pass

if __name__ == "__main__":
    YARNJobMonitorMaster().execute()
