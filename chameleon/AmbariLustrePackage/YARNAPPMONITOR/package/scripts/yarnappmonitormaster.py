import sys
import os
import subprocess
import ConfigParser
import time
import random
import commands
import socket
import json
import signal
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

class YARNAppMonitorMaster(Script):
    def __init__(self):
        self.sbtls = SubTools()

    def install(self, env):
        print 'Install YARNAppMonitorMaster.'

        # Load the all configuration files
        config = Script.get_config()
        # Bind to a local variable
        # HadoopLustreAdapterMgmtService_user = config['configurations']['lustrefs-config-env']['HadoopLustreAdapterMgmtService_user']
        file_path = os.path.dirname(os.path.realpath(__file__))
        #Execute(format('cp '+file_path+'/modules/Ambari_View/Metric-Registry-View-2.0.0.0-SNAPSHOT.war /var/lib/ambari-server/resources/views/ '));
        # install timescaledb
        # copy chameleon web
        # copy yarnmonitor view
        
        print 'Installation complete.'

    def stop(self, env):
        print 'Stop YARNAppMonitorMaster'
        file_path = os.path.dirname(os.path.realpath(__file__))
        try:
            f = open("/tmp/chameleon.pid", 'r')
            pid = f.readlines()[0]
            os.remove("/tmp/chameleon.pid")
            os.kill(int(pid), signal.SIGKILL)
            f.close()
            pass
        except Exception as e:
            pass

        #Execute(format('kill -9 `cat /tmp/chameleon.pid` > /dev/null'));
        #self.sbtls.sp_open('python '+file_path+'/daemon-yarnappmonitor.py stop')

    def start(self, env):
        print 'Start YARNAppMonitorMaster'
        file_path = os.path.dirname(os.path.realpath(__file__))
        # try:
        #     process1 = subprocess.Popen(['nohup', '/usr/jdk64/jdk1.8.0_112/bin/java' , '-jar' , file_path+'/modules/chameleon/chameleon-0.0.1-SNAPSHOT.jar'],stdout=open('/dev/null', 'w'),stderr=open('logfile.log', 'a'),preexec_fn=os.setpgrp)
        #     f = open("/tmp/chameleon.pid", 'w')
        #     f.write(str(process1.pid))
        #     f.close()
        #     pass
        # except Exception as e:
        #     pass
        
        self.sbtls.sp_open('python '+file_path+'/daemon-yarnappmonitor.py start')

    def status(self, env):
        file_path = os.path.dirname(os.path.realpath(__file__))
        check = self.sbtls.sp_open('python '+file_path+'/daemon-yarnappmonitor.py status')
        # check = self.sbtls.sp_open('python /var/lib/ambari-agent/cache/stacks/YARNAPPMONITOR/0.1.0/package/scripts/daemon-yarnappmonitor.py status')
        print check
        if 'not' in str(check):
            raise ComponentIsNotRunning
        pass
        
        # check = os.path.exists('/tmp/chameleon.pid');
        # print check
        # if check:
        #     pass;
        # else:
        #     raise ComponentIsNotRunning;

if __name__ == "__main__":
    YARNAppMonitorMaster().execute()
