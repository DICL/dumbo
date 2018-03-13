# -*- coding: utf-8 -*-
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

class LustreMDSMgmtService(Script):
    def __init__(self):
        self.sbtls = SubTools()
    def install(self, env):
        import mdt_params;

        mds_wget = mdt_params.mds_wget;

        # File(
        #     os.path.join("/var/lib/ambari-agent","LustreMDSMgmtService_status.conf"),
        #     owner='root',
        #     group='root',
        #     mode=0644,
        #     content="status=0",
        # )


        print 'Install LustreMDSMgmtService.'

        # Load the all configuration files
        config = Script.get_config()
        # Bind to a local variable
        #LustreMDSMgmtService_user = config['configurations']['my-config-env']['lustremdsmgmtservice_user']


        # Install packages
        self.install_packages(env)
        Execute(format("yum install -y wget, libesmtp libyaml net-snmp-agent-libs opensm opensm-libs sg3_utils tcl tk"))


        for pkg in mds_wget:
            download_file = 'wget '+pkg['url']
            Execute(format(download_file))

        pkg_file = 'yum install -y'
        for pkg in mds_wget:
            pkg_file = pkg_file +' '+ pkg['name'] + ' '

        try:
            Execute(format(pkg_file))
        except ExecutionFailed as e:
            print(e)
            pass

        print 'Installation complete.'


        self.configure(env)
        # file_path = os.path.dirname(os.path.realpath(__file__))
        # self.sbtls.sp_open('python '+file_path+'/daemon-lustre-mds.py start')
    def stop(self, env):
        print 'Stop LustreMDSMgmtService'
        # Stop your service

        #Since we have not installed a real service, there is no PID file created by it.
        #Therefore we are going to artificially remove the PID.
        # Execute( "rm -f /tmp/LustreMDSMgmtService.pid" )
        file_path = os.path.dirname(os.path.realpath(__file__))
        self.sbtls.sp_open('python '+file_path+'/daemon-lustre-mds.py stop')

    def start(self, env):
        print 'Start LustreMDSMgmtService'
        # Reconfigure all files
        # Start your service

        #Since we have not installed a real service, there is no PID file created by it.
        #Therefore we are going to artificially create the PID.
        # Execute( "touch /tmp/LustreMDSMgmtService.pid" )
        file_path = os.path.dirname(os.path.realpath(__file__))
        self.sbtls.sp_open('python '+file_path+'/daemon-lustre-mds.py start')

    def status(self, env):
        # print 'Status of LustreMDSMgmtService'
        # LustreMDSMgmtService_pid_file = "/tmp/LustreMDSMgmtService.pid"
        # #check_process_status(dummy_master_pid_file)
        # Execute( format("cat {LustreMDSMgmtService_pid_file}") )
        file_path = os.path.dirname(os.path.realpath(__file__))
        check = self.sbtls.sp_open('python '+file_path+'/daemon-lustre-mds.py status')
        print check
        if 'not' in str(check):
            raise ComponentIsNotRunning
        pass

    def mountmdts(self, env):
        # self.sbtls.excuteDaemon('sample02',5680)
        print("mountmdts!!")

    def unmountmdts(self, env):
        # self.sbtls.excuteDaemon('sample02',5680)
        print("unmountmdts!!")

    def mkfsmdts(self, env):
        # self.sbtls.excuteDaemon('sample02',5680)
        print("mkfsmdts!!")


    def configure(self, env):
        print 'LustreMDSMgmtService Configure start....'
        import mdt_params;
        config = Script.get_config();
        mdt_server_device = mdt_params.mdt_server_device;

        local_hostname = mdt_params.local_hostname;
        mdt_fsname = mdt_params.mdt_fsname;
        mdt_index = mdt_params.mdt_index;
        mdt_mount = mdt_params.mdt_mount;

        modprobe_networks = mdt_server_device['network_mds'];

        Lustrecontent = format('options lnet networks="'+modprobe_networks+'"')
        File(
            os.path.join("/etc/modprobe.d","lustre.conf"),
            owner='root',
            group='root',
            mode=0644,
            content=Lustrecontent,
        )
        Execute(format('modprobe lustre'))
        Execute(format('lsmod | egrep "lustre|lnet"'))
        Execute(format( 'mkfs.lustre --fsname='+mdt_fsname+'  --mdt --mgs --index='+mdt_index+' '+mdt_server_device['device_mds'] ))

        Directory(mdt_mount, create_parents=True, owner='root', group='root')

        Execute(format('mount -t lustre '+mdt_server_device['device_mds']+' '+mdt_mount))

        # File(
        #     os.path.join("/var/lib/ambari-agent","LustreMDSMgmtService_status.conf"),
        #     owner='root',
        #     group='root',
        #     mode=0644,
        #     content="status=1",
        # )

        print 'Configure complete.'

if __name__ == "__main__":
    LustreMDSMgmtService().execute()
