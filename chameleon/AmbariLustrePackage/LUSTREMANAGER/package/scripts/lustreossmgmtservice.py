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
from resource_management.libraries.functions.default import default



class LustreOSSMgmtService(Script):
    def __init__(self):
        self.sbtls = SubTools()
    def install(self, env):
        import oss_params;
        print 'Install LustreOSSMgmtService.'

        oss_wget = oss_params.oss_wget

        # Install packages
        self.install_packages(env)
        Execute(format("yum install -y wget, libesmtp libyaml net-snmp-agent-libs opensm opensm-libs sg3_utils tcl tk lsof psmisc attr"))

        try:
            # pip 설치
            Execute(format("yum install -y python-pip"))
        except ExecutionFailed as e:
            print(e)
            pass

        try:
            # requests 등의 python 패키지 설치
            Execute(format("pip install requests"))
        except ExecutionFailed as e:
            print(e)
            pass


        # for pkg in oss_wget:
        #     download_file = 'wget '+pkg['url']
        #     Execute(format(download_file))

        # 테스트 (패키지를 사전에 다운로드함)
        file_path = os.path.dirname(os.path.realpath(__file__));
        try:
            Execute(format("mkdir ~/lustre_oss_module"))
        except ExecutionFailed as e:
            print(e)
            pass

        # Execute(format('cp -f ' +file_path + '/modules/lustre_packages/*' + ' ' + '~/lustre_oss_module'))

        for pkg in oss_wget:
            download_file = 'wget '+pkg['url']+' -P ~/lustre_oss_module'
            Execute(format(download_file))

        pkg_file = 'yum localinstall -y'
        for pkg in oss_wget:
            pkg_file = pkg_file +' ~/lustre_oss_module/'+ pkg['name'] + ' '
            # pkg_file = pkg_file +' '+ pkg['name'] + ' '

        try:
            Execute(format(pkg_file))
        except ExecutionFailed as e:
            print(e)
            pass

        print 'Installation complete.'


        self.configure(env)
        # file_path = os.path.dirname(os.path.realpath(__file__))
        # self.sbtls.sp_open('python '+file_path+'/daemon-lustre-oss.py start')

    def stop(self, env):
        print 'Stop LustreOSSMgmtService'
        # Stop your service

        #Since we have not installed a real service, there is no PID file created by it.
        #Therefore we are going to artificially remove the PID.
        # Execute( "rm -f /tmp/LustreOSSMgmtService.pid" )
        file_path = os.path.dirname(os.path.realpath(__file__))
        self.sbtls.sp_open('python '+file_path+'/daemon-lustre-oss.py stop')
        try:
            self.sbtls.sp_open('python '+file_path+'/modules/lustre_producer_oss/lustre_producer_oss.py stop')
            pass
        except Exception as e:
            print(e)
            pass

    def start(self, env):
        print 'Start LustreOSSMgmtService'
        # Reconfigure all files
        # Start your service

        #Since we have not installed a real service, there is no PID file created by it.
        #Therefore we are going to artificially create the PID.
        #Execute( "touch /tmp/LustreOSSMgmtService.pid" )
        file_path = os.path.dirname(os.path.realpath(__file__))
        self.sbtls.sp_open('python '+file_path+'/daemon-lustre-oss.py start')
        try:
            #self.sbtls.sp_open('python '+file_path+'/modules/lustre_producer_oss/lustre_producer_oss.py start')
            pass
        except Exception as e:
            print(e)
            pass


    def status(self, env):
        print 'Status of LustreOSSMgmtService'
        # LustreOSSMgmtService_pid_file = "/tmp/LustreOSSMgmtService.pid"
        # #check_process_status(dummy_slave_pid_file)
        # # Execute( format("cat {LustreOSSMgmtService_pid_file}") )
        # # pass

        file_path = os.path.dirname(os.path.realpath(__file__))
        check = self.sbtls.sp_open('python '+file_path+'/daemon-lustre-oss.py status')
        print check
        if 'not' in str(check):
            raise ComponentIsNotRunning
        pass

    # def mountosts(self, env):
    #     # self.sbtls.excuteDaemon('sample02',5681)
    #     import oss_params;

    #     local_hostname = oss_params.local_hostname;
    #     oss_server_device = oss_params.oss_server_device[local_hostname];
    #     modprobe_networks = oss_server_device['network_oss']
    #     mds_host = oss_params.mds_host;
    #     mdt_fsname = oss_params.mdt_fsname;
    #     network_device_mds = oss_params.network_device_mds;
    #     oss_mount = oss_params.oss_mount;

    #     for i in range(0, oss_server_device['device_num']):
    #         if oss_server_device['server_device'][i] == 'none':
    #             continue;
    #         Execute(format('mount -t lustre '+oss_server_device['server_device'][i]+' '+oss_mount+str(oss_server_device['server_device_index'][i])));
    #     print("mountosts!!")

    # def umountosts(self, env):
    #     # self.sbtls.excuteDaemon('sample02',5681)
    #     import oss_params;

    #     local_hostname = oss_params.local_hostname;
    #     oss_server_device = oss_params.oss_server_device[local_hostname];
    #     modprobe_networks = oss_server_device['network_oss']
    #     mds_host = oss_params.mds_host;
    #     mdt_fsname = oss_params.mdt_fsname;
    #     network_device_mds = oss_params.network_device_mds;
    #     oss_mount = oss_params.oss_mount;

    #     for i in range(0, oss_server_device['device_num']):
    #         if oss_server_device['server_device'][i] == 'none':
    #             continue;
    #         Execute(format('umount '+oss_mount+str(oss_server_device['server_device_index'][i])));
    #     print("umountosts!!")

    # def mkfsosts(self, env):
    #     # self.sbtls.excuteDaemon('sample02',5681)
    #     import oss_params;

    #     local_hostname = oss_params.local_hostname;
    #     oss_server_device = oss_params.oss_server_device[local_hostname];
    #     modprobe_networks = oss_server_device['network_oss']
    #     mds_host = oss_params.mds_host;
    #     mdt_fsname = oss_params.mdt_fsname;
    #     network_device_mds = oss_params.network_device_mds;
    #     oss_mount = oss_params.oss_mount;

    #     for i in range(0, oss_server_device['device_num']):
    #         if oss_server_device['server_device'][i] == 'none':
    #             continue;
    #         Execute(format('mkfs.lustre --fsname='+mdt_fsname+'  --ost --mgsnode='+mds_host+'@'+network_device_mds+' --index='+str(oss_server_device['server_device_index'][i])+' --reformat '+oss_server_device['server_device'][i]))
    #     print("mkfsosts!!")

    def configure(self, env):
        print 'LustreOSSMgmtService Configure start....'
        # import oss_params;


        # local_hostname = oss_params.local_hostname;
        # oss_server_device = oss_params.oss_server_device[local_hostname];
        # modprobe_networks = oss_server_device['network_oss']
        # mds_host = oss_params.mds_host;
        # mdt_fsname = oss_params.mdt_fsname;
        # network_device_mds = oss_params.network_device_mds;
        # oss_mount = oss_params.oss_mount;

        # Lustrecontent = format('options lnet networks="'+modprobe_networks+'"')
        # File(
        #     os.path.join("/etc/modprobe.d","lustre.conf"),
        #     owner='root',
        #     group='root',
        #     mode=0644,
        #     content=Lustrecontent,
        # )
        # Execute(format('modprobe lustre'))
        # Execute(format('modprobe ost'))


        # for i in range(0, oss_server_device['device_num']):
        #     if oss_server_device['server_device'][i] == 'none':
        #         continue;
        #     Execute(format('mkfs.lustre --fsname='+mdt_fsname+'  --ost --mgsnode='+mds_host+'@'+network_device_mds+' --index='+str(oss_server_device['server_device_index'][i])+' --reformat '+oss_server_device['server_device'][i]))

        #     Directory(
        #         oss_mount + str(oss_server_device['server_device_index'][i]),
        #         create_parents=True,
        #         owner='root',
        #         group='root')
        #     Execute(format('mount -t lustre '+oss_server_device['server_device'][i]+' '+oss_mount+str(oss_server_device['server_device_index'][i])));

        print 'Configure complete.'


if __name__ == "__main__":
    LustreOSSMgmtService().execute()
