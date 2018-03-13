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
from resource_management.core.exceptions import ClientComponentHasNoStatus



class LustreClient(Script):

    def install(self, env):
        print 'install'

        import params

        print 'Install LustreMDSMgmtService.'

        # Load the all configuration files
        config = Script.get_config()
        # Bind to a local variable
        #LustreMDSMgmtService_user = config['configurations']['my-config-env']['lustremdsmgmtservice_user']


        # Install packages
        self.install_packages(env)
        Execute(format("yum install -y wget, libyaml net-snmp-agent-libs sg3_utils"))


        for pkg in params.client_wget:
            download_file = 'wget '+pkg['url']
            Execute(format(download_file))

        pkg_file = 'yum install -y'
        for pkg in params.client_wget:
            pkg_file = pkg_file +' '+ pkg['name'] + ' '

        try:
            Execute(format(pkg_file))
        except ExecutionFailed as e:
            print(e)
            pass

        print 'Installation complete.'


        self.configure(env)


    def unmount(self, env):
        # import params
        # Execute(format('umount /lustre'));
        # Execute(format('rm -rf  /lustre'));
        print 'unmount'
        pass

    def mount(self, env):
        # import params
        # Directory('/lustre', create_parents=True, owner='root', group='root');
        # Execute(format('mount -t lustre  '+params.mds_host+'@tcp:/'+params.mdt_fsname+'  /lustre'));
        print 'mount'
        pass

    def status(self, env):
        print 'status'
        raise ClientComponentHasNoStatus()

    def configure(self, env):
        print 'LustreMDSMgmtService Configure start....'
        import params
        local_hostname = params.local_hostname;
        modprobe_networks = params.network_client;
        Lustrecontent = format('options lnet networks="'+modprobe_networks+'"')

        if(os.path.isfile('/etc/modprobe.d')):
            pass;
        else:
            File(
                os.path.join("/etc/modprobe.d","lustre.conf"),
                owner='root',
                group='root',
                mode=0644,
                content=Lustrecontent,
            )
        print 'Configure complete.'

if __name__ == "__main__":
    LustreClient().execute()
