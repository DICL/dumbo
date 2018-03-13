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

class LustreFSKernelMaster(Script):
    def __init__(self):
        self.sbtls = SubTools()

    def install(self, env):
        print 'Installation complete.'

    def stop(self, env):
        print 'Stop LustreFSKernelMaster'
        # Stop your service
        self.sbtls.sp_open('python /var/lib/ambari-agent/cache/stacks/HDP/2.6/services/LUSTREFSKERNEL/package/scripts/daemon-lustrefs-kernel.py stop')

    def start(self, env):
        print 'Start LustreFSKernelMaster'
        # Reconfigure all files
        # Start your service
        self.sbtls.sp_open('python /var/lib/ambari-agent/cache/stacks/HDP/2.6/services/LUSTREFSKERNEL/package/scripts/daemon-lustrefs-kernel.py start')

    def reboot(self, env):
        self.sbtls.excuteDaemon('kernelinstall',5700)
        print("LustreFS kernel installed!!")

    def status(self, env):
        check = self.sbtls.sp_open('python /var/lib/ambari-agent/cache/stacks/HDP/2.6/services/LUSTREFSKERNEL/package/scripts/daemon-lustrefs-kernel.py status')
        print check
        if 'not' in str(check):
            raise ComponentIsNotRunning
        pass

if __name__ == "__main__":
    LustreFSKernelMaster().execute()
