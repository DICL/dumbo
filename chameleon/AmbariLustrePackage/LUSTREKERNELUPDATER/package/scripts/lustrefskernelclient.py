import sys
import os
import subprocess
import ConfigParser
import time
import random
import commands
from subtools import SubTools
from resource_management import *
from resource_management.core.resources.system import File, Execute, Directory
from resource_management.libraries.functions.format import format
from resource_management.core.resources.service import Service
from resource_management.core.exceptions import ComponentIsNotRunning
from resource_management.core import shell
from resource_management.libraries.script.script import Script
from ambari_agent import AmbariConfig
 
class LustreFSKernelClient(Script):
  def __init__(self):
    self.sbtls = SubTools()

  def install(self, env):
    try:
      Execute(format("mkdir ~/lustre_kernel"))
      Execute(format("yum install -y wget libesmtp libyaml net-snmp-agent-libs opensm opensm-libs sg3_utils tcl tk "))
      Execute(format("wget https://downloads.hpdd.intel.com/public/lustre/lustre-2.10.0/el7.3.1611/server/RPMS/x86_64/kernel-devel-3.10.0-514.21.1.el7_lustre.x86_64.rpm -P ~/lustre_kernel"))
      Execute(format("wget https://downloads.hpdd.intel.com/public/lustre/lustre-2.10.0/el7.3.1611/server/RPMS/x86_64/kernel-3.10.0-514.21.1.el7_lustre.x86_64.rpm -P ~/lustre_kernel"))

      # Execute(format("mkdir ~/lustre_kernel"))
      # Execute(format("yum install -y wget libesmtp libyaml net-snmp-agent-libs opensm opensm-libs sg3_utils tcl tk "))
      # Execute(format("wget http://150.183.249.30/intel_lustre/2.10.1/kernel-3.10.0-693.2.2.el7_lustre.x86_64.rpm -P ~/lustre_kernel"))

      
      Execute(format("yum -y localinstall ~/lustre_kernel/*.rpm "))

    except ExecutionFailed as e:
      print(e)
      pass
    
    print 'install'

  def status(self, env):
    raise ClientComponentHasNoStatus()
 
if __name__ == "__main__":
  LustreFSKernelClient().execute()