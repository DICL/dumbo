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
 
class YARNAppMonitorClient(Script):
  def __init__(self):
    self.sbtls = SubTools()

  def install(self, env):
    time.sleep(random.randint(0,15))
    file_path = os.path.dirname(os.path.realpath(__file__))
    config = Script.get_config()
    Yarnappmonitor_sctipt_locatton = config['configurations']['yarnjobmonitor-config-env']['location']
    Execute(format("yum install -y sysstat perf"));
    Execute(format("pip install psycopg2-binary"));
    Execute(format("cp "+ file_path + "/yarn_appmonitor_send.py " + Yarnappmonitor_sctipt_locatton));
    Execute(format("cp "+ file_path + "/crontab /etc/"));
    Execute(format("service crond restart"));
    print 'install'

  def umount(self, env):
    # Execute(format("ssh root@localhost -T \"nohup umount -f /mnt/lustre/hadoop > /tmp/mounthistory \""))
    # Execute(format("ssh root@localhost -T \"nohup umount -f /filedisk2 > /tmp/mounthistory \""))
    print 'umount'

  def mount(self, env):
    #Execute(format("ssh root@localhost -T \"nohup umount -f /mnt/lustre/hadoop > /tmp/mounthistory \""))
    #Execute(format("ssh root@localhost -T \"nohup umount -f /filedisk2 > /tmp/mounthistory \""))
    HadoopLustreAdapterMgmtService_user = config['configurations']['lustrefs-config-env']['hadooplustreadaptermgmtservice_user']
    # Execute(format("ssh root@localhost -T \"nohup mount -t lustre -o flock {HadoopLustreAdapterMgmtService_user}@tcp:/testfs  /mnt/lustre/hadoop > /tmp/mounthistory \""))
    # Execute(format("ssh root@localhost -T \"nohup mount -B /mnt/lustre/temp_disk/$HOSTNAME /filedisk2 > /tmp/mounthistory \""))
    print 'mount'

  def status(self, env):
    raise ClientComponentHasNoStatus()
 
if __name__ == "__main__":
  YARNAppMonitorClient().execute()