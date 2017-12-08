import sys
import os
import subprocess
import ConfigParser
import time
import random
import commands
from resource_management import *
from resource_management.core.resources.system import File, Execute, Directory
from resource_management.libraries.functions.format import format
from resource_management.core.resources.service import Service
from resource_management.core.exceptions import ComponentIsNotRunning
from resource_management.core import shell
from resource_management.libraries.script.script import Script
from ambari_agent import AmbariConfig


class Client(Script):
  def install(self, env):
    time.sleep(random.randint(0,15))
    print 'install'
  def umount(self, env):
    pass
  def mount(self, env):
    pass
  def useHDFS(self, env):
    print 'useHDFS';

  def useLUSTRE(self, env):
    print 'userLUSTRE';

  def status(self, env):
    raise ClientComponentHasNoStatus()

if __name__ == "__main__":
  Client().execute()
