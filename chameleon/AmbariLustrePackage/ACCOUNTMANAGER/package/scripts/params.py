# -*- coding: utf-8 -*-
#!/usr/bin/env python
import sys
import os
import ConfigParser
import json
import socket
from resource_management import *
from resource_management.core.resources.system import File, Execute, Directory
from resource_management.libraries.functions.format import format
from resource_management.core.resources.service import Service
from resource_management.core.exceptions import ComponentIsNotRunning
from resource_management.core import shell
from resource_management.libraries.script.script import Script
from ambari_agent import AmbariConfig

config = Script.get_config()

ldap_manager_name = config['configurations']['usersync-config-env']['ldap_manager_name'];
ldap_manager_pass = config['configurations']['usersync-config-env']['ldap_manager_pass'];
ldap_domain = config['configurations']['usersync-config-env']['ldap_domain'];
