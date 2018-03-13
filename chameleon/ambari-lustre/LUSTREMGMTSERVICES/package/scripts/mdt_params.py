# -*- coding: utf-8 -*-
#!/usr/bin/env python
import sys
import os
import ConfigParser
import json
import socket

import params;

from resource_management import *
from resource_management.core.resources.system import File, Execute, Directory
from resource_management.libraries.functions.format import format
from resource_management.core.resources.service import Service
from resource_management.core.exceptions import ComponentIsNotRunning
from resource_management.core import shell
from resource_management.libraries.script.script import Script
from ambari_agent import AmbariConfig

# server configurations
config = Script.get_config()


mds_wget = [
    {
        'name':
        'libcom_err-1.42.13.wc6-7.el7.x86_64.rpm',
        'url':
        'https://downloads.hpdd.intel.com/public/e2fsprogs/1.42.13.wc6/el7/RPMS/x86_64/libcom_err-1.42.13.wc6-7.el7.x86_64.rpm',
    },
    {
        'name':
        'e2fsprogs-libs-1.42.13.wc6-7.el7.x86_64.rpm',
        'url':
        'https://downloads.hpdd.intel.com/public/e2fsprogs/1.42.13.wc6/el7/RPMS/x86_64/e2fsprogs-libs-1.42.13.wc6-7.el7.x86_64.rpm',
    },
    {
        'name':
        'libss-1.42.13.wc6-7.el7.x86_64.rpm',
        'url':
        'https://downloads.hpdd.intel.com/public/e2fsprogs/1.42.13.wc6/el7/RPMS/x86_64/libss-1.42.13.wc6-7.el7.x86_64.rpm',
    },
    {
        'name':
        'e2fsprogs-1.42.13.wc6-7.el7.x86_64.rpm',
        'url':
        'https://downloads.hpdd.intel.com/public/e2fsprogs/1.42.13.wc6/el7/RPMS/x86_64/e2fsprogs-1.42.13.wc6-7.el7.x86_64.rpm',
    },
    {
        'name':
        'kmod-lustre-2.10.0-1.el7.x86_64.rpm',
        'url':
        'https://downloads.hpdd.intel.com/public/lustre/lustre-2.10.0/el7/server/RPMS/x86_64/kmod-lustre-2.10.0-1.el7.x86_64.rpm',
    },
    {
        'name':
        'lustre-osd-ldiskfs-mount-2.10.0-1.el7.x86_64.rpm',
        'url':
        'https://downloads.hpdd.intel.com/public/lustre/lustre-2.10.0/el7/server/RPMS/x86_64/lustre-osd-ldiskfs-mount-2.10.0-1.el7.x86_64.rpm',
    },
    {
        'name':
        'kmod-lustre-osd-ldiskfs-2.10.0-1.el7.x86_64.rpm',
        'url':
        'https://downloads.hpdd.intel.com/public/lustre/lustre-2.10.0/el7/server/RPMS/x86_64/kmod-lustre-osd-ldiskfs-2.10.0-1.el7.x86_64.rpm',
    },
    {
        'name':
        'lustre-2.10.0-1.el7.x86_64.rpm',
        'url':
        'https://downloads.hpdd.intel.com/public/lustre/lustre-2.10.0/el7/server/RPMS/x86_64/lustre-2.10.0-1.el7.x86_64.rpm',
    },
    {
        'name':
        'lustre-iokit-2.10.0-1.el7.x86_64.rpm',
        'url':
        'https://downloads.hpdd.intel.com/public/lustre/lustre-2.10.0/el7/server/RPMS/x86_64/lustre-iokit-2.10.0-1.el7.x86_64.rpm',
    },
]

local_hostname = params.local_hostname;

mdt_server_device = {};




mds_host = config['configurations']['lustrefs-config-env']['mds_host'];
mdt_index = config['configurations']['lustrefs-config-env']['mdt_index'];
mdt_fsname = config['configurations']['lustrefs-config-env']['mdt_fsname'];
mdt_mount = config['configurations']['lustrefs-config-env']['mdt_mount'];


mdt_server_device['device_mds'] = config['configurations']['lustre-default-config']['device_mds-' + mds_host];
# mdt_server_device['device_mds_size'] = config['configurations']['lustre-default-config']['device_mds_size-' + mds_host];
mdt_server_device['network_mds'] = config['configurations']['lustre-default-config']['network_mds-' + mds_host];
