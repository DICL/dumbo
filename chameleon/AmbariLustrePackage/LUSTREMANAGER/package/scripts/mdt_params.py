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

# mds_wget = [
#     {'name':'e2fsprogs-1.42.13.wc6-7.el7.x86_64.rpm','url':'http://150.183.249.30/intel_lustre/2.10.1/e2fsprogs-1.42.13.wc6-7.el7.x86_64.rpm'},
#     {'name':'e2fsprogs-libs-1.42.13.wc6-7.el7.x86_64.rpm','url':'http://150.183.249.30/intel_lustre/2.10.1/e2fsprogs-libs-1.42.13.wc6-7.el7.x86_64.rpm'},
#     {'name':'e2fsprogs-static-1.42.13.wc6-7.el7.x86_64.rpm','url':'http://150.183.249.30/intel_lustre/2.10.1/e2fsprogs-static-1.42.13.wc6-7.el7.x86_64.rpm'},
#     {'name':'kernel-3.10.0-693.2.2.el7_lustre.x86_64.rpm','url':'http://150.183.249.30/intel_lustre/2.10.1/kernel-3.10.0-693.2.2.el7_lustre.x86_64.rpm'},
#     {'name':'kernel-tools-3.10.0-693.2.2.el7_lustre.x86_64.rpm','url':'http://150.183.249.30/intel_lustre/2.10.1/kernel-tools-3.10.0-693.2.2.el7_lustre.x86_64.rpm'},
#     {'name':'kernel-tools-libs-3.10.0-693.2.2.el7_lustre.x86_64.rpm','url':'http://150.183.249.30/intel_lustre/2.10.1/kernel-tools-libs-3.10.0-693.2.2.el7_lustre.x86_64.rpm'},
#     {'name':'kmod-lustre-2.10.1-1.el7.x86_64.rpm','url':'http://150.183.249.30/intel_lustre/2.10.1/kmod-lustre-2.10.1-1.el7.x86_64.rpm'},
#     {'name':'kmod-lustre-osd-ldiskfs-2.10.1-1.el7.x86_64.rpm','url':'http://150.183.249.30/intel_lustre/2.10.1/kmod-lustre-osd-ldiskfs-2.10.1-1.el7.x86_64.rpm'},
#     {'name':'kmod-lustre-tests-2.10.1-1.el7.x86_64.rpm','url':'http://150.183.249.30/intel_lustre/2.10.1/kmod-lustre-tests-2.10.1-1.el7.x86_64.rpm'},
#     {'name':'libcom_err-1.42.13.wc6-7.el7.x86_64.rpm','url':'http://150.183.249.30/intel_lustre/2.10.1/libcom_err-1.42.13.wc6-7.el7.x86_64.rpm'},
#     {'name':'libnvpair1-0.7.1-1.el7.x86_64.rpm','url':'http://150.183.249.30/intel_lustre/2.10.1/libnvpair1-0.7.1-1.el7.x86_64.rpm'},
#     {'name':'libss-1.42.13.wc6-7.el7.x86_64.rpm','url':'http://150.183.249.30/intel_lustre/2.10.1/libss-1.42.13.wc6-7.el7.x86_64.rpm'},
#     {'name':'libuutil1-0.7.1-1.el7.x86_64.rpm','url':'http://150.183.249.30/intel_lustre/2.10.1/libuutil1-0.7.1-1.el7.x86_64.rpm'},
#     {'name':'libzpool2-0.7.1-1.el7.x86_64.rpm lustre-2.10.1-1.el7.x86_64.rpm','url':'http://150.183.249.30/intel_lustre/2.10.1/libzpool2-0.7.1-1.el7.x86_64.rpm'},
#     {'name':'lustre-iokit-2.10.1-1.el7.x86_64.rpm','url':'http://150.183.249.30/intel_lustre/2.10.1/lustre-iokit-2.10.1-1.el7.x86_64.rpm'},
#     {'name':'lustre-osd-ldiskfs-mount-2.10.1-1.el7.x86_64.rpm','url':'http://150.183.249.30/intel_lustre/2.10.1/lustre-osd-ldiskfs-mount-2.10.1-1.el7.x86_64.rpm'},
#     {'name':'lustre-resource-agents-2.10.1-1.el7.x86_64.rpm','url':'http://150.183.249.30/intel_lustre/2.10.1/lustre-resource-agents-2.10.1-1.el7.x86_64.rpm'},
#     {'name':'perf-3.10.0-693.2.2.el7_lustre.x86_64.rpm','url':'http://150.183.249.30/intel_lustre/2.10.1/perf-3.10.0-693.2.2.el7_lustre.x86_64.rpm'},
#     {'name':'python-perf-3.10.0-693.2.2.el7_lustre.x86_64.rpm','url':'http://150.183.249.30/intel_lustre/2.10.1/python-perf-3.10.0-693.2.2.el7_lustre.x86_64.rpm'}
# ]

local_hostname = params.local_hostname;

mdt_server_device = {};



# shkim 20181016 start
# mds_host = config['configurations']['lustrefs-config-env']['mds_host'];
# mdt_index = config['configurations']['lustrefs-config-env']['mdt_index'];
# mdt_fsname = config['configurations']['lustrefs-config-env']['mdt_fsname'];
# mdt_mount = config['configurations']['lustrefs-config-env']['mdt_mount'];
# shkim 20181016 end

# shkim 20181016 start
#mdt_server_device['device_mds'] = config['configurations']['lustre-default-config']['device_mds-' + mds_host];
# shkim 20181016 end
# mdt_server_device['device_mds_size'] = config['configurations']['lustre-default-config']['device_mds_size-' + mds_host];
# shkim 20181016 start
#mdt_server_device['network_mds'] = config['configurations']['lustre-default-config']['network_mds-' + mds_host];
# shkim 20181016 end
