# -*- coding: utf-8 -*-
#!/usr/bin/env python
import sys
import os
import ConfigParser
import json
import socket

import params
# server configurations

from resource_management import *
from resource_management.core.resources.system import File, Execute, Directory
from resource_management.libraries.functions.format import format
from resource_management.core.resources.service import Service
from resource_management.core.exceptions import ComponentIsNotRunning
from resource_management.core import shell
from resource_management.libraries.script.script import Script
from ambari_agent import AmbariConfig

config = Script.get_config();


oss_wget = [
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


# oss_wget = [
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



oss_server_device = {};
local_hostname = params.local_hostname;
# shkim 20181016 start
# oss_hostnum = 0;
# ost_index = 1;

# mds_host = config['configurations']['lustrefs-config-env']['mds_host']
# mdt_index = config['configurations']['lustrefs-config-env']['mdt_index'];
# mdt_fsname = config['configurations']['lustrefs-config-env']['mdt_fsname'];

# oss_hosts = config['configurations']['lustrefs-config-env']['oss_host'].split();
# oss_mount = '/mnt/ost';

# network_mds = config['configurations']['lustre-default-config']['network_mds-' + mds_host];
# network_device_mds = network_mds.split("(")[0];


# for oss_host in oss_hosts:
#     if(oss_host != ''):
#         host = oss_host.split('|')[0];
#         device_num = oss_host.split('|')[1];
#         oss_server_device[host] = {};
#         oss_server_device[host]['network_oss'] =  config['configurations']['lustre-default-config']['network_oss-' + host];
#         oss_server_device[host]['server_device'] = [];
#         # oss_server_device[host]['server_device_size'] = [];
#         oss_server_device[host]['server_device_index'] = [];
#         oss_server_device[host]['device_num'] = int(device_num)

#         for i in range(1, oss_server_device[host]['device_num']+1):
#             if config['configurations']['lustre-default-config']['device_oss'+str(i)+'-' + host] == 'none' or config['configurations']['lustre-default-config']['index_oss'+str(i)+'-' + host] == 'none':
#                 oss_server_device[host]['server_device'].append('none');
#                 oss_server_device[host]['server_device_index'].append('none');

#             oss_server_device[host]['server_device'].append(config['configurations']['lustre-default-config']['device_oss'+str(i)+'-' + host]);
#             oss_server_device[host]['server_device_index'].append(config['configurations']['lustre-default-config']['index_oss'+str(i)+'-' + host]);
# shkim 20181016 end