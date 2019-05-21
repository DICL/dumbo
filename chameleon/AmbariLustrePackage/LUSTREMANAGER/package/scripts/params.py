# -*- coding: utf-8 -*-
#!/usr/bin/env python
import sys
import os
import ConfigParser
import json
import socket
# import requests
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

# mds_wget = [
#  {
#   'name':'libcom_err-1.42.13.wc6-7.el7.x86_64.rpm',
#   'url':'https://downloads.hpdd.intel.com/public/e2fsprogs/1.42.13.wc6/el7/RPMS/x86_64/libcom_err-1.42.13.wc6-7.el7.x86_64.rpm',
#  },
#  {
#   'name':'e2fsprogs-libs-1.42.13.wc6-7.el7.x86_64.rpm',
#   'url':'https://downloads.hpdd.intel.com/public/e2fsprogs/1.42.13.wc6/el7/RPMS/x86_64/e2fsprogs-libs-1.42.13.wc6-7.el7.x86_64.rpm',
#  },
#  {
#   'name':'libss-1.42.13.wc6-7.el7.x86_64.rpm',
#   'url':'https://downloads.hpdd.intel.com/public/e2fsprogs/1.42.13.wc6/el7/RPMS/x86_64/libss-1.42.13.wc6-7.el7.x86_64.rpm',
#  },
#  {
#   'name':'e2fsprogs-1.42.13.wc6-7.el7.x86_64.rpm',
#   'url':'https://downloads.hpdd.intel.com/public/e2fsprogs/1.42.13.wc6/el7/RPMS/x86_64/e2fsprogs-1.42.13.wc6-7.el7.x86_64.rpm',
#  },
#  {
#   'name':'kmod-lustre-2.10.0-1.el7.x86_64.rpm',
#   'url':'https://downloads.hpdd.intel.com/public/lustre/lustre-2.10.0/el7/server/RPMS/x86_64/kmod-lustre-2.10.0-1.el7.x86_64.rpm',
#  },
#  {
#   'name':'lustre-osd-ldiskfs-mount-2.10.0-1.el7.x86_64.rpm',
#   'url':'https://downloads.hpdd.intel.com/public/lustre/lustre-2.10.0/el7/server/RPMS/x86_64/lustre-osd-ldiskfs-mount-2.10.0-1.el7.x86_64.rpm',
#  },
#  {
#   'name':'kmod-lustre-osd-ldiskfs-2.10.0-1.el7.x86_64.rpm',
#   'url':'https://downloads.hpdd.intel.com/public/lustre/lustre-2.10.0/el7/server/RPMS/x86_64/kmod-lustre-osd-ldiskfs-2.10.0-1.el7.x86_64.rpm',
#  },
#  {
#   'name':'lustre-2.10.0-1.el7.x86_64.rpm',
#   'url':'https://downloads.hpdd.intel.com/public/lustre/lustre-2.10.0/el7/server/RPMS/x86_64/lustre-2.10.0-1.el7.x86_64.rpm',
#  },
#  {
#   'name':'lustre-iokit-2.10.0-1.el7.x86_64.rpm',
#   'url':'https://downloads.hpdd.intel.com/public/lustre/lustre-2.10.0/el7/server/RPMS/x86_64/lustre-iokit-2.10.0-1.el7.x86_64.rpm',
#  },
# ]

client_wget = [
 {
  'name':'kmod-lustre-client-2.10.0-1.el7.x86_64.rpm',
  'url':'https://downloads.hpdd.intel.com/public/lustre/lustre-2.10.0/el7/client/RPMS/x86_64/kmod-lustre-client-2.10.0-1.el7.x86_64.rpm',
 },
 {
  'name':'lustre-client-2.10.0-1.el7.x86_64.rpm',
  'url':'https://downloads.hpdd.intel.com/public/lustre/lustre-2.10.0/el7/client/RPMS/x86_64/lustre-client-2.10.0-1.el7.x86_64.rpm',
 },
 {
  'name':'lustre-iokit-2.10.0-1.el7.x86_64.rpm',
  'url':'https://downloads.hpdd.intel.com/public/lustre/lustre-2.10.0/el7/client/RPMS/x86_64/lustre-iokit-2.10.0-1.el7.x86_64.rpm',
 },
]


# client_wget = [
#  {
#   'name':'kernel-3.10.0-693.2.2.el7.x86_64.rpm',
#   'url':'http://150.183.249.30/intel_lustre/2.10.1/kernel-3.10.0-693.2.2.el7.x86_64.rpm',
#  },
#  {
#   'name':'kmod-lustre-client-2.10.1-1.el7.x86_64.rpm',
#   'url':'http://150.183.249.30/intel_lustre/2.10.1/kmod-lustre-client-2.10.1-1.el7.x86_64.rpm',
#  },
#  {
#   'name':'lustre-client-2.10.1-1.el7.x86_64.rpm',
#   'url':'http://150.183.249.30/intel_lustre/2.10.1/lustre-client-2.10.1-1.el7.x86_64.rpm',
#  },
#  {
#   'name':'lustre-iokit-2.10.1-1.el7.x86_64.rpm',
#   'url':'http://150.183.249.30/intel_lustre/2.10.1/lustre-iokit-2.10.1-1.el7.x86_64.rpm',
#  },
# ]

# hosts = []
# server_device = {}
# host_file = open('/etc/hosts','r')
# for line in host_file:
#     if(line.split()[0] not in ["127.0.0.1" ,"::1"]):
#         hosts.append(line.split()[2]);

# for host in hosts:
#     server_device['device_select-'+host] = config['configurations']['lustre-default-config']['device_select-'+host]
#     server_device['device_size-'+host] = config['configurations']['lustre-default-config']['device_size-'+host]
#     server_device['network_select-'+host] = config['configurations']['lustre-default-config']['network_select-'+host]
#     server_device['network_device-'+host] = server_device['network_select-'+host].split('(')[0]


# shkim 20181016 start
# mds_host = config['configurations']['lustrefs-config-env']['mds_host']
# mdt_index = config['configurations']['lustrefs-config-env']['mdt_index']
# mdt_fsname = config['configurations']['lustrefs-config-env']['mdt_fsname']


# network_client = config['configurations']['lustre-default-config']['network_client']


# oss_host = config['configurations']['lustrefs-config-env']['oss_host']
# shkim 20181016 end

# oss_hosts
# [
#     ['slave1.hadoop.com':'1'],
#     ['slave2.hadoop.com':'2'],
#     ['slave3.hadoop.com':'3'],
# ]



local_hostname = socket.gethostname()



# local_hostname = socket.gethostname()
# print "Local Hostname : " + local_hostname
# ambari server hostname
# config = ConfigParser.ConfigParser()
# config.read("/etc/ambari-agent/conf/ambari-agent.ini")
# global ambari_url
# ambari_url = config.get('server','hostname')
# print "Ambari server Hostname : " + ambari_url
# # cluster_name
# headers = {
# 	'X-Requested-By': 'ambari',
# }
# r = requests.get('http://'+ambari_url+':8080/api/v1/clusters', headers=headers, auth=('admin', 'admin'))
# j = json.loads(r.text)
# items = j["items"][0]["Clusters"]
# print "Cluster Name : " + items["cluster_name"]
# cluster_name = items["cluster_name"]

# # LustreClient HostNames
# # curl -u admin:admin http://192.168.1.194:8080/api/v1/clusters/+cluster_name+/services/LUSTRE/components/LUSTRE_CLIENT
# r = requests.get('http://'+ambari_url+':8080/api/v1/clusters/'+cluster_name+'/hosts', headers=headers, auth=('admin', 'admin'))
# j = json.loads(r.text)

# host_names =[]
# for component in j["items"]:
# 	host_names.append(component["Hosts"]["host_name"])