#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import re
import requests
import json
import time
import socket
import random
import ConfigParser
import subprocess
import os
import logging
import logging.handlers

from daemon_producer_oss import Daemon

# Ambari server url
ambari_url = "cn7"
# Ambari Metrics Collect url
amc_hostname = "cn9"
#local hostname
local_hostname = ""

count = 0
logger = None

class MyDaemon(Daemon):
  def run(self):
    global logger;
    oss_producer = OssProducer();
    oss_producer.init()
    oss_producer.getOssIndex()
    while True:
      try:
        line = subprocess.check_output(format("lctl get_param obdfilter.*.stats") , shell=True)

        data = oss_producer.parse(line)
        pass
      except Exception as e:
        logger.error(e)
        pass

      try:
        oss_producer.produce(data)
        pass
      except Exception as e:
        logger.error(e)
        pass

      time.sleep(1)

class OssProducer:

  def parse(self,data):
    ost_info = {}
    lustre_name_list = [];
    ost_status = '';
    for obdfilter in data.split('obdfilter'):
      for line in obdfilter.split('\n'):
        if(len(line.split()) > 0 and len(line.split()) < 3 ):
          ost_status = line.split('-')[1].split('.')[0]
          lustrefs_name = line.split('-')[0].split('.')[1]
          if lustrefs_name in ost_info :
            pass
          else :
            ost_info[lustrefs_name] = {}
          ost_info[lustrefs_name][ost_status] = {}
          lustre_name_list.append(lustrefs_name);
          print lustrefs_name
        elif(len(line.split()) > 0 and len(line.split()) > 3 ):
          if 'read_bytes' in line.split()[0] :
            ost_info[lustrefs_name][ost_status]['read_bytes'] = line.split()[6]
          elif  'write_bytes' in line.split()[0] :
            ost_info[lustrefs_name][ost_status]['write_bytes'] = line.split()[6]
          elif  'setattr' in line.split()[0] :
            ost_info[lustrefs_name][ost_status]['setattr'] = line.split()[1]
          elif  'sync' in line.split()[0] :
            ost_info[lustrefs_name][ost_status]['sync'] = line.split()[1]
          elif  'destroy' in line.split()[0] :
            ost_info[lustrefs_name][ost_status]['destroy'] = line.split()[1]
          elif  'create' in line.split()[0] :
            ost_info[lustrefs_name][ost_status]['create'] = line.split()[1]
          elif  'statfs' in line.split()[0] :
            ost_info[lustrefs_name][ost_status]['statfs'] = line.split()[1]
          elif  'get_info' in line.split()[0] :
            ost_info[lustrefs_name][ost_status]['get_info'] = line.split()[1]
          elif  'connect' in line.split()[0] :
            ost_info[lustrefs_name][ost_status]['connect'] = line.split()[1]
          elif  'statfs' in line.split()[0] :
            ost_info[lustrefs_name][ost_status]['statfs'] = line.split()[1]
          elif  'preprw' in line.split()[0] :
            ost_info[lustrefs_name][ost_status]['preprw'] = line.split()[1]
          elif  'commitrw' in line.split()[0] :
            ost_info[lustrefs_name][ost_status]['commitrw'] = line.split()[1]
          elif  'ping' in line.split()[0] :
            ost_info[lustrefs_name][ost_status]['ping'] = line.split()[1]

    lustre_read_bytes_avg = {}
    lustre_write_bytes_avg = {}
    for lustre_name in lustre_name_list:
      total_read_bytes = 0
      total_write_bytes = 0
      for ost_data in ost_info[lustre_name]:
        if 'read_bytes' in ost_info[lustre_name][ost_data]:
          total_read_bytes += int(ost_info[lustre_name][ost_data]['read_bytes'])
        if 'write_bytes' in ost_info[lustre_name][ost_data]:
          total_write_bytes += int(ost_info[lustre_name][ost_data]['write_bytes'])
      if (len(ost_info[lustre_name]) > 0) & (total_read_bytes > 0) :
        lustre_read_bytes_avg[lustre_name] = total_read_bytes / len(ost_info[lustre_name])
        lustre_write_bytes_avg[lustre_name] = total_write_bytes / len(ost_info[lustre_name])
      else :
        lustre_read_bytes_avg[lustre_name] = 0
        lustre_write_bytes_avg[lustre_name] = 0
    return (lustre_read_bytes_avg, lustre_write_bytes_avg , lustre_name_list)

  def produce(self,data):

    # (read_bytes_avg,write_bytes_avg) = data
    (lustre_read_bytes_avg, lustre_write_bytes_avg , lustre_name_list) = data

    curtime = int(round(time.time() * 1000))
  #  local_hostname ="cn7"

    index = str(oss_index);

    metrics_list = [];
    tmp_metrics = {};
    for lustre_name in lustre_name_list:
      tmp_metrics = {
        "metricname": "Oss"+index+"-"+lustre_name+"-"+"Read",
        "appid": "lustremanager",
        "hostname": local_hostname,
        "timestamp": curtime,
        "starttime": curtime,
        "metrics": { curtime: lustre_read_bytes_avg[lustre_name] }
      }
      metrics_list.append(tmp_metrics);

      tmp_metrics = {
        "metricname": "Oss"+index+"-"+lustre_name+"-"+"Write",
        "appid": "lustremanager",
        "hostname": local_hostname,
        "timestamp": curtime,
        "starttime": curtime,
        "metrics": { curtime: lustre_write_bytes_avg[lustre_name] }
      }
      metrics_list.append(tmp_metrics);
      pass

    payload ={}
    payload['metrics'] = metrics_list




    # payload = {
    #  "metrics": [{
    #    "metricname": "Oss"+index+"Read",
    #    "appid": "lustremanager",
    #    "hostname": local_hostname,
    #    "timestamp": curtime,
    #    "starttime": curtime,
    #    "metrics": { curtime: read_bytes_avg }
    #  },{
    #    "metricname": "Oss"+index+"Write",
    #    "appid": "lustremanager",
    #    "hostname": local_hostname,
    #    "timestamp": curtime,
    #    "starttime": curtime,
    #    "metrics": { curtime: write_bytes_avg }
    #  }]
    #   }

    headers = {'content-type': 'application/json'}
    url = "http://" + amc_hostname + ":6188/ws/v1/timeline/metrics"

    result = json.dumps(payload)
    print str
    r = requests.post(url, data=result, headers=headers)
    print r
    print r.text

  def init(self):

    global local_hostname
    local_hostname = socket.gethostname()
    print "Local Hostname : " + local_hostname

  # ambari server hostname
    config = ConfigParser.ConfigParser()
    config.read("/etc/ambari-agent/conf/ambari-agent.ini")
    global ambari_url
    ambari_url = config.get('server','hostname')
    print "Ambari server Hostname : " + ambari_url
  #

  # cluster_name
    headers = {
      'X-Requested-By': 'ambari',
    }
    r = requests.get('http://'+ambari_url+':8080/api/v1/clusters', headers=headers, auth=('admin', 'admin'))
    j = json.loads(r.text)
    items = j["items"][0]["Clusters"]
    global cluster_name
    cluster_name = items["cluster_name"]
    print "Cluster Name : " + items["cluster_name"]
  #

  # Ambari Metrics Collector hostname
    r = requests.get('http://'+ambari_url+':8080/api/v1/clusters/'+items["cluster_name"]+'/services/AMBARI_METRICS/components/METRICS_COLLECTOR', headers=headers, auth=('admin', 'admin'))
    j = json.loads(r.text)
    global amc_hostname
    amc_hostname = j["host_components"][0]["HostRoles"]["host_name"]
    print "Ambari Metrics Collector Hostname : " + amc_hostname
  #


  # def getOssIndex():
  #   headers = {
  #     'X-Requested-By': 'ambari',
  #   }
  #   global oss_index
  #   global local_hostname
  #   local_hostname = socket.gethostname()
  #   # lustrefs-config-env 의 최근버전 추출
  #   r = requests.get('http://'+ambari_url+':8080/api/v1/clusters/'+cluster_name+'?fields=Clusters/desired_configs', headers=headers, auth=('admin', 'admin'))
  #   j = json.loads(r.text)
  #   lustrefs_config_env_tag = j['Clusters']['desired_configs']["lustrefs-config-env"]["tag"];
  #   # lustrefs-config-env 의 환경설정내용 추출
  #   r = requests.get('http://'+ambari_url+':8080/api/v1/clusters/'+cluster_name+'/configurations?type=lustrefs-config-env&tag='+lustrefs_config_env_tag, headers=headers, auth=('admin', 'admin'));
  #   j = json.loads(r.text)
  #   global lustrefs_config_env
  #   lustrefs_config_env =  j["items"][0]["properties"];
  #   oss_host_list = lustrefs_config_env['oss_host'].split('\n')

  #   index = 0;
  #   for idx, val in enumerate(oss_host_list):
  #     if(val.split('|')[0] in local_hostname):
  #       index = idx + 1;
  #       break
  #   global oss_index
  #   oss_index = index;
  #   print 'OSS NAME : ' + 'Oss' + str(oss_index)

  def getOssIndex(self):
    headers = {
      'X-Requested-By': 'ambari',
    }
    global oss_index
    global local_hostname
    local_hostname = socket.gethostname()
    # lustrefs-config-env 의 최근버전 추출
    r = requests.get('http://'+ambari_url+':8080/api/v1/clusters/'+cluster_name+'/services/LUSTREMANAGER/components/LustreOSSMgmtService', headers=headers, auth=('admin', 'admin'))
    j = json.loads(r.text)
    oss_host_list = j['host_components'];

    index = 0;
    global oss_index;
    for oss_info in oss_host_list:
      if(oss_info['HostRoles']['host_name'] in local_hostname):
        oss_index = index + 1;
        print 'OSS NAME : ' + 'Oss' + str(oss_index)
        break;
        pass
      pass
      index = index + 1;
    
    
    print 'OSS NAME : ' + 'Oss' + str(oss_index)

  # def getOssIndex(self):
  #   headers = {
  #     'X-Requested-By': 'ambari',
  #   }
  #   global oss_index
  #   global local_hostname
  #   local_hostname = socket.gethostname()
  #   # lustre manager 의 내용 추출
  #   instance_name = 'Lustre_View'
  #   version_name = '1.0.0'

  #   r = requests.get('http://'+ambari_url+':8080/views/Lustre_View/'+version_name+'/'+instance_name+'/api/v1/ambari/getLustreNodes', headers=headers, auth=('admin', 'admin'))
  #   nodes = json.loads(r.text)
  #   for node in nodes:
  #     if node['host_name'] == local_hostname:
  #       oss_index = node['index'] + 1;
  #       print 'OSS NAME : ' + 'Oss' + str(oss_index)
  #       break;
  #       pass
  #     pass



  #   pass;


# oss_producer = OssProducer();
# oss_producer.init()
# oss_producer.getOssIndex()
# while 1:
#   try:
#       time.sleep(1)
#       try:
#         line = subprocess.check_output(format("lctl get_param obdfilter.*.stats") , shell=True)
#         data = oss_producer.parse(line)
#         pass
#       except Exception as e:
#         pass

#       try:
#         oss_producer.produce(data)
#         pass
#       except Exception as e:
#         pass
#   except KeyboardInterrupt:
#       break



if __name__ == "__main__":

  file_path = os.path.dirname(os.path.realpath(__file__));
  logger = logging.getLogger('mylogger')
  fileMaxByte = 1024 * 1024 * 100 #100MB
  log_file_name = file_path+'/oss_producer.log';
  filehander = logging.handlers.RotatingFileHandler(log_file_name, maxBytes=fileMaxByte, backupCount=10)
  streamHandler = logging.StreamHandler()
  fomatter = logging.Formatter('[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s > %(message)s')
  filehander.setFormatter(fomatter)
  logger.addHandler(filehander)

  daemon = MyDaemon('/tmp/daemon-oss-producer.pid')
  if len(sys.argv) == 2:
    if 'start' == sys.argv[1]:
      daemon.start()
    elif 'stop' == sys.argv[1]:
      daemon.stop()
    elif 'restart' == sys.argv[1]:
      daemon.restart()
    else:
      print "Unknown command"
      sys.exit(2)
      sys.exit(0)
  else:
    print "usage: %s start|stop|restart" % sys.argv[0]
    sys.exit(2)

