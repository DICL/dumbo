# -*- coding: utf-8 -*-
#!/usr/bin/env python

import sys
import re
import requests
import json
import time
import socket
import random
import ConfigParser
import subprocess
import time
import os
import logging
import logging.handlers

from daemon_producer_mds import Daemon





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
    mds_producer = MdsProducer();
    while True:
      try:
        mds_producer.init();
        line = subprocess.check_output(format("lctl get_param mdt.*.md_stats") , shell=True)
        data = mds_producer.parse(line)
        mds_producer.produce(data)
        pass
      except Exception as e:
        logger.error(e)
        pass
      time.sleep(1)

class MdsProducer:
  def parse(self,data):

    # Open            = None;
    # Close           = None;
    # Mknod           = None;
    # Unlink          = None;
    # Mkdir           = None;
    # Rmdir           = None;
    # Rename          = None;
    # Getattr         = None;
    # Setattr         = None;
    # Getxattr        = None;
    # Statfs          = None;
    # Sync            = None;
    # Samedir_rename  = None;
    # Crossdir_rename = None;

    Open            = {};
    Close           = {};
    Mknod           = {};
    Unlink          = {};
    Mkdir           = {};
    Rmdir           = {};
    Rename          = {};
    Getattr         = {};
    Setattr         = {};
    Getxattr        = {};
    Statfs          = {};
    Sync            = {};
    Samedir_rename  = {};
    Crossdir_rename = {};

    lustrefs_name   = None;
    lustrefs_name_list = [];

    #print data
    lines = data.split("\n");

    for line in lines:
      words = line.split();
    #   # print words
      if len(words) > 0:
        if 'md_stats' in words[0]:
          lustrefs_name = words[0].split('-')[0].split('.')[1];
          lustrefs_name_list.append(lustrefs_name);

        if 'open'  in words[0]:
          Open[lustrefs_name] = words[1];
        elif 'close' in words[0]:
          Close = words[1];
        elif 'mknod' in words[0]:
          Mknod[lustrefs_name] = words[1];
        elif 'unlink' in words[0]:
          Unlink[lustrefs_name] = words[1];
        elif 'mkdir' in words[0]:
          Mkdir[lustrefs_name] = words[1];
        elif 'rmdir' in words[0]:
          Rmdir[lustrefs_name] = words[1];
        elif 'rename' in words[0]:
          Rename[lustrefs_name] = words[1];
        elif 'getattr' in words[0]:
          Getattr[lustrefs_name] = words[1];
        elif 'setattr' in words[0]:
          Setattr[lustrefs_name] = words[1];
        elif 'getxattr' in words[0]:
          Getxattr[lustrefs_name] = words[1];
        elif 'statfs' in words[0]:
          Statfs[lustrefs_name] = words[1];
        elif 'sync' in words[0]:
          Sync[lustrefs_name] = words[1];
        elif 'samedir_rename' in words[0]:
          Samedir_rename[lustrefs_name] = words[1];
        elif 'crossdir_rename' in words[0]:
          Crossdir_rename[lustrefs_name] = words[1];


        # if words[0] in 'open':
        #   Open = words[1];
        # elif words[0] in 'close':
        #   Close = words[1];
        # elif words[0] in 'mknod':
        #   Mknod = words[1];
        # elif words[0] in 'unlink':
        #   Unlink = words[1];
        # elif words[0] in 'mkdir':
        #   Mkdir = words[1];
        # elif words[0] in 'rmdir':
        #   Rmdir = words[1];
        # elif words[0] in 'rename':
        #   Rename = words[1];
        # elif words[0] in 'getattr':
        #   Getattr = words[1];
        # elif words[0] in 'setattr':
        #   Setattr = words[1];
        # elif words[0] in 'getxattr':
        #   Getxattr = words[1];
        # elif words[0] in 'statfs':
        #   Statfs = words[1];
        # elif words[0] in 'sync':
        #   Sync = words[1];
        # elif words[0] in 'samedir_rename':
        #   Samedir_rename = words[1];
        # elif words[0] in 'crossdir_rename':
        #   Crossdir_rename = words[1];
      else:
        continue

    # print line

    #Ost             KBRead   Reads  SizeKB    KBWrite  Writes  SizeKB
    # data = re.split(' +', line)
  #  print data

  # mdt.mylustre-MDT0000.md_stats
  # snapshot_time             1522120769.086667926 secs.nsecs
  # open                      4505 samples [reqs]
  # close                     4417 samples [reqs]
  # mknod                     2255 samples [reqs]
  # unlink                    365 samples [reqs]
  # mkdir                     493 samples [reqs]
  # rmdir                     170 samples [reqs]
  # rename                    34 samples [reqs]
  # getattr                   9800 samples [reqs]
  # setattr                   4416 samples [reqs]
  # getxattr                  316 samples [reqs]
  # statfs                    11613 samples [reqs]
  # sync                      265 samples [reqs]
  # samedir_rename            27 samples [reqs]
  # crossdir_rename           7 samples [reqs]

    # try:
    #   (open,close,mkdir,getattr,setattr,statfs) = data
    # except:
    #   open = None
    #   close = None
    #   mkdir = None
    #   getattr = None
    #   setattr = None
    #   statfs = None



    #return (Open,Close,Mknod,Unlink,Mkdir,Rmdir,Rename,Getattr,Setattr,Getxattr,Statfs,Sync,Samedir_rename,Crossdir_rename)
    return (Open,Close,Mknod,Unlink,Mkdir,Rmdir,Rename,Getattr,Setattr,Getxattr,Statfs,Sync,Samedir_rename,Crossdir_rename,lustrefs_name_list)


  def produce(self,data):

    (Open,Close,Mknod,Unlink,Mkdir,Rmdir,Rename,Getattr,Setattr,Getxattr,Statfs,Sync,Samedir_rename,Crossdir_rename,lustrefs_name_list) = data

    if Open            is None:
      Open = 0;
    if Close           is None:
      Close = 0;
    if Mknod           is None:
      Mknod = 0;
    if Unlink          is None:
      Unlink = 0;
    if Mkdir           is None:
      Mkdir = 0;
    if Rmdir           is None:
      Rmdir = 0;
    if Rename          is None:
      Rename = 0;
    if Getattr         is None:
      Getattr = 0;
    if Setattr         is None:
      Setattr = 0;
    if Getxattr        is None:
      Getxattr = 0;
    if Statfs          is None:
      Statfs = 0;
    if Sync            is None:
      Sync = 0;
    if Samedir_rename  is None:
      Samedir_rename = 0;
    if Crossdir_rename is None:
      Crossdir_rename = 0;



    curtime = int(round(time.time() * 1000))
  #  local_hostname ="cn7"

    metrics_list = []
    tmp_metrics = {}
    for lustrefs_name in lustrefs_name_list:
      try:
        tmp_text = Open[lustrefs_name];
        pass
      except Exception as e:
        tmp_text = 0;
        pass
      tmp_metrics = {
         "metricname": lustrefs_name + "-" + "open",
         "appid": "lustremanager",
         "hostname": local_hostname,
         "timestamp": curtime,
         "starttime": curtime,
         "metrics": { curtime: tmp_text }
      }
      metrics_list.append(tmp_metrics)

      try:
        tmp_text = Close[lustrefs_name];
        pass
      except Exception as e:
        tmp_text = 0;
        pass
      tmp_metrics ={
         "metricname": lustrefs_name + "-" + "close",
         "appid": "lustremanager",
         "hostname": local_hostname,
         "timestamp": curtime,
         "starttime": curtime,
         "metrics": { curtime: tmp_text }
        }
      metrics_list.append(tmp_metrics)

      try:
        tmp_text = Mknod[lustrefs_name];
        pass
      except Exception as e:
        tmp_text = 0;
        pass
      tmp_metrics = {
         "metricname": lustrefs_name + "-" + "mknod",
         "appid": "lustremanager",
         "hostname": local_hostname,
         "timestamp": curtime,
         "starttime": curtime,
         "metrics": { curtime: tmp_text }
        }
      metrics_list.append(tmp_metrics)

      try:
        tmp_text = Mkdir[lustrefs_name];
        pass
      except Exception as e:
        tmp_text = 0;
        pass
      tmp_metrics = {
         "metricname": lustrefs_name + "-" + "mkdir",
         "appid": "lustremanager",
         "hostname": local_hostname,
         "timestamp": curtime,
         "starttime": curtime,
         "metrics": { curtime: tmp_text }
        }
      metrics_list.append(tmp_metrics)

      try:
        tmp_text = Rmdir[lustrefs_name];
        pass
      except Exception as e:
        tmp_text = 0;
        pass
      tmp_metrics = {
         "metricname": lustrefs_name + "-" + "rmdir",
         "appid": "lustremanager",
         "hostname": local_hostname,
         "timestamp": curtime,
         "starttime": curtime,
         "metrics": { curtime: tmp_text }
       }
      metrics_list.append(tmp_metrics)

      try:
        tmp_text = Rename[lustrefs_name];
        pass
      except Exception as e:
        tmp_text = 0;
        pass
      tmp_metrics = {
         "metricname": lustrefs_name + "-" + "rename",
         "appid": "lustremanager",
         "hostname": local_hostname,
         "timestamp": curtime,
         "starttime": curtime,
         "metrics": { curtime: tmp_text }
       }
      metrics_list.append(tmp_metrics)

      try:
        tmp_text = Getattr[lustrefs_name];
        pass
      except Exception as e:
        tmp_text = 0;
        pass
      tmp_metrics = {
         "metricname": lustrefs_name + "-" + "getattr",
         "appid": "lustremanager",
         "hostname": local_hostname,
         "timestamp": curtime,
         "starttime": curtime,
         "metrics": { curtime: tmp_text }
       }
      metrics_list.append(tmp_metrics)

      try:
        tmp_text = Setattr[lustrefs_name];
        pass
      except Exception as e:
        tmp_text = 0;
        pass
      tmp_metrics = {
         "metricname": lustrefs_name + "-" + "setattr",
         "appid": "lustremanager",
         "hostname": local_hostname,
         "timestamp": curtime,
         "starttime": curtime,
         "metrics": { curtime: tmp_text }
       }
      metrics_list.append(tmp_metrics)
      pass

    payload ={}
    payload['metrics'] = metrics_list

    # payload = {
    #  "metrics": [{
    #    "metricname": "open",
    #    "appid": "lustremanager",
    #    "hostname": local_hostname,
    #    "timestamp": curtime,
    #    "starttime": curtime,
    #    "metrics": { curtime: Open }
    #  },{
    #    "metricname": "close",
    #    "appid": "lustremanager",
    #    "hostname": local_hostname,
    #    "timestamp": curtime,
    #    "starttime": curtime,
    #    "metrics": { curtime: Close }
    #  },{
    #    "metricname": "mknod",
    #    "appid": "lustremanager",
    #    "hostname": local_hostname,
    #    "timestamp": curtime,
    #    "starttime": curtime,
    #    "metrics": { curtime: Mknod }
    #  },{
    #    "metricname": "mkdir",
    #    "appid": "lustremanager",
    #    "hostname": local_hostname,
    #    "timestamp": curtime,
    #    "starttime": curtime,
    #    "metrics": { curtime: Mkdir }
    #  },{
    #    "metricname": "rmdir",
    #    "appid": "lustremanager",
    #    "hostname": local_hostname,
    #    "timestamp": curtime,
    #    "starttime": curtime,
    #    "metrics": { curtime: Rmdir }
    #  },{
    #    "metricname": "rename",
    #    "appid": "lustremanager",
    #    "hostname": local_hostname,
    #    "timestamp": curtime,
    #    "starttime": curtime,
    #    "metrics": { curtime: Rename }
    #  },{
    #    "metricname": "getattr",
    #    "appid": "lustremanager",
    #    "hostname": local_hostname,
    #    "timestamp": curtime,
    #    "starttime": curtime,
    #    "metrics": { curtime: Getattr }
    #  },{
    #    "metricname": "setattr",
    #    "appid": "lustremanager",
    #    "hostname": local_hostname,
    #    "timestamp": curtime,
    #    "starttime": curtime,
    #    "metrics": { curtime: Setattr }
    #  }]
    #   }

    headers = {'content-type': 'application/json'}
    url = "http://" + amc_hostname + ":6188/ws/v1/timeline/metrics"

    str = json.dumps(payload)
    print str
    r = requests.post(url, data=str, headers=headers)
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
    print "Cluster Name : " + items["cluster_name"]
  #

  # Ambari Metrics Collector hostname
    r = requests.get('http://'+ambari_url+':8080/api/v1/clusters/'+items["cluster_name"]+'/services/AMBARI_METRICS/components/METRICS_COLLECTOR', headers=headers, auth=('admin', 'admin'))
    j = json.loads(r.text)
    global amc_hostname
    amc_hostname = j["host_components"][0]["HostRoles"]["host_name"]
    print "Ambari Metrics Collector Hostname : " + amc_hostname
#

# init()
# while 1:
#   try:
#       time.sleep(1)
#       line = subprocess.check_output(format("lctl get_param mdt.*.md_stats") , shell=True)
#       # line = '';
#       # line = sys.stdin.readline()
#   except KeyboardInterrupt:
#       break

#   if not line:
#      break
# #  print line
#   data = parse(line)
#   # parse(line)
#   # print data

#   produce(data)
#   # produce()

if __name__ == "__main__":
  file_path = os.path.dirname(os.path.realpath(__file__));
  logger = logging.getLogger('mylogger')
  fileMaxByte = 1024 * 1024 * 100 #100MB
  log_file_name = file_path+'/mds_producer.log';
  filehander = logging.handlers.RotatingFileHandler(log_file_name, maxBytes=fileMaxByte, backupCount=10)
  streamHandler = logging.StreamHandler()
  fomatter = logging.Formatter('[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s > %(message)s')
  filehander.setFormatter(fomatter)
  logger.addHandler(filehander)

  daemon = MyDaemon('/tmp/daemon-mds-producer.pid')
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
    pass
pass
