# -*- coding: utf-8 -*-
import sys
import os
import subprocess
import ConfigParser
import time
import random
import commands
import socket
import json
from subtools import SubTools
from resource_management import *
from resource_management.core.resources.system import File, Execute, Directory
from resource_management.libraries.functions.format import format
from resource_management.core.resources.service import Service
from resource_management.core.exceptions import ComponentIsNotRunning
from resource_management.core import shell
from resource_management.libraries.script.script import Script
from ambari_agent import AmbariConfig


try:
    import requests;
except ImportError:
    file_path = os.path.dirname(os.path.realpath(__file__));
    sys.path.append(file_path+'/modules');
    import requests;

class SwitchConfig:
    def __init__(self):

        # set ambari_url
        self.sbtls = SubTools()
        config = ConfigParser.ConfigParser()
        config.read("/etc/ambari-agent/conf/ambari-agent.ini")
        self.ambari_url = config.get('server','hostname')

        # set cluster_name
        headers = {
          'X-Requested-By': 'ambari',
        }
        r = requests.get('http://'+self.ambari_url+':8080/api/v1/clusters', headers=headers, auth=('admin', 'admin'))
        j = json.loads(r.text)
        items = j["items"][0]["Clusters"]
        print "2 - Cluster Name : " + items["cluster_name"]
        self.cluster_name = items["cluster_name"]

        # change version
        file_path = os.path.dirname(os.path.realpath(__file__));
        json_data=open(file_path+"/config-version.json").read()
        data = json.loads(json_data)

        data["version"] = data["version"]+1
        with open(file_path+'/config-version.json', 'w') as f:
            json.dump(data, f)

        self.checkVersion = str(data["version"])

    def toHdfs(self):
        confInfo = self.sbtls.sp_open("curl -u admin:admin -X GET  http://"+self.ambari_url+":8080/api/v1/clusters/"+self.cluster_name+"?fields=Clusters/desired_configs")
        j = json.loads(confInfo[0])
        desiredConf = j['Clusters']['desired_configs']
        targetSites = ('mapred-site','yarn-site','core-site')
        
        hdfs_name_node = self.get_name_node_host();
        #self.subprocess_open('echo "lustrefs_config_env hdfs_name_node-> '+str(hdfs_name_node)+'" >> /tmp/switch-config-log.info')

        # targetSites = ('mapred-site')
        for targetSite in targetSites:
            print "\n\n\n"
            print "target site : "+targetSite
            print "target version"+desiredConf[targetSite]['tag']
            print "----- target conf -----"
            print desiredConf
            print "----- target conf end -----"
            print "\n"
            try:
                versionNum = desiredConf[targetSite]['tag'].split('version')[1]
            except Exception, e:
                versionNum = desiredConf[targetSite]['tag']


            # nextVersionNum = str(int(versionNum)+1)
            nextVersionNum = self.checkVersion

            targetConf = self.sbtls.sp_open('curl -u admin:admin -H "X-Requested-By: ambari" -X GET "http://'+self.ambari_url+':8080/api/v1/clusters/'+self.cluster_name+'/configurations?type='+targetSite+'&tag='+desiredConf[targetSite]['tag']+'"')

            targetJson = json.loads(targetConf[0])

            prop = targetJson['items'][0]['properties']
            print targetJson
            #print prop
            if targetSite == 'core-site':
                # 190220 je.kim NAMENODE 호스트 네임으로 변경
                prop['fs.defaultFS'] = u"hdfs://"+hdfs_name_node+":8020" #modify
                prop.pop('fs.lustrefs.impl') #remove
                prop.pop('fs.AbstractFileSystem.lustrefs.impl') #remove
                prop.pop('fs.lustrefs.mount') #remove
                prop.pop('hadoop.tmp.dir') #remove

            elif targetSite == 'yarn-site':
                prop['yarn.nodemanager.container-executor.class'] = u"org.apache.hadoop.yarn.server.nodemanager.DefaultContainerExecutor" #modify
                prop['yarn.nodemanager.local-dirs'] = u"/hadoop/yarn/local" #modify
                prop['yarn.nodemanager.log-dirs'] = u"/hadoop/yarn/log" #modify
                # prop['yarn.timeline-service.leveldb-state-store.path'] = u"/hadoop/yarn/timeline"#modify
                # prop['yarn.timeline-service.leveldb-timeline-store.path'] = u"/hadoop/yarn/timeline"#modify
                prop.pop('yarn.nodemanager.linux-container-executor.nonsecure-mode.local-user') #remove
                prop.pop('yarn.nodemanager.linux-container-executor.nonsecure-mode.limit-users') #remove

            elif targetSite == 'mapred-site':
                prop['yarn.app.mapreduce.am.staging-dir'] = u"/user"#modify

                # print json.dumps(prop)

            reqHead = 'curl -u admin:admin -H "X-Requested-By: ambari" -X PUT -d '
            reqBody = '[{"Clusters":{"desired_config":[{"type":"'+targetSite+'","tag":"version'+nextVersionNum+'","service_config_version_note":"New config version"}]}}]'
            reqTail = ' "http://'+self.ambari_url+':8080/api/v1/clusters/'+self.cluster_name+'"'

            body = json.loads(reqBody)
            body[0]['Clusters']['desired_config'][0][u'properties']=prop
            reqBody = json.dumps(body)
            print "======= Req Api ======="
            print reqHead + reqBody+ reqTail
            print "======= Req Api End ======="
            #TEST
            #getData = reqHead + reqBody+ reqTail
            #f = open("/home/lustre-daemon/test_20170510_2.txt", 'w')
            #f.write(getData)
            #f.close()
            reqResult = self.sbtls.sp_open(reqHead+"'"+reqBody+"'"+reqTail)
            print str(reqResult)

    def toLustrefs(self):
        confInfo = self.sbtls.sp_open("curl -u admin:admin -X GET  http://"+self.ambari_url+":8080/api/v1/clusters/"+self.cluster_name+"?fields=Clusters/desired_configs")
        j = json.loads(confInfo[0])
        desiredConf = j['Clusters']['desired_configs']
        targetSites = ('mapred-site','yarn-site','core-site')

        lustre_config = self.get_lustre_configs();
        #self.subprocess_open('echo "lustrefs_config_env result'+str(lustre_config)+'" >> /tmp/switch-config-log.info')

        # targetSites = ('mapred-site')
        for targetSite in targetSites:
            print "\n\n\n"
            print "target site : "+targetSite
            print "target version"+desiredConf[targetSite]['tag']
            print "----- target conf -----"
            print desiredConf
            print "----- target conf end -----"
            print "\n"
            try:
                versionNum = desiredConf[targetSite]['tag'].split('version')[1]
            except Exception, e:
                versionNum = desiredConf[targetSite]['tag']


            # nextVersionNum = str(int(versionNum)+1)
            nextVersionNum = self.checkVersion

            targetConf = self.sbtls.sp_open('curl -u admin:admin -H "X-Requested-By: ambari" -X GET "http://'+self.ambari_url+':8080/api/v1/clusters/'+self.cluster_name+'/configurations?type='+targetSite+'&tag='+desiredConf[targetSite]['tag']+'"')

            targetJson = json.loads(targetConf[0])

            prop = targetJson['items'][0]['properties']

            # print prop
            if targetSite == 'core-site':
                prop['fs.lustrefs.impl'] = u"org.apache.hadoop.fs.lustrefs.LustreFileSystem"
                prop['fs.AbstractFileSystem.lustrefs.impl'] = u"org.apache.hadoop.fs.local.LustreFs"
                prop['fs.defaultFS'] = u"lustrefs:///"
                prop['fs.lustrefs.mount'] = "/"+lustre_config['items'][0]['properties']['lustrefs_mountpoint']+"/hadoop"
                prop['hadoop.tmp.dir'] = u"/tmp2/hadoop-${user.name}"

            elif targetSite == 'yarn-site':
                prop['yarn.nodemanager.container-executor.class'] = u"org.apache.hadoop.yarn.server.nodemanager.LinuxContainerExecutor"
                prop['yarn.nodemanager.linux-container-executor.nonsecure-mode.local-user'] = u"yarn"
                prop['yarn.nodemanager.linux-container-executor.nonsecure-mode.limit-users'] = u"false"
                prop['yarn.nodemanager.local-dirs'] = u"/tmp2/yarn/local"
                prop['yarn.nodemanager.log-dirs'] = u"/tmp2/yarn/log"
                # prop['yarn.timeline-service.leveldb-state-store.path'] = u"/tmp2/yarn/timeline"
                # prop['yarn.timeline-service.leveldb-timeline-store.path'] = u"/tmp2/yarn/timeline"

            elif targetSite == 'mapred-site':
                prop['yarn.app.mapreduce.am.staging-dir'] = u"/tmp2/hadoop-${user.name}/staging"

                # print json.dumps(prop)

            reqHead = 'curl -u admin:admin -H "X-Requested-By: ambari" -X PUT -d '
            reqBody = '[{"Clusters":{"desired_config":[{"type":"'+targetSite+'","tag":"version'+nextVersionNum+'","service_config_version_note":"New config version"}]}}]'
            reqTail = ' "http://'+self.ambari_url+':8080/api/v1/clusters/'+self.cluster_name+'"'

            body = json.loads(reqBody)
            body[0]['Clusters']['desired_config'][0][u'properties']=prop
            reqBody = json.dumps(body)
            print "======= Req Api ======="
            print reqHead + reqBody+ reqTail
            print "======= Req Api End ======="
            #TEST
            #getData = reqHead + reqBody+ reqTail
            #f = open("/home/lustre-daemon/test.txt", 'w')
            #f.write(getData)
            #f.close()
            reqResult = self.sbtls.sp_open(reqHead+"'"+reqBody+"'"+reqTail)
            print '\n'
            print '+=+=+=+=+=+=+=+'
            print 'ReqResult : '+str(reqResult)
            print '+=+=+=+=+=+=+=+'

    def get_lustre_configs(self):
        confInfo = self.sbtls.sp_open("curl -u admin:admin -X GET  http://"+self.ambari_url+":8080/api/v1/clusters/"+self.cluster_name+"?fields=Clusters/desired_configs")
        j = json.loads(confInfo[0])
        desiredConf = j['Clusters']['desired_configs']
        #self.subprocess_open('echo "lustrefs_config_env desiredConf'+str(desiredConf)+'" >> /tmp/switch-config-log.info')
        targetSite = 'lustrefs-config-env'

        targetConf = self.sbtls.sp_open('curl -u admin:admin -H "X-Requested-By: ambari" -X GET "http://'+self.ambari_url+':8080/api/v1/clusters/'+self.cluster_name+'/configurations?type='+targetSite+'&tag='+desiredConf[targetSite]['tag']+'"')
        #self.subprocess_open('echo "lustrefs_config_env targetConf'+str(targetConf)+'" >> /tmp/switch-config-log.info')
        targetJson = json.loads(targetConf[0])
        return targetJson;

    def get_name_node_host(self):
        confInfo = self.sbtls.sp_open("curl -u admin:admin -X GET  http://"+self.ambari_url+":8080/api/v1/clusters/"+self.cluster_name+"/services/HDFS/components/NAMENODE?fields=host_components")
        j = json.loads(confInfo[0])
        return str(j['host_components'][0]['HostRoles']['host_name']);
        