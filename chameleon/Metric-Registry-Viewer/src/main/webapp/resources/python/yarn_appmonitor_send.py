#!/usr/bin/env python
# -*- coding: utf-8 -*-


import psycopg2
import json
import socket
import ConfigParser

import os, sys, time
from datetime import datetime

import logging
import logging.handlers

import subprocess

database_url = '192.168.1.190';
database_port = '5432';
database_user = 'postgres';
database_password = 'postgres';
database_neme = 'ambari';
database_table_name = 'apptable_81_20181120135509';


pidstatcpu_list = None;

perfbranchmisses_list = None;


logger = None;
now = None;
create_time = None;
local_hostname = None;


class Chameleon_agent:

    # get jsp data
    def getApplicationDataForJps(self):
        proc1 = subprocess.Popen(['/usr/jdk64/jdk1.8.0_112/bin/jps', '-v'], stdout=subprocess.PIPE)
        proc2 = subprocess.Popen(['grep', 'application_'], stdin=proc1.stdout,stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        proc1.stdout.close() # Allow proc1 to receive a SIGPIPE if proc2 exits.
        out, err = proc2.communicate()
        return out;
        pass

    def getPIDForString(self,data):
        result_list = [];
        jps_list = data.split('\n');
        for jps_data in jps_list:
            if jps_data:
                result = jps_data.split()
                result_list.append(result[0]);
        return result_list;
        pass

    # get application id
    def getApplicationIDForString(self,data):
        result_list = [];
        jps_list = data.split('\n');
        for jps_data in jps_list:
            if jps_data:
                arrayString = jps_data.split('/');
                for string in arrayString:
                    if 'application_' in string:
                        #print string
                        result_list.append(string)
                        break;
        
        return result_list;
        pass

    # get container id
    def getContainerIDForString(self,data):
        result_list = [];
        jps_list = data.split('\n');
        for jps_data in jps_list:
            if jps_data:
                arrayString = jps_data.split('/');
                for string in arrayString:
                    if 'container_' in string:
                        #print string
                        result_list.append(string)
                        break;
        
        return result_list;
        pass
    
    def get_pidstatcpu(self,PID_list) :
    	global pidstatcpu_list;
    	global logger;
    	
    	pidstatcpu_list = [];
    	result_list = [];
    	for PID in PID_list:
            command = 'pidstat -l -p '+PID+' 1 1 | awk \'$1 ~ /^Average/{ print $7 }\'';
            proc1 = subprocess.Popen(command , shell = True , stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            out, err = proc1.communicate()
            tmp = out.split('\n')[0]
            result_list.append(tmp);
        pidstatcpu_list = result_list;
        pass
    
    def get_perfbranchmisses(self,PID_list) :
    	global perfbranchmisses_list;
    	global logger;
    	
    	perfbranchmisses_list = [];
    	result_list = [];
    	for PID in PID_list:
            command = '(perf stat -x\' \' -p '+PID+' -e branch-misses sleep 3) 2>&1 | awk \'{print $1}\'';
            proc1 = subprocess.Popen(command , shell = True , stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            out, err = proc1.communicate()
            tmp = out.split('\n')[0]
            result_list.append(tmp);
        perfbranchmisses_list = result_list;
        pass
    

    def sendServer(self,cur,PID,application_id,container_id):
        json_body = []

        global local_hostname;

        global database_url;
        global database_port;
        global database_user;
        global database_password;
        global database_neme;
        global database_table_name;
        global database_type;
        global logger;
        global create_time;
        
        
        global pidstatcpu_list;
        
        global perfbranchmisses_list;
        


        query_string = '''
        	insert INTO apptable_81_20181120135509 (
        		create_date
        		,pid
        		,application_id
        		,container_id
        		,node
        		
        		,pidstatcpu
        		
        		,perfbranchmisses
        		
        	) 
        	VALUES 
        ''';
        for i in range(0 , len(PID)):
            i = i - 1;
            if i > -1:
                query_string += ',';
            query_string += '( \''+create_time+'\' , \''+PID[i]+'\' , \''+application_id[i]+'\' , \''+container_id[i]+'\'  , \''+local_hostname+'\'   ,\''+pidstatcpu_list[i]+'\'  ,\''+perfbranchmisses_list[i]+'\' )';

        #print(query_string)
        try:
            logger.info(PID);
            cur.execute(query_string);
        except Exception as ex:
            logger.error(ex)


if __name__ == "__main__":
	
    local_hostname = socket.gethostname();
    file_path = os.path.dirname(os.path.realpath(__file__));
    now = time.localtime()
    create_time = "%04d-%02d-%02d %02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec);
    # logger config
    logger = logging.getLogger('mylogger')
    fileMaxByte = 1024 * 1024 * 100 #100MB
    log_file_name = file_path+'/chameleon.log';
    filehander = logging.handlers.RotatingFileHandler(log_file_name, maxBytes=fileMaxByte, backupCount=10)
    streamHandler = logging.StreamHandler()
    fomatter = logging.Formatter('[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s > %(message)s')
    filehander.setFormatter(fomatter)
    logger.addHandler(filehander)
	
    try:
        conn = psycopg2.connect(dbname=database_neme, host=database_url, port=database_port, user=database_user , password=database_password);
        cur = conn.cursor()
    except Exception as ex:
        logger.error(ex)	
    
    chameleon_agent = Chameleon_agent();
    json_body = [];
    jps_data = chameleon_agent.getApplicationDataForJps();
    
    if jps_data:
    	logger.info('find PID' + jps_data)
    	PID = chameleon_agent.getPIDForString(jps_data);
        application_id = chameleon_agent.getApplicationIDForString(jps_data);
        container_id = chameleon_agent.getContainerIDForString(jps_data);
        
        chameleon_agent.get_pidstatcpu(PID);
        
        chameleon_agent.get_perfbranchmisses(PID);
        
        
        try:
            chameleon_agent.sendServer(cur,PID,application_id,container_id);
            conn.commit();
        except Exception as ex:
            logger.error(ex)
            pass
    pass;