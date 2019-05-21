#!/usr/bin/env python
# -*- coding: utf-8 -*-
{{!--
jps 을 통하여 데이터를 수집하는 파이썬 코드의 템플릿 파일;
* 이 파일은 Handlebars 에 의하여 yarn_appmonitor_send.py 파일로 저장됩니다.
--}}

import psycopg2
import json
import socket
import ConfigParser

import os, sys, time
from datetime import datetime

import logging
import logging.handlers

import subprocess

database_url = '{{{database.url}}}';
database_port = '{{{database.port}}}';
database_user = '{{{database.user}}}';
database_password = '{{{database.password}}}';
database_neme = '{{{database.neme}}}';
database_table_name = '{{{database.table_name}}}';

{{#each col_list}}
{{{col_name}}}_list = None;
{{/each}}

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
    {{#each col_list}}
    def get_{{col_name}}(self,PID_list) :
    	global {{col_name}}_list;
    	global logger;

    	{{col_name}}_list = [];
    	result_list = [];
    	for PID in PID_list:
            command = '{{{parser}}}';
            proc1 = subprocess.Popen(command , shell = True , stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            out, err = proc1.communicate()
            tmp = out.split('\n')[0]
            {{!--
             출력결과가 수치가 아닐경우 공란처리
            --}}
            if not tmp.replace('.','',1).isdigit():
                tmp = ''
            result_list.append(tmp);
        {{col_name}}_list = result_list;
        pass
    {{/each}}

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

        {{#each col_list}}
        global {{{col_name}}}_list;
        {{/each}}


        query_string = '''
        	insert INTO {{{database.table_name}}} (
        		create_date
        		,pid
        		,application_id
        		,container_id
        		,node
        		{{#each col_list}}
        		,{{{col_name}}}
        		{{/each}}
        	)
        	VALUES
        ''';
        for i in range(0 , len(PID)):
            i = i - 1;
            if i > -1:
                query_string += ',';
            query_string += '( \''+create_time+'\' , \''+PID[i]+'\' , \''+application_id[i]+'\' , \''+container_id[i]+'\'  , \''+local_hostname+'\'  {{#each col_list}} ,\''+{{{col_name}}}_list[i]+'\' {{/each}})';

        #print(query_string)
        try:
            logger.info(PID);
            logger.info(query_string);
            cur.execute(query_string);
        except Exception as ex:
            logger.error(ex)


if __name__ == "__main__":

    now = time.localtime()
    local_hostname = socket.gethostname();
    file_path = os.path.dirname(os.path.realpath(__file__));
    # create_time = "%04d-%02d-%02d %02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec);
    create_time = str(datetime.utcnow().isoformat());
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
        {{#each col_list}}
        chameleon_agent.get_{{col_name}}(PID);
        {{/each}}

        try:
            chameleon_agent.sendServer(cur,PID,application_id,container_id);
            conn.commit();
        except Exception as ex:
            logger.error(ex)
            pass
    pass;
