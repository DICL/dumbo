#!/usr/bin/env python

import sys
import os
import re
import argparse
import subprocess
import itertools
import collections
import copy

env = {
    'IP_MASTERS' : (
        '10.0.5.100',
    ),

    'IP_AGENTS' : (
        '10.0.5.100',
        '10.0.5.101',
        '10.0.5.102',
        #'10.0.5.103',
        '10.0.5.104',
        '10.0.5.105',
        '10.0.5.106',
    ),

    'PREFIX_PROJECT' : '/home/jojaeeon/bigdata',
    'PREFIX_CONFIG' : '{PREFIX_PROJECT}/run/conf',
    'PREFIX_LOG' : '{PREFIX_PROJECT}/run/log',

    'PREFIX_MESOS' : '{PREFIX_PROJECT}/mesos/dist',
    'PREFIX_MESOS_BUILD' : '{PREFIX_PROJECT}/mesos/build',
    'PREFIX_MESOS_WORK' : '/var/lib/mesos',

    'PREFIX_HTRACE' : '{PREFIX_PROJECT}/htrace/htrace-htraced',
    'PREFIX_HTRACE_BUILD' : '{PREFIX_PROJECT}/htrace',

    'PREFIX_HADOOP' : '{PREFIX_PROJECT}/hadoop/hadoop-dist/target/hadoop-2.6.0-cdh5.11.1',
    'PREFIX_HADOOP_BUILD' : '{PREFIX_PROJECT}/hadoop',

    'PREFIX_HADOOP_OLD' : '{PREFIX_PROJECT}/hadoop/hadoop-mapreduce1-project/build/hadoop-2.6.0-mr1-cdh5.11.1',
    'PREFIX_HADOOP_OLD_BUILD' : '{PREFIX_PROJECT}/hadoop/hadoop-mapreduce1-project',

    'PREFIX_MESOS_HADOOP_BUILD' : '{PREFIX_PROJECT}/mesos-hadoop',

    'PATH_HADOOP_OLD_EXAMPLES' : '{PREFIX_HADOOP_OLD}/../hadoop-examples-2.6.0-mr1-cdh5.11.1.jar',
    'PATH_HADOOP_EXAMPLES' : '{PREFIX_HADOOP}/share/hadoop/mapreduce/hadoop-mapreduce-examples-2.6.0-cdh5.11.1.jar',
}

avail_actions = {
    'single' : [
        'build',
        'start',
        'stop',
        'clean',
    ],
    'compound' : {
        'reset' : ['stop', 'clean', 'start',],
    },
}
avail_targets = {
    'single' : [
        'htrace',
        'hdfs',
        'mesos',
        'yarn',
        'hadoop',
        'terasort',
    ],
    'compound' : {
        'all' : ['hdfs', 'mesos', 'hadoop'],
    },
}

parser = argparse.ArgumentParser()
parser.add_argument('-v', '--verbose', action='store_true')
parser.add_argument('-d', '--dryrun', action='store_true')
parser.add_argument('-m', '--actions', type=str, required=True,
        help=','.join(avail_actions['single']+avail_actions['compound'].keys()))
parser.add_argument('-t', '--targets', type=str, required=True,
        help=','.join(avail_targets['single']+avail_targets['compound'].keys()))
args = parser.parse_args()

actions = args.actions.split(',')
assert all(action in avail_actions['single']+avail_actions['compound'].keys() for action in actions)
if any(action in avail_actions['compound'] for action in actions):
    assert len(actions) == 1
    actions = avail_actions['compound'][actions[0]]

targets = args.targets.split(',')
assert all(target in avail_targets['single']+avail_targets['compound'].keys() for target in targets)
if any(target in avail_targets['compound'] for target in targets):
    assert len(targets) == 1
    targets = avail_targets['compound'][targets[0]]

def eformat(conf, fmt):
    while re.search('\{\w*\}', fmt):
        fmt = fmt.format(**conf)
    return fmt

def ip2host(ip):
    proc = subprocess.Popen(('getent', 'hosts', ip), stdout=subprocess.PIPE)
    return proc.stdout.readline().split()[-1]

def run(env, cmd, bg=False, redir=False):
    cmd = eformat(env, cmd)
    if redir:
        cmd += ' >{} 2>&1'.format(eformat(env, redir))
    if bg:
        cmd += ' &'
    if args.verbose:
        print cmd
    if args.dryrun:
        return
    os.system(cmd)

def sshrun(env, ip, cmd, bg=False, redir=False):
    sshcmd = 'ssh {ip} "{cmd}"'.format(ip=ip, cmd=cmd.replace('"', r'\"'))
    run(env, sshcmd, bg, redir)

if 'start' in args.actions:
    for path in subprocess.check_output(eformat(env, 'find {PREFIX_CONFIG} -name masters | grep -v backup'), shell=True).rstrip().split():
        with open(path, 'w') as fp:
            for ip in env['IP_MASTERS']:
                fp.write('{}\n'.format(ip))
    for path in subprocess.check_output(eformat(env, 'find {PREFIX_CONFIG} -name slaves | grep -v backup'), shell=True).rstrip().split():
        with open(path, 'w') as fp:
            for ip in env['IP_AGENTS']:
                fp.write('{}\n'.format(ip))

    if 'start' in actions:
        env['log_dir'] = '{PREFIX_LOG}/{host}'
        for ip in env['IP_MASTERS'] + env['IP_AGENTS']:
            env.update({'ip' : ip, 'host' : ip2host(ip)})
            log_dir = eformat(env, '{log_dir}')
            if not os.path.isdir(log_dir):
                if args.verbose:
                    print log_dir
                os.makedirs(log_dir)

def start_or_stop(action, target):
    if target == 'hdfs':
        if action == 'build':
            run(env, 'cd {PREFIX_HADOOP_BUILD}; mvn -e -X package -Pdist,native -DskipTests -Dmaven.javadoc.skip=true -Dtar')
        elif action == 'start':
            run(env, '{PREFIX_HADOOP}/sbin/start-dfs.sh')
        elif action == 'stop':
            run(env, '{PREFIX_HADOOP}/sbin/stop-dfs.sh')
        elif action == 'clean':
            run(env, 'rm {PREFIX_LOG}/*/hadoop-*namenode-* {PREFIX_LOG}/*/hadoop-*-datanode-*')

    elif target == 'mesos':
        if action == 'build':
            if not os.path.isdir(eformat(env, '{PREFIX_MESOS_BUILD}')):
                os.makedirs(eformat(env, '{PREFIX_MESOS_BUILD}'))
                run(env, 'cd {PREFIX_MESOS_BUILD}/..; ./bootstrap')
                run(env, 'cd {PREFIX_MESOS_BUILD}; ../configure --prefix={PREFIX_MESOS}')
            run(env, 'make -C {PREFIX_MESOS_BUILD} -j16 install')
        elif action == 'start':
            run(env, '{PREFIX_MESOS}/sbin/mesos-start-cluster.sh')
        elif action == 'stop':
            run(env, '{PREFIX_MESOS}/sbin/mesos-stop-cluster.sh')
        elif action == 'clean':
            run(env, 'rm {PREFIX_LOG}/*/*mesos*')
            for ip in env['IP_MASTERS'] + env['IP_AGENTS']:
                print ip, eformat(env, 'rm -rf {PREFIX_MESOS_WORK}/*')
                sshrun(env, ip, 'rm -rf {PREFIX_MESOS_WORK}/*')

    elif target == 'yarn':
        if action == 'build':
            run(env, 'cd {PREFIX_HADOOP_BUILD}; mvn -e -X package -Pdist,native -DskipTests -Dmaven.javadoc.skip=true -Dtar')
        elif action == 'start':
            run(env, '{PREFIX_HADOOP}/sbin/start-yarn.sh')
        elif action == 'stop':
            run(env, '{PREFIX_HADOOP}/sbin/stop-yarn.sh')
        elif action == 'clean':
            run(env, 'rm {PREFIX_LOG}/*/yarn-*')

    elif target == 'htrace':
        if action == 'build':
            run(env, 'cd {PREFIX_HTRACE_BUILD}; mvn install -DskipTests -Dmaven.javadoc.skip=true assembly:single -Pdist')
            run(env, 'cp {PREFIX_HTRACE}/target/htrace-htraced-4.2.0-incubating.jar {PREFIX_HADOOP}/share/hadoop/common/lib')
            run(env, 'cp {PREFIX_HTRACE}/target/htrace-htraced-4.2.0-incubating.jar {PREFIX_HADOOP_OLD}/lib')
        elif action == 'start':
            run(env, 'HTRACED_CONF_DIR={PREFIX_HTRACE}/example {PREFIX_HTRACE}/go/build/htraced >/dev/null 2>&1 &')
        elif action == 'stop':
            run(env, 'killall -9 htraced')
        elif action == 'clean':
            run(env, 'rm {PREFIX_LOG}/*/htrace*')

    elif target == 'hadoop':
        if action == 'build':
            run(env, 'cd {PREFIX_HADOOP_OLD_BUILD}; ant -v -d binary examples')
            run(env, 'cd {PREFIX_MESOS_HADOOP_BUILD}; mvn package')
            run(env, 'cp {PREFIX_MESOS_HADOOP_BUILD}/target/hadoop-mesos-*.jar {PREFIX_HADOOP_OLD}/lib')
        elif action == 'start':
            run(env, '{PREFIX_HADOOP_OLD}/bin/start-mesos-jobtracker.sh')
        elif action == 'stop':
            run(env, '{PREFIX_HADOOP_OLD}/bin/stop-mesos-jobtracker.sh')
        elif action == 'clean':
            run(env, 'rm {PREFIX_LOG}/*/hadoop-*-jobtracker-*')
            run(env, 'rm -rf {PREFIX_LOG}/*/history {PREFIX_LOG}/*/userlogs {PREFIX_LOG}/*/job_*')

    elif target == 'terasort':
        if action == 'build':
            run(env, '{PREFIX_HADOOP}/bin/hadoop dfs -rm -r -f /ts_in')
            run(env, '{PREFIX_HADOOP}/bin/hadoop jar {PATH_HADOOP_EXAMPLES} teragen 1000 /ts_in')
        elif action == 'start':
            run(env, '{PREFIX_HADOOP}/bin/hadoop jar {PATH_HADOOP_EXAMPLES} terasort /ts_in /ts_out')
        elif action == 'stop':
            pass
        elif action == 'clean':
            run(env, '{PREFIX_HADOOP}/bin/hadoop dfs -rm -r -f /ts_out')

for (action, target) in itertools.product(actions, targets):
    start_or_stop(action, target)
