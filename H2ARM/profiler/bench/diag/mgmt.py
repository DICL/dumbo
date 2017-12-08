#!/usr/bin/env python

import sys, os, re
import argparse
import collections
import itertools
import subprocess

def eformat(conf, fmt):
    while re.search('\{\w*\}', fmt):
        fmt = fmt.format(**conf)
    return fmt

genv = {
    'HOME' : os.environ['HOME'],
    'PROJECT_PREFIX' : '{HOME}/bigdata',
    'DIAG_PREFIX' : '{PROJECT_PREFIX}/bench/diag',
    'HADOOP_PREFIX' : '{PROJECT_PREFIX}/hadoop/hadoop-dist/target/hadoop-2.6.0-cdh5.11.1',
    'YARN_UNMANAGED_AM_LAUNCHER' : '{HADOOP_PREFIX}/share/hadoop/yarn/hadoop-yarn-applications-unmanaged-am-launcher-2.6.0-cdh5.11.1.jar',
    'DIAG_JAR' : 'diag-0.0.1-jar-with-dependencies.jar',
    'DIAG_JAR_PATH' : '{DIAG_PREFIX}/target/{DIAG_JAR}',
    'DIAG_JAR_HDFS_PATH' : '/tmp/{DIAG_JAR}',
    'DIAG_PACKAGE' : 'hpcs.bigdata',
    'MESOS_MASTER' : 'TRUSTY100:5050',
    'NUM_CONTAINERS' : 4,

    'HDFS_MASTER'   : 'TRUSTY100',
    'HDFS_GEN_DIR'  : '/file_access_app',
    #'HDFS_GEN_SIZE' : 134217728,
    'HDFS_GEN_SIZE' : 67108864,
    'HDFS_GEN_CNT'  : 8,
}
os.environ['HADOOP_HOME'] = eformat(genv, '{HADOOP_PREFIX}')
nodes = [
    'TRUSTY100',
    'TRUSTY101',
    'TRUSTY102',
    'TRUSTY104',
    'TRUSTY105',
    'TRUSTY106',
]

parser = argparse.ArgumentParser()
parser.add_argument('TARGET', choices=['build', 'prepare', 'yarn-um', 'yarn', 'mesos'])
parser.add_argument('-v', '--verbose', action='store_true')
parser.add_argument('-d', '--dryrun', action='store_true')
args = parser.parse_args()

def runs(env, cmd):
    fcmd = eformat(env, cmd)
    if args.verbose:
        print fcmd
    if not args.dryrun:
        os.system(fcmd)

def runp(env, cmd, **kwargs):
    fcmd = eformat(env, cmd)
    if args.verbose:
        print fcmd
    if args.dryrun:
        fcmd = 'echo ""'
    proc = subprocess.Popen(fcmd, shell=True, **kwargs)
    return proc

def get_hdfs_dist():
    proc = runp(genv, '{HADOOP_PREFIX}/bin/hadoop jar {DIAG_JAR_PATH} {DIAG_PACKAGE}.app.FileStatApp', stdout=subprocess.PIPE)
    all_hosts = set()
    file2hosts = {}
    hosts2files = collections.defaultdict(set)
    for line in itertools.imap(str.rstrip, proc.stdout):
        #print line
        key, val = line.split()
        if key == 'D':
            dir = val
        elif key == 'F':
            file = val
            file2hosts[file] = []
        elif key == 'O':
            offset = val
        elif key == 'H':
            host = val
            all_hosts.add(host)
            hosts = file2hosts[file]
            hosts.append(host)
            if len(hosts) == 3:
                hosts2files[tuple(sorted(hosts))].add(file)
    #for file in sorted(file2hosts.keys()):
    #    print file, file2hosts[file]
    for hosts in itertools.combinations(sorted(all_hosts), 3):
        print hosts, sorted(hosts2files[hosts])
    return [(hosts, sorted(hosts2files[hosts])) for hosts in itertools.combinations(sorted(all_hosts), 3)]

if args.TARGET == 'build':
    runs(genv, 'mvn clean compile assembly:single')

elif args.TARGET == 'prepare':
    runs(genv, '{HADOOP_PREFIX}/bin/hdfs dfs -rm -r -f {HDFS_GEN_DIR}')
    runs(genv, '{HADOOP_PREFIX}/bin/hdfs dfs -mkdir {HDFS_GEN_DIR}')
    nid = 0
    procs = []
    for fid in itertools.count():
        env = genv.copy()
        env.update({
            'NODE' : nodes[nid],
            'FILE_NAME' : '{:02d}'.format(fid),
            'FILE_PATH' : os.path.join('{HDFS_GEN_DIR}/{FILE_NAME}')
        })
        proc = runp(env, 'ssh {NODE} "head -c {HDFS_GEN_SIZE} </dev/urandom | {HADOOP_PREFIX}/bin/hdfs dfs -put - {FILE_PATH}"')
        procs.append(proc)
        nid = (nid + 1) % len(nodes)
        if nid == 0:
            for proc in procs:
                proc.wait()
            if (fid + 1) % (4 * len(nodes)) == 0:
                all_full = True
                for hosts, files in get_hdfs_dist():
                    all_full &= len(files) >= genv['HDFS_GEN_CNT']
                    for file in files[genv['HDFS_GEN_CNT']:]:
                        env.update({
                            'FILE_NAME' : file,
                            'FILE_PATH' : os.path.join('{HDFS_GEN_DIR}/{FILE_NAME}')
                        })
                        runp(env, '{HADOOP_PREFIX}/bin/hdfs dfs -rm {FILE_PATH}')
                if all_full:
                    break
    get_hdfs_dist()

elif args.TARGET == 'yarn-um':
    runs(genv, '{HADOOP_PREFIX}/bin/hadoop jar {YARN_UNMANAGED_AM_LAUNCHER} Client -classpath {DIAG_JAR_PATH} -cmd "java {DIAG_PACKAGE}.yarn.ApplicationMaster /bin/date {NUM_CONRAINERS}"')

elif args.TARGET == 'yarn':
    runs(genv, '{HADOOP_PREFIX}/bin/hdfs dfs -copyFromLocal -f {DIAG_JAR_PATH} {DIAG_JAR_HDFS_PATH}')
    runs(genv, '{HADOOP_PREFIX}/bin/hadoop jar {DIAG_JAR_PATH} {DIAG_PACKAGE}.yarn.Client /bin/date {NUM_CONTAINERS} hdfs://TRUSTY100{DIAG_JAR_HDFS_PATH}')

elif args.TARGET == 'mesos':
    runs(genv, 'java -cp {DIAG_JAR_PATH} {DIAG_PACKAGE}.mesos.MesosDiagMain {MESOS_MASTER} {NUM_CONTAINERS}')

else:
    raise NotImplemented
