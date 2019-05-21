#!/bin/bash

# chkconfig: 345 88 08
# description: mds producer Daemon


WORKDIR=/var/lib/ambari-agent/cache/stacks/HDP/2.6/services/LUSTREMANAGER/package/scripts/modules/lustre_producer_oss
DAEMON=lustre_producer_oss.py
LOG=/tmp/lustre_producer_oss.log

function do_start()
{
        nohup python ${WORKDIR}/${DAEMON} & >> ${LOG}
}



function do_stop()
{
        PID=`ps -ef | grep ${DAEMON} | grep -v grep | awk '{print $2}'`
        if [ "$PID" != "" ]; then
                kill -9 $PID
        fi
}



case "$1" in
    start|stop)
        do_${1}
        ;;
    reload|restart)
        do_stop
        do_start
        ;;
    *)

        echo "Usage: /etc/init.d/tunnel {start|stop|restart}"
        exit 1
        ;;
esac