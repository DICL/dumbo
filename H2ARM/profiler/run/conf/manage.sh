#!/bin/bash

# A POSIX variable
OPTIND=1         # Reset in case getopts has been used previously in the shell.

# Initialize our own variables:
BACKUP_DIR="backup"
mode="backup"
dryrun=0
verbose=0

while getopts ":h?f:rdv" opt; do
    case "$opt" in
    h|\?)
        echo "Usage: $(basename $0) [-f BACKUP_DIR] [-v] [-r]"
        echo "  -f BACKUP_DIR   backup the files under BACKUP_DIR (default: ${BACKUP_DIR})"
        echo "  -r              restore instead of backup"
        echo "  -d              dryrun"
        echo "  -v              verbose"
        exit 0
        ;;
    f)  BACKUP_DIR="$OPTARG"
        ;;
    r)  mode="restore"
        ;;
    d)  dryrun=1
        ;;
    v)  verbose=1
        ;;
    esac
done

shift $((OPTIND-1))

if [ $mode == "backup" ] && [ -d "${BACKUP_DIR}" ]; then
    echo "ERROR: cannot overwrite to '${BACKUP_DIR}'"
    exit 1
fi

for link in $(find -L . -xtype l); do
    src=$(readlink -f $link)
    dst="$BACKUP_DIR/$link"
    if [ $mode == "backup" ]; then
        [ $verbose -eq 1 ] && echo "$src -> $dst"
        [ $dryrun -eq 0 ] && mkdir -p "${BACKUP_DIR}/$(dirname $link)" && cp $src $dst
    else
        [ $verbose -eq 1 ] && echo "$dst -> $src"
        [ $dryrun -eq 0 ] && cp $dst $src
    fi
done
