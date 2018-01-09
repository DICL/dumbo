THISFILE=$(readlink -f $BASH_SOURCE)
PDIR=${THISFILE%/*}

CPS=$(echo $CLASSPATH|tr ':' '\n')
for CP in $PDIR/bin \
    $PDIR/lib/jikesrvm3.1.4.jar \
    ;do 
    for P in $CPS;do
        if [[ $P = $CP ]];then
            continue 2
        fi
    done
    CLASSPATH=$CP:$CLASSPATH
done

LPS=$(echo $LD_LIBRARY_PATH|tr ':' '\n')
for LP in $PDIR/lib \
          $PDIR/bin \
          ;do
    for P in $LPS;do
        if [[ $LP = $P ]];then
            continue 2
        fi
    done
    LD_LIBRARY_PATH=$LP:$LD_LIBRARY_PATH
done

