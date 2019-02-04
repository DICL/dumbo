package hetri.tool;

import oshi.SystemInfo;
import oshi.hardware.GlobalMemory;

public class MemoryInfo {

    final static double RESERVED_RATIO = 0.05;

    static SystemInfo si = new SystemInfo();
    static GlobalMemory gm = si.getHardware().getMemory();
    static long total = gm.getTotal();
    final static long RESERVED = (long) (total * RESERVED_RATIO);


    static public long getUsedMem(){
        Runtime runtime = Runtime.getRuntime();
        return runtime.totalMemory() - runtime.freeMemory();
    }

    static public long getFreeMem(){
        long free = gm.getAvailable();

        if(free < total * 0.2){
            System.gc();
            System.gc();
        }

        free = gm.getAvailable();
        return free - RESERVED;
    }
}
