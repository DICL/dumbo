/*
 * HeTri: Multi-level Node Coloring for Efficient Triangle Enumeration on Heterogeneous Clusters
 * Authors: Ha-Myung Park and U Kang
 *
 * -------------------------------------------------------------------------
 * File: MemoryInfo.java
 * - System memory state monitor.
 */

package hetri.tool;

import oshi.SystemInfo;
import oshi.hardware.GlobalMemory;

public class MemoryInfo {

    final static double RESERVED_RATIO = 0.05;

    static SystemInfo si = new SystemInfo();
    static GlobalMemory gm = si.getHardware().getMemory();
    static long total = gm.getTotal();
    final static long RESERVED = (long) (total * RESERVED_RATIO);

    /**
     * @return total memory - free memory
     */
    static public long getUsedMem(){
        Runtime runtime = Runtime.getRuntime();
        return runtime.totalMemory() - runtime.freeMemory();
    }

    /**
     * @return free memory - reserved memory
     */
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
