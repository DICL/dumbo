/**
 * Lustre file system for Hadoop
 * 
 * Uses lustrefs:/// as the file system.
 *
 * Based on the glusterfs-hadoop plugin
 * https://forge.gluster.org/hadoop/pages/Architecture
 * https://forge.gluster.org/hadoop
 *
 * Modifications by:
 * Seagate Technology
 * December 2014
 */

package org.apache.hadoop.fs.lustrefs;

import java.io.File;
import java.io.IOException;

import org.apache.hadoop.fs.FileUtil;
import org.apache.hadoop.util.Shell;

public class Util{

    public static String execCommand(File f,String...cmd) throws IOException{
        String[] args=new String[cmd.length+1];
        System.arraycopy(cmd, 0, args, 0, cmd.length);
        args[cmd.length]=FileUtil.makeShellPath(f, true);
        String output=Shell.execCommand(args);
        return output;
    }

    /* copied from unstalbe hadoop API org.apache.hadoop.Shell */
    public static String[] getGET_PERMISSION_COMMAND(){
        // force /bin/ls, except on windows.
        return new String[]{(WINDOWS ? "ls" : "/bin/ls"),"-ld"};
    }
    
    /* copied from unstalbe hadoop API org.apache.hadoop.Shell */
    
    public static final boolean WINDOWS /* borrowed from Path.WINDOWS */
    =System.getProperty("os.name").startsWith("Windows");
    // / loads permissions, owner, and group from `ls -ld`
}
