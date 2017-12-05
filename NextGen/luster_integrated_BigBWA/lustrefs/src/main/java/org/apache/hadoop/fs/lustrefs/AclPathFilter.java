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

import java.io.IOException;
import java.util.ListIterator;
import java.util.Vector;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.security.UserGroupInformation;


public class AclPathFilter {
	
	Vector<String> paths = null;
	
	public AclPathFilter(){
		paths = new Vector<String>();
	}
	/* generates a white list of ACL path regular expressions */
	public AclPathFilter(Configuration conf){
		this();
	    UserGroupInformation ugi = null;
	    try {
			ugi = UserGroupInformation.getCurrentUser();
		} catch (IOException e) {
			
		}
		String stagingRootDir = new Path(conf.get("yarn.app.mapreduce.am.staging-dir",conf.get("mapreduce.jobtracker.staging.root.dir", "/tmp/hadoop/mapred/staging"))).toString();
		String user;
		String randid = "\\d*";
		if (ugi != null) {
			user = ugi.getShortUserName();
		} else {
		    user = "dummy";
		}
		paths.add("^" + new Path(stagingRootDir, user +  randid +"/.staging").toString() + ".*");
		paths.add("^" + (new Path(stagingRootDir, user +  randid +"/.staging")).toUri().getPath() + ".*");
		stagingRootDir = new Path(conf.get("mapreduce.jobtracker.staging.root.dir","/tmp/hadoop/mapred/staging")).toString();
        paths.add("^" + new Path(stagingRootDir, user+"/.staging").toString() + ".*");
        paths.add("^" + (new Path(stagingRootDir, user+"/.staging")).toUri().getPath() + ".*");
		
	}
	
	public boolean matches(String path){
		
		boolean needsAcl = false;
		ListIterator<String> list = paths.listIterator();
		String filterPath = null;
		while(list.hasNext() && !needsAcl){
			filterPath = list.next();
			if(path.matches(filterPath)){
				needsAcl = true;
			}
		}
		return needsAcl;
	}
	public boolean matches(Path path){
		return matches(path.toString());
	}
}
