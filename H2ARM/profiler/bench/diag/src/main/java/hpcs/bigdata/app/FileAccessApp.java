package hpcs.bigdata.app;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.RemoteIterator;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.LocatedFileStatus;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.BlockLocation;
import java.lang.System;
import java.io.PrintWriter;
import java.util.Arrays;

public class FileAccessApp {

  static Configuration conf;
  static FileSystem fs;

  public void run(String[] args) throws Exception {
    /*
    RemoteIterator<LocatedFileStatus> itor = fs.listFiles(new Path("/"), false);
    while (itor.hasNext()) {
      LocatedFileStatus child = itor.next();
      System.out.println(child.toString());
    }
    */
    String prefix = "/file_access_app";
    System.out.println("D "+fs.getName()+prefix);
    for (FileStatus stat: fs.listStatus(new Path(prefix))) {
      System.out.println("F "+stat.getPath().getName());
      BlockLocation[] locs = fs.getFileBlockLocations(stat, 0, stat.getLen());
      for (BlockLocation loc: locs) {
        System.out.println("O "+loc.getOffset());
        for (String host: loc.getNames())
          System.out.println("H "+host);
      }
    }
  }

  public static void main(String[] args) throws Exception {
    conf = new Configuration();
    //Configuration.dumpConfiguration(conf, new PrintWriter(System.out));
    //System.out.println("\n");
    fs = FileSystem.get(conf);
    FileAccessApp app = new FileAccessApp();
    app.run(args);
  }
}

