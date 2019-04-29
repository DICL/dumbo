package client;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.hbase.HBaseConfiguration;
import org.apache.hadoop.hbase.HColumnDescriptor;
import org.apache.hadoop.hbase.HTableDescriptor;
import org.apache.hadoop.hbase.client.HBaseAdmin;

public class myCreate {
//	public myCreate(){
//		Configuration conf = HBaseConfiguration.create();
//		HTableDescriptor desc = new HTableDescriptor("test");
//		
//		HColumnDescriptor hcd = new HColumnDescriptor("aa");
//		desc.addFamily(hcd);
//		desc.addIndexColumn("aa");
//		
//		try{
//			HBaseAdmin admin = new HBaseAdmin(conf);
//			admin.createTable(desc);
//			admin.close();
//		}catch(Exception e){
//			e.printStackTrace();
//		}
//		
//	}
//	
//	public myCreate(String tableName){
//		Configuration conf = HBaseConfiguration.create();
//		HTableDescriptor desc = new IdxHTableDescriptor(tableName);
//		
//		HColumnDescriptor hcd = new HColumnDescriptor("aa");
//		desc.addFamily(hcd);
//		
//		try{
//			HBaseAdmin admin = new HBaseAdmin(conf);
//			admin.createTable(desc);
//			admin.close();
//		}catch(Exception e){
//			e.printStackTrace();
//		}
//		
//	}
}
