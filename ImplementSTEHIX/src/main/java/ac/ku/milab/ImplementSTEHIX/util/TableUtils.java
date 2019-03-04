package ac.ku.milab.ImplementSTEHIX.util;

import org.apache.hadoop.hbase.TableName;
import org.apache.hadoop.hbase.util.Bytes;

/* This class is for Utility of table functions */
public class TableUtils {

	/**
	 * @param tableName
	 * @return whether this table is meta table or root table
	 */

	public static boolean isSystemTable(byte[] tableName) {
		TableName tName = TableName.valueOf(tableName);

		if(tName.isSystemTable()){
			return true;
		}else{
			return false;
		}
	}
	
	/**
	 * @param tableName
	 * @return whether this table is meta table or root table
	 */
	
	public static boolean isSystemTable(TableName tableName) {

		if(tableName.isSystemTable()){
			return true;
		}else{
			return false;
		}
	}
	
	/**
	 * @param tableName
	 * @return whether this table is meta table or root table
	 */
	
	public static boolean isSystemTable(String tableName) {

		TableName tName = TableName.valueOf(tableName);
		if(tName.isSystemTable()){
			return true;
		}else{
			return false;
		}
	}

	/**
	 * @param tableName
	 *            user table name
	 * @param startKey
	 *            region's start key
	 * @param regionServer
	 *            region server having region
	 * @return index table name of user table
	 */
//
//	public static HRegion getIndexTableRegion(String tableName, byte[] startKey, HRegionServer regionServer) {
//		String indexTableName = getIndexTableName(tableName);
//		Collection<HRegion> idxTableRegions = regionServer.getOnlineRegions(Bytes.toBytes(indexTableName));
//		for (HRegion idxTableRegion : idxTableRegions) {
//			if (Bytes.equals(idxTableRegion.getStartKey(), startKey)) {
//				return idxTableRegion;
//			}
//		}
//		return null;
//	}
}
