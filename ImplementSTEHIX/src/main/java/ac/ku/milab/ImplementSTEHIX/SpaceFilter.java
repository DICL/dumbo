package ac.ku.milab.ImplementSTEHIX;

import java.io.IOException;
import java.util.List;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.hbase.Cell;
import org.apache.hadoop.hbase.CellUtil;
import org.apache.hadoop.hbase.exceptions.DeserializationException;
import org.apache.hadoop.hbase.filter.FilterBase;
import org.apache.hadoop.hbase.util.Bytes;

public class SpaceFilter extends FilterBase {
	
	private double x1 = 0.0;
	private double y1 = 0.0;
	private double x2 = 0.0;
	private double y2 = 0.0;
	private boolean filterRow = false;
	
  private final byte[] bLat = Bytes.toBytes("lat");
  private final byte[] bLon = Bytes.toBytes("lon");
	
	private static final Log LOG = LogFactory.getLog(SpaceFilter.class.getName());
	
	public SpaceFilter() {
		super();
	}
	
	public SpaceFilter(double x1, double y1, double x2, double y2){
		this.x1 = x1;
		this.y1 = y1;
		this.x2 = x2;
		this.y2 = y2;
	}
	
	public SpaceFilter(double x1, double y1, double x2, double y2, boolean filterRow){
		this.x1 = x1;
		this.y1 = y1;
		this.x2 = x2;
		this.y2 = y2;
		this.filterRow = filterRow;
	}
	
	@Override
	public boolean filterRowKey(byte[] buffer, int offset, int length) throws IOException {
		// TODO Auto-generated method stub
//		byte[] rowkey = Bytes.copy(buffer, offset, length);
		
		//LOG.info("buffer is : "+Bytes.toString(rowkey));
//		byte[] carNum = Bytes.copy(rowkey, 0, 9);
//		byte[] time = Bytes.copy(rowkey, 10, 8);
//		
//		if(Bytes.equals(carNum, this.carNum)){
//			LOG.info("COLLECT");
//		}
//		String rowKey = Bytes.toString(rowkey);
//		byte[] query = Bytes.add(Bytes.toBytes("idx"),this.qualNum,this.value);
//		byte[] query = Bytes.add(query, Bytes.toBytes("2v"));
		//String qualValue = rowKey.split("idx")[1];
		//String val = Bytes.toString(this.value);
		//LOG.info("qual value is : "+ qualValue);
		
//		if(rowKey.contains(val)){
//			return false;
//		}else{
//			return true;
//		}
		
//		if(Bytes.contains(rowkey, query)){
//			return false;
//		}else{
//			return true;
//		}
	
		return super.filterRowKey(buffer, offset, length);
		
	}
	
	
	@Override
	public boolean filterRow() {
		// TODO Auto-generated method stub
		return filterRow;
	}
	
//	@Override
//	public boolean filterAllRemaining() {
//		// TODO Auto-generated method stub
//	}
	
	@Override
	public void reset() {
		// TODO Auto-generated method stub
		this.filterRow = false;
	}
	

	@Override
	public ReturnCode filterKeyValue(Cell c) throws IOException {
		// TODO Auto-generated method stub
	  byte[] qualifier = CellUtil.cloneQualifier(c);
		//LOG.info(Bytes.toString(qualifier));
		if(Bytes.compareTo(qualifier, bLat)==0){
		  byte[] val = CellUtil.cloneValue(c);
		  double lat = Bytes.toDouble(val);
		  if(lat>y1 && lat < y2){
		    filterRow = false;
		    return ReturnCode.INCLUDE_AND_NEXT_COL;
		  }else{
		    return ReturnCode.NEXT_ROW;
		  }
		}
		else if(Bytes.compareTo(qualifier, bLon)==0){
		  byte[] val = CellUtil.cloneValue(c);
		  double lon = Bytes.toDouble(val);
      if(lon > x1 && lon < x2){
        filterRow = false;
        return ReturnCode.INCLUDE_AND_NEXT_COL;
      }else{
        return ReturnCode.NEXT_ROW;
      }
		}else{
		  return ReturnCode.INCLUDE;
		}
	}

	
	@Override
	public void filterRowCells(List<Cell> ignored) throws IOException {
	  // TODO Auto-generated method stub
	  //super.filterRowCells(ignored);
	  if(ignored.size()>=2){
	    super.filterRowCells(ignored);
	  }else{
	    ignored.clear();
	  }
	}
	
	@Override
	public boolean hasFilterRow() {
	  // TODO Auto-generated method stub
	  return true;
	}
	
	public void setFilterRow(boolean filterRow) {
		this.filterRow = filterRow;
	}
	
	@Override
	public byte[] toByteArray(){
		
		byte[] array = new byte[0];
		array = Bytes.add(array, Bytes.toBytes(this.x1));
		array = Bytes.add(array, Bytes.toBytes(this.y1));
		array = Bytes.add(array, Bytes.toBytes(this.x2));
		array = Bytes.add(array, Bytes.toBytes(this.y2));
		array = Bytes.add(array, Bytes.toBytes(this.filterRow));
		return array;
	}
	
	public static SpaceFilter parseFrom(byte[] bytes) throws DeserializationException{
		SpaceFilter filter = null;
		int length = bytes.length;
		
		double x1 = Bytes.toDouble(Bytes.copy(bytes, 0, 8));
		double y1 = Bytes.toDouble(Bytes.copy(bytes, 8, 8));
		double x2 = Bytes.toDouble(Bytes.copy(bytes, 16, 8));
		double y2 = Bytes.toDouble(Bytes.copy(bytes, 24, 8));
		
		boolean filterRow = Bytes.toBoolean(Bytes.copy(bytes, 32, 1));
		
		filter = new SpaceFilter(x1, y1, x2, y2, filterRow);
		
		return filter;
	}
	
	public double getX1(){
		return this.x1;
	}
	
	public double getY1(){
		return this.y1;
	}
	
	public double getX2(){
		return this.x2;
	}
	
	public double getY2(){
		return this.y2;
	}	
}
