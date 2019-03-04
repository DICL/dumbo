package ac.ku.milab.ImplementSTEHIX;

public class StoreFileTrace {

  // store file name
  private String storeFileName;
  // the number of data in storeFile
  private int numData;
  
  // Constructor
  public StoreFileTrace(){
    this.storeFileName = "";
    this.numData = 0;
  }
  
  //Constructor
  public StoreFileTrace(String name){
    this();
    this.setStoreFileName(name);
  }
  
  //Constructor
  public StoreFileTrace(String name, int num){
    this();
    this.setStoreFileName(name);
    this.setNumData(num);
  }

  // get StoreFileName
  public String getStoreFileName() {
    return storeFileName;
  }

  //set StoreFileName
  public void setStoreFileName(String storeFileName) {
    this.storeFileName = storeFileName;
  }

  //get NumData
  public int getNumData() {
    return numData;
  }

  //set NumData
  public void setNumData(int numData) {
    this.numData = numData;
  }
  
  // whether numData is 0
  public boolean hasNoData(){
    if(this.numData==0){
      return true;
    }else{
      return false;
    }
  }
  
  // minus 1, and then, return nuData becomes 0
  public boolean decrementNumber(){
    if(this.numData>0){
      this.numData--;
    }
    return hasNoData();
  }
  
  public boolean decrementNumber(int num){
    if(this.numData>0){
      this.numData = this.numData - num;
      if(this.numData<0){
        this.numData = 0;
      }
    }
    return hasNoData();
  }
  
  // print information
  public String printTrace(){
    StringBuilder sb = new StringBuilder();
    sb.append("StoreFile : ");
    sb.append(this.storeFileName);
    sb.append(" has ");
    sb.append("" + this.numData);
    sb.append(" data");
    String trace = sb.toString();
    return trace;
  }
  
  @Override
  public boolean equals(Object object){
    if(object instanceof StoreFileTrace){
      StoreFileTrace trace = (StoreFileTrace)object;
      if(this.storeFileName.equals(trace.getStoreFileName())){
        return true;
      }
    }
    return false;
  }
}
