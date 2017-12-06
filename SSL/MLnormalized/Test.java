import weka.core.FastVector;
import weka.core.DenseInstance;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.Instance;
import java.io.File;

class Test{
	public static String getName(String nameData){
		String result = "";
		int indx = nameData.lastIndexOf('/');
		int begin = 0;
		int end;
		if(indx >= 0)
			begin = indx + 1;

		end = nameData.lastIndexOf('.');
		if(end == -1)
			end = nameData.length();
		result = nameData.substring(begin, end);
		return result;
		
		
	}
	public static void main(String args[]) throws Exception{
				
			
			String model = args[0];
			String strTestData = args[1]; 
			System.out.println(args[1]);	

			WekaTrainTest wkClassifier = new WekaTrainTest();
			String strTestSave = args[2];
		//	wkClassifier.setModel(model);
			wkClassifier.test(model,strTestData, strTestSave);
				//wkClassifier.test(strWholeData, strWholeSave);
			
	}

}
