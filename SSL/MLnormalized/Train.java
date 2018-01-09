import weka.core.FastVector;
import weka.core.DenseInstance;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.Instance;
import java.io.File;

class Train{
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
		String strResultDir = args[1];
		String strSubResult = "";
		String strModelSave = "";	
		String strTestSave  = "";
		String strWholeSave = "";
		String strTrainSummary = "";
		String strSubName = "";
		
		String strTrainData = args[0]; 
		WekaTrainTest wkClassifier = new WekaTrainTest();
		// create several type to check
		String[] type = {"ANN", "Gauss", "SMO", "REPTree", "RandomTree"};
//		String[] type = { "REPTree", "RandomTree"};
		
		File fResult = new File(strResultDir);

		if(!fResult.exists()){
			fResult.mkdir();
		}
		for(int i = 0; i < type.length; i++){
				strSubResult = strResultDir + "/" + type[i];
				File fSaveResult2 = new File(strSubResult);
				if(!fSaveResult2.exists()){
					fSaveResult2.mkdir();
				}

				strSubName = getName(strTrainData) + "_" + type[i];
				strModelSave = strSubResult + "/" + strSubName + ".model";
				strTrainSummary = strSubResult + "/" + "TrainCV_" + strSubName +  ".txt";
 
				wkClassifier.train(strTrainData, type[i], strModelSave);
				wkClassifier.saveInfoTrain(strTrainSummary);

			
			}
		}
}

