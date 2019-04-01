import weka.core.FastVector;
import weka.core.DenseInstance;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.Instance;
import java.io.File;
import java.util.Scanner;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.text.DecimalFormat;
class Help{
	

	public static final String[] AppsName = {"grepspark", "wcspark", "grephadoop", "teragen", "cg", "lammps", "namd", "132"};
	public static final double[] soloRuntime0 = {134.787, 122.189 , 189.697, 198.764, 400.373, 235.079, 140.051,  238.567}; //i72222
	public static final double[] soloRuntime1 = {297.610, 277.24 , 348.818,  338.565, 275.250, 102.722,  69.381,  339.189}; //NUMA44
	public static final double[] maxRealRuntime = {220.142, 225.721, 275.462, 198.395, 429.343, 235.079,140.051, 359.372  };
	//public static final double[] maxRealRuntime = {276.532, 245.077, 356.594, 338.565, 429.343, 235.079,140.051, 359.372  };
	public static final double[] soloI7_11111111 = { 79.599, 80.495, 178.230, 169.761, 429.343, 192.826, 123.689, 205.882};
	public static final double[] soloNUMA_2222 = { 159.140, 144.339,325.227,246.090,263.520,103.967,53.965,359.372};
	public static final double[] soloI7_1111_NUMA_1111 = { 68.133, 88.033, 190.469, 202.912, 406.841, 227.185, 134.965, 342.965};
	public static final double[] soloi7_22_NUMA_22 = { 117.434, 129.928, 206.233, 201.164, 415.733, 177.133, 112.547, 358.268};


	public static double computeGeomean(double[] arr){
                double times = 1.0;
                for(int i = 0; i < arr.length; i++){
                        times =  times*arr[i];
                }

                return Math.pow(times, 1.0/arr.length);

        }

	public static int checkApplicationType(String name){
		
		for(int i = 0; i < 8; i++)
		{
			if(name.contains(AppsName[i]))
				return i;
		}	
		return -1;

	}	
	public static void main(String args[]) throws Exception{

		int NumberFile = Integer.parseInt(args[0]);
		//System.out.println(NumberFile);
		String[] applications = new String[NumberFile];
		Instances[] isTest = new Instances[NumberFile];
		String nameModel = new String();
		int indexApp;
		double[] speedUp = new double[NumberFile];
		double[] predict = new double[NumberFile];
		double[] maxRun  = new double[NumberFile];
		double geoMean;
/*		
		for(int i = 0; i < NumberFile; i++){
			applications[i] = args[i+1];
			System.out.println(applications[i]);
			System.out.println(checkApplicationType(applications[i]));
			System.out.println(readFile("WhereModel/"+AppsName[checkApplicationType(applications[i])]));

		}
*/		
		WekaTrainTest[] wkClassifier = new WekaTrainTest[NumberFile];

//		System.out.println("aaa");
//		System.out.println(NumberFile);
		for(int i = 0; i < NumberFile; i++){
			applications[i] = args[i+1];
			wkClassifier[i] = new WekaTrainTest();
			isTest[i] = WekaHelper.readData(applications[i]);
			indexApp = checkApplicationType(applications[i]);
			nameModel = readFile("WhereModel/" + AppsName[indexApp]);
			wkClassifier[i].setModel(nameModel);
			predict[i] = wkClassifier[i].predict(isTest[i].instance(0));

//			System.out.println(nameModel);
			if (indexApp == 4 || indexApp == 5 || indexApp == 6 || indexApp == 7 ) {
				if((NumberFile == 1) || (isTest[i].instance(0).value(8) == 0 && isTest[i].instance(0).value(23) == 0)){
				
				
 					if(isTest[i].instance(0).value(0) == 8 && isTest[i].instance(0).value(1) == 8) predict[i] = soloI7_11111111[indexApp];	
 					if(isTest[i].instance(0).value(0) == 0 && isTest[i].instance(0).value(2) == 4) predict[i] = soloNUMA_2222[indexApp];	
 					if(isTest[i].instance(0).value(0) == 4 && isTest[i].instance(0).value(1) == 4 && isTest[i].instance(0).value(2) == 4) predict[i] = soloI7_1111_NUMA_1111[indexApp];	
 					if(isTest[i].instance(0).value(0) == 4 && isTest[i].instance(0).value(1) == 2 && isTest[i].instance(0).value(2) == 2) predict[i] = soloi7_22_NUMA_22[indexApp];	
					if(isTest[i].instance(0).value(0) == 8 && isTest[i].instance(0).value(1) == 4) predict[i] = soloRuntime0[indexApp];
 					if(isTest[i].instance(0).value(0) == 0 && isTest[i].instance(0).value(2) == 2) predict[i] = soloRuntime1[indexApp];	
				}
				else
				{	
					predict[i] = wkClassifier[i].predict(isTest[i].instance(0));
				//predict[i] = Math.pow(10, predict[i]);
			
//			if(isTest[i].instance(0).value(0) == 8 && isTest[i].instance(0).value(1) == 4) predict[i] = soloRuntime0[indexApp];
//			if(isTest[i].instance(0).value(0) == 0 && isTest[i].instance(0).value(2) == 2) predict[i] = soloRuntime1[indexApp];	
				}	

			}
			maxRun[i] = maxRealRuntime[indexApp];
			speedUp[i] = maxRun[i]/predict[i];
			
		//System.out.println( predict[i] + "  "  );
		//System.out.println(applications[i] + " maxrun: " + maxRun[i] + ",predict: " + predict[i] + ", speedup: " +  speedUp[i] );
		}
			
	
			geoMean = computeGeomean(speedUp);
			DecimalFormat f = new DecimalFormat("##.000");
			System.out.println(f.format(geoMean));
			//System.out.println(geoMean);



	}
	
	public static String readFile(String filename){
		BufferedReader br = null;
		FileReader fr = null;
		
			String sCurrentLine = null;
		try{
			fr = new FileReader(filename);
			br = new BufferedReader(fr);
			br = new BufferedReader(new FileReader(filename));

			sCurrentLine = br.readLine();// read only first line
		} catch (IOException e) {

			e.printStackTrace();
		}

		return sCurrentLine;
	}



}
