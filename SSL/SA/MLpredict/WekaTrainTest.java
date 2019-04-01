import weka.core.FastVector;
import weka.core.DenseInstance;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.Instance;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.meta.FilteredClassifier; 
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.RandomizableClassifier;
import weka.classifiers.trees.REPTree;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMOreg;
import weka.classifiers.functions.GaussianProcesses;
import weka.classifiers.trees.RandomTree;
import weka.filters.Filter;
//import weka.filters.unsupervised.instance.Denormalize;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.Standardize;
import java.io.File;


//import weka.classifiers.trees.SimpleCart;


//import java.lange.Object.lang.Math;
public class WekaTrainTest{
		Classifier cModel;
		Instances isTrain;
		Instances isTest;
		Evaluation eTest;
		Evaluation eTrain;
		Classifier cModel1;
		Normalize norm;
		double MinNorm;
		double MaxNorm;
		double scaleNorm;
		double translationNorm;
		boolean flagMPI=false;
		// train a new data set from file name
	 	public void  train(String filename, String type) throws Exception{
			Instances oTrain = WekaHelper.readData(filename);
			norm = new Normalize();
			//norm.setInputFormat(oTrain);
			norm.setIgnoreClass(true);
			norm.setInputFormat(oTrain);
			isTrain = Filter.useFilter(oTrain, norm);
			//System.out.println(isTrain);		
			switch(type){
				case "ANN":
					cModel	 = (Classifier) new MultilayerPerceptron(); 
					cModel1	 = (Classifier) new MultilayerPerceptron();
					break;
				case "Gauss":
					cModel	 = (Classifier) new GaussianProcesses();
					cModel1	 = (Classifier) new GaussianProcesses();
					break;
				case "SMO":
					cModel	 = (Classifier) new SMOreg(); 
					cModel1	 = (Classifier) new SMOreg();
					break;
				case "REPTree":
					cModel	 = (Classifier) new REPTree();
					cModel1	 = (Classifier) new REPTree();
					break;
				//case "SimpleCart":
				//	cModel 	 = (Classifier) new SimpleCart();
				//	cModel1  = (Classifier) new SimpleCart();
				//	break;
				case "RandomTree":
					cModel   = (Classifier) new RandomTree();
					cModel1  = (Classifier) new RandomTree();
			}

//			System.out.println(isTrain);
			cModel.buildClassifier(isTrain);
//			eTest = new Evaluation(isTrain);


                    //    eTest.evaluateModel(cModel, isTrain);
                  //      WekaHelper.writeFile("TRAINABC", eTest);


		//	double[] Min = norm.getMinArray();
                       // double[] Max = norm.getMaxArray();

                       // System.out.println("TRAIN");
                       // double scale = norm.getScale();
                       // double translation = norm.getTranslation();
                       // for(int i = 0; i < Min.length ; i++)
                       //         System.out.println("Min: " + Min[i] + " Max: " + Max[i]);



		}
		
		// train a new data set from strInputFile and save the Model to strOutputFile
		public void train(String strInputFile, String type,String strOutputFile) throws Exception{
			train(strInputFile, type);
			WekaHelper.saveModel(cModel, strOutputFile);
			WekaHelper.saveNorm(norm, strOutputFile + "norm");

		}

		public  void test(String sInputFile, String sOutputFile) throws Exception{ 
			isTest = WekaHelper.readData(sInputFile);
			eTest=null;
			if(eTest == null)
				eTest = new Evaluation(isTest);
			eTest.evaluateModel(cModel, isTest);
			WekaHelper.writeFile(sOutputFile, eTest);

		}
		public void test(String strModelFile,String strInputFile, String strOutputFile) throws Exception{
			System.out.println(strModelFile);
			cModel = (Classifier) WekaHelper.readModel(strModelFile);
			norm = (Normalize) WekaHelper.readNorm(strModelFile+"norm");
			Instances isTest1 = WekaHelper.readData(strInputFile);
			isTest = Filter.useFilter(isTest1, norm);
			double[] Min = norm.getMinArray();
			double[] Max = norm.getMaxArray();

			System.out.println("TEST");
			double scale = norm.getScale();
			double translation = norm.getTranslation();
			for(int i = 0; i < Min.length ; i++)
				System.out.println("Min: " + Min[i] + " Max: " + Max[i]);
			
			

			int classIdx  = isTest1.numAttributes() - 1;
			int numInst   = isTest1.numInstances();
			double actual[] = new double[numInst];
			double denormalize;
			double scale1;
			for(int i = 0; i < numInst; i++){
				actual[i] = isTest1.instance(i).value(classIdx);
				//System.out.println(isTest1.instance(i).value(classIdx) + " " + actual[i]);
				scale1 = isTest.instance(i).value(classIdx);
				denormalize =  (scale1 - translation)*(Max[classIdx] - Min[classIdx])*scale + Min[classIdx];	
				System.out.println(isTest1.instance(i).value(classIdx) + " " + actual[i] + " " + denormalize);
			}
	//		System.out.println(isTest);
		//	Denormalize dn = new Denormalize();
	//		dn.setOptions(norm.getOptions());
			
//			Instances aaa = Filter.useFilter(isTest, dn);
//			System.out.println(aaa);

			//System.out.println(isTest);
			eTest = new Evaluation(isTest);
			eTest.evaluateModel(cModel, isTest);
			//WekaHelper.writeFile(strOutputFile, eTest);
			WekaHelper.writeFile1(strOutputFile, eTest, Min[classIdx] , Max[classIdx], scale, translation, actual);
		}
		public double predict1(double[] Arr) throws Exception{
			Instance iTest1 = WekaHelper.convertArr2InstancePredict(Arr);

			double result = cModel.classifyInstance(iTest1);

			if(!flagMPI){
			//Instance iTest = Filter.useFilter(iTest1, norm);

			if(norm.input(iTest1)){
				Instance iTest =  norm.output();
				result = WekaHelper.predictConvert(result, MinNorm, MaxNorm, scaleNorm, translationNorm);	
				}
			}
			return result;
		}
		public double predict(Instance a) throws Exception{
			double result =  cModel.classifyInstance(a);
			if(!flagMPI){
				if(norm.input(a)){
                                Instance iTest =  norm.output();
				result = cModel.classifyInstance(iTest);
                                result = WekaHelper.predictConvert(result, MinNorm, MaxNorm, scaleNorm, translationNorm);
                                }
				return result;	
			}	
			return Math.pow(10, result);
			
		}
		public void setModel(String strModelFile) throws Exception{
			cModel = (Classifier) WekaHelper.readModel(strModelFile);

			File f = new File(strModelFile+"norm");
			if (!f.exists()){
				flagMPI = true;

			}
			else{
				norm = (Normalize) WekaHelper.readNorm(strModelFile+"norm");

                       	 	double[] Min = norm.getMinArray();
                        	double[] Max = norm.getMaxArray();

				MinNorm = Min[Min.length - 1];
				MaxNorm = Max[Max.length - 1];
                        	scaleNorm = norm.getScale();
                        	translationNorm = norm.getTranslation();
			}
		
		}

		public void saveInfoTrain(String strOutFile) throws Exception{
			WekaHelper.writeTrainSummary(cModel,cModel1, isTrain, strOutFile);
		}
		

		

		
}
