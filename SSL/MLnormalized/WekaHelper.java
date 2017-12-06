import weka.core.FastVector;
import weka.core.DenseInstance;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.Instance;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.REPTree;
import weka.classifiers.evaluation.NumericPrediction;
import java.io.BufferedWriter;
import java.io.FileWriter;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.Standardize;





public class WekaHelper{
	
	// read data from file arff


	static double MinNorm;
	static double MaxNorm;
	static double scaleNorm;
	static double translationNorm;
	static double[] actualValue;
	static double[] errorArr;//error for cross validation
	public static Instances readData(String filename) throws Exception{
		// Read from data source
		DataSource source = new DataSource(filename);
		Instances data = source.getDataSet();
		if(data.classIndex() == -1)
			data.setClassIndex(data.numAttributes() - 1);
		return data;
	}
	// convert array double to Instance
	public static Instance convertArr2Instance(double[] Arr) throws Exception{
		int n = Arr.length;
		String name ;
		FastVector fvWekaAttributes = new FastVector(n);
		for(int i = 0; i < n; i++){
			name = "" + i;
			fvWekaAttributes.addElement(new Attribute(name));
		}
		Instances isTemp = new Instances("Temp", fvWekaAttributes, 10);
		isTemp.setClassIndex(n - 1);
		Instance iUse = new DenseInstance(n);
		for(int i = 0; i < n; i++){
			iUse.setValue((Attribute)fvWekaAttributes.elementAt(i),Arr[i]);
		}
	        iUse.setDataset(isTemp);	
		return iUse;
 	}
	public static Instance convertArr2InstancePredict(double[] Arr) throws Exception{
                int n = Arr.length;
                String name ;
		int iNumberOfFeature = n + 1;
                FastVector fvWekaAttributes = new FastVector(iNumberOfFeature);
                for(int i = 0; i <= n; i++){
                        name = "" + i;
                        fvWekaAttributes.addElement(new Attribute(name));
                }
                Instances isTemp = new Instances("Temp", fvWekaAttributes, 10);
                isTemp.setClassIndex(n);
                Instance iUse = new DenseInstance(iNumberOfFeature);
                for(int i = 0; i < n; i++){
                        iUse.setValue((Attribute)fvWekaAttributes.elementAt(i),Arr[i]);
                }
                iUse.setDataset(isTemp);
                return iUse;
        }

		
	// use crossValidaion for esstimate a result
	public static Instances[][] crossValidationSplit (Instances data, int numberOfFolds) throws Exception{
		Instances[][] split = new Instances[2][numberOfFolds];
		for(int i = 0; i < numberOfFolds; i++){
			split[0][i] = data.trainCV(numberOfFolds, i);
			split[1][i] = data.testCV(numberOfFolds, i);

		}
		return split;
	
	}

	// compute error of cross validation
	public static double implementCV(Classifier model, Instances data, int numberOfFolds) throws Exception{
		Instances[][] split = crossValidationSplit(data, numberOfFolds);
		Instances[] trainingSplits = split[0];
		Instances[] testingSplits = split[1];
		errorArr= new double[trainingSplits.length];
		double sum = 0;	
		for(int i = 0; i < trainingSplits.length; i++)
		{
			Evaluation validation = new Evaluation(trainingSplits[i]);
			model.buildClassifier(trainingSplits[i]);
			validation.evaluateModel(model, testingSplits[i]);
			errorArr[i] =  calculateError(validation);
			sum = sum + errorArr[i];
			
		}
		
		return (sum/errorArr.length);
	}
	// compute array of error rate
	public static double calculateError(Evaluation eSample) throws Exception{
		FastVector predictions = new FastVector();
		predictions.appendElements(eSample.predictions());
		double[] dArrErr = new double[predictions.size()];
		for(int i = 0; i < predictions.size(); i++){
			NumericPrediction np = (NumericPrediction) predictions.elementAt(i);
			//dArrErr[i] = error(np.actual(), Math.pow(10.0,np.predicted()));
			dArrErr[i] = error(np.actual(), np.predicted());
		}
		double error = computeErr(dArrErr);
		return error;	
	}


	  public static double calculateError1(Evaluation eSample) throws Exception{
                FastVector predictions = new FastVector();
                predictions.appendElements(eSample.predictions());
                double[] dArrErr = new double[predictions.size()];
                for(int i = 0; i < predictions.size(); i++){
                        NumericPrediction np = (NumericPrediction) predictions.elementAt(i);
                        //dArrErr[i] = error(np.actual(), Math.pow(10.0,np.predicted()));
			double denormalize = predictConvert(np.predicted(), MinNorm, MaxNorm,scaleNorm, translationNorm);
                        dArrErr[i] = error(actualValue[i], denormalize);
                }
                double error = computeErr(dArrErr);
                return error;
        }


	// compute mean of err rate
	public static double computeErr(double[] dArrErr) throws Exception{
		double error;
		double sum = 0;	
		for (int i = 0; i < dArrErr.length; i++)
			sum += dArrErr[i];

		error = sum/dArrErr.length;
		return error;
	}

	//function to compute error
	private static double error(double actual, double predict) throws Exception{
		double result = Math.abs(actual - predict)/actual;
		return result;
	}

	// write the result or Evaluation to File
	public static void writeFile(String sOutFile, Evaluation eVal) throws Exception{
		BufferedWriter writer = new BufferedWriter(new FileWriter(sOutFile));
		writer.write(eVal.toSummaryString());
		double  err = calculateError(eVal);
		String str = "Our method Error:" + err + "\n";
		writer.write(str); 

		writer.write("\n===========================\n");
		writer.write("Result:\n");
		FastVector predictions = new FastVector();
		predictions.appendElements(eVal.predictions());
		str = "No\tActual\tPredicted\tError\n";
		writer.write(str);
		for(int i = 0; i < predictions.size(); i++){
			NumericPrediction np = (NumericPrediction) predictions.elementAt(i);
			//err = error(np.actual(),Math.pow(10.0, np.predicted()));
			err = error(np.actual(), np.predicted());
			str = i + "\t" + np.actual() + "\t" + np.predicted() + "\t" + err + "\n";
			writer.write(str); 
			
		}
		writer.flush();
		writer.close();
	}
	

	public static void writeFile1(String sOutFile, Evaluation eVal, double Min, double Max,double scale, double translation, double[] actual) throws Exception{
                MinNorm = Min;
		MaxNorm = Max;
		scaleNorm = scale;
		translationNorm = translation;
		actualValue = new double [actual.length];
		for(int i = 0; i < actual.length; i++)
			actualValue[i] = actual[i];
			
		


		BufferedWriter writer = new BufferedWriter(new FileWriter(sOutFile));
                writer.write(eVal.toSummaryString());
                double  err = calculateError1(eVal);
                String str = "Our method Error:" + err + "\n";
                writer.write(str);

                writer.write("\n===========================\n");
                writer.write("Result:\n");
                FastVector predictions = new FastVector();
                predictions.appendElements(eVal.predictions());
                str = "No\tActual\tPredicted\tError\n";
                writer.write(str);
                for(int i = 0; i < predictions.size(); i++){
                        NumericPrediction np = (NumericPrediction) predictions.elementAt(i);
                        //err = error(np.actual(),Math.pow(10.0, np.predicted()));

			double predict = predictConvert(np.predicted(), Min, Max, scale, translation);
                        err = error(actual[i], predict);
                        str = i + "\t" + actual[i] + "\t" + predict + "\t" + err + "\n";
                        writer.write(str);

                }
                writer.flush();
                writer.close();
        }

	public static double predictConvert(double normPred, double Min, double Max,double scale, double translation) throws Exception{

		double denormalize = (normPred - translation)*(Max - Min)*scale + Min;
		return denormalize;


	}


	public static void saveModel(Classifier model, String strOutputFile) throws Exception{
		weka.core.SerializationHelper.write(strOutputFile, model);
	}



	public static void saveNorm(Normalize norm, String strOutputFile) throws Exception{
		weka.core.SerializationHelper.write(strOutputFile, norm);

	}

	public static Normalize readNorm(String strNormFile) throws Exception{
			Normalize norm = (Normalize) weka.core.SerializationHelper.read(strNormFile);
			return norm;

	}

	public static Classifier readModel(String strModelFile) throws Exception{
		Classifier cls = (Classifier) weka.core.SerializationHelper.read(strModelFile);
		return cls;
	}

	public static void writeTrainSummary(Classifier model,Classifier model1, Instances train, String strOutFile) throws Exception{
		BufferedWriter writer = new BufferedWriter(new FileWriter(strOutFile));	
		double errCV = implementCV(model1, train, 5);
		String s = "Error Cross Validation:" + errCV + "\n";
		
		for (int i = 0; i < errorArr.length; i++)
			s += "Error in " + i + " is : " + errorArr[i] + "\n";
		writer.write(s);
		
		writer.write(model.toString());
		writer.write(s); 
		writer.flush();
		writer.close();
	}
}
