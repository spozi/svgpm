#include <shark/Algorithms/Trainers/CSvmTrainer.h>
#include <shark/Algorithms/Trainers/NormalizeComponentsUnitVariance.h>
#include <shark/Algorithms/Trainers/NormalizeComponentsUnitInterval.h>
#include <shark/Models/Kernels/GaussianRbfKernel.h>
#include <shark/Models/Kernels/PolynomialKernel.h>
#include <shark/Models/Kernels/AbstractKernelFunction.h>
#include <shark/Models/Normalizer.h>
#include <shark/Models/Kernels/KernelExpansion.h>
#include <shark/ObjectiveFunctions/Loss/ZeroOneLoss.h>
#include <shark/Data/DataDistribution.h>
#include <shark/Data/Csv.h>
#include <shark/Data/CVDatasetTools.h>
#include <shark/Data/DataDistribution.h>
#include <boost/archive/polymorphic_text_oarchive.hpp>


#include <iostream>
#include <list>
#include <fstream>
#include <string>
#include <cmath>
#include <algorithm>
#include <tuple>


using namespace std;
using namespace shark;


#include <cstdlib>
#include <cmath>


class BinaryConfusionMatrix
{
public:
	int TP; int FP;
	int FN; int TN;

	BinaryConfusionMatrix()
	{
		TP = 0; FP = 0;
		FN = 0; TN = 0;
	}

	double getAccuracy()
	{
		return ((double)TP + double(TN)) / ((double)TP + double(TN) + (double)FP + double(FN));
	}

	double getSpecificity()
	{
		return (double(TN) / double(TN) + double(FP));
	}

	double getSensitivity()
	{
		return (double(TP) / double(TP) + double(FN));
	}

	double getGeometricMean()
	{
		return std::sqrt(getSensitivity() * getSpecificity());
	}
};

//Function prototype
ClassificationDataset makeLabeledData(Data<RealVector> &dataset, bool normalize);
ClassificationDataset makeLabeledData(Data<RealVector> &trainingdataset, Data<RealVector> &testingdataset, bool normalize);



//GSVM
// double GSVM(ClassificationDataset training, ClassificationDataset testing)
// {

// 	Data<RealVector> Z_w = kc.decisionFunction()(training.inputs());	//<z,w>
// }

std::tuple<int,int,int,int> GSVM(ClassificationDataset trainingdata, ClassificationDataset testingdata, int C, int gamma)
{

	std::vector<BinaryConfusionMatrix> result;

	// std::string filename = "GSVM_C"/*_+ std::to_string(C)+ "_g_" + std::to_string(gamma) */ + std::string("_GSVM.txt");
	// std::ofstream myfile;
	// myfile.open(filename, ios::app);

	//SVM start here
	GaussianRbfKernel<RealVector> rbfKernel(std::pow(2, gamma));							//gamma {-4 to 6}

	SquaredHingeCSvmTrainer<RealVector> trainer(&rbfKernel, std::pow(2,C), true); 	//C {-5 to 6}	//Unconstrained or constrained svm are purely computational
	
	//Let's try cost-sensitive SVM
	// CSvmTrainer<RealVector> trainer(&rbfKernel, std::pow(2, C), unconstrained); //C {-5 to 6}
	//CSvmTrainer<RealVector> trainer(&rbfKernel, total_pos * std::pow(2,C), total_neg * std::pow(2,C), unconstrained); //C {-5 to 6}
	//trainer.sparsify() = false;
	
	KernelClassifier<RealVector> kc;

	BinaryConfusionMatrix ConfusionMatrix;
	ConfusionMatrix.TP = 0;
	ConfusionMatrix.FP = 0;
	ConfusionMatrix.FN = 0;
	ConfusionMatrix.TN = 0;

	// ClassificationDataset training = trainingdata;
	// ClassificationDataset validation = testingdata;

	trainer.train(kc, trainingdata);

	//GSVM starts here
	Data<RealVector> Z_w = kc.decisionFunction()(trainingdata.inputs());	//<z,w>

																		//GSVM Post-processing
	std::vector<double> Zp_w;
	Zp_w.clear();
	std::vector<double> Zn_w;
	Zn_w.clear();

	for (int k = 0; k < trainingdata.numberOfElements(); ++k)
	{
		if (trainingdata.element(k).label != 0)	// Positive class
			Zp_w.push_back(Z_w.element(k)(0));
		else if (trainingdata.element(k).label == 0) // Negative class
			Zn_w.push_back(Z_w.element(k)(0));
	}

	//myfile << s.element(i)(0) << "\t" << testing_output.element(i) << "\n";
	double beta = *std::min_element(Zp_w.begin(), Zp_w.end()), beta_star = *std::max_element(Zp_w.begin(), Zp_w.end());
	double alpha_star = *std::min_element(Zn_w.begin(), Zn_w.end()), alpha = *std::max_element(Zn_w.begin(), Zn_w.end());
	double lambda = std::max(alpha_star, beta);
	double theta = std::max(alpha, beta_star);

	//define bm and bM (min)

	//double bm = Zp_w[0];
	//for(int k = 0; k < Zp_w.size(); ++k)
	//{
	//	if((Zp_w[k] > lambda) && (Zp_w[k] <= bm))
	//		bm = Zp_w[k];
	//}

	//double bM = Zp_w[0];
	//for(int k = 0; k < Zp_w.size(); ++k)
	//{
	//	if((Zp_w[k] > theta) && (Zp_w[k] <= bM))
	//		bM = Zp_w[k];
	//}

	double bm = *std::min_element(Zp_w.begin(), Zp_w.end(), [&lambda](double a, double b)
	{
		if (a < b)
		{
			if (a > lambda)
				return a;
			else if (a >= lambda)
				return a;
		}
		else if ((a <= b))
		{
			if (a > lambda)
				return a;
			else if (a >= lambda)
				return a;
		}
	});

	double bM = *std::min_element(Zp_w.begin(), Zp_w.end(), [&theta](double a, double b)
	{
		if (a < b)
		{
			if (a > theta)
				return a;
			else if (a >= theta)
				return a;
		}
		else if ((a <= b))
		{
			if (a > theta)
				return a;
			else if (a >= theta)
				return a;
		}
	});

	double bi = 0.0;
	double bGSVM = 0.0;

	//Then we define max GM(b_(GSVM))
	BinaryConfusionMatrix GSVMCFM;
	GSVMCFM.TP = 0;
	GSVMCFM.FP = 0;
	GSVMCFM.FN = 0;
	GSVMCFM.TN = 0;

	double maxGM = 0.0;
	for (int k = 0; k < Zp_w.size(); ++k)
	{
		if (alpha < beta)
		{
			bGSVM = (alpha + beta) / 2.0;
		}
		else if (alpha >= beta)
		{
			if ((Zp_w[k] >= bm) && (Zp_w[k] <= bM))
			{
				bi = Z_w.element(k)[0];
				for (int l = 0; l < Z_w.numberOfElements(); ++l)
				{
					int gsvmLabel = ((Z_w.element(l)[0] + bi) >= 0 ? 1 : 0);

					if (gsvmLabel == 0 && trainingdata.element(l).label == 0)
					{
						++GSVMCFM.TN;
					}
					if (gsvmLabel == 0 && trainingdata.element(l).label != 0)
					{
						++GSVMCFM.FN; //if actual is positive, but output is negative, then it is false negative
					}
					if (gsvmLabel != 0 && trainingdata.element(l).label == 0)
					{
						++GSVMCFM.FP; //if actual is negative, but output is positive, then it is false positive
					}
					if (gsvmLabel != 0 && trainingdata.element(l).label != 0)
					{
						++GSVMCFM.TP;
					}
				}
			}

			if (GSVMCFM.getGeometricMean() > maxGM)
			{
				maxGM = GSVMCFM.getGeometricMean();
				bGSVM = bi;
			}
		}
	}

	//And for validation set
	Data<RealVector> validation_w = kc.decisionFunction()(testingdata.inputs());

	//Fisher exact test uhhhh.

	// std::string filename1 = "GSVM_C_" + std::to_string(C) + "_g_" + std::to_string(gamma) + std::string("_fisher.txt");
	// std::ofstream myfile1;
	// myfile1.open(filename1);
	for (int k = 0; k < validation_w.numberOfElements(); ++k)
	{
		// cout << validation_w.element(k)[0] << " + " << bGSVM << "\n";
		int gsvmLabel = ((validation_w.element(k)[0] + bGSVM) >= 0 ? 1 : 0);

		// myfile1 << testingdata.element(k).label << "\t" << gsvmLabel << "\n";

		if (gsvmLabel == 0 && testingdata.element(k).label == 0)
		{
			++ConfusionMatrix.TN;
		}
		else if (gsvmLabel == 0 && testingdata.element(k).label != 0)
		{
			++ConfusionMatrix.FN; //if actual is positive, but output is negative, then it is false negative
		}
		else if (gsvmLabel != 0 && testingdata.element(k).label == 0)
		{
			++ConfusionMatrix.FP; //if actual is negative, but output is positive, then it is false positive
		}
		else if (gsvmLabel != 0 && testingdata.element(k).label != 0)
		{
			++ConfusionMatrix.TP;
		}
	}
	// myfile1.close();
	// myfile << C << "\t " << gamma << "\t" << ConfusionMatrix.TP << "\t" << ConfusionMatrix.FP << "\n\t\t" << ConfusionMatrix.FN << "\t" << ConfusionMatrix.TN << "\n";
	// myfile.close();

	return {ConfusionMatrix.TP, ConfusionMatrix.FP, ConfusionMatrix.FN, ConfusionMatrix.TN};
}



int main(int argc, char ** argv)
{
	double C = 0;
	double gamma = 0;
	bool l1 = true;
	bool costSensitive = false;
	int folds = 2;
	std::string trainingFile;
	std::string validationFile;
	std::string datasetFile;
	bool wholeDataset = true;
	bool bias = true;
	bool unconstrained = true;

	if (argc < 2)
	{
		// Check the value of argc. If not enough parameters have been passed, inform user and exit.
		std::cout << "Usage is -in <infile> -out <outdir>\n"; // Inform the user of how to use the program
		std::cin.get();
		exit(0);
	}
	else
	{
		// if we got enough parameters...
		for (int i = 1; i < argc; i++)
		{ /* We will iterate over argv[] to get the parameters stored inside.
		  * Note that we're starting on 1 because we don't need to know the
		  * path of the program, which is stored in argv[0] */
		  //if (i + 1 != argc) // Check that we haven't finished parsing already
			if (string(argv[i]) == "-tr")
			{
				trainingFile = argv[i + 1];
				//std::cout << trainingFile << "\n";
				wholeDataset = false;
			}
			else if (string(argv[i]) == "-ts")
			{
				validationFile = argv[i + 1];
				//std::cout << validationFile << "\n";
			}
			else if (string(argv[i]) == "-ds")
			{
				datasetFile = argv[i + 1];
				wholeDataset = true;
			}
			else if (string(argv[i]) == "-c")
			{
				C = atof(argv[i + 1]);
			}
			else if (string(argv[i]) == "-g")
			{
				gamma = atof(argv[i + 1]);
			}
			else if (string(argv[i]) == "-cs")
			{
				costSensitive = true;
			}
			else if (string(argv[i]) == "-l1")
			{
				l1 = true;
			}
			else if (string(argv[i]) == "-l2")
			{
				l1 = false;
			}
			else if (string(argv[i]) == "-cv")
			{
				folds = atof(argv[i + 1]);
			}
		}
	}

	if (wholeDataset)
	{
		//This is for wholedataset, which is going to be split into k-fold dataset
		Data<RealVector> dataset;
		importCSV(dataset, datasetFile, ',','#',shark::Data<RealVector>::DefaultBatchSize, 1);
		
		//Prepare to make labeled dataset
		typedef Data<RealVector>::element_range Elements;
		Elements elementsTraining = dataset.elements();
		double total_pos = 0;
		double total_neg = 0;
		for (Elements::iterator pos = elementsTraining.begin(); pos != elementsTraining.end(); ++pos)
		{
			if ((*pos)[pos->size() - 1] != 0)
			{
				++total_pos;
				(*pos)[pos->size() - 1] = 1;
			}
			else
				++total_neg;
		}
		ClassificationDataset dataset_labeled = makeLabeledData(dataset, false); //The second option is: True is to normalize the data, false is not. However, this option is not used for now, as data is expected to be normalized.
		dataset_labeled.makeIndependent();
		CVFolds<ClassificationDataset> cvfolds = createCVSameSizeBalanced(dataset_labeled, folds);
		int TP = 0; int FP = 0; int FN = 0; int TN = 0;
		for(int fold = 0; fold != cvfolds.size(); ++fold)
		{
			ClassificationDataset training = cvfolds.training(fold);
			ClassificationDataset testing = cvfolds.validation(fold);
			training.makeIndependent();
			testing.makeIndependent();

			auto [tp, fp, fn, tn] = GSVM(training, testing, C, gamma);
			TP += tp;
			FP += fp;
			FN += fn;
			TN += tn;
		}

		std::cout << TP << "\t" << FP << "\n" << FN << "\t" << TN << "\n"; 
	}

/*********************Skip training and testing dataset for now***********************************************/
	// This is for training and testing dataset
	// Data<RealVector> trainingDataset, testingDataset;
	// importCSV(trainingDataset, trainingFile, ',', '#', shark::Data<RealVector>::DefaultBatchSize, 1);
	// importCSV(testingDataset, validationFile, ',', '#', shark::Data<RealVector>::DefaultBatchSize, 1);

	// typedef Data<RealVector>::element_range Elements;
	// Elements elementsTraining = trainingDataset.elements();
	// double total_pos = 0;
	// double total_neg = 0;
	// for (Elements::iterator pos = elementsTraining.begin(); pos != elementsTraining.end(); ++pos)
	// {
	// 	if ((*pos)[pos->size() - 1] != 0)
	// 	{
	// 		++total_pos;
	// 		(*pos)[pos->size() - 1] = 1;
	// 	}
	// 	else
	// 		++total_neg;
	// }
	// ClassificationDataset training_dataset_labeled = makeLabeledData(trainingDataset, true);
	// training_dataset_labeled.makeIndependent();

	// Elements elementsTesting = testingDataset.elements();
	// double test_total_pos = 0;
	// double test_total_neg = 0;
	// for (Elements::iterator pos = elementsTesting.begin(); pos != elementsTesting.end(); ++pos)
	// {
	// 	if ((*pos)[pos->size() - 1] != 0)
	// 	{
	// 		++test_total_pos;
	// 		(*pos)[pos->size() - 1] = 1;
	// 	}
	// 	else
	// 		++test_total_neg;
	// }
	// ClassificationDataset testing_dataset_labeled = makeLabeledData(trainingDataset, testingDataset, true);
	// testing_dataset_labeled.makeIndependent();




/**************************************SVM/GSVM Start here*******************************************************/
	// std::vector<BinaryConfusionMatrix> result;

	// std::string filename = "GSVM_C"/*_+ std::to_string(C)+ "_g_" + std::to_string(gamma) */ + std::string("_GSVM.txt");
	// std::ofstream myfile;
	// myfile.open(filename, ios::app);




	// //SVM start here
	// GaussianRbfKernel<RealVector> rbfKernel(std::pow(2, gamma));							//gamma {-4 to 6}

	// SquaredHingeCSvmTrainer<RealVector> trainer(&rbfKernel, std::pow(2,C), unconstrained); 	//C {-5 to 6}
	
	// //Let's try cost-sensitive SVM
	// // CSvmTrainer<RealVector> trainer(&rbfKernel, std::pow(2, C), unconstrained); //C {-5 to 6}
	// 																			//CSvmTrainer<RealVector> trainer(&rbfKernel, total_pos * std::pow(2,C), total_neg * std::pow(2,C), unconstrained); //C {-5 to 6}

	// 																			//trainer.sparsify() = false;
	// KernelClassifier<RealVector> kc;

	// BinaryConfusionMatrix ConfusionMatrix;
	// ConfusionMatrix.TP = 0;
	// ConfusionMatrix.FP = 0;
	// ConfusionMatrix.FN = 0;
	// ConfusionMatrix.TN = 0;

	// ClassificationDataset training = training_dataset_labeled;
	// ClassificationDataset validation = testing_dataset_labeled;

	// trainer.train(kc, training);

	// //GSVM starts here
	// Data<RealVector> Z_w = kc.decisionFunction()(training.inputs());	//<z,w>

	// 																	//GSVM Post-processing
	// std::vector<double> Zp_w;
	// Zp_w.clear();
	// std::vector<double> Zn_w;
	// Zn_w.clear();

	// for (int k = 0; k < training.numberOfElements(); ++k)
	// {
	// 	if (training.element(k).label == 1)	// Positive class
	// 		Zp_w.push_back(Z_w.element(k)(0));
	// 	else if (training.element(k).label == 0) // Negative class
	// 		Zn_w.push_back(Z_w.element(k)(0));
	// }

	// //myfile << s.element(i)(0) << "\t" << testing_output.element(i) << "\n";
	// double beta = *std::min_element(Zp_w.begin(), Zp_w.end()), beta_star = *std::max_element(Zp_w.begin(), Zp_w.end());
	// double alpha_star = *std::min_element(Zn_w.begin(), Zn_w.end()), alpha = *std::max_element(Zn_w.begin(), Zn_w.end());
	// double lambda = std::max(alpha_star, beta);
	// double theta = std::max(alpha, beta_star);

	// //define bm and bM (min)

	// //double bm = Zp_w[0];
	// //for(int k = 0; k < Zp_w.size(); ++k)
	// //{
	// //	if((Zp_w[k] > lambda) && (Zp_w[k] <= bm))
	// //		bm = Zp_w[k];
	// //}

	// //double bM = Zp_w[0];
	// //for(int k = 0; k < Zp_w.size(); ++k)
	// //{
	// //	if((Zp_w[k] > theta) && (Zp_w[k] <= bM))
	// //		bM = Zp_w[k];
	// //}

	// double bm = *std::min_element(Zp_w.begin(), Zp_w.end(), [&lambda](double a, double b)
	// {
	// 	if (a < b)
	// 	{
	// 		if (a > lambda)
	// 			return a;
	// 		else if (a >= lambda)
	// 			return a;
	// 	}
	// 	else if ((a <= b))
	// 	{
	// 		if (a > lambda)
	// 			return a;
	// 		else if (a >= lambda)
	// 			return a;
	// 	}
	// });

	// double bM = *std::min_element(Zp_w.begin(), Zp_w.end(), [&theta](double a, double b)
	// {
	// 	if (a < b)
	// 	{
	// 		if (a > theta)
	// 			return a;
	// 		else if (a >= theta)
	// 			return a;
	// 	}
	// 	else if ((a <= b))
	// 	{
	// 		if (a > theta)
	// 			return a;
	// 		else if (a >= theta)
	// 			return a;
	// 	}
	// });

	// double bi = 0.0;
	// double bGSVM = 0.0;

	// //Then we define max GM(b_(GSVM))
	// BinaryConfusionMatrix GSVMCFM;
	// GSVMCFM.TP = 0;
	// GSVMCFM.FP = 0;
	// GSVMCFM.FN = 0;
	// GSVMCFM.TN = 0;

	// double maxGM = 0.0;
	// for (int k = 0; k < Zp_w.size(); ++k)
	// {
	// 	if (alpha < beta)
	// 	{
	// 		bGSVM = (alpha + beta) / 2.0;
	// 	}
	// 	else if (alpha >= beta)
	// 	{
	// 		if ((Zp_w[k] >= bm) && (Zp_w[k] <= bM))
	// 		{
	// 			bi = Z_w.element(k)[0];
	// 			for (int l = 0; l < Z_w.numberOfElements(); ++l)
	// 			{
	// 				int gsvmLabel = ((Z_w.element(l)[0] + bi) >= 0 ? 1 : 0);

	// 				if (gsvmLabel == 0 && training.element(l).label == 0)
	// 				{
	// 					++GSVMCFM.TN;
	// 				}
	// 				if (gsvmLabel == 0 && training.element(l).label != 0)
	// 				{
	// 					++GSVMCFM.FN; //if actual is positive, but output is negative, then it is false negative
	// 				}
	// 				if (gsvmLabel != 0 && training.element(l).label == 0)
	// 				{
	// 					++GSVMCFM.FP; //if actual is negative, but output is positive, then it is false positive
	// 				}
	// 				if (gsvmLabel != 0 && training.element(l).label != 0)
	// 				{
	// 					++GSVMCFM.TP;
	// 				}
	// 			}
	// 		}

	// 		if (GSVMCFM.getGeometricMean() > maxGM)
	// 		{
	// 			maxGM = GSVMCFM.getGeometricMean();
	// 			bGSVM = bi;
	// 		}
	// 	}
	// }

	// //And for validation set
	// Data<RealVector> validation_w = kc.decisionFunction()(validation.inputs());

	// //Fisher exact test uhhhh.
	// std::string filename1 = "GSVM_C_" + std::to_string(C) + "_g_" + std::to_string(gamma) + std::string("_fisher.txt");
	// std::ofstream myfile1;
	// myfile1.open(filename1);
	// for (int k = 0; k < validation_w.numberOfElements(); ++k)
	// {
	// 	cout << validation_w.element(k)[0] << " + " << bGSVM << "\n";
	// 	int gsvmLabel = ((validation_w.element(k)[0] + bGSVM) >= 0 ? 1 : 0);

	// 	myfile1 << validation.element(k).label << "\t" << gsvmLabel << "\n";

	// 	if (gsvmLabel == 0 && validation.element(k).label == 0)
	// 	{
	// 		++ConfusionMatrix.TN;
	// 	}
	// 	else if (gsvmLabel == 0 && validation.element(k).label != 0)
	// 	{
	// 		++ConfusionMatrix.FN; //if actual is positive, but output is negative, then it is false negative
	// 	}
	// 	else if (gsvmLabel != 0 && validation.element(k).label == 0)
	// 	{
	// 		++ConfusionMatrix.FP; //if actual is negative, but output is positive, then it is false positive
	// 	}
	// 	else if (gsvmLabel != 0 && validation.element(k).label != 0)
	// 	{
	// 		++ConfusionMatrix.TP;
	// 	}
	// }
	// myfile1.close();
	// myfile << C << "\t " << gamma << "\t" << ConfusionMatrix.TP << "\t" << ConfusionMatrix.FP << "\n\t\t" << ConfusionMatrix.FN << "\t" << ConfusionMatrix.TN << "\n";
	// myfile.close();

	/**************************************************SVM/GSVM ends here*****************************************************/
}

ClassificationDataset makeLabeledData(Data<RealVector> &dataset, bool normalize) //Make labeled data from unlabeled data
{
	// create and train data normalizer
	std::vector<RealVector> inputs;
	std::vector<unsigned int> labels;

	inputs.resize(dataset.numberOfElements());
	for (int i = 0; i < inputs.size(); ++i)
		inputs[i].resize(dataset.element(i).size() - 1);

	labels.resize(dataset.numberOfElements());

	//Fill the input and the label manually
	for (int i = 0; i < dataset.numberOfElements(); ++i)
	{
		for (int j = 0; j < dataset.element(i).size(); ++j)
		{
			if (j == (dataset.element(i).size() - 1))
			{
				if (dataset.element(i)[j] == 0)
					labels.at(i) = 0;
				else if (dataset.element(i)[j] != 0)
					labels.at(i) = 1;
			}
			else
				inputs[i](j) = dataset.element(i)[j];
		}
	}
	ClassificationDataset normalizedData = createLabeledDataFromRange(inputs, labels);
	normalizedData.makeIndependent();

	// bool removeMean = true;
	// NormalizeComponentsUnitInterval<RealVector> normalizingTrainer;
	// Normalizer<RealVector> normalizer;
	// normalizingTrainer.train(normalizer, normalizedData.inputs());
	// normalizedData = transformInputs(normalizedData, normalizer);
	return normalizedData;
}

ClassificationDataset makeLabeledData(Data<RealVector> &trainingdataset, Data<RealVector> &testingdataset, bool normalize) //Make labeled data from unlabeled data
{
	// create and train data normalizer
	std::vector<RealVector> inputs;
	std::vector<unsigned int> labels;

	inputs.resize(trainingdataset.numberOfElements());
	for (int i = 0; i < inputs.size(); ++i)
		inputs[i].resize(trainingdataset.element(i).size() - 1);

	labels.resize(trainingdataset.numberOfElements());

	//Fill the input and the label manually
	for (int i = 0; i < trainingdataset.numberOfElements(); ++i)
	{
		for (int j = 0; j < trainingdataset.element(i).size(); ++j)
		{
			if (j == (trainingdataset.element(i).size() - 1))
			{
				if (trainingdataset.element(i)[j] == 0)
					labels.at(i) = 0;
				else if (trainingdataset.element(i)[j] != 0)
					labels.at(i) = 1;
			}
			else
				inputs[i](j) = trainingdataset.element(i)[j];
		}
	}
	ClassificationDataset normalizedTrainingData = createLabeledDataFromRange(inputs, labels);
	normalizedTrainingData.makeIndependent();

	inputs.clear();
	labels.clear();

	inputs.resize(testingdataset.numberOfElements());
	for (int i = 0; i < inputs.size(); ++i)
		inputs[i].resize(testingdataset.element(i).size() - 1);

	labels.resize(testingdataset.numberOfElements());

	//Fill the input and the label manually
	for (int i = 0; i < testingdataset.numberOfElements(); ++i)
	{
		for (int j = 0; j < testingdataset.element(i).size(); ++j)
		{
			if (j == (testingdataset.element(i).size() - 1))
			{
				if (testingdataset.element(i)[j] == 0)
					labels.at(i) = 0;
				else if (testingdataset.element(i)[j] != 0)
					labels.at(i) = 1;
			}
			else
				inputs[i](j) = testingdataset.element(i)[j];
		}
	}
	ClassificationDataset normalizedTestingData = createLabeledDataFromRange(inputs, labels);
	normalizedTestingData.makeIndependent();

	// bool removeMean = true;
	// NormalizeComponentsUnitInterval<RealVector> normalizingTrainer;
	// Normalizer<RealVector> normalizer;
	// normalizingTrainer.train(normalizer, normalizedTrainingData.inputs());
	// normalizedTrainingData = transformInputs(normalizedTrainingData, normalizer);
	// normalizedTestingData = transformInputs(normalizedTestingData, normalizer);
	return normalizedTestingData;
}