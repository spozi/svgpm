// #define SHARK_USE_OPENMP 1
// #define BOOST_UBLAS_NDEBUG 1

// #include <svgpm-libs/GPSVMGM.h>

#include <GPSVMGM.h>

#include <iostream>
#include <fstream>
#include <list>
#include <cmath>

using namespace std;

// void writeDataToFile(ClassificationDataset &dataset, std::string &filename)
// {
// 	std::ofstream ofs(filename);
// 	ofs << dataset;
// 	ofs.close();
// }

void WriteResultToFile(std::string filename, BinaryConfusionMatrix& confusionMatrix)
{
	std::ofstream ofs;
	ofs.open (filename, std::ofstream::out | std::ofstream::app);

	ofs << confusionMatrix.powerC	<< "\t"		<< confusionMatrix.powerGamma	<< "\t"		<<  confusionMatrix.TP	<< "\t" << confusionMatrix.FP << "\n";
	ofs <<  "\t\t"																			<<  confusionMatrix.FN  << "\t" << confusionMatrix.TN << "\n";

	ofs.close();
}

// void WriteResultToFile(std::string filename, std::vector<BinaryConfusionMatrix>& confusionMatrix)
// {
// 	std::ofstream ofs;
// 	ofs.open (filename, std::ofstream::out | std::ofstream::app);

// 	for(int i = 0; i < confusionMatrix.size(); ++i)
// 	{
// 		ofs << /*confusionMatrix[i].powerC << "\t" << confusionMatrix[i].powerGamma << "\t" <<*/ confusionMatrix[i].TP << "\t" << confusionMatrix[i].FP << "\n";
// 		ofs	<</*						        "\t" <<                                  "\t" <<*/ confusionMatrix[i].FN << "\t" << confusionMatrix[i].TN << "\n";
// 	}	

// 	ofs.close();
// }

	// void printBestGPFeatures(BinaryConfusionMatrix& confusionMatrix)
	// {
	// 	// std::cout << *confusionMatrix.DerivedFeatures;
	// 	// std::vector<Tree> bestFeatures = confusionMatrix.DerivedFeatures;
	// 	// std::cout << "Tree length: " << bestFeatures.size();
	// 	// for(int j = 0; j < bestFeatures.size(); ++j)
	// 	// {
	// 	// 	std::cout << bestFeatures[j] << "\t";
	// 	// 	// if(bestFeatures[j].mFitness != 0)
	// 	// 	// {
				
	// 	// 	// }
	// 	// }
	// }

// void WriteDerivedFeaturesToFile(std::string filename, BinaryConfusionMatrix& confusionMatrix)
// {
// 	std::ofstream ofs;
// 	ofs.open (filename, std::ofstream::out | std::ofstream::app);

// 	std::vector<Tree> bestFeatures = confusionMatrix.DerivedFeatures;
// 	for(int j = 0; j < bestFeatures.size(); ++j)
// 	{
// 		if(bestFeatures[j].mFitness != 0)
// 		{
// 			ofs << bestFeatures[j];
// 			if(j != bestFeatures.size() - 1)
// 				ofs << "\t\t"; 
// 		}
// 	}
// 	ofs << "\n";
// 	ofs.close();
// }

// void WriteDerivedFeaturesToFile(std::string filename, std::vector<BinaryConfusionMatrix> & confusionMatrix)
// {
// 	std::ofstream ofs;
// 	ofs.open (filename, std::ofstream::out | std::ofstream::app);
	
	
// 	for(int i = 0; i < confusionMatrix.size(); ++i)
// 	{
// 		//ofs << confusionMatrix[i].DerivedFeatures.size();
// 		std::vector<Tree> bestFeatures = confusionMatrix[i].DerivedFeatures;
// 		for(int j = 0; j < bestFeatures.size(); ++j)
// 		{
// 			if(bestFeatures[j].mFitness != 0)
// 			{
// 				ofs << bestFeatures[j];
// 				if(j != bestFeatures.size() - 1)
// 					ofs << "\t\t"; 
// 			}
// 		}
// 		ofs << "\n";
// 	}
// 	ofs.close();
// }

ClassificationDataset makeLabeledData(Data<RealVector> &dataset, bool normalize) //Make labeled data from unlabeled data
{
	// create and train data normalizer
	std::vector<RealVector> inputs;
	std::vector<unsigned int> labels;

	inputs.resize(dataset.numberOfElements());
	for(int i = 0; i < inputs.size(); ++i)
		inputs[i].resize(dataset.element(i).size()-1);

	labels.resize(dataset.numberOfElements());

	//Fill the input and the label manually
	for(int i = 0; i < dataset.numberOfElements(); ++i)
	{
		for(int j = 0; j < dataset.element(i).size(); ++j)
		{
			if(j == (dataset.element(i).size() - 1))
			{
				if(dataset.element(i)[j] == 0)
					labels.at(i) = 0;
				else if(dataset.element(i)[j] == 1)
					labels.at(i) = 1;
			}	
			else	
				inputs[i](j) = dataset.element(i)[j];				
		}
	}
	ClassificationDataset normalizedData = createLabeledDataFromRange(inputs, labels);
	normalizedData.makeIndependent();

	bool removeMean = true;
	NormalizeComponentsUnitInterval<RealVector> normalizingTrainer;	//Data Normalization.
	Normalizer<RealVector> normalizer;
	normalizingTrainer.train(normalizer, normalizedData.inputs());
	normalizedData = transformInputs(normalizedData, normalizer);
	return normalizedData;
}

ClassificationDataset makeLabeledData(Data<RealVector> &trainingdataset, Data<RealVector> &testingdataset, bool normalize) //Make labeled data from unlabeled data
{
	// create and train data normalizer
	std::vector<RealVector> inputs;
	std::vector<unsigned int> labels;

	inputs.resize(trainingdataset.numberOfElements());
	for(int i = 0; i < inputs.size(); ++i)
		inputs[i].resize(trainingdataset.element(i).size()-1);

	labels.resize(trainingdataset.numberOfElements());

	//Fill the input and the label manually
	for(int i = 0; i < trainingdataset.numberOfElements(); ++i)
	{
		for(int j = 0; j < trainingdataset.element(i).size(); ++j)
		{
			if(j == (trainingdataset.element(i).size() - 1))
			{
				if(trainingdataset.element(i)[j] == 0)
					labels.at(i) = 0;
				else if(trainingdataset.element(i)[j] == 1)
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
	for(int i = 0; i < inputs.size(); ++i)
		inputs[i].resize(testingdataset.element(i).size()-1);

	labels.resize(testingdataset.numberOfElements());

	//Fill the input and the label manually
	for(int i = 0; i < testingdataset.numberOfElements(); ++i)
	{
		for(int j = 0; j < testingdataset.element(i).size(); ++j)
		{
			if(j == (testingdataset.element(i).size() - 1))
			{
				if(testingdataset.element(i)[j] == 0)
					labels.at(i) = 0;
				else if(testingdataset.element(i)[j] == 1)
					labels.at(i) = 1;
			}	
			else	
				inputs[i](j) = testingdataset.element(i)[j];				
		}
	}
	ClassificationDataset normalizedTestingData = createLabeledDataFromRange(inputs, labels);
	normalizedTestingData.makeIndependent();

	bool removeMean = true;
	NormalizeComponentsUnitInterval<RealVector> normalizingTrainer;
	Normalizer<RealVector> normalizer;
	normalizingTrainer.train(normalizer, normalizedTrainingData.inputs());
	normalizedTrainingData = transformInputs(normalizedTrainingData, normalizer);
	normalizedTestingData = transformInputs(normalizedTestingData, normalizer);
	return normalizedTestingData;
}

int main(int argc, char ** argv)
{
	int generation = 0;
	int population = 0;
	double C = 1;
	double gamma = 1;
	bool l1 = true;
	bool l2 = false;
	bool costSensitive = false;
	std::string trainingFile;
	std::string validationFile;
	std::string datasetFile;
	bool wholeDataset = true;


	if (argc < 3) 
	{ 
		// Check the value of argc. If not enough parameters have been passed, inform user and exit.
		std::cout << "Usage is -in <infile> -out <outdir>\n"; // Inform the user of how to use the program
		std::cin.get();
		exit(0);
	} 
	else 
	{ // if we got enough parameters...
		char* myFile, myPath, myOutPath;
		std::cout << argv[0];
		for (int i = 1; i < argc; i++) 
		{ /* We will iterate over argv[] to get the parameters stored inside.
		   * Note that we're starting on 1 because we don't need to know the
		   * path of the program, which is stored in argv[0] */
			//if (i + 1 != argc) // Check that we haven't finished parsing already
				if (string(argv[i]) == "-gen") 
				{
					// We know the next argument *should* be the filename:
					generation = atoi(argv[i + 1]);
					//std::cout << generation << "\n";
				} 
				else if (string(argv[i]) == "-p") 
				{
					population = atoi(argv[i + 1]);
				} 
				else if (string(argv[i]) == "-tr") 
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
					//std::cout << datasetFile << "\n";
					wholeDataset = true;
				} 
				else if (string(argv[i]) == "-c") 
				{
					//SVM Parameter C
					C = atof(argv[i + 1]);
					//std::cout << C << "\n";
				} 
				else if (string(argv[i]) == "-g") 
				{
					//SVM RBF Parameter gamma
					gamma = atof(argv[i + 1]);
					//std::cout << gamma << "\n";
				}
				else if(string(argv[i]) == "-cs")
				{
					costSensitive = true;
				}
				else if(string(argv[i]) == "-l1")
				{
					l1 = true;
					l2 = false;
				}
				else if(string(argv[i]) == "-l2")
				{
					l1 = false;
					l2 = true;
				}
		}
	}

	std::cout << "\nC: " << C << "\nGamma: " << gamma << "\nL1: " << l1 <<"\n";
	BinaryConfusionMatrix cfMatrix(C, gamma);
	std::string resultFile = "";
	if(wholeDataset)
	{
		resultFile = datasetFile + "_C_" + std::to_string(C) + "_g_" + std::to_string(gamma) + "_l1_" + std::to_string(l1) +  + "_l2_" + std::to_string(l2) + "_cs_" +  std::to_string(costSensitive) + ".result";
		// int folds = 10;	//Set to 10-fold cross validation #This is already set in the main file
		Data<RealVector> dataset;
		importCSV(dataset, datasetFile, ',','#',shark::Data<RealVector>::DefaultBatchSize, 1);
		std::cout << datasetFile << "\nNumber of Generation: "  << generation << " Population Size: " <<  population   << "\n";
	
		//Change the dataset into one-zero
		//zero - negative
		//one - positive
		typedef Data<RealVector>::element_range Elements;
		Elements elements = dataset.elements();
		double total_pos = 0;
		double total_neg = 0;
		for(Elements::iterator pos = elements.begin(); pos != elements.end(); ++pos)
		{
			if((*pos)[pos->size()-1] != 0)
			{
				++total_pos;
				(*pos)[pos->size()-1] = 1;
			}
			else
				++total_neg;
		}
		ClassificationDataset dataset_labeled = makeLabeledData(dataset, true);
		dataset_labeled.makeIndependent();

		if(costSensitive)
		{
			if(l1)
			{
				GaussianRbfKernel<RealVector> rbfKernel(std::pow(2,gamma));
				CSvmTrainer<RealVector> svm(&rbfKernel, total_pos * std::pow(2,C), total_neg * std::pow(2,C), true);
				GPSVMGM gpsvm(dataset_labeled, svm, cfMatrix, generation, population);
				// printBestGPFeatures(cfMatrix);
			}
			else
			{
				GaussianRbfKernel<RealVector> rbfKernel(std::pow(2,gamma));
				SquaredHingeCSvmTrainer<RealVector> svm(&rbfKernel, total_pos * std::pow(2,C), total_neg * std::pow(2,C), true);
				GPSVMGM gpsvm(dataset_labeled, svm, cfMatrix, generation, population);
				// printBestGPFeatures(cfMatrix);
			}
		}
		else if(!costSensitive)
		{
			if(l1)
			{
				GaussianRbfKernel<RealVector> rbfKernel(std::pow(2,gamma));
				CSvmTrainer<RealVector> svm(&rbfKernel, std::pow(2,C), true);
				GPSVMGM gpsvm(dataset_labeled, svm, cfMatrix, generation, population);
				// printBestGPFeatures(cfMatrix);
			}
			else
			{
				GaussianRbfKernel<RealVector> rbfKernel(std::pow(2,gamma));
				SquaredHingeCSvmTrainer<RealVector> svm(&rbfKernel, std::pow(2,C), true);
				GPSVMGM gpsvm(dataset_labeled, svm, cfMatrix, generation, population);
				// printBestGPFeatures(cfMatrix);
			}
		}
	}

	else if(!wholeDataset)
	{
		resultFile = validationFile + ".result";
		Data<RealVector> trainingDataset, testingDataset;
		importCSV(trainingDataset, trainingFile, ',','#',shark::Data<RealVector>::DefaultBatchSize, 1);
		importCSV(testingDataset, validationFile, ',','#',shark::Data<RealVector>::DefaultBatchSize, 1);
		std::cout << trainingFile << "\t" <<  validationFile << "\nNumber of Generation: "  << generation << " Population Size: " <<  population   << "\n";

		typedef Data<RealVector>::element_range Elements;
		Elements elementsTraining = trainingDataset.elements();
		double total_pos = 0;
		double total_neg = 0;
		for(Elements::iterator pos = elementsTraining.begin(); pos != elementsTraining.end(); ++pos)
		{
			if((*pos)[pos->size()-1] != 0)
			{
				++total_pos;
				(*pos)[pos->size()-1] = 1;
			}
			else
				++total_neg;
		}
		ClassificationDataset training_dataset_labeled = makeLabeledData(trainingDataset, true);
		training_dataset_labeled.makeIndependent();

		Elements elementsTesting = testingDataset.elements();
		double test_total_pos = 0;
		double test_total_neg = 0;
		for(Elements::iterator pos = elementsTesting.begin(); pos != elementsTesting.end(); ++pos)
		{
			if((*pos)[pos->size()-1] != 0)
			{
				++test_total_pos;
				(*pos)[pos->size()-1] = 1;
			}
			else
				++test_total_neg;
		}
		ClassificationDataset testing_dataset_labeled = makeLabeledData(trainingDataset, testingDataset, true);
		testing_dataset_labeled.makeIndependent();

		if(costSensitive)
		{
			if(l1)
			{
				GaussianRbfKernel<RealVector> rbfKernel(std::pow(2,gamma));
				CSvmTrainer<RealVector> svm(&rbfKernel, total_pos * std::pow(2,C), total_neg * std::pow(2,C), true);
				GPSVMGM gpsvm(training_dataset_labeled, testing_dataset_labeled, svm, cfMatrix, generation, population);
				// printBestGPFeatures(cfMatrix);
			}
			else
			{
				GaussianRbfKernel<RealVector> rbfKernel(std::pow(2,gamma));
				SquaredHingeCSvmTrainer<RealVector> svm(&rbfKernel, total_pos * std::pow(2,C), total_neg * std::pow(2,C), true);
				GPSVMGM gpsvm(training_dataset_labeled, testing_dataset_labeled, svm, cfMatrix, generation, population);
				// printBestGPFeatures(cfMatrix);
			}
		}
		else if(!costSensitive)
		{
			if(l1)
			{
				GaussianRbfKernel<RealVector> rbfKernel(std::pow(2,gamma));
				CSvmTrainer<RealVector> svm(&rbfKernel, std::pow(2,C), true);
				GPSVMGM gpsvm(training_dataset_labeled, testing_dataset_labeled, svm, cfMatrix, generation, population);
				// printBestGPFeatures(cfMatrix);
			}
			else
			{
				GaussianRbfKernel<RealVector> rbfKernel(std::pow(2,gamma));
				SquaredHingeCSvmTrainer<RealVector> svm(&rbfKernel, std::pow(2,C), true);
				GPSVMGM gpsvm(training_dataset_labeled, testing_dataset_labeled, svm, cfMatrix, generation, population);
				// printBestGPFeatures(cfMatrix);
			}
		}
	}
	WriteResultToFile(resultFile, cfMatrix);
	
	// std::string featureFile = trainingFile + "_C_" + std::to_string(C) + "_g_" + std::to_string(gamma)  + ".feature";
	// WriteDerivedFeaturesToFile(featureFile, temp);
	return 0;
}