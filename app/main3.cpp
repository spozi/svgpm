#define SHARK_USE_OPENMP 1
#define BOOST_UBLAS_NDEBUG 1

#include <GPSVMGM.h>

#include <iostream>
#include <fstream>
#include <list>
#include <cmath>

using namespace std;

void writeDataToFile(ClassificationDataset &dataset, std::string &filename)
{
	std::ofstream ofs(filename);
	ofs << dataset;
	ofs.close();
}

void WriteResultToFile(std::string filename, BinaryConfusionMatrix& confusionMatrix)
{
	std::ofstream ofs;
	ofs.open (filename, std::ofstream::out | std::ofstream::app);

	ofs << confusionMatrix.TP << "\t" << confusionMatrix.FP << "\n";
	ofs << confusionMatrix.FN << "\t" << confusionMatrix.TN << "\n";

	ofs.close();
}

void WriteResultToFile(std::string filename, std::vector<BinaryConfusionMatrix>& confusionMatrix)
{
	std::ofstream ofs;
	ofs.open (filename, std::ofstream::out | std::ofstream::app);

	for(int i = 0; i < confusionMatrix.size(); ++i)
	{
		ofs << /*confusionMatrix[i].powerC << "\t" << confusionMatrix[i].powerGamma << "\t" <<*/ confusionMatrix[i].TP << "\t" << confusionMatrix[i].FP << "\n";
		ofs	<</*						        "\t" <<                                  "\t" <<*/ confusionMatrix[i].FN << "\t" << confusionMatrix[i].TN << "\n";
	}	

	ofs.close();
}

void WriteDerivedFeaturesToFile(std::string filename, std::vector<BinaryConfusionMatrix> & confusionMatrix)
{
	std::ofstream ofs;
	ofs.open (filename, std::ofstream::out | std::ofstream::app);
	
	
	for(int i = 0; i < confusionMatrix.size(); ++i)
	{
		//ofs << confusionMatrix[i].DerivedFeatures.size();
		std::vector<Tree> bestFeatures = confusionMatrix[i].DerivedFeatures;
		for(int j = 0; j < bestFeatures.size(); ++j)
		{
			if(bestFeatures[j].mFitness != 0)
			{
				ofs << bestFeatures[j];
				if(j != bestFeatures.size() - 1)
					ofs << "\t\t"; 
			}
		}
		ofs << "\n";
	}
	ofs.close();
}

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

    // Data already normalized
	bool removeMean = true;
	// NormalizeComponentsUnitInterval<RealVector> normalizingTrainer;
	// Normalizer<RealVector> normalizer;
	// normalizingTrainer.train(normalizer, normalizedTrainingData.inputs());
	// normalizedTrainingData = transformInputs(normalizedTrainingData, normalizer);
	// normalizedTestingData = transformInputs(normalizedTestingData, normalizer);
	return normalizedTestingData;
}

int main(int argc, char ** argv)
{
	int generation = 0;
	int population = 0;
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
				if (string(argv[i]) == "-g") 
				{
					// We know the next argument *should* be the filename:
					generation = atoi(argv[i + 1]);
					std::cout << generation << "\n";
				} 
				else if (string(argv[i]) == "-p") 
				{
					population = atoi(argv[i + 1]);
				} 
				else if (string(argv[i]) == "-tr") 
				{
					trainingFile = argv[i + 1];
					std::cout << trainingFile << "\n";
					wholeDataset = false;
				} 
				else if (string(argv[i]) == "-ts") 
				{
					validationFile = argv[i + 1];
					std::cout << validationFile << "\n";
				} 
				else if (string(argv[i]) == "-ds") 
				{
					datasetFile = argv[i + 1];
					std::cout << datasetFile << "\n";
					wholeDataset = true;
				} 
				//else 
				//{
					//std::cout << "Not enough or invalid arguments, please try again.\n";
					//Sleep(2000); 
					//exit(0);
				//}
			//std::cout << argv[i] << " ";
		}
	}

	if(wholeDataset)
	{
		// int folds = 10; //Folds is set to 3
		Data<RealVector> dataset;
		importCSV(dataset, datasetFile, ',','#',shark::Data<RealVector>::DefaultBatchSize, 1);
		std::cout << datasetFile << " Number of Generation: "  << generation << " Population Size: " <<  population   << "\n";
	
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

		//writeDataToFile(dataset_labeled, std::string("Kurt.csv"));

		////Cost Sensitive SVM
		//bool bias = true;
		//bool unconstrained = false;

		//std::vector<BinaryConfusionMatrix> result;
		//
		////SVM Trainer
		//std::string dataFile(argv[1]);
		//std::string resultFile = datasetFile + "GP_l1_acc_svm.result";
		std::string resultFile = datasetFile + "GP_l1_svm.result";
		//GaussianRbfKernel<RealVector> rbfKernel;
		//LinearKernel<RealVector> lKernel;
		//KernelClassifier<RealVector> kc;
		//
		////CSvmTrainer<RealVector> *L1trainer;//(&rbfKernel, std::pow(2,i), true);
		////SquaredHingeCSvmTrainer<RealVector> *L2trainer;

		std::vector<BinaryConfusionMatrix> ConfusionMatrix;
		//ConfusionMatrix.resize(132);

	#pragma omp parallel for
		for(int i = -5; i < 7; ++i)
		{	
		#pragma omp parallel for
			for(int j = -4; j < 7; ++j)
			{
				std::cout << "L1SVM C: 2^" << i << "gamma: 2^" << j << "\n"; 
				GaussianRbfKernel<RealVector> rbfKernel(std::pow(2,j));
				CSvmTrainer<RealVector> svm(&rbfKernel, std::pow(2,i), true);
				BinaryConfusionMatrix temp(i, j);
				GPSVMGM gpsvm(dataset_labeled, svm, temp, generation, population);
				ConfusionMatrix.push_back(temp);
			}
		}
		std::sort(ConfusionMatrix.begin(), ConfusionMatrix.end());
		WriteResultToFile(resultFile, ConfusionMatrix);
		ConfusionMatrix.clear();
	
		//Cost Sensitive
		//resultFile = datasetFile + "GP_l1_acc_CS_svm.result";
		resultFile = datasetFile + "GP_l1_CS_svm.result";
	#pragma omp parallel for
		for(int i = -5; i < 7; ++i)
		{	
		#pragma omp parallel for
			for(int j = -4; j < 7; ++j)
			{
				cout << "Cost Sensitive L1SVM C: 2^" << i << "gamma: 2^" << j << "\n"; 
				GaussianRbfKernel<RealVector> rbfKernel(std::pow(2,j));
				CSvmTrainer<RealVector> svm(&rbfKernel, total_pos * std::pow(2,i), total_neg * std::pow(2,i), true);
				BinaryConfusionMatrix temp(i, j);
				GPSVMGM gpsvm(dataset_labeled, svm, temp, generation, population);
				ConfusionMatrix.push_back(temp);
			}
		}
		std::sort(ConfusionMatrix.begin(), ConfusionMatrix.end());
		WriteResultToFile(resultFile, ConfusionMatrix);
		ConfusionMatrix.clear();

		//resultFile = datasetFile + "GP_l2_acc_svm.result";
		resultFile = datasetFile + "GP_l2_svm.result";
	#pragma omp parallel for
		for(int i = -5; i < 7; ++i)
		{	
		#pragma omp parallel for
			for(int j = -4; j < 7; ++j)
			{
				std::cout << "L2SVM C: 2^" << i << "gamma: 2^" << j << "\n"; 
				GaussianRbfKernel<RealVector> rbfKernel(std::pow(2,j));
				SquaredHingeCSvmTrainer<RealVector> svm(&rbfKernel, std::pow(2,i), true);
				BinaryConfusionMatrix temp(i, j);
				GPSVMGM gpsvm(dataset_labeled, svm, temp, generation, population);
				ConfusionMatrix.push_back(temp);
			}
		}
		std::sort(ConfusionMatrix.begin(), ConfusionMatrix.end());
		WriteResultToFile(resultFile, ConfusionMatrix);
		ConfusionMatrix.clear();

		//Cost Sensitive
		//resultFile = datasetFile + "GP_l2_acc_CS_svm.result";
		resultFile = datasetFile + "GP_l2_CS_svm.result";
	#pragma omp parallel for
		for(int i = -5; i < 7; ++i)
		{	
		#pragma omp parallel for
			for(int j = -4; j < 7; ++j)
			{
				cout << "Cost Sensitive L2SVM C: 2^" << i << "gamma: 2^" << j << "\n"; 
				GaussianRbfKernel<RealVector> rbfKernel(std::pow(2,j));
				SquaredHingeCSvmTrainer<RealVector> svm(&rbfKernel, total_pos * std::pow(2,i), total_neg * std::pow(2,i), true);
				BinaryConfusionMatrix temp(i, j);
				GPSVMGM gpsvm(dataset_labeled, svm, temp, generation, population);
				ConfusionMatrix.push_back(temp);
			}
		}
		std::sort(ConfusionMatrix.begin(), ConfusionMatrix.end());
		WriteResultToFile(resultFile, ConfusionMatrix);
		ConfusionMatrix.clear();
	}


	else if(!wholeDataset)
	{
		Data<RealVector> trainingDataset, testingDataset;
		importCSV(trainingDataset, trainingFile, ',','#',shark::Data<RealVector>::DefaultBatchSize, 1);
		importCSV(testingDataset, validationFile, ',','#',shark::Data<RealVector>::DefaultBatchSize, 1);
		std::cout << trainingFile << "\t" <<  validationFile << " Number of Generation: "  << generation << " Population Size: " <<  population   << "\n";

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

		std::vector<BinaryConfusionMatrix> ConfusionMatrix;

		
		//std::string resultFile = validationFile + "_GP_l1_acc_svm.result";
		std::string resultFile = validationFile + "_GP_l1_svm.result";
		//std::string featureFile = validationFile + "_GP_l1_acc_svm.GPModel";
		std::string featureFile = validationFile + "_GP_l1_svm.GPModel";

	#pragma omp parallel for
		for(int i = -5; i < 7; ++i)
		{	
		#pragma omp parallel for
			for(int j = -4; j < 7; ++j)
			{
				std::cout << "L1SVM C: 2^" << i << "gamma: 2^" << j << "\n"; 
				GaussianRbfKernel<RealVector> rbfKernel(std::pow(2,j));
				CSvmTrainer<RealVector> svm(&rbfKernel, std::pow(2,i), true);
				BinaryConfusionMatrix temp(i, j);
				GPSVMGM gpsvm(training_dataset_labeled, testing_dataset_labeled, svm, temp, generation, population);
				ConfusionMatrix.push_back(temp);
			}
		}
		std::sort(ConfusionMatrix.begin(), ConfusionMatrix.end());
		WriteResultToFile(resultFile, ConfusionMatrix);
		WriteDerivedFeaturesToFile(featureFile, ConfusionMatrix);
		ConfusionMatrix.clear();
	
		//Cost Sensitive
		//resultFile = validationFile + "_GP_l1_acc_CS_svm.result";
		resultFile = validationFile + "_GP_l1_CS_svm.result";
		//featureFile = validationFile + "_GP_l1_acc_CS_svm.GPModel";
		featureFile = validationFile + "_GP_l1_CS_svm.GPModel";
	#pragma omp parallel for
		for(int i = -5; i < 7; ++i)
		{	
		#pragma omp parallel for
			for(int j = -4; j < 7; ++j)
			{
				cout << "Cost Sensitive L1SVM C: 2^" << i << "gamma: 2^" << j << "\n"; 
				GaussianRbfKernel<RealVector> rbfKernel(std::pow(2,j));
				CSvmTrainer<RealVector> svm(&rbfKernel, total_pos * std::pow(2,i), total_neg * std::pow(2,i), true);
				BinaryConfusionMatrix temp(i, j);
				GPSVMGM gpsvm(training_dataset_labeled, testing_dataset_labeled, svm, temp, generation, population);
				ConfusionMatrix.push_back(temp);
			}
		}
		std::sort(ConfusionMatrix.begin(), ConfusionMatrix.end());
		WriteResultToFile(resultFile, ConfusionMatrix);
		WriteDerivedFeaturesToFile(featureFile, ConfusionMatrix);
		ConfusionMatrix.clear();

		//resultFile = validationFile + "_GP_l2_acc_svm.result";
		resultFile = validationFile + "_GP_l2_svm.result";
		//featureFile = validationFile + "_GP_l2_acc_svm.GPModel";
		featureFile = validationFile + "_GP_l2_svm.GPModel";
	#pragma omp parallel for
		for(int i = -5; i < 7; ++i)
		{	
		#pragma omp parallel for
			for(int j = -4; j < 7; ++j)
			{
				std::cout << "L2SVM C: 2^" << i << "gamma: 2^" << j << "\n"; 
				GaussianRbfKernel<RealVector> rbfKernel(std::pow(2,j));
				SquaredHingeCSvmTrainer<RealVector> svm(&rbfKernel, std::pow(2,i), true);
				BinaryConfusionMatrix temp(i, j);
				GPSVMGM gpsvm(training_dataset_labeled, testing_dataset_labeled, svm, temp, generation, population);
				ConfusionMatrix.push_back(temp);
			}
		}
		std::sort(ConfusionMatrix.begin(), ConfusionMatrix.end());
		WriteResultToFile(resultFile, ConfusionMatrix);
		WriteDerivedFeaturesToFile(featureFile, ConfusionMatrix);
		ConfusionMatrix.clear();

		//Cost Sensitive
		//resultFile = validationFile + "_GP_l2_acc_CS_svm.result";
		resultFile = validationFile + "_GP_l2_CS_svm.result";
		//featureFile = validationFile + "_GP_l2_acc_CS_svm.GPModel";
		featureFile = validationFile + "_GP_l2_CS_svm.GPModel";
	#pragma omp parallel for
		for(int i = -5; i < 7; ++i)
		{	
		#pragma omp parallel for
			for(int j = -4; j < 7; ++j)
			{
				cout << "Cost Sensitive L2SVM C: 2^" << i << "gamma: 2^" << j << "\n"; 
				GaussianRbfKernel<RealVector> rbfKernel(std::pow(2,j));
				SquaredHingeCSvmTrainer<RealVector> svm(&rbfKernel, total_pos * std::pow(2,i), total_neg * std::pow(2,i), true);
				BinaryConfusionMatrix temp(i, j);
				GPSVMGM gpsvm(training_dataset_labeled, testing_dataset_labeled, svm, temp, generation, population);
				ConfusionMatrix.push_back(temp);
			}
		}
		std::sort(ConfusionMatrix.begin(), ConfusionMatrix.end());
		WriteResultToFile(resultFile, ConfusionMatrix);
		WriteDerivedFeaturesToFile(featureFile, ConfusionMatrix);
		ConfusionMatrix.clear();
	}
	return 0;
}