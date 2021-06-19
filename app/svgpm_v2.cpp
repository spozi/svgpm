#define SHARK_USE_OPENMP 1
#define BOOST_UBLAS_NDEBUG 1

#include <GPSVMGM.h>

#include <iostream>
#include <fstream>
#include <list>
#include <cmath>
#include <chrono>
using namespace std::chrono;

using namespace std;


// Function prototype
ClassificationDataset makeLabeledData(Data<RealVector> dataset); //Make labeled data from unlabeled data


int main(int argc, char ** argv)
{
	int generation = 0;
	int population = 0;
    int C = 0;
    int gamma = 0;
    int fold = 2;
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
		{ 
          /* We will iterate over argv[] to get the parameters stored inside.
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
            else if (string(argv[i]) == "-C") 
            {
                C = atoi(argv[i + 1]);
            } 
            else if (string(argv[i]) == "-g") 
            {
                gamma = atoi(argv[i + 1]);
            }
            else if (string(argv[i]) == "--gamma") 
            {
                gamma = atoi(argv[i + 1]);
            } 
            else if (string(argv[i]) == "-cv") 
            {
                fold = atoi(argv[i + 1]);
            } 
            else 
            {
                std::cout << "Not enough or invalid arguments, please try again.\n";
                exit(0);
            }
		}
	}

    if(wholeDataset)
	{
		// int folds = 10; //Folds is set to 5
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
		
        ClassificationDataset dataset_labeled = makeLabeledData(dataset);
		dataset_labeled.makeIndependent();
		std::vector<BinaryConfusionMatrix> ConfusionMatrix;

        cout << "Cost Sensitive L2SVM C: 2^" << C << "gamma: 2^" << gamma << "\n"; 
        // auto start = high_resolution_clock::now();
        GaussianRbfKernel<RealVector> rbfKernel(std::pow(2,gamma));
        SquaredHingeCSvmTrainer<RealVector> svm(&rbfKernel, total_pos * std::pow(2,C), total_neg * std::pow(2,C), true);
        // auto stop = high_resolution_clock::now();
        BinaryConfusionMatrix temp(C, gamma);
        GPSVMGM gpsvm(dataset_labeled, svm, temp, generation, population);
        temp.powerC = C;
        temp.powerGamma = gamma;
        ConfusionMatrix.push_back(temp);
    }
}


























ClassificationDataset makeLabeledData(Data<RealVector> dataset) //Make labeled data from unlabeled data
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
	ClassificationDataset labeledData = createLabeledDataFromRange(inputs, labels);
	return labeledData;
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




