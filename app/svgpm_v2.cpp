#define SHARK_USE_OPENMP 1
#define BOOST_UBLAS_NDEBUG 1

#include <shark/Algorithms/Trainers/CSvmTrainer.h>
#include <shark/Algorithms/Trainers/NormalizeComponentsUnitVariance.h>
#include <shark/Algorithms/Trainers/NormalizeComponentsUnitInterval.h>
#include <shark/Models/Kernels/GaussianRbfKernel.h>
#include <shark/Models/Kernels/PolynomialKernel.h>
#include <shark/Models/Kernels/LinearKernel.h>
#include <shark/Models/Kernels/AbstractKernelFunction.h>
#include <shark/Models/Normalizer.h>
#include <shark/Models/Kernels/KernelExpansion.h>
#include <shark/ObjectiveFunctions/Loss/ZeroOneLoss.h>
#include <shark/Data/DataDistribution.h>
#include <shark/Data/Csv.h>
#include <shark/Data/CVDatasetTools.h>
#include <shark/Data/DataDistribution.h>
#include <boost/archive/polymorphic_text_oarchive.hpp>
#include <boost/serialization/binary_object.hpp>
#include <shark/Algorithms/Trainers/CARTTrainer.h>     // the CART trainer has been replaced by RFTrainer
#include <shark/Algorithms/Trainers/RFTrainer.h>     // the RF trainer



#include <iostream>
#include <fstream>
#include <list>
#include <cmath>
#include <chrono>
#include <string>

#include "puppy/Puppy.hpp"
#include "GPAttributePrimitive.h"

using namespace std::chrono;
using namespace std;

using namespace shark;
using namespace Puppy;


//Struct Binary Confusion Matrix
typedef struct _BinaryConfusionMatrix{
	int TP; int FP;
	int FN; int TN;
} BinaryConfusionMatrix;


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
            else if (string(argv[i]) == "--cv") 
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

BinaryConfusionMatrix SVGPM(ClassificationDataset dataset, int C, int gamma, int generations, int populationsize, int kfold)
{
    int TotalNumberOfOriginalAttributes = dataset.element(0).input.size();  //Total original features
    int TotalNumberOfInstances		    = dataset.numberOfElements();       //Total instances we have right now

    std::vector<int> Labels;
    for(int i = 0; i < TotalNumberOfInstances; ++i)
		Labels.push_back(dataset.element(i).label);

    //SVGPM Start here
    //1. Initialization
    Puppy::Context svgpm_context;
    svgpm_context.mRandom.seed(1);
	svgpm_context.insert(new GPAttributePrimitive::Add);
	svgpm_context.insert(new GPAttributePrimitive::Subtract);
	svgpm_context.insert(new GPAttributePrimitive::Multiply);
	svgpm_context.insert(new GPAttributePrimitive::Divide);	
	svgpm_context.insert(new GPAttributePrimitive::Ln);
	svgpm_context.insert(new GPAttributePrimitive::SquareRoot);
	svgpm_context.insert(new GPAttributePrimitive::Power);
	svgpm_context.insert(new GPAttributePrimitive::Exponent);
	svgpm_context.insert(new GPAttributePrimitive::ErrorFunction);
	svgpm_context.insert(new GPAttributePrimitive::Log);
	svgpm_context.insert(new GPAttributePrimitive::Cosine);
	svgpm_context.insert(new GPAttributePrimitive::Sine);
	svgpm_context.insert(new GPAttributePrimitive::Ephemeral);
	
	for(int i = 0; i <  TotalNumberOfOriginalAttributes; ++i)
	{
		std::string x = std::string("A") + std::to_string(i + 1);
		svgpm_context.insert(new TokenT<double>(x));
	}
    std::vector<Tree> svgpm_Population(populationsize);

    //2. Evolution (super long process)
    ClassificationDataset internalData = dataset;
	internalData.makeIndependent();

    CVFolds<ClassificationDataset> folds = createCVSameSizeBalanced(internalData, kfold);
    for(int fold = 0; fold != folds.size(); ++fold)
	{
		ClassificationDataset Training = folds.training(fold);
		ClassificationDataset Validation = folds.validation(fold);


    /***************************************Training start here***************************************************************************/
        std::vector<unsigned int> TrainingLabels(Training.numberOfElements());
        for(int i = 0; i < Training.numberOfElements(); ++i)
		    TrainingLabels[i] = Training.element(i).label;

        double pct = 0.0;
        double temp = 0.0;
        
        //Evolution start
        //Initialize population
        Puppy::initializePopulation(svgpm_Population, svgpm_context);



        std::vector<RealVector> OriginalData;
        std::vector<RealVector> GPData;

        GPData.resize(Training.numberOfElements());
        OriginalData.resize(Training.numberOfElements());

        //Filling the original data
        for(int i = 0; i < Training.numberOfElements(); ++i)
        {
            GPData[i].resize(svgpm_Population.size());
            OriginalData[i].resize(Training.element(i).input.size());
            for(int j = 0; j < Training.element(i).input.size(); ++j)
            {
                OriginalData[i](j) = Training.element(i).input(j);
            }
        }

        for(unsigned int i = 0; i < Training.numberOfElements(); ++i)
        {
            //Set the value to each node of GP Tree
            for(unsigned int j = 0; j < Training.element(i).input.size(); ++j)
            {
                svgpm_context.mPrimitiveMap[std::string("A" + std::to_string(j+1))]->setValue(&(Training.element(i).input(j)));
            }

            for(unsigned int j = 0; j < svgpm_Population.size(); ++j)
            {
                /*	if(populationTree[j].mValid)
                        continue;
                */	
                double xds = 0.0;
                svgpm_Population[j].interpret(&xds, svgpm_context);
                //std::cout << xds << "\n";
                if(__isnan(xds))
                    xds = 0.0;
                if(!__finite(xds))
                    xds = 0.0;
                GPData[i](j) = xds;
            }
        }

        //Merge original data
        std::vector<RealVector> ds;
        ds.resize(OriginalData.size());
        for(int i = 0; i < ds.size(); ++i)
        {
            ds[i].resize(OriginalData[i].size() + GPData[i].size());
            //Dataset 1;
            for(int j = 0; j < OriginalData[i].size(); ++j)
            {
                ds[i](j) = OriginalData[i](j);
            }
            //Dataset 2;
            for(int j = 0; j < GPData[i].size(); ++j)
            {
                ds[i](j + OriginalData[i].size()) = GPData[i](j);
            }
        }

        ClassificationDataset mergedLabeledData = createLabeledDataFromRange(ds, TrainingLabels);
        mergedLabeledData.makeIndependent();

        //Critical Training the most crucial code, lot days of debugging

        //Define CART
        CARTTrainer CART;
        CARTClassifier<RealVector> svgpm_CARTModel;
        CART.train(svgpm_CARTModel, mergedLabeledData);
        UIntVector histogram = svgpm_CARTModel.countAttributes();	//Histogram consists of totaloriginalattributes + GPAttributes
        
        /***************************The code portion below is very important, source of memory violation / bugs****************************************/
        //Check each attributes
        std::vector<int> svgpm_Indices;
        for(int i = 0; i < histogram.size(); ++i)	//What about if histogram.size is actually bigger than size of population?
        {
            if(i < TotalNumberOfOriginalAttributes)	//Ignore original attributes
                continue;
            if(histogram[i] > 0)	// GP Attributes start from m_TotalNumberOfOriginalAttributes If GP attributes exist in the CART Model
            {
                svgpm_Indices.push_back(i - TotalNumberOfOriginalAttributes);
            }
        }

        for(int i = 0; i < svgpm_Indices.size(); ++i)
        {
            svgpm_Population[svgpm_Indices[i]].mFitness = 1.0;
            svgpm_Population[svgpm_Indices[i]].mValid = true;
        }

        for(int i = 0; i < svgpm_Population.size(); ++i)
        {
            if(svgpm_Population[i].mValid != true)
            {
                svgpm_Population[i].mFitness = 0.0;
                svgpm_Population[i].mValid = true;
            }
        }

        //Merging and obtaining the GPData based on the best fitness function
        std::vector<RealVector> OriginalData;
        std::vector<RealVector> GPFullData;
        std::vector<RealVector> GPData;

        OriginalData.resize(mergedLabeledData.numberOfElements());
        GPFullData.resize(mergedLabeledData.numberOfElements());
        GPData.resize(mergedLabeledData.numberOfElements());
        for(int i = 0; i < mergedLabeledData.numberOfElements(); ++i)
        {	
            OriginalData[i].resize(TotalNumberOfOriginalAttributes);
            GPFullData[i].resize(mergedLabeledData.element(i).input.size());
            GPData[i].resize(svgpm_Indices.size());
            for(int j = 0; j < TotalNumberOfOriginalAttributes; ++j)
            {
                OriginalData[i](j) = mergedLabeledData.element(i).input(j);
            }

            for(int j = 0; j < mergedLabeledData.element(i).input.size(); ++j)
            {
                GPFullData[i](j) = mergedLabeledData.element(i).input(j);
            }

            for(unsigned int j = 0; j < svgpm_Indices.size(); ++j)
            {
                int index = svgpm_Indices[j];
                //std::cout << "Index: " << index << "\t" <<  labeledDataset.element(i).input.size() << "\t" <<  labeledDataset.element(i).input(index) << "\t" << GPFullData[i](index) << "\n"; 
                GPData[i][j] = mergedLabeledData.element(i).input(index);
            }
        }

        //Select the SVM
        


















        pct = EvaluatePopulation(Training, TrainingLabels);

        std::vector<Puppy::Tree> svgpm_BestPopulation = svgpm_Population;
        // std::vector<int> svgpm_BestIndices = 
        m_BestPopulation	= m_Population;
        //std::cout << "Best Indices\n";
        m_BestIndices		= m_Indices;
        //std::cout << "Best Kernel Classifier\n";
        m_BestKernelClassifier = m_KernelClassifier;
        //std::cout << "Best Dataset\n";
        m_BestDataset		= m_GPLabeledDataset;
        //std::cout << "Make Independet\n";
        m_BestDataset.makeIndependent();
        //std::cout << "CalculateStats\n";
        //Puppy::calculateStats(m_Population, 0);
	
	
	    //Iterations
        for(unsigned int i=1; i <= m_TotalGenerations; ++i) 
        {	
            //std::cout << "Selection Start:\n" << m_Population.size() << "\n";
            //if(performanceMetric > 0.0)
            applySelectionTournament(m_Population, m_Context);
            //else
            //applySelectionRoulette(m_Population, m_Context);		
            //std::cout << "Selection\n";
            applyCrossover(m_Population, m_Context);
            //std::cout << "Crossover\n";
            applyMutationStandard(m_Population, m_Context);
            //std::cout << "Mutation\n";
            applyMutationSwap(m_Population, m_Context);
            //std::cout << "Evaluating.. \n";
            double temp = EvaluatePopulation(labeledDataset, labels);
            if(pct <= temp)
            {
                pct						= temp;
                m_BestPopulation		= m_Population;
                m_BestIndices			= m_Indices;
                m_BestKernelClassifier	= m_KernelClassifier;
                m_BestDataset			= m_GPLabeledDataset;
                m_BestDataset.makeIndependent();
            }	
            if(pct == 1.0)
                break;
            //Puppy::calculateStats(m_BestPopulation, i);
        }    

		//training.makeIndependent();
		//validation.makeIndependent();
		std::cout << fold << ", ";
		Train(training);
		//std::cout << "Writing to training file\n";
		//WriteDataToFile("TrainingCV.csv", training);
		cfm = cfm + Test(validation);
		//WriteDataToFile("TestingCV.csv", validation);
	}

}


GPSVMGM::GPSVMGM(ClassificationDataset& dataset, SquaredHingeCSvmTrainer<RealVector> &svm, BinaryConfusionMatrix &confusion, int generations, int populationsize, int kfold) : m_L1SVM(nullptr), m_L2SVM(&svm)
{
	m_SVM = 2 ;
	// std::cout << "Welcome to GPSVMGM\n";
	m_LabeledDataset = dataset;
	m_LabeledDataset.makeIndependent();
	//m_filename = filename;
	m_TotalNumberOfOriginalAttributes = m_LabeledDataset.element(0).input.size();					//Store the total of original attributes

	m_TotalNumberOfInstances		= m_LabeledDataset.numberOfElements();

	for(int i = 0; i < m_TotalNumberOfInstances; ++i)
		m_Labels.push_back(m_LabeledDataset.element(i).label);


	m_TotalGenerations = generations;
	m_TotalIndividuals = populationsize;

	std::cout << "Init GPSVMGM\n";
	InitGPSVMGM();
	std::cout << "Training\n";
	//EvolveGPSVMGM();
	BuildAndEvaluateModelCV(m_LabeledDataset, kfold, confusion);	//Using 5 fold cross validation
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