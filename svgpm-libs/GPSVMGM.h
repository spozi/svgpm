#pragma once
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


#include <string>


#include "puppy/Puppy.hpp"

#include "BinaryConfusionMatrix.h"
#include "GPAttributePrimitive.h"

using namespace shark;
using namespace Puppy;

class GPSVMGM
{
public:
	//GPSVMGM(void);
	GPSVMGM(ClassificationDataset& dataset, CSvmTrainer<RealVector> &svm, BinaryConfusionMatrix &cfm, int generations, int populationsize);
	GPSVMGM(ClassificationDataset& dataset, SquaredHingeCSvmTrainer<RealVector> &svm, BinaryConfusionMatrix &cfm, int generations, int populationsize);
	GPSVMGM(ClassificationDataset& trainingSet, ClassificationDataset& testingSet, CSvmTrainer<RealVector> &svm, BinaryConfusionMatrix &cfm, int generations, int populationsize);
	GPSVMGM(ClassificationDataset& trainingSet, ClassificationDataset& testingSet, SquaredHingeCSvmTrainer<RealVector> &svm, BinaryConfusionMatrix &cfm, int generations, int populationsize);
	~GPSVMGM(void);

	void BuildAndEvaluateModelCV(int numberofCrossValidation);
	void BuildAndEvaluateModelCV(ClassificationDataset& dataset, int numberofCrossValidation, BinaryConfusionMatrix &cfm);
	
	void BuildAndEvaluateModelDataset(ClassificationDataset& trainingLabeledDataset, ClassificationDataset& validationLabeledDataset, BinaryConfusionMatrix &cfm);

	void Train(ClassificationDataset& dataset);
	BinaryConfusionMatrix Test(ClassificationDataset& dataset);

	std::vector<Tree> getBestPopulation();


	ClassificationDataset NormalizeData(ClassificationDataset &dataset);
	void MakeLabeledData();

	void WriteDataToFile(std::string filename, ClassificationDataset& dataset);
	void WriteResultToFile(std::string filename, BinaryConfusionMatrix& confusionMatrix);


private:
	int m_SVM;

	//Source file
	std::string m_filename;

	//POD
	int m_TotalNumberOfOriginalAttributes;
	int m_TotalNumberOfInstances;
	int m_TotalNumberOfTrainingInstances;
	int m_TotalNumberOfTestingInstances;
	int m_CrossValidationTotal;


	//GP
	int m_TotalIndividuals;
	int m_TotalGenerations;

	Context m_Context;
	std::vector<Tree> m_Population;
	std::vector<Tree> m_BestPopulation;

	std::vector<int> m_Indices;
	std::vector<int> m_BestIndices;
	
	//Dataset
	ClassificationDataset m_LabeledDataset; 
	ClassificationDataset m_GPLabeledDataset;
	ClassificationDataset m_LabeledTrainingSet;
	//ClassificationDataset m_GPTrainingSet;
	ClassificationDataset m_LabeledTestingSet;
	ClassificationDataset m_GPTestingSet;

	ClassificationDataset m_BestDataset;
	ClassificationDataset m_BestTrainingDataset;
	ClassificationDataset m_BestTestingDataset;

	Data<RealVector> m_Dataset;

	
	std::vector<RealVector> CombinedDataset;

	std::vector<unsigned int> m_Labels;
	std::vector<unsigned int> m_TrainingLabels;
	std::vector<unsigned int> m_TestingLabels;

	//Classifier Trainer
	CSvmTrainer<RealVector>* m_L1SVM;
	SquaredHingeCSvmTrainer<RealVector>* m_L2SVM;
	CARTTrainer CART;
	Normalizer<RealVector> normalizer;
	

	//Classifier Model
	CARTClassifier<RealVector> m_CARTModel;
	KernelClassifier<RealVector> m_KernelClassifier;
	KernelClassifier<RealVector> m_BestKernelClassifier;
	// NormalizeComponentsUnitVariance<RealVector> normalizingTrainer;

	//Performance Measure
	BinaryConfusionMatrix m_ConfusionMatrix;

private:
	
	//Core methods
	void InitGPSVMGM();																						//Initialize GPSVMGM
	void EvolveGPSVMGM(ClassificationDataset& labeledDataset, std::vector<unsigned int>& labels);			//Evolve GPSVMGM
	double EvaluatePopulation(ClassificationDataset& labeledDataset, std::vector<unsigned int>& labels);
	
	ClassificationDataset TrainInternal(ClassificationDataset& labeledDataset, std::vector<unsigned int>& labels, BinaryConfusionMatrix& confusionMatrix);
	BinaryConfusionMatrix EvaluateKernelModelOnDataset(ClassificationDataset& labeledDataset, std::vector<unsigned int>& labels, KernelClassifier<RealVector>& kc);
	
	void ResizeDataset();		//Resize Dataset
	
	
	
	ClassificationDataset MergeOriginalDataWithGPData(std::vector<RealVector>& OriginalData, std::vector<RealVector>& GPData, std::vector<unsigned int>& labels);
	ClassificationDataset MakeDataFromBestFitness(ClassificationDataset& FullGPlabeledDataset, std::vector<unsigned int>& labels, std::vector<Tree> bestPopulation, std::vector<int>& bestIndices);	//FullGPLabeledDataset attributes = original + population size
	ClassificationDataset InterpretIndividualsIntoLabeledDataset(ClassificationDataset& labeledDataset, std::vector<unsigned int>& labels, Context& GPContext, std::vector<Tree> &populationTree);

};

