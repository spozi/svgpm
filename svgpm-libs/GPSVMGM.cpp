#include "GPSVMGM.h"


// GPSVMGM::GPSVMGM(void) 
// {
// }

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


GPSVMGM::GPSVMGM(ClassificationDataset& dataset, CSvmTrainer<RealVector> &svm, BinaryConfusionMatrix &confusion, int generations, int populationsize) : m_L1SVM(&svm), m_L2SVM(nullptr)
{
	m_SVM = 1;
	std::cout << "Welcome to GPSVMGM\n";
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
	BuildAndEvaluateModelCV(m_LabeledDataset, 5, confusion); 	//Using 5 fold cross validation
}

GPSVMGM::GPSVMGM(ClassificationDataset& dataset, SquaredHingeCSvmTrainer<RealVector> &svm, BinaryConfusionMatrix &confusion, int generations, int populationsize) : m_L1SVM(nullptr), m_L2SVM(&svm)
{
	m_SVM = 2 ;
	std::cout << "Welcome to GPSVMGM\n";
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
	BuildAndEvaluateModelCV(m_LabeledDataset, 5, confusion);	//Using 5 fold cross validation
}


GPSVMGM::GPSVMGM(ClassificationDataset& trainingSet, ClassificationDataset& testingSet, CSvmTrainer<RealVector> &svm, BinaryConfusionMatrix &confusion, int generations, int populationsize) : m_L1SVM(&svm), m_L2SVM(nullptr)
{
	m_SVM = 1;
	m_LabeledTrainingSet = trainingSet;
	m_LabeledTrainingSet.makeIndependent();
	m_LabeledTestingSet  = testingSet;
	m_LabeledTestingSet.makeIndependent();

	m_TotalNumberOfOriginalAttributes	= m_LabeledTrainingSet.element(0).input.size();			//Store the total of original attributes
	m_TotalNumberOfTrainingInstances	= m_LabeledTrainingSet.numberOfElements();
	m_TotalNumberOfTestingInstances		= m_LabeledTestingSet.numberOfElements();

	for(int i = 0; i < m_TotalNumberOfTrainingInstances; ++i)
		m_TrainingLabels.push_back(m_LabeledTrainingSet.element(i).label);
	for(int i = 0; i < m_TotalNumberOfTestingInstances; ++i)
		m_TestingLabels.push_back(m_LabeledTestingSet.element(i).label);

	m_TotalGenerations = generations;
	m_TotalIndividuals = populationsize;

	InitGPSVMGM();
	BuildAndEvaluateModelDataset(m_LabeledTrainingSet, m_LabeledTestingSet, confusion);
	//EvolveGPSVMGM();
	//WriteDataToFile("All.csv");
}

GPSVMGM::GPSVMGM(ClassificationDataset& trainingSet, ClassificationDataset& testingSet, SquaredHingeCSvmTrainer<RealVector> &svm, BinaryConfusionMatrix &confusion, int generations, int populationsize) : m_L1SVM(nullptr), m_L2SVM(&svm)
{
	m_SVM = 2;
	m_LabeledTrainingSet = trainingSet;
	m_LabeledTrainingSet.makeIndependent();
	m_LabeledTestingSet  = testingSet;
	m_LabeledTestingSet.makeIndependent();

	m_TotalNumberOfOriginalAttributes	= m_LabeledTrainingSet.element(0).input.size();			//Store the total of original attributes
	m_TotalNumberOfTrainingInstances	= m_LabeledTrainingSet.numberOfElements();
	m_TotalNumberOfTestingInstances		= m_LabeledTestingSet.numberOfElements();

	for(int i = 0; i < m_TotalNumberOfTrainingInstances; ++i)
		m_TrainingLabels.push_back(m_LabeledTrainingSet.element(i).label);
	for(int i = 0; i < m_TotalNumberOfTestingInstances; ++i)
		m_TestingLabels.push_back(m_LabeledTestingSet.element(i).label);

	m_TotalGenerations = generations;
	m_TotalIndividuals = populationsize;

	InitGPSVMGM();
	BuildAndEvaluateModelDataset(m_LabeledTrainingSet, m_LabeledTestingSet, confusion);
	//EvolveGPSVMGM();
	//WriteDataToFile("All.csv");
}


void GPSVMGM::BuildAndEvaluateModelCV(ClassificationDataset& labeledDataset, int numCV, BinaryConfusionMatrix &confusion)
{
	ClassificationDataset internalData = labeledDataset;
	internalData.makeIndependent();
	CVFolds<ClassificationDataset> folds = createCVSameSizeBalanced(internalData, numCV);
	BinaryConfusionMatrix cfm = confusion;
	std::cout << "Fold:\t";
	for(int fold = 0; fold != folds.size(); ++fold)
	{
		ClassificationDataset training = folds.training(fold);
		ClassificationDataset validation = folds.validation(fold);

		//training.makeIndependent();
		//validation.makeIndependent();
		std::cout << fold << ", ";
		Train(training);
		//std::cout << "Writing to training file\n";
		//WriteDataToFile("TrainingCV.csv", training);
		cfm = cfm + Test(validation);
		//WriteDataToFile("TestingCV.csv", validation);
	}

	std::cout << "\n";
	confusion = cfm;
	//WriteResultToFile(m_filename, cfm);
}

void GPSVMGM::BuildAndEvaluateModelDataset(ClassificationDataset& traininglabeledDataset, ClassificationDataset& validationlabeledDataset, BinaryConfusionMatrix &confusion)
{
	ClassificationDataset internalTrainingData = traininglabeledDataset;
	internalTrainingData.makeIndependent();
	ClassificationDataset internalTestingData = validationlabeledDataset;
	internalTestingData.makeIndependent();
	
	//internalData.makeIndependent();
	//CVFolds<ClassificationDataset> folds = createCVSameSizeBalanced(internalData, numCV);
	BinaryConfusionMatrix cfm = confusion;
	//std::cout << "Fold:\t";
	//for(int fold = 0; fold != folds.size(); ++fold)
	//{
		//ClassificationDataset training = folds.training(fold);
		//ClassificationDataset validation = folds.validation(fold);

		//training.makeIndependent();
		//validation.makeIndependent();
		//std::cout << fold << ", ";
		std::cout << "Training begin\n";
		Train(internalTrainingData);
		std::cout << "Training Finish\n";
		//std::cout << "Writing to training file\n";
		//WriteDataToFile("TrainingCV.csv", training);
		cfm = cfm + Test(internalTestingData);
		//WriteDataToFile("TestingCV.csv", validation);
	//}

	//std::cout << "\n";
	confusion = cfm;
	confusion.DerivedFeatures = m_BestPopulation;
	//WriteResultToFile(m_filename, cfm);
}



std::vector<Tree> GPSVMGM::getBestPopulation()
{
	std::vector<Tree> BestIndividuals = m_BestPopulation;
	for(int i = BestIndividuals.size() - 1; i >= 0; --i)
	{
		if(BestIndividuals[i].mFitness == 0)
			BestIndividuals.erase(BestIndividuals.begin() + i);
	}

	return BestIndividuals;

}


//The following functions would be refactored


void GPSVMGM::Train(ClassificationDataset& labeledTrainDataset)
{
	m_LabeledTrainingSet = labeledTrainDataset;
	m_LabeledTrainingSet.makeIndependent();
	m_TrainingLabels.resize(m_LabeledTrainingSet.numberOfElements());
	for(int i = 0; i < m_LabeledTrainingSet.numberOfElements(); ++i)
		m_TrainingLabels[i] = m_LabeledTrainingSet.element(i).label;

	EvolveGPSVMGM(m_LabeledTrainingSet, m_TrainingLabels);

	labeledTrainDataset = m_BestDataset;
	labeledTrainDataset.makeIndependent();
}

BinaryConfusionMatrix GPSVMGM::Test(ClassificationDataset& labeledTestDataset)
{
	m_LabeledTestingSet = labeledTestDataset;
	m_LabeledTestingSet.makeIndependent();
	m_TestingLabels.resize(m_LabeledTestingSet.numberOfElements());
	for(int i = 0; i < m_LabeledTestingSet.numberOfElements(); ++i)
		m_TestingLabels[i] = m_LabeledTestingSet.element(i).label;


	ClassificationDataset fullGP = InterpretIndividualsIntoLabeledDataset(m_LabeledTestingSet, m_TestingLabels, m_Context, m_BestPopulation);
	fullGP.makeIndependent();
	//WriteDataToFile("FullGP.csv", fullGP);
	//std::cout << "Size of best indices: " << m_BestIndices.size() << "\n";
	ClassificationDataset bestGP = MakeDataFromBestFitness(fullGP, m_TestingLabels, m_BestPopulation, m_BestIndices);
	bestGP.makeIndependent();
	//WriteDataToFile("BestGP.csv", bestGP);
	labeledTestDataset = bestGP;
	labeledTestDataset.makeIndependent();
	BinaryConfusionMatrix x = EvaluateKernelModelOnDataset(bestGP, m_TestingLabels, m_BestKernelClassifier);
	return x;
}

void GPSVMGM::InitGPSVMGM()
{
	m_Context.mRandom.seed(1);
	m_Context.insert(new GPAttributePrimitive::Add);
	m_Context.insert(new GPAttributePrimitive::Subtract);
	m_Context.insert(new GPAttributePrimitive::Multiply);
	m_Context.insert(new GPAttributePrimitive::Divide);	
	m_Context.insert(new GPAttributePrimitive::Ln);
	m_Context.insert(new GPAttributePrimitive::SquareRoot);
	m_Context.insert(new GPAttributePrimitive::Power);
	m_Context.insert(new GPAttributePrimitive::Exponent);
	m_Context.insert(new GPAttributePrimitive::ErrorFunction);
	m_Context.insert(new GPAttributePrimitive::Log);
	m_Context.insert(new GPAttributePrimitive::Cosine);
	m_Context.insert(new GPAttributePrimitive::Sine);
	m_Context.insert(new GPAttributePrimitive::Ephemeral);
	
	for(int i = 0; i <  m_TotalNumberOfOriginalAttributes; ++i)
	{
		std::string x = std::string("A") + std::to_string(i + 1);
		m_Context.insert(new TokenT<double>(x));
	}
	m_Population.resize(m_TotalIndividuals);
}

void GPSVMGM::EvolveGPSVMGM(ClassificationDataset& labeledDataset, std::vector<unsigned int>& labels)
{
	double pct = 0.0;
	double temp = 0.0;
	
	//Initialize population
	//std::cout << "Initializing population\n";
	Puppy::initializePopulation(m_Population, m_Context);
	//std::cout << "Evaluating\n";
	pct = EvaluatePopulation(labeledDataset, labels);
	//std::cout << "Best Population\n";
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
}

double GPSVMGM::EvaluatePopulation(ClassificationDataset& labeledDataset, std::vector<unsigned int>& labels)
{
	ClassificationDataset internalData = labeledDataset;
	//internalData.makeIndependent();
	m_GPLabeledDataset = InterpretIndividualsIntoLabeledDataset(internalData, labels, m_Context, m_Population);
	//m_GPLabeledDataset.makeIndependent();	//Break free from m_LabeledDataset
	//m_GPLabeledDataset = &mergedData;
	//ClassificationDataset NormalizedMergedData = NormalizeData(mergedData);
	//NormalizedMergedData.makeIndependent();
	m_GPLabeledDataset = TrainInternal(m_GPLabeledDataset, labels, m_ConfusionMatrix);

	//WriteDataToFile("MergedData.csv", m_GPLabeledDataset);

	//Performance metrics
	double pct = std::min(m_ConfusionMatrix.getAccuracy(), m_ConfusionMatrix.getGeometricMean()) / m_ConfusionMatrix.nSV;
	//double pct = m_ConfusionMatrix.getGeometricMean();
	//double pct = m_ConfusionMatrix.getAccuracy();
	//double pct = m_ConfusionMatrix.getAccuracy()  / m_ConfusionMatrix.nSV;
	
	for(int i = 0; i < m_Population.size(); ++i)
	{
		if(m_Population[i].mFitness == 1.0)
			m_Population[i].mFitness = pct;
		else
			m_Population[i].mFitness = 0.0;
	}
	//std::cout << "Population size: " << m_Population.size() << "\t" << pct;

	return pct;
}

ClassificationDataset GPSVMGM::MergeOriginalDataWithGPData(std::vector<RealVector>& OriginalData, std::vector<RealVector>& GPData, std::vector<unsigned int>& labels)
{
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

	ClassificationDataset x = createLabeledDataFromRange(ds, labels);
	//x.makeIndependent();
	return x;
		
}

ClassificationDataset GPSVMGM::MakeDataFromBestFitness(ClassificationDataset& FullGPlabeledDataset, std::vector<unsigned int>& labels, std::vector<Tree> bestPopulation, std::vector<int>& bestIndices)
{
	std::vector<RealVector> OriginalData;
	std::vector<RealVector> GPFullData;
	std::vector<RealVector> GPData;

	OriginalData.resize(FullGPlabeledDataset.numberOfElements());
	GPFullData.resize(FullGPlabeledDataset.numberOfElements());
	GPData.resize(FullGPlabeledDataset.numberOfElements());
	for(int i = 0; i < FullGPlabeledDataset.numberOfElements(); ++i)
	{	
		OriginalData[i].resize(m_TotalNumberOfOriginalAttributes);
		GPFullData[i].resize(FullGPlabeledDataset.element(i).input.size());
		GPData[i].resize(bestIndices.size());
		for(int j = 0; j < m_TotalNumberOfOriginalAttributes; ++j)
		{
			OriginalData[i](j) = FullGPlabeledDataset.element(i).input(j);
		}

		for(int j = 0; j < FullGPlabeledDataset.element(i).input.size(); ++j)
		{
			GPFullData[i](j) = FullGPlabeledDataset.element(i).input(j);
		}

		for(unsigned int j = 0; j < bestIndices.size(); ++j)
		{
			int index = bestIndices[j];
			//std::cout << "Index: " << index << "\t" <<  labeledDataset.element(i).input.size() << "\t" <<  labeledDataset.element(i).input(index) << "\t" << GPFullData[i](index) << "\n"; 
			GPData[i][j] = FullGPlabeledDataset.element(i).input(index);
		}
	}

	//for(unsigned int i = 0; i < m_TotalNumberOfInstances; ++i)
	//{
	//	for(unsigned int j = 0; j < bestIndices.size(); ++j)
	//	{
	//		int index = bestIndices[j];
	//		std::cout << "Index: " << index << "\t" <<  GPFullData[i].size() << "\t" << GPFullData[i][index] << "\n"; 
	//		GPData[i][j] = GPFullData[i][index];
	//	}
	//}

	
	//for(unsigned int i = 0; i < m_TotalNumberOfInstances; ++i)
	//{
	//	//Set the value to each node of GP Tree
	//	for(unsigned int j = 0; j < m_TotalNumberOfOriginalAttributes; ++j)
	//	{
	//		m_Context.mPrimitiveMap[std::string("A" + std::to_string(j+1))]->setValue(&(labeledDataset.element(i).input(j)));
	//	}

	//	for(unsigned int j = 0; j < m_BestIndices.size(); ++j)
	//	{
	//		double xds = 0.0;
	//		//std::cout << m_BestIndices[j] << "\t " << xds << "\n";
	//		m_Population[m_BestIndices[j]].interpret(&xds, m_Context);

	//		GPData[i][j] = xds;
	//	}
	//}
	ClassificationDataset x = MergeOriginalDataWithGPData(OriginalData, GPData, labels);
	//x.makeIndependent();
	//WriteDataToFile("bestX.csv", x);
	return x;
}

ClassificationDataset GPSVMGM::NormalizeData(ClassificationDataset& dataset)
{
	bool removeMean = false;
	Normalizer<RealVector> normalizer;
	NormalizeComponentsUnitVariance<RealVector> normalizingTrainer(removeMean);
	normalizingTrainer.train(normalizer, dataset.inputs());
	return transformInputs(dataset, normalizer);
}

void GPSVMGM::WriteResultToFile(std::string filename, BinaryConfusionMatrix& confusionMatrix)
{
	std::ofstream ofs;
	ofs.open (filename, std::ofstream::out | std::ofstream::app);

	ofs << confusionMatrix.TP << "\t" << confusionMatrix.FP << "\n";
	ofs << confusionMatrix.FN << "\t" << confusionMatrix.TN << "\n";

	ofs.close();
}

void GPSVMGM::WriteDataToFile(std::string filename, ClassificationDataset& dataset)
{
	std::ofstream ofs(filename);
	ofs << dataset;
	ofs.close();
}


ClassificationDataset GPSVMGM::InterpretIndividualsIntoLabeledDataset(ClassificationDataset& labeledDataset, std::vector<unsigned int>& labels, Context& GPContext, std::vector<Tree> &populationTree)
{
	std::vector<RealVector> OriginalData;
	std::vector<RealVector> GPData;

	GPData.resize(labeledDataset.numberOfElements());
	OriginalData.resize(labeledDataset.numberOfElements());

	//Filling the original data
	for(int i = 0; i < labeledDataset.numberOfElements(); ++i)
	{
		GPData[i].resize(populationTree.size());
		OriginalData[i].resize(labeledDataset.element(i).input.size());
		for(int j = 0; j < labeledDataset.element(i).input.size(); ++j)
		{
			OriginalData[i](j) = labeledDataset.element(i).input(j);
		}
	}

	//std::cout << "Hello:\n";
	for(unsigned int i = 0; i < labeledDataset.numberOfElements(); ++i)
	{
		//Set the value to each node of GP Tree
		for(unsigned int j = 0; j < labeledDataset.element(i).input.size(); ++j)
		{
			GPContext.mPrimitiveMap[std::string("A" + std::to_string(j+1))]->setValue(&(labeledDataset.element(i).input(j)));
		}

		for(unsigned int j = 0; j < populationTree.size(); ++j)
		{
/*			if(populationTree[j].mValid)
				continue;*/	
			double xds = 0.0;
			populationTree[j].interpret(&xds, GPContext);
			//std::cout << xds << "\n";
			if(__isnan(xds))
				xds = 0.0;
			if(!__finite(xds))
				xds = 0.0;
			GPData[i](j) = xds;
		}
	}
	ClassificationDataset mergedData = MergeOriginalDataWithGPData(OriginalData, GPData, labels);
	mergedData.makeIndependent();
	return mergedData;
}



/*The most crucial code, lot days of debugging*/
ClassificationDataset GPSVMGM::TrainInternal(ClassificationDataset& labeledDataset, std::vector<unsigned int>& labels, BinaryConfusionMatrix& ConfusionMatrix)
{
	CART.train(m_CARTModel, labeledDataset);
	UIntVector histogram = m_CARTModel.countAttributes();	//Histogram consists of totaloriginalattributes + GPAttributes
	//std::cout << "Attribute selection finish\n";
	
	/***************************The code portion below is very important, source of memory violation / bugs****************************************/
	//Check each attributes
	m_Indices.clear();
	for(int i = 0; i < histogram.size(); ++i)	//What about if histogram.size is actually bigger than size of population?
	{
		if(i < m_TotalNumberOfOriginalAttributes)	//Ignore original attributes
			continue;
		if(histogram[i] > 0)	// GP Attributes start from m_TotalNumberOfOriginalAttributes If GP attributes exist in the CART Model
		{
			//m_Population[i].mFitness = 1.0;	//Set the fitness to 1.0 if this individual is contained in the CARTModel
			m_Indices.push_back(i - m_TotalNumberOfOriginalAttributes);
		}
		//else 
			//m_Population[i].mFitness = 0.0; //Set the fitness to 0.0 if this individual is not in the CARTModel
	}

	for(int i = 0; i < m_Indices.size(); ++i)
	{
		m_Population[m_Indices[i]].mFitness = 1.0;
		m_Population[m_Indices[i]].mValid = true;
	}

	for(int i = 0; i < m_Population.size(); ++i)
	{
		if(m_Population[i].mValid != true)
		{
			m_Population[i].mFitness = 0.0;
			m_Population[i].mValid = true;
		}
	}

	/**********************************************************************************************************************************/

	ClassificationDataset BestMergedGPData = MakeDataFromBestFitness(labeledDataset, labels, m_Population, m_Indices);

	//ClassificationDataset BestNormalizedMergedData = NormalizeData(BestMergedData);
	//BestNormalizedMergedData.makeIndependent();
	//WriteDataToFile("BestMergedData.csv", BestMergedGPData);

	//std::cout << "SVM Training\n";
	if(m_SVM == 1)	
		m_L1SVM->train(m_KernelClassifier, BestMergedGPData);
	else if (m_SVM == 2)
		m_L2SVM->train(m_KernelClassifier, BestMergedGPData);
	ConfusionMatrix = EvaluateKernelModelOnDataset(BestMergedGPData, labels, m_KernelClassifier);
	ConfusionMatrix.calculateAll();
	return BestMergedGPData;
}


BinaryConfusionMatrix GPSVMGM::EvaluateKernelModelOnDataset(ClassificationDataset& labeledDataset, std::vector<unsigned int>& labels, KernelClassifier<RealVector>& kc)
{
	Data<unsigned int> testing_output = kc(labeledDataset.inputs());
	BinaryConfusionMatrix ConfusionMatrix;
	ConfusionMatrix.TP = 0; ConfusionMatrix.FP = 0; ConfusionMatrix.TN = 0; ConfusionMatrix.FN = 0;
	
	std::string filename1 = std::string("fisher.txt");
	std::ofstream myfile1;
	myfile1.open(filename1);
	for(int i = 0; i < testing_output.numberOfElements(); ++i)
	{
		myfile1 << labeledDataset.element(i).label << "\t" << testing_output.element(i) << "\n";
		if(labeledDataset.element(i).label == 0 && testing_output.element(i) == 0)
		{
			++ConfusionMatrix.TN;
		}
		if(labeledDataset.element(i).label == 0 && testing_output.element(i) == 1)
		{
			++ConfusionMatrix.FP; //if actual is negative, but output is positive, then it is false positive
		}			
		if(labeledDataset.element(i).label == 1 && testing_output.element(i) == 0)
		{
			++ConfusionMatrix.FN; //if actual is positive, but output is negative, then it is false negative
		}			
		if(labeledDataset.element(i).label == 1 && testing_output.element(i) == 1)
		{
			++ConfusionMatrix.TP;
		}
	}
	myfile1.close();
	ConfusionMatrix.calculateAll();
	ConfusionMatrix.nSV = kc.decisionFunction().alpha().size1();	//Number of support vector

	return ConfusionMatrix;
}

GPSVMGM::~GPSVMGM(void)
{
	//delete m_Context;
	m_Population.clear();
	m_BestPopulation.clear();
	m_BestIndices.clear();
}
