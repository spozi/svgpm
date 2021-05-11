#include "GPAProgram.h"
#include <shark/Algorithms/Trainers/CARTTrainer.h>


GPAProgram::GPAProgram(void)
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

	m_numPopulation = 300;
	m_numParticipant = 2;
	m_minDepth = 3;
	m_maxDepth = 7;
	m_GrowProbability = 0.5;
	m_CrossOverDistribProbablity = 0.9;
	m_MutationStandard = 0.1;
	m_MutationSwap = 0.1;

}

//The number of population is the number of features. Hence, each individual represents one feature
std::vector<Tree> GPAProgram::evolve(shark::ClassificationDataset& dataset, shark::CSvmTrainer<shark::RealVector>& svm, shark::KernelClassifier<shark::RealVector>& kc, int numberOfGeneration)
{

	double performanceMetric = 0.0;

	//Primitive setup
	for(int i = 0; i < dataset.element(0).input.size(); ++i)
	{
		std::string x = std::string("A") + std::to_string(i + 1);
		m_Context.insert(new TokenT<double>(x));
	}

	for(int i = 0; i < kc.decisionFunction().parameterVector().size(); ++i)
	{
		std::string x = std::string("ALPHA") + std::to_string(i + 1);
		m_Context.insert(new TokenT<double>(x));
	}

	m_Population.resize(m_numPopulation);
	initializePopulation(m_Population, m_Context, m_GrowProbability, m_minDepth, m_maxDepth);
	evaluateGPAttribute(dataset, svm, kc);
	calculateStats(m_Population, 0);

	for(unsigned int i = 1; i <= numberOfGeneration; ++i) 
	{
		applySelectionTournament(m_Population, m_Context, m_numParticipant);
		applyCrossover(m_Population, m_Context, m_CrossOverProbability, m_CrossOverDistribProbablity);
		applyMutationStandard(m_Population, m_Context, m_MutationStandard);
		applyMutationSwap(m_Population, m_Context, m_MutationSwap);
		evaluateGPAttribute(dataset, svm, kc);
		calculateStats(m_Population, i);

		std::vector<Tree>::const_iterator lBestIndividual =
			std::max_element(m_Population.begin(), m_Population.end());

		double temp = lBestIndividual->mFitness;
		if(temp >= performanceMetric)
		{
			performanceMetric = temp;
			bestPopulation = m_Population;
		}	
	}

	//Return the best population (the final and cut version)
	std::sort(bestPopulation.begin(), bestPopulation.end());
	std::vector<Tree> veryBest;
	for(int i = bestPopulation.size()/2; i < bestPopulation.size(); ++i)
	{
		veryBest.push_back(bestPopulation[i]);
	}
	bestPopulation.clear();
	bestPopulation = veryBest;
	return bestPopulation;
}

std::vector<Tree> GPAProgram::getBestPopulation()
{
	return bestPopulation;
}

double GPAProgram::evaluateGPAttribute(shark::ClassificationDataset &dataset, shark::CSvmTrainer<shark::RealVector> &svm, shark::KernelClassifier<shark::RealVector> &kc)
{
	std::vector<shark::RealVector> GPInputs;
	GPInputs.resize(dataset.numberOfElements());
	std::vector<unsigned int> labels;

	//Fill the label in seperate vector container
	for(int i = 0; i < dataset.numberOfElements(); ++i)
	{
		labels.push_back(dataset.element(i).label);
	}

	//Copy the training set into a GPInputs vector
	for(int i = 0; i < dataset.numberOfElements(); ++i)
	{
		GPInputs[i].resize(m_Population.size());
	}

	for(unsigned int i = 0; i < dataset.numberOfElements(); ++i)
	{
		//Set the value to each node of GP Tree (Features)
		for(unsigned int j = 0; j < dataset.element(i).input.size(); ++j)
		{
			m_Context.mPrimitiveMap[std::string("A" + std::to_string(j+1))]->setValue(&(dataset.element(i).input(j)));
		}
		//Set the value of each node of GP Tree (SVM Alpha Value)
		for(unsigned int j = 0; j < kc.decisionFunction().parameterVector().size(); ++j)
		{
			m_Context.mPrimitiveMap[std::string("ALPHA" + std::to_string(j+1))]->setValue(&kc.decisionFunction().parameterVector()(j));
		}

		for(unsigned int j = 0; j < m_Population.size(); ++j)
		{
			if(m_Population[j].mValid)
				continue;					
			m_Population[j].interpret(&GPInputs[i](j), m_Context);
		}
	}

	std::vector<shark::RealVector> GPData;
	GPData.resize(dataset.numberOfElements());
	for(int i = 0; i < GPData.size(); ++i)
	{
		GPData[i].resize(dataset.element(i).input.size() + m_Population.size());		//Resize to #population features/attributes
		
		//Fill the original attributes
		for(unsigned int j = 0; j < dataset.element(i).input.size(); ++j)
		{
			GPData[i](j) =  dataset.element(i).input(j);
		}
		//Fill the GPAttribute at the end after the original attributes
		for(unsigned int j = 0; j < m_numPopulation; ++j)
		{
			GPData[i](j + dataset.element(i).input.size()) =  GPInputs[i][j];
		}
	}

	shark::ClassificationDataset GPClassificationData = shark::createLabeledDataFromRange(GPData, labels); 

	//Decision Tree	(Feature Selection)
	shark::CARTTrainer CART;
	CART.setNumberOfFolds(3);	//Just set to 3 fold
	shark::CARTClassifier<shark::RealVector> CARTModel;

	CART.train(CARTModel, GPClassificationData);
	shark::UIntVector histogram = CARTModel.countAttributes();
	std::vector<int> GPOnlyHistogram;
	int originalAttributes = 0;
	for(int i = 0; i < histogram.size(); ++i)
	{
		originalAttributes = histogram.size() - m_Population.size();	//Histogram.size() must be greater than population size
		if(i < originalAttributes)
			continue;
		m_Population[i - originalAttributes].mFitness = histogram[i];	//m_Population start from 0, histogram m_Population start from originalAttributes.size
		m_Population[i - originalAttributes].mValid = false;
		GPOnlyHistogram.push_back(histogram[i]);
	}

	int newSVMAttributes = m_Population.size() * m_PopulationPercentage;

	//Lambda Sort
	auto sortPopulation = [&GPOnlyHistogram](std::vector<int> &v)
	{
		  // initialize original index locations
		  std::vector<int> idx(v.size());
		  for (int i = 0; i != idx.size(); ++i) idx[i] = i;
		
		  // sort indexes based on comparing values in v
		  sort(idx.begin(), idx.end(),
		       [&v](int i1, int i2) {return v[i1] < v[i2];});
		       
		  return idx;
	};

	std::vector<int> oldIndex = sortPopulation(GPOnlyHistogram);

	//Reminder
	//The dataset need to be resized first
	std::vector<shark::RealVector> GPSet;
	GPSet.resize(dataset.numberOfElements());
	for(int i = 0; i < dataset.numberOfElements(); ++i)
	{
		GPSet[i].resize(dataset.element(i).input.size() + newSVMAttributes);
		for(int j = 0; j < dataset.element(i).input.size(); ++j)
		{
			GPSet[i](j) =  dataset.element(i).input(j);
		}
		//Fill the GPAttribute at the end after the original attributes
		for(unsigned int j = 0; j < newSVMAttributes; ++j)
		{
			GPSet[i](j + dataset.element(j).input.size()) =  GPData[i][oldIndex[j+newSVMAttributes]];
		}
	}

	GPClassificationData = shark::createLabeledDataFromRange(GPSet, labels);

	//Evaluate
	double stdeviation = 0.0;
	double avg = 0.0;
	double sdsum = 0.0;
	double sdpow2sum = 0.0;
	double pct = 0.0;

	int ctr = 0;
	//svm.train(kc, dt);
	//kc.decisionFunction().offset(0);
	shark::Data<unsigned int> testing_output = kc(GPClassificationData.inputs());
	shark::Data<shark::RealVector> testing_distance = kc.decisionFunction()(GPClassificationData.inputs());
	BinaryConfusionMatrix ConfusionMatrix;
	for(int i = 0; i < testing_output.numberOfElements(); ++i)
	{
		//std::cout << testing_distance.element(j)(0) << "\n";
		if(labels[i] == 0 && testing_output.element(i) == 0)
		{
			//if(testing_distance.element(j)(0) <= -1.0)
				++ConfusionMatrix.TN;
			//else
				//++ConfusionMatrix.FP;
		}
		if(labels[i] == 0 && testing_output.element(i) == 1)
		{
			++ConfusionMatrix.FP; //if actual is negative, but output is positive, then it is false positive
		}			
		if(labels[i] == 1 && testing_output.element(i) == 0)
		{
			++ConfusionMatrix.FN; //if actual is positive, but output is negative, then it is false negative
		}			
		if(labels[i] == 1 && testing_output.element(i) == 1)
		{
			//if(testing_distance.element(j)(0) >= 1.0)
				++ConfusionMatrix.TP;
			//else
				//++ConfusionMatrix.FN;
		} 
	}

	double temp = std::min(ConfusionMatrix.getAccuracy(), ConfusionMatrix.getGeometricMean());
	if(temp > pct)
		pct = temp;

	for(int i = 0; i < newSVMAttributes; ++i)
	{	
		m_Population[oldIndex[i+newSVMAttributes]].mFitness = m_Population[oldIndex[i+newSVMAttributes]].mFitness * pct;
		m_Population[oldIndex[i+newSVMAttributes]].mValid = true;
	}

	std::vector<Tree>::const_iterator lBestIndividual =
			std::max_element(m_Population.begin(), m_Population.end());

	for(int i = 0; i < m_Population.size(); ++i)
	{	
		if(m_Population[i].mValid == false)
		{
			m_Population[i].mFitness = 0;
			m_Population[i].mValid = true;
		}

		if(lBestIndividual->mFitness == 0.0)
			m_Population[i].mFitness = 0.0;
		else
			m_Population[i].mFitness /= lBestIndividual->mFitness;
	}

	return pct;
}


GPAProgram::~GPAProgram(void)
{
	
}
