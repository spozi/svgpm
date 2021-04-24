// #pragma once
#include <puppy/Puppy.hpp>
#include <vector>
#include <shark/Data/Dataset.h>
#include <shark/Algorithms/Trainers/CSvmTrainer.h>

#include "GPAttributePrimitive.h"
#include "BinaryConfusionMatrix.h"

using namespace Puppy;
class GPAProgram
{
public:
	GPAProgram(void);
	
	void setPopulation(int& numPopulation);
	void setGrowProbability(double& growProbability);
	void setMinInitialTreeDepth(double& minDepth);
	void setMaxInitialTreeDepth(double& maxDepth);

	std::vector<Tree>  evolve(shark::ClassificationDataset& dataset, shark::CSvmTrainer<shark::RealVector>& svm, shark::KernelClassifier<shark::RealVector>& kc, int numOfIteration);
	std::vector<Tree>  evolve(shark::ClassificationDataset& dataset, shark::SquaredHingeCSvmTrainer<shark::RealVector>& svm, shark::KernelClassifier<shark::RealVector>& kc, int numOfIteration);
	// std::vector<Tree>  evolve(shark::ClassificationDataset& dataset, shark::CSvmTrainer<shark::RealVector>& svm, shark::KernelClassifier<shark::RealVector>& kc, int numOfIteration);

	std::vector<Tree> getBestPopulation();

	shark::ClassificationDataset createGPData(shark::ClassificationDataset& dataset);


	double evaluateGPAttribute(shark::ClassificationDataset &dataset, shark::CSvmTrainer<shark::RealVector>& svm, shark::KernelClassifier<shark::RealVector> &kc);

	~GPAProgram(void);
private:
	Puppy::Context m_Context;
	std::vector<Puppy::Tree> m_Population;
	std::vector<Puppy::Tree> bestPopulation;
	int m_numPopulation;
	int m_numParticipant;
	double m_GrowProbability;
	double m_minDepth;
	double m_maxDepth;
	double m_CrossOverProbability;
	double m_CrossOverDistribProbablity;
	double m_MutationSwap;
	double m_MutationStandard;
	double m_PopulationPercentage;


};

