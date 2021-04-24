#pragma once
#include <shark/Algorithms/Trainers/CARTTrainer.h>

using namespace shark;
class DecisionTree : public CARTTrainer
{
public:
	DecisionTree(void);

	std::string name() const
	{	return "DecisionTree";	}

	std::vector<int> getAttributes()
	{
		std::vector<int> listOfUsedAttributes;

	}
	~DecisionTree(void);
};

