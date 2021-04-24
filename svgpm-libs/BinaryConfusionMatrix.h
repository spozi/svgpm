#ifndef BINARY_CONFUSION_MATRIX
#define BINARY_CONFUSION_MATRIX

#include <cstdlib>
#include <cmath>
#include "puppy/Tree.hpp"

class BinaryConfusionMatrix
{
public:
	int powerGamma; int powerC;
	int TP; int FP;
	int FN; int TN;
	int nSV;

	std::vector<Puppy::Tree> DerivedFeatures;

private:
	double mAccuracy;
	double mSensitivity;
	double mSpecificity;
	double mGeometricMean;



public:
	BinaryConfusionMatrix(void)
	{
		powerGamma = 0; powerC = 0;
		TP = 0; FP = 0;
		FN = 0; TN = 0;
		nSV = 0;
	}

	BinaryConfusionMatrix(int expC, int expGamma)
	{
		powerGamma = expGamma; powerC = expC;
		TP = 0; FP = 0;
		FN = 0; TN = 0;
		nSV = 0;
	}

	BinaryConfusionMatrix(const BinaryConfusionMatrix& obj)
	{
		powerGamma = obj.powerGamma; powerC = obj.powerC;
		DerivedFeatures = obj.DerivedFeatures;
		nSV = obj.nSV;

		TP = obj.TP; FP = obj.FP;
		FN = obj.FN; TN = obj.TN;

		mAccuracy = obj.mAccuracy;
		mSpecificity = obj.mSpecificity;
		mSensitivity = obj.mSensitivity;
		mGeometricMean = obj.mGeometricMean;

	}

	bool operator<( BinaryConfusionMatrix& cfm)
	{
		if (powerC != cfm.powerC)
			return powerC < cfm.powerC;
		if (powerGamma != cfm.powerGamma)
			return powerGamma < cfm.powerGamma;
	}

	BinaryConfusionMatrix operator+(const BinaryConfusionMatrix& obj)
	{
		BinaryConfusionMatrix bc;
		bc.powerGamma = this->powerGamma + obj.powerGamma;	bc.powerC = this->powerC + obj.powerC;
		bc.TP =	this->TP + obj.TP;	bc.FP = this->FP + obj.FP;
		bc.FN =	this->FN + obj.FN;	bc.TN = this->TN + obj.TN;

		bc.calculateAll();

		return bc;
	}

	void calculateAll()
	{
		mAccuracy = ((double)TP + double(TN)) / ((double)TP + double(TN) + (double)FP + double(FN));
		mSpecificity = (double(TN) / double(TN) + double(FP));
		mSensitivity = (double(TP) / double(TP) + double(FN));
		mGeometricMean = std::sqrt(getSensitivity() * getSpecificity());
	}

	double getAccuracy()
	{
		return mAccuracy;
	}

	double getSpecificity()
	{
		return mSpecificity;
	}

	double getSensitivity()
	{
		return mSensitivity;
	}

	double getGeometricMean()
	{
		return mGeometricMean;
	}
};

#endif 