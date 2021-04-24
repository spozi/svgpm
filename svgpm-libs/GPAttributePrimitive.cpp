#include "GPAttributePrimitive.h"

#include <cmath>
#include <math.h>
#include <sstream>


using namespace Puppy;

/*!
 *  \brief Construct Add GP primitive.
 */
namespace GPAttributePrimitive
{
	Add::Add() :
	  Primitive(2, "ADD")
	{ }

	/*!
	 *  \brief Execute characteristic operation of Add primitive.
	 *  \param outDatum Result of the Add operation.
	 *  \param ioContext Evolutionary context.
	 */
	void Add::execute(void* outDatum, Context& ioContext)
	{
	  double& lResult = *(double*)outDatum;
	  double lArg2;
	  getArgument(0, &lResult, ioContext);
	  getArgument(1, &lArg2, ioContext);
	  lResult += lArg2;
	}


	/*!
	 *  \brief Construct Subtract GP primitive.
	 */
	Subtract::Subtract() :
	  Primitive(2, "SUB")
	{ }


	/*!
	 *  \brief Execute characteristic operation of Subtract primitive.
	 *  \param outDatum Result of the Subtract operation.
	 *  \param ioContext Evolutionary context.
	 */
	void Subtract::execute(void* outDatum, Context& ioContext)
	{
	  double& lResult = *(double*)outDatum;
	  double lArg2;
	  getArgument(0, &lResult, ioContext);
	  getArgument(1, &lArg2, ioContext);
	  lResult -= lArg2;
	}


	/*!
	 *  \brief Construct Multiply GP primitive.
	 */
	Multiply::Multiply() :
	  Primitive(2, "MUL")
	{ }


	/*!
	 *  \brief Execute characteristic operation of Multiply primitive.
	 *  \param outDatum Result of the Multiply operation.
	 *  \param ioContext Evolutionary context.
	 */
	void Multiply::execute(void* outDatum, Context& ioContext)
	{
	  double& lResult = *(double*)outDatum;
	  double lArg2;
	  getArgument(0, &lResult, ioContext);
	  getArgument(1, &lArg2, ioContext);
	  lResult *= lArg2;
	}


	/*!
	 *  \brief Construct Divide GP primitive.
	 */
	Divide::Divide() :
	  Primitive(2, "DIV")
	{ }


	/*!
	 *  \brief Execute characteristic operation of Divide primitive.
	 *  \param outDatum Result of the Divide operation.
	 *  \param ioContext Evolutionary context.
	 */
	void Divide::execute(void* outDatum, Context& ioContext)
	{
	  double& lResult = *(double*)outDatum;
	  double lArg2;
	  getArgument(1, &lArg2, ioContext);
	  if(std::fabs(lArg2) < 0.001) lResult = 1.0;
	  else {
		getArgument(0, &lResult, ioContext);
		lResult /= lArg2;
	  }
	}

	//ErrorFunction
	ErrorFunction::ErrorFunction() :
		Primitive(1, "ERF")
	{}

	void ErrorFunction::execute(void* outDatum, Context& ioContext)
	{
		double& lResult = *(double*)outDatum;
		getArgument(0, &lResult, ioContext);
		auto errorFunction = [&]() -> double
		{
			// constants
			double a1 =  0.254829592;
			double a2 = -0.284496736;
			double a3 =  1.421413741;
			double a4 = -1.453152027;
			double a5 =  1.061405429;
			double p  =  0.3275911;

			// Save the sign of x
			int sign = 1;
			if (lResult < 0)
				sign = -1;
			lResult = fabs(lResult);

			// A&S formula 7.1.26
			double t = 1.0/(1.0 + p*lResult);
			double y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*exp(-lResult*lResult);

			return sign*y;
		};

		lResult = errorFunction();
	}

	Ln::Ln () : 
		Primitive(1, "LN")
	{ }

	void Ln::execute(void* outDatum, Context& ioContext)
	{
		double& lResult = *(double*)outDatum;
		getArgument(0, &lResult, ioContext);
		lResult = std::log(lResult); //log e
	}

	Log::Log() :
		Primitive(1, "LOG")
	{ }

	void Log::execute(void* outDatum, Context& ioContext)
	{
		double& lResult = *(double*)outDatum;
		getArgument(0, &lResult, ioContext);
		lResult = std::log10(lResult); //log 10
	}

	SquareRoot::SquareRoot() :
		Primitive(1, "SQRT")
	{ }

	void SquareRoot::execute(void* outDatum, Context& ioContext)
	{
		double& lResult = *(double*)outDatum;
		getArgument(0, &lResult, ioContext);
		lResult = std::sqrt(lResult); //log 10
	}

	Power::Power() :
		Primitive(2, "POWER")
	{ }

	void Power::execute(void* outDatum, Context& ioContext)
	{
		double& lResult = *(double*)outDatum;
		double lArg2 = 0.0;
		getArgument(0, &lResult, ioContext);
		getArgument(1, &lArg2, ioContext);
		lResult = std::pow(lResult, lArg2);
	}

	Exponent::Exponent() :
		Primitive(1, "EXP")
	{ }

	void Exponent::execute(void* outDatum, Context& ioContext)
	{
		double& lResult = *(double*)outDatum;
		getArgument(0, &lResult, ioContext);
		lResult = std::exp(lResult);
	}

	//Cosine
	Cosine::Cosine() :
		Primitive(1, "COS")
	{ }

	void Cosine::execute(void* outDatum, Context& ioContext)
	{
		double& lResult = *(double*)outDatum;
		getArgument(0, &lResult, ioContext);
		lResult = std::cos(lResult);
	}

	//Cosine
	Sine::Sine() :
		Primitive(1, "SIN")
	{ }

	void Sine::execute(void* outDatum, Context& ioContext)
	{
		double& lResult = *(double*)outDatum;
		getArgument(0, &lResult, ioContext);
		lResult = std::sin(lResult);
	}


	/*!
	 *  \brief Construct ephemeral random constant generator primitive.
	 */
	Ephemeral::Ephemeral() :
	  Primitive(0, "E")
	{ }


	/*!
	 *  \brief Dummy function, ephemeral primitive is used only to generate constants.
	 *  \param outDatum Result of the Divide operation.
	 *  \param ioContext Evolutionary context.
	 */
	void Ephemeral::execute(void* outDatum, Context& ioContext)
	{ }

	/*!
	 *  \brief Generate random constant and return as primitive handle.
	 *  \param ioContext Evolutionary context.
	 */
	PrimitiveHandle Ephemeral::giveReference(Context& ioContext)
	{
	  double lValue = ioContext.mRandom.rollUniform(0.0, 1.0);
	  std::ostringstream lOSS;
	  lOSS << lValue;
	  return new TokenT<double>(lOSS.str(), lValue);
	}
}