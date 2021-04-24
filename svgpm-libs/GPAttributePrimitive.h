#pragma once
#pragma once

#include "puppy/Puppy.hpp"

/*!
 * \class Add GPAttributePrimitive.h
 * \brief Add two doubles GP primitive
 * \ingroup GPAttributePrimitive
 */
namespace GPAttributePrimitive
{
	class Add : public Puppy::Primitive
	{
	public:
		Add();
		virtual ~Add() {}

		virtual void execute(void* outDatum, Puppy::Context& ioContext);
	};

	class Subtract : public Puppy::Primitive
	{
	public:
		Subtract();
		virtual ~Subtract(){}

		virtual void execute(void* outDatum, Puppy::Context& ioContext);
	};


	class Multiply : public Puppy::Primitive 
	{
	public:
		Multiply();
		virtual ~Multiply() { }

		virtual void execute(void* outDatum, Puppy::Context& ioContext);

	};

	/*!
	 *  \class Divide SymbRegPrimits.hpp "SymbRegPrimits.hpp"
	 *  \brief Protected division of two doubles GP primitive.
	 *  \ingroup SymbReg
	 */
	class Divide : public Puppy::Primitive 
	{

	public:
		Divide();
		virtual ~Divide() { }

		virtual void execute(void* outDatum, Puppy::Context& ioContext);

	};

	class ErrorFunction : public Puppy::Primitive
	{
	public:
		ErrorFunction();
		virtual ~ErrorFunction(){}

		virtual void execute(void* outDatum, Puppy::Context& ioContext);
	};


	class Ln : public Puppy::Primitive
	{
	public:
		Ln();
		virtual ~Ln(){}

		virtual void execute(void* outDatum, Puppy::Context& ioContext);
	};

	class Log : public Puppy::Primitive
	{
	public:
		Log();
		virtual ~Log(){}

		virtual void execute(void* outDatum, Puppy::Context& ioContext);
	};

	class SquareRoot : public Puppy::Primitive
	{
	public:
		SquareRoot();
		virtual ~SquareRoot(){}

		virtual void execute(void* outDatum, Puppy::Context& ioContext);
	};


	class Power : public Puppy::Primitive
	{

	public:
		Power();
		virtual ~Power(){}

		virtual void execute(void* outDatum, Puppy::Context& ioContext);
	};


	class Exponent : public Puppy::Primitive
	{

	public:
		Exponent();
		virtual ~Exponent(){}

		virtual void execute(void* outDatum, Puppy::Context& ioContext);
	};

	class Cosine : public Puppy::Primitive
	{

	public:
		Cosine();
		virtual ~Cosine(){}

		virtual void execute(void* outDatum, Puppy::Context& ioContext);
	};

	class Sine : public Puppy::Primitive
	{
	public:
		Sine();
		virtual ~Sine(){}
	
		virtual void execute(void* outDatum, Puppy::Context& ioContext);

	};

	/*!
	 *  \class Ephemeral SymbRegPrimits.hpp "SymbRegPrimits.hpp"
	 *  \brief Ephemeral random constant.
	 *  \ingroup SymbReg
	 */
	class Ephemeral : public Puppy::Primitive 
	{

	public:
		Ephemeral();
		virtual ~Ephemeral() { }
	
		virtual void execute(void* outDatum, Puppy::Context& ioContext);
		virtual Puppy::PrimitiveHandle giveReference(Puppy::Context& ioContext);
  
	};
}

