#include "RNGenerator.h"
#include <ctime> 

RNGenerator::RNGenerator(void)
	: m_rng(time(0))
{
}

RNGenerator::~RNGenerator(void)
{
}

int RNGenerator::getUniform(int minVal, int maxVal)
{
	boost::random::uniform_int_distribution<int> uni(minVal, maxVal);

	return uni(m_rng); 
}

double RNGenerator::getUniform01(void)
{
	boost::random::uniform_01<double> uni;

	return uni(m_rng);
}

double RNGenerator::getNormal(double mean, double std)
{
	boost::normal_distribution<double> nd(mean, std);

	return nd(m_rng);
}
