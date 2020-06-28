#pragma once
#include <boost/random.hpp>

class RNGenerator
{
public:
	RNGenerator(void);
	~RNGenerator(void);

	int getUniform(int minVal = 1, int maxVal = 6);
	double getUniform01(void);
	double getNormal(double mean = 0.0, double std = 1.0);

private:

	boost::mt19937 m_rng;
};
