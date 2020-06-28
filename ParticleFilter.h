#pragma once
#include "RNGenerator.h"
#include <algorithm>
#include <opencv2/opencv.hpp>

#define M_PI 3.141592653589793
#define M2_PI 6.283185307179586

class ParticleFilter
{
public:
	ParticleFilter(int n, int width, int height);
	~ParticleFilter(void);

	void createParticles();
	void updateParticles(double *F, double *X, int n);
	void calcLogLikelihood(double* X, double* Y, double* C, double* L);
	void resampleParticles(double* X, double* L_log);
	void run(double* Y);

	int m_n;
	int Y_m;
	int Y_n;

	double A;
	double B;
	double Xstd_rgb;
	double Xstd_pos;
	double Xstd_vec;

	double *X;
	double *Z;
	double* Y;
	double* L; 
	double* R;
	double* T;
	int* I;

	RNGenerator *m_rng;
};
