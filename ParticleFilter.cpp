#include "ParticleFilter.h"

void Multiply(int ma, int na,
			  int mb, int nb, double* A, double* B, double* C)
{	
	for(int i = 0; i < mb*nb; i++)
	{
		C[i] = 0.0;
	}

	for(int i = 0; i < ma; i++)
	{
		for(int j = 0; j < nb; j++)
		{
			for(int k = 0; k < na; k++)
			{
				C[i*ma + j] += A[i*ma + k]*B[k*nb + j];
			}
		}
	}
}

void PrintMatrix(int m, int n, double* A)
{	
	for(int i = 0; i < m; i++)
	{
		for(int j = 0; j < n; j++)
		{
			std::cout << " " << A[i*m + j];
		}
		std::cout << std::endl;
	}
}

ParticleFilter::ParticleFilter(int n, int width, int height)
	: m_rng(new RNGenerator()),
	m_n(n),
	Y_n(width),
	Y_m(height)
{
	X = new double[4*n];
	Z = new double[4*n];
	Y = new double[4*n];
	L = new double[n];
	R = new double[n];
	T = new double[n];
	I = new int[n];

	Xstd_rgb = 25.0;
	Xstd_pos = 25.0;
	Xstd_vec = 15.0;

	A = -log(sqrt(M2_PI) * Xstd_rgb);
	B = -0.5 / (Xstd_rgb * Xstd_rgb);

	createParticles();
}

ParticleFilter::~ParticleFilter(void)
{
	delete m_rng;
	delete Y;
	delete L;
	delete R;
	delete X;
	delete Z;
	delete T;
	delete I;
}

void ParticleFilter::createParticles()
{
	for(int p = 0; p < m_n; p++)
	{
		X[p		   ] = m_rng->getUniform(0, Y_n - 1);
		X[p +   m_n] = m_rng->getUniform(0, Y_m - 1);
		X[p + 2*m_n] = 0;
		X[p + 3*m_n] = 0;

		/*Z[p		   ] = 0;
		Z[p +   m_n] = 0;
		Z[p + 2*m_n] = 0;
		Z[p + 3*m_n] = 0;*/
	}
}

void ParticleFilter::updateParticles(double *F, double *X, int n)
{
	cv::Mat A = cv::Mat(4, 4, CV_64F, F);
	cv::Mat B = cv::Mat(4, n, CV_64F, X);

	cv::Mat C = A*B;

	double *x = C.ptr<double>(0);

	for(int p = 0; p < n; p++)
	{
		X[p		 ] = x[p	  ] + Xstd_pos*m_rng->getNormal();
		X[p + n  ] = x[p + n  ] + Xstd_pos*m_rng->getNormal();
		X[p + 2*n] = x[p + 2*n] + Xstd_vec*m_rng->getNormal();
		X[p + 3*n] = x[p + 3*n] + Xstd_vec*m_rng->getNormal();
	}
}

void ParticleFilter::calcLogLikelihood(double* X, double* Y, double *C, double* L)
{
	// Output Array
	int	k, j1, j2, j3, m, n;

	double dR, dG, dB, D2;
	
	for(k = 0; k < m_n; k++)
	{
		m = int(X[m_n + k]);
		n = int(X[k]);

		if(m >= 0 && n >= 0 && m < Y_m && n < Y_n)
		{
			j1 = (Y_n*m + n)*3;
			j2 = j1 + 1;
			j3 = j2 + 1;

			dR = Y[j3] - C[0];
			dG = Y[j2] - C[1];
			dB = Y[j1] - C[2];

			D2	= dR * dR + dG * dG + dB * dB;
			
			L[k] = A + B * D2;
		}
		else
		{
			L[k] = -DBL_MAX;
		}
	}
}

void ParticleFilter::resampleParticles(double* X, double* L_log)
{
	// Calculating Cumulative Distribution
	double max = -DBL_MAX;

	for(int i = 0; i < m_n; i++)
	{
		if(L_log[i] > max)
		{
			max = L_log[i];
		}
	}

	double sum = 0.0;

	for(int i = 0; i < m_n; i++)
	{
		L[i] = std::exp(L_log[i] - max);
		sum += L[i];
	}

	double cumsum = 0.0;

	// Generating Random Numbers
	for(int i = 0; i < m_n; i++)
	{
		L[i] /= sum;
		cumsum += L[i];
		R[i] = cumsum;
		T[i] = m_rng->getUniform01();
	}

	// Resampling
	double value = 0.0;

	for(int h = 0; h < m_n; h++)
	{
		value = T[h];
		/* Use a binary search */
		int nbins = m_n;
		int k0 = 0;
		int k1 = nbins - 1;
		int k = 0;
		
		if(value >= R[0] && value < R[nbins - 1])
		{
			k = (k0 + k1)/2;
			while(k0 < k1 - 1)
		    {
				if(value >= R[k])
				{
					k0 = k;
				}
				else
				{
					k1 = k;
				}
				k = (k0 + k1)/2;
			}
			k = k0;
		}
		/* Check for special case */
		if(value == R[nbins - 1])
		{
			k = nbins-1;
		}
		I[h] = k + 1;
	}

	for(int i = 0; i < m_n; i++)
	{
		Z[i		   ] = X[I[i]		 ];
		Z[i +	m_n] = X[I[i] +   m_n];
		Z[i + 2*m_n] = X[I[i] + 2*m_n];
		Z[i + 3*m_n] = X[I[i] + 3*m_n];
	}
	std::swap(Z, X);

	/*for(int i = 0; i < m_n; i++)
	{
		X[i		   ] = Z[i		   ];
		X[i +	m_n] = Z[i +   m_n];
		X[i + 2*m_n] = Z[i + 2*m_n];
		X[i + 3*m_n] = Z[i + 3*m_n];
	}*/
}

void ParticleFilter::run(double* Y)
{	
	double F[16] = {1, 0, 1, 0,
					0, 1, 0, 1,
					0, 0, 1, 0,
					0, 0, 0, 1};

	double C[3] = {0, 0, 255};

	// 1. Forecasting
	updateParticles(F, X, m_n);
	// 2. Calculating Likelihood
	calcLogLikelihood(X, Y, C, L);
	// 3. Resampling
	resampleParticles(X, L);
}