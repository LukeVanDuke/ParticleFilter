#include <iostream>
#include <opencv2/opencv.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include "ParticleFilter.h"

int main(void)
{
	cv::Mat Y, y;

	cv::VideoCapture vid(CV_CAP_ANY);

	int width = vid.get(CV_CAP_PROP_FRAME_WIDTH);
	int height = vid.get(CV_CAP_PROP_FRAME_HEIGHT);

	int nParticles = 5000;

	ParticleFilter *pf = new ParticleFilter(nParticles, width, height);

	cv::Mat labels, normlabelsMat, colormap;
	for(int i = 0; i < nParticles; i++)
	{
		labels.push_back(i);
	}
	cv::normalize(labels, normlabelsMat, 0, 255, cv::NORM_MINMAX, CV_8U);
	cv::applyColorMap(normlabelsMat, colormap, cv::COLORMAP_JET);

	for(;;)
	{
		// Get image
		vid >> Y;

		Y.convertTo(y, CV_64FC3);
		//cv::normalize(y, y, 0, 1, cv::NORM_MINMAX);

		if(!Y.isContinuous())
		{
			break;
		}
		pf->run(y.ptr<double>(0));	
		
		// Showing Image
		
		int x_m = 0;
		int y_m = 0;

		for(int i = 0; i < nParticles; i++)
		{
			x_m += int(pf->X[i]);
			y_m += int(pf->X[i + nParticles]);
		}
		x_m /= nParticles;
		y_m /= nParticles;
		
		cv::Mat image = cv::Mat::zeros(Y.size(), CV_8UC3);

		for(int i = 0; i < nParticles; i++)
		{
			cv::circle(image, cv::Point(int(pf->X[i]), int(pf->X[i + nParticles])), 1, cv::Scalar(255, 255, 255), -1);
		}
		cv::circle(image, cv::Point(x_m, y_m), 10, cv::Scalar(0, 0, 255), -1);

		std::stringstream ssParticles;
		ssParticles <<  "#Particles: " << nParticles;
		cv::putText(Y, ssParticles.str(), cv::Point(10, 20), CV_FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(255, 255, 255), 1, CV_AA);
		std::stringstream ssPosition;
		ssPosition << "Position: (" << x_m << ", " << y_m << ")";
		cv::putText(Y, ssPosition.str(), cv::Point(10, 40), CV_FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(255, 255, 255), 1, CV_AA);

		cv::imshow("Image", image);
		cv::imshow("Particle Filter", Y);

		if(cv::waitKey(1) == 27)
		{
			break;
		}
	}
	delete pf;

	return 0;
}