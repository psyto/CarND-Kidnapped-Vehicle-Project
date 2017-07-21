/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	//	struct Particle {
	//
	//		int id;
	//		double x;
	//		double y;
	//		double theta;
	//		double weight;
	//		std::vector<int> associations;
	//		std::vector<double> sense_x;
	//		std::vector<double> sense_y;
	//	};

	// Set the number of particles.
	// Number of particles to draw
	// int num_particles;
	num_particles = 100;
	particles.resize(num_particles);
	weights.resize(num_particles);

	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	// std::default_random_engine
	default_random_engine gen;

	// Initialize all particles to first position (based on estimates of x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// Vector of weights of all particles
	// std::vector<double> weights;
	for (int i=0; i<num_particles; i++)
	{
		Particle p;
		p.id = i;
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		p.weight = 1.0;
		particles.push_back(p);
		weights.push_back(p.weight);
	}

	// Flag, if filter is initialized
	// bool is_initialized;
	is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	// std::normal_distribution
	normal_distribution<double> dist_x(0, std_pos[0]);
	normal_distribution<double> dist_y(0, std_pos[1]);
	normal_distribution<double> dist_theta(0, std_pos[2]);

	// std::default_random_engine
	default_random_engine gen;

	// Add measurements to each particle.
	for (auto &p : particles)
	{
		if (fabs(yaw_rate) < 0.00001)
		{
			p.x += velocity * delta_t * cos(p.theta);
			p.y += velocity * delta_t * sin(p.theta);
		}
		else
		{
			double y = velocity / yaw_rate;
			double new_theta = p.theta + yaw_rate * delta_t;
			p.x += y * ( sin(new_theta) - sin(p.theta) );
			p.y += y * ( cos(p.theta) - cos(new_theta) );
			p.theta = new_theta;
		}

		// Add random Gaussian noise.
		p.x += dist_x(gen);
		p.y += dist_y(gen);
		p.theta += dist_theta(gen);

	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.

    for (auto &obs : observations) {
        double min_dist = numeric_limits<double>::max();
        int map_id = -1;

        for (auto &pred : predicted) {
            double cur_dist = dist(obs.x, obs.y, pred.x, pred.y);
            if (cur_dist < min_dist) {
                min_dist = cur_dist;
                map_id = pred.id;
            }
        }
        obs.id = map_id;
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
		weights.clear();

    for (auto &p : particles) {
        vector<LandmarkObs> predictions;

        for (auto &lm : map_landmarks.landmark_list) {
            if (fabs(lm.x_f - p.x) <= sensor_range && fabs(lm.y_f - p.y) <= sensor_range) {
                predictions.push_back(LandmarkObs{lm.id_i, lm.x_f, lm.y_f});
            }
        }

        vector<LandmarkObs> transformed_os;
        for (auto &obs : observations) {
            double t_x = cos(p.theta) * obs.x - sin(p.theta) * obs.y + p.x;
            double t_y = sin(p.theta) * obs.x + cos(p.theta) * obs.y + p.y;
            transformed_os.push_back(LandmarkObs{obs.id, t_x, t_y});
        }

        dataAssociation(predictions, transformed_os);
        p.weight = 1.0;
        for (auto &tos : transformed_os) {
            double pr_x, pr_y;
            int association_prediction = tos.id;
            for (auto &pred : predictions) {
                if (pred.id == association_prediction) {
                    pr_x = pred.x;
                    pr_y = pred.y;
                }
            }

            double std_x = std_landmark[0];
            double std_y = std_landmark[1];
						double num = exp(-0.5 * (pow(pr_x-tos.x,2)/(pow(std_x,2)) + (pow(pr_y-tos.y,2)/(pow(std_y,2)))));
						double denom = 2 * M_PI * std_x * std_y;
						double wt = num / denom;

            p.weight *= wt;
        }
				weights.push_back(p.weight);
    }
}

void ParticleFilter::resample() {
	// Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

		// Resample particles with replacement with probability proportional to their weight.
		// std::discrete_distribution
		default_random_engine gen;
		discrete_distribution<int> discrete_dist(weights.begin(), weights.end());
		vector<Particle> resampled_particles;

		for (int i=0; i<num_particles; i++)
		{
			auto new_id = discrete_dist(gen);
			resampled_particles.push_back(particles[new_id]);
		}
		particles = resampled_particles;

}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
