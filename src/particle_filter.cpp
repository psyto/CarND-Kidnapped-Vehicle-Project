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
	for (auto &particle : particles)
	{
		if (fabs(yaw_rate) < 0.00001)
		{
			particle.x += velocity * delta_t * cos(particle.theta);
			particle.y += velocity * delta_t * sin(particle.theta);
		}
		else
		{
			double y = velocity / yaw_rate;
			double new_theta = particle.theta + yaw_rate * delta_t;
			particle.x += y * ( sin(new_theta) - sin(particle.theta) );
			particle.y += y * ( cos(particle.theta) - cos(new_theta) );
			particle.theta = new_theta;
		}

		// Add random Gaussian noise.
		particle.x += dist_x(gen);
		particle.y += dist_y(gen);
		particle.theta += dist_theta(gen);

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
    for (auto &particle : particles) {
        vector<LandmarkObs> predictions;

        for (auto &lm : map_landmarks.landmark_list) {
            if (fabs(lm.x_f - particle.x) <= sensor_range && fabs(lm.y_f - particle.y) <= sensor_range) {
                predictions.push_back(LandmarkObs{lm.id_i, lm.x_f, lm.y_f});
            }
        }

        vector<LandmarkObs> transformed_os;
        for (auto &obs : observations) {
            double t_x = cos(particle.theta) * obs.x - sin(particle.theta) * obs.y + particle.x;
            double t_y = sin(particle.theta) * obs.x + cos(particle.theta) * obs.y + particle.y;
            transformed_os.push_back(LandmarkObs{obs.id, t_x, t_y});
        }

        dataAssociation(predictions, transformed_os);
        particle.weight = 1.0;
        for (auto &tos : transformed_os) {
            double o_x, o_y, pr_x, pr_y;
            o_x = tos.x;
            o_y = tos.y;
            int association_prediction = tos.id;
            for (auto &pred : predictions) {
                if (pred.id == association_prediction) {
                    pr_x = pred.x;
                    pr_y = pred.y;
                }
            }

            double s_x = std_landmark[0];
            double s_y = std_landmark[1];
            double obs_w = (1/(2 * M_PI*s_x*s_y)) * exp(-(pow(pr_x-o_x,2)/(2*pow(s_x, 2)) + (pow(pr_y-o_y,2)/(2*pow(s_y, 2)))));

            particle.weight *= obs_w;
        }
    }
}

void ParticleFilter::resample() {
	// Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    vector<Particle> new_particles;

		// std::default_random_engine
		default_random_engine gen;

    vector<double> weights;
    for (auto &particle : particles) {
        weights.push_back(particle.weight);
    }

    uniform_int_distribution<int> uniintdist(0, num_particles-1);
    auto index = uniintdist(gen);
    double max_weight = *max_element(weights.begin(), weights.end());
    uniform_real_distribution<double> unirealdist(0.0, max_weight);
    double beta = 0.0;

    for (int i=0; i < num_particles; i++) {
        beta += unirealdist(gen) * 2.0;
        while (beta > weights[index]) {
            beta -= weights[index];
            index = (index + 1) % num_particles;
        }
        new_particles.push_back(particles[index]);
    }
    particles = new_particles;
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
