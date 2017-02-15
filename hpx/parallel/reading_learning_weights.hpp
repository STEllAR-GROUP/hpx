//  Copyright (c) 2017 Zahra Khatami and Lukas Troska
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/reading_learning_weights.hpp
#include <fstream>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>

#if !defined(HPX_PARALLEL_READING_LEARNING_WEIGHTS_FEB_8_2017_0300PM)
#define HPX_PARALLEL_READING_LEARNING_WEIGHTS_FEB_8_2017_0300PM

namespace hpx { namespace parallel {

    // weights values
    static std::vector<double> weights_seq_par;
    static std::vector<std::vector<double>> weights_chunk_size;
    static std::vector<std::vector<double>> weights_prefetching_distance;    

    void reading_weights_seq_par() {
        assert(weights_seq_par.size() == 0 && "ERROR : weights information has been already stored on the disk.\n");     
                
        // instructions for users
        std::cout<<"Please include normalization parameters (variance and average) in the first line. \n";
        std::ifstream infile("/home/zahra/Desktop/runtime_opt_with_compiler_and_ML/Learning_Alg/learning_weights/weights_seq_par.dat");

        // first line includes (variance and average) of each features
        std::string line_normalization;
        getline(infile, line_normalization);

        // normalization parameters (variance and average)
        std::vector<double> normalization_params_seq_par;
            
        std::istringstream normalization_buffer(line_normalization);
        std::transform(std::istream_iterator<std::string>(normalization_buffer), 
                    std::istream_iterator<std::string>(),
                    std::back_inserter(normalization_params_seq_par),
                    [](std::string s) {return std::stod(s);});

        std::string line; 
        getline( infile, line );
        std::istringstream buffer(line);
        std::size_t t = 0;

        std::transform(std::istream_iterator<std::string>(buffer), 
                    std::istream_iterator<std::string>(),
                    std::back_inserter(weights_seq_par),
                    [&](std::string s) {
                        double w = double((std::stod(s) - normalization_params_seq_par[t * 2])/normalization_params_seq_par[2 * t + 1]);
                        t++;
                        return w;
                    });
    }

    void reading_weights_chunk_size() {
        assert(weights_chunk_size.size() == 0 && "ERROR : weights information has been already stored on the disk.\n");     
                
        // instructions for users
        std::cout<<"Please include normalization parameters (variance and average) in the first line. \n";
        std::ifstream infile("/home/zahra/Desktop/runtime_opt_with_compiler_and_ML/Learning_Alg/learning_weights/weights_chunk_size.dat");

        // first line includes (variance and average) of each features
        std::string line_normalization;
        getline(infile, line_normalization);

        // normalization parameters (variance and average)
        std::vector<double> normalization_params_chunk_size;
            
        std::istringstream normalization_buffer(line_normalization);
        std::transform(std::istream_iterator<std::string>(normalization_buffer), 
                    std::istream_iterator<std::string>(),
                    std::back_inserter(normalization_params_chunk_size),
                    [](std::string s) {return std::stod(s);});

        // the rest of the lines include the values of weights
        for(std::string line; getline( infile, line ); )
        {
            std::vector<double> weights_row;
            std::istringstream buffer(line);
            std::size_t t = 0;

            std::transform(std::istream_iterator<std::string>(buffer), 
                    std::istream_iterator<std::string>(),
                    std::back_inserter(weights_row),
                    [&](std::string s) {
                        double w = double((std::stod(s) - normalization_params_chunk_size[t * 2])/normalization_params_chunk_size[2 * t + 1]);
                        t++;
                        return w;
                    });
            weights_chunk_size.push_back(weights_row);
        }        
    }

    void reading_weights_prefetching_distance() {
        assert(weights_prefetching_distance.size() == 0 && "ERROR : weights information has been already stored on the disk.\n");     
                
        // instructions for users
        std::cout<<"Please include normalization parameters (variance and average) in the first line. \n";
        std::ifstream infile("/home/zahra/Desktop/runtime_opt_with_compiler_and_ML/Learning_Alg/learning_weights/weights_prefetcher_distance_factor.dat");

        // first line includes (variance and average) of each features
        std::string line_normalization;
        getline(infile, line_normalization);

        // normalization parameters (variance and average)
        std::vector<double> normalization_params_prefetching_distance;
            
        std::istringstream normalization_buffer(line_normalization);
        std::transform(std::istream_iterator<std::string>(normalization_buffer), 
                    std::istream_iterator<std::string>(),
                    std::back_inserter(normalization_params_prefetching_distance),
                    [](std::string s) {return std::stod(s);});

        // the rest of the lines include the values of weights
        for(std::string line; getline( infile, line ); )
        {
            std::vector<double> weights_row;
            std::istringstream buffer(line);
            std::size_t t = 0;

            std::transform(std::istream_iterator<std::string>(buffer), 
                    std::istream_iterator<std::string>(),
                    std::back_inserter(weights_row),
                    [&](std::string s) {
                        double w = double((std::stod(s) - normalization_params_prefetching_distance[t * 2])/normalization_params_prefetching_distance[2 * t + 1]);
                        t++;
                        return w;
                    });
            weights_prefetching_distance.push_back(weights_row);
        }
    }

    std::vector<double> retreiving_weights_seq_par() {
        if(weights_seq_par.size() == 0) {
            reading_weights_seq_par();
        }

        return weights_seq_par;
    }

    std::vector<std::vector<double>> retreiving_weights_chunk_size() {
        if(weights_chunk_size.size() == 0) {
            reading_weights_chunk_size();
        }

        return weights_chunk_size;
    }

    std::vector<std::vector<double>> retreiving_weights_prefetching_distance() {
        if(weights_prefetching_distance.size() == 0) {
            reading_weights_prefetching_distance();
        }

        return weights_prefetching_distance;
    }
}}
#endif
