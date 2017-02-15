//  Copyright (c) 2017 Zahra Khatami and Lukas Troska
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/prefetching_distance_determination.hpp
#include <iostream>
#include <cmath>
#include "reading_learning_weights.hpp"

#if !defined(HPX_PARALLEL_PREFETCHING_DISTANCE_DETERMINATION_FEB_1_2017_0300PM)
#define HPX_PARALLEL_PREFETCHING_DISTANCE_DETERMINATION_FEB_1_2017_0300PM

namespace hpx { namespace parallel {
    
	std::size_t prefetching_distance_determination(std::vector<std::size_t>&& features)
	{   
    	auto && weights = retreiving_weights_prefetching_distance();    
    	assert(weights.size() > 0 && "ERROR : File is not readable or it is not is the defined format.\n");

    	features[0] = hpx::get_os_thread_count();
    
    	//initial class = 0
    	std::size_t determined_class = 0;
    	double max = 0.0;

    	//max of (wi * f)
    	for(std::size_t w = 0; w < weights.size(); w++) {
    		double sum = 0.0;
    		for(std::size_t f = 0; f < features.size(); f++) {
    			sum += weights[w][f] * features[f];
    		}
    		if(max < sum) {
    			max = sum;
    			determined_class = w;
    		}
    	}

        std::size_t dist = std::pow(10, determined_class); //* something!

    	//return dist;
        return dist;
	}
}}

#endif
