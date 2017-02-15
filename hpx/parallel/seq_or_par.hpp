//  Copyright (c) 2017 Lukas Troska and Zahra Khatami
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/seq_or_par.hpp
#include "reading_learning_weights.hpp"

#if !defined(HPX_PARALLEL_SEQ_OR_PAR_JAN_30_2017_0300PM)
#define HPX_PARALLEL_SEQ_OR_PAR_JAN_30_2017_0300PM

namespace hpx { namespace parallel {
    
    bool seq_or_par(std::vector<std::size_t>&& features)
    {   
        auto && weights = retreiving_weights_seq_par();
        assert(weights.size() > 0 && "ERROR : File is not readable or it is not is the defined format.\n");   

        features[0] = hpx::get_os_thread_count();
    
        double result = weights[0];

        auto w = weights.begin() + 1;
        auto f = features.begin();

        for (;f != features.end(); ++f, ++w)
            result += *w * *f;
    
        return result < 0;
    }
}}

#endif
