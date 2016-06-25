//  Copyright (c) 2007-2015 Hartmut Kaiser
//  Copyright (c) 2013-2015 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/util/batch_environments/alps_environment.hpp>
#include <hpx/util/safe_lexical_cast.hpp>

#include <string>
#include <vector>

namespace hpx { namespace util { namespace batch_environments
{
    alps_environment::alps_environment(std::vector<std::string> & nodelist,
            bool debug)
      : node_num_(0)
      , num_threads_(0)
      , num_localities_(0)
      , valid_(false)
    {
        char *node_num = std::getenv("ALPS_APP_PE");
        valid_ = node_num != nullptr;
        if(valid_)
        {
            // Initialize our node number
            node_num_ = safe_lexical_cast<std::size_t>(node_num);

            // Get the number of threads
            char *num_threads = std::getenv("ALPS_APP_DEPTH");
            if(!num_threads)
            {
                valid_ = false;
                return;
            }
            num_threads_ = safe_lexical_cast<std::size_t>(num_threads);

            // Get the number of localities
            char *total_num_threads = std::getenv("PBS_NP");
            if(!total_num_threads)
            {
                valid_ = false;
                return;
            }
            num_localities_
                = safe_lexical_cast<std::size_t>(total_num_threads) / num_threads_;
        }
    }
}}}
