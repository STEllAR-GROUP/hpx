//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2013-2015 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_ALPS_ENVIRONMENT_HPP)
#define HPX_UTIL_ALPS_ENVIRONMENT_HPP

#include <hpx/config.hpp>

#include <hpx/util/safe_lexical_cast.hpp>

#include <string>
#include <vector>

namespace hpx { namespace util { namespace batch_environments {

    struct alps_environment
    {
        alps_environment(std::vector<std::string> & nodelist, bool debug)
          : node_num_(0)
          , num_threads_(0)
          , num_localities_(0)
          , valid_(false)
        {
            char *node_num = std::getenv("ALPS_APP_PE");
            valid_ = node_num != 0;
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

        bool valid() const
        {
            return valid_;
        }

        std::size_t node_num() const
        {
            return node_num_;
        }

        std::size_t num_threads() const
        {
            return num_threads_;
        }

        std::size_t num_localities() const
        {
            return num_localities_;
        }

    private:
        std::size_t node_num_;
        std::size_t num_threads_;
        std::size_t num_localities_;
        bool valid_;
    };
}}}

#endif
