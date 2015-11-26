//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2013-2015 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_ALPS_ENVIRONMENT_HPP)
#define HPX_UTIL_ALPS_ENVIRONMENT_HPP

#include <hpx/config.hpp>

#include <string>
#include <vector>

namespace hpx { namespace util { namespace batch_environments {

    struct alps_environment
    {
        HPX_EXPORT alps_environment(std::vector<std::string> & nodelist,
            bool debug);

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
