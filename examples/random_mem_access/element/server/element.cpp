//  Copyright (c) 2011 Matt Anderson
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/include/iostreams.hpp>

#include "element.hpp"

#include <boost/format.hpp>

namespace random_mem_access { namespace server
{
    void element::init(std::size_t i)
    {
        char const* fmt
            = "Initializing element %1% on locality %2%\n";
        std::size_t const here = hpx::applier::get_prefix_id();

        hpx::cout << (boost::format(fmt) % i % here)
                  << hpx::flush; 

        mutex_type::scoped_lock l(mtx_);

        arg_ = i;
        arg_init_ = i;
    }

    void element::add()
    {
        mutex_type::scoped_lock l(mtx_);

        char const* fmt
            = "Incrementing element %1% on locality %2% from %3% to %4%\n";
        std::size_t const here = hpx::applier::get_prefix_id();

        hpx::cout << (boost::format(fmt) % arg_init_ % here % arg_ % (arg_ + 1))
                  << hpx::flush; 

        arg_ += 1;
    }

    void element::print()
    {
        mutex_type::scoped_lock l(mtx_);

        char const* fmt
            = "Element %1% on locality %2% has value %3%\n";
        std::size_t const here = hpx::applier::get_prefix_id();

        hpx::cout << (boost::format(fmt) % arg_init_ % here % arg_)
                  << hpx::flush; 
    }
}}

