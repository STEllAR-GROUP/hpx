////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2016 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#ifndef HPX_UTIL_SCOPED_TIMER_HPP
#define HPX_UTIL_SCOPED_TIMER_HPP

#include <hpx/util/high_resolution_clock.hpp>

namespace hpx { namespace util
{
    template <typename T>
    struct scoped_timer
    {
        scoped_timer(T& t)
          : started_at_(hpx::util::high_resolution_clock::now())
          , t_(t)
        {}

        ~scoped_timer()
        {
            t_ += (hpx::util::high_resolution_clock::now() - started_at_);
        }

        boost::uint64_t started_at_;
        T& t_;
    };
}}

#endif
