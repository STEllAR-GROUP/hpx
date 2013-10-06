//  Copyright (c) 2011-2012 Bryce Adelstein-Lelbach
//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c)      2013 Thomas Heller
//  Copyright (c)      2013 Patricia Grubel
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_TESTS_PERFORMANCE_WORKER_HPP
#define HPX_TESTS_PERFORMANCE_WORKER_HPP

#include <hpx/config.hpp>
#include <hpx/util/high_resolution_timer.hpp>
#include <boost/cstdint.hpp>

using hpx::util::high_resolution_timer;

HPX_SYMBOL_EXPORT void worker_timed(double delay_sec, volatile int * i);

#endif
