//  Copyright (c) 2011-2012 Bryce Adelstein-Lelbach
//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c)      2013 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_TESTS_PERFORMANCE_WORKER_HPP
#define HPX_TESTS_PERFORMANCE_WORKER_HPP

#include <hpx/config.hpp>
#include <boost/cstdint.hpp>

HPX_SYMBOL_EXPORT void worker(boost::uint64_t delay, volatile double * d);

#endif
