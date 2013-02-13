//  Copyright (c) 2011-2012 Bryce Adelstein-Lelbach
//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c)      2013 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "worker.hpp"

HPX_SYMBOL_EXPORT void worker(boost::uint64_t delay, volatile double * d)
{
    for (boost::uint64_t i = 0; i < delay; ++i)
        *d += 1. / (2. * i + 1.);
}

///////////////////////////////////////////////////////////////////////////////
// avoid to have one single volatile variable to become a contention point
HPX_SYMBOL_EXPORT double invoke_worker(boost::uint64_t delay)
{
    volatile double d = 0;
    worker(delay, &d);
    return d;
}

