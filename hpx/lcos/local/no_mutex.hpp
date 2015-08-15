//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_LOCAL_NO_MUTEX_SEP_14_2012_0114PM)
#define HPX_LCOS_LOCAL_NO_MUTEX_SEP_14_2012_0114PM

#include <hpx/config.hpp>

#include <boost/thread/locks.hpp>

namespace hpx { namespace lcos { namespace local
{
    struct no_mutex
    {
        typedef boost::unique_lock<no_mutex> scoped_lock;
        typedef boost::detail::try_lock_wrapper<no_mutex> scoped_try_lock;

        void lock() {}
        void unlock() {}
    };
}}}

#endif
