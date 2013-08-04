//  Copyright (c) 2007-2008 Chirag Dekate, Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_SCOPED_UNLOCK_JUN_17_2008_1131AM)
#define HPX_UTIL_SCOPED_UNLOCK_JUN_17_2008_1131AM

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    // This is a helper structure to make sure a lock gets unlocked and locked
    // again in a scope.
    template <typename Lock>
    struct scoped_unlock
    {
        scoped_unlock(Lock& l) : l_(l)
        {
            l_.unlock();
        }
        ~scoped_unlock()
        {
            l_.lock();
        }

        Lock& l_;
    };

///////////////////////////////////////////////////////////////////////////////
}}

#endif
