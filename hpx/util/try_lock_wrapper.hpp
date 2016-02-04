////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2007-2015 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_5CE1FE44_73E2_43D6_8BC9_E5FC27251C33)
#define HPX_5CE1FE44_73E2_43D6_8BC9_E5FC27251C33

#include <boost/thread/locks.hpp>

namespace hpx { namespace util
{

///////////////////////////////////////////////////////////////////////
// This try_lock_wrapper is essentially equivalent to the template
// boost::thread::detail::try_lock_wrapper with the one exception, that
// the lock() function always calls base::try_lock(). This allows us to
// skip lock acquisition while exiting the condition variable.
template <typename Mutex>
struct try_lock_wrapper : public boost::detail::try_lock_wrapper<Mutex>
{
    typedef boost::detail::try_lock_wrapper<Mutex> base;

    explicit try_lock_wrapper(Mutex& m_) : base(m_, boost::try_to_lock) {}

    void lock()
    { base::try_lock(); } // this is different
};

}}

#endif // HPX_5CE1FE44_73E2_43D6_8BC9_E5FC27251C33

