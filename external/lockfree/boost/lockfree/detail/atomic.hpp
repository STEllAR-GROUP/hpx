//  Copyright (C) 2007, 2008 Tim Blechmann
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_LOCKFREE_DETAIL_ATOMIC_HPP
#define BOOST_LOCKFREE_DETAIL_ATOMIC_HPP

#if !defined(BOOST_NO_0X_HDR_ATOMIC)
#ifdef __GNUC__
# if __GNUC__ < 4 || (__GNUC__ == 4 && __GNUC_MINOR__ < 6)
  || !defined(__GXX_EXPERIMENTAL_CXX0X__)
#  define BOOST_NO_0X_HDR_ATOMIC
# endif
#else
# define BOOST_NO_0X_HDR_ATOMIC
#endif
#endif

#ifdef BOOST_NO_0X_HDR_ATOMIC
#include <boost/atomic.hpp>
#else
#include <atomic>
#endif

namespace boost {
namespace lockfree {
namespace detail {

#ifdef BOOST_NO_0X_HDR_ATOMIC
using boost::atomic;
using boost::memory_order_acquire;
using boost::memory_order_consume;
using boost::memory_order_relaxed;
using boost::memory_order_release;
#else
using std::atomic;
using std::memory_order_acquire;
using std::memory_order_consume;
using std::memory_order_relaxed;
using std::memory_order_release;
#endif

}
using detail::atomic;
using detail::memory_order_acquire;
using detail::memory_order_consume;
using detail::memory_order_relaxed;
using detail::memory_order_release;

}}

#endif /* BOOST_LOCKFREE_DETAIL_ATOMIC_HPP */
