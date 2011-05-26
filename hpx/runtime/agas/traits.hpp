////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_7D2054F6_DBA9_4D70_82FB_32D284A3CCB4)
#define HPX_7D2054F6_DBA9_4D70_82FB_32D284A3CCB4

#include <boost/mpl/integral_c.hpp>

#include <hpx/version.hpp>
#include <hpx/util/spinlock_pool.hpp>
#include <hpx/runtime/agas/traits_fwd.hpp>
#include <hpx/runtime/agas/network/traits.hpp>
#include <hpx/runtime/agas/database/traits.hpp>

namespace hpx { namespace agas {

struct empty { };

namespace traits {

///////////////////////////////////////////////////////////////////////////////
template <typename T, typename Enable>
struct serialization_version
  : boost::mpl::integral_c<unsigned, HPX_AGAS_VERSION> {};

///////////////////////////////////////////////////////////////////////////////
template <typename Mutex, typename Enable>
struct initialize_mutex_hook
{
    typedef void result_type;

    static void call(Mutex&) {}
};

template <> 
struct initialize_mutex_hook<boost::detail::spinlock>
{ 
    typedef void result_type;

    static void call(boost::detail::spinlock& m)
    {
        boost::detail::spinlock l = BOOST_DETAIL_SPINLOCK_INIT;
        m = l;
    }
};

template <typename Mutex>
inline void initialize_mutex (Mutex& m)
{ initialize_mutex_hook<Mutex>::call(m); }

}}}

#endif // HPX_7D2054F6_DBA9_4D70_82FB_32D284A3CCB4

