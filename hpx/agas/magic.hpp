////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_7D2054F6_DBA9_4D70_82FB_32D284A3CCB4)
#define HPX_7D2054F6_DBA9_4D70_82FB_32D284A3CCB4

#include <boost/optional.hpp>

#include <hpx/lcos/mutex.hpp>
#include <hpx/util/spinlock_pool.hpp>
#include <hpx/agas/magic_fwd.hpp>

namespace hpx { namespace agas { namespace magic
{

template <typename Tag, typename Enable> 
struct mutex_type
{ typedef typename hpx::lcos::mutex type; };


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

template <typename Tag, typename Enable> 
struct key_type
{ typedef typename registry_type<Tag>::key_type type; };

template <typename Tag, typename Enable> 
struct mapped_type
{ typedef typename registry_type<Tag>::mapped_type type; };

// TODO: implement default definitions for bind, unbind and resolve

}}}

#endif // HPX_7D2054F6_DBA9_4D70_82FB_32D284A3CCB4

