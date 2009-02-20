//  Copyright (c) 2007-2009 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(BOOST_LOCKFREE_PRIMITIVES_JUL_11_2008_0411PM)
#define BOOST_LOCKFREE_PRIMITIVES_JUL_11_2008_0411PM

#include <boost/config.hpp>

#include <boost/detail/lightweight_mutex.hpp>
#include <boost/lockfree/detail/static_lw_mutex.hpp>

namespace boost { namespace lockfree { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    struct lw_CAS_mutex_tag {};
    inline boost::detail::lightweight_mutex& get_CAS_mutex()
    {
        detail::static_<boost::detail::lightweight_mutex, lw_CAS_mutex_tag> mtx;
        return mtx;
    }

    ///////////////////////////////////////////////////////////////////////////
    struct lw_ilce_mutex_tag {};
    inline boost::detail::lightweight_mutex& get_ilce_mutex()
    {
        detail::static_<boost::detail::lightweight_mutex, lw_ilce_mutex_tag> mtx;
        return mtx;
    }

    ///////////////////////////////////////////////////////////////////////////
    struct lw_CAS2_mutex_tag {};
    inline boost::detail::lightweight_mutex& get_CAS2_mutex()
    {
        detail::static_<boost::detail::lightweight_mutex, lw_CAS2_mutex_tag> mtx;
        return mtx;
    }

}}}

#define BOOST_LOCKFREE_CACHELINE_BYTES 64

#if defined(USE_ATOMIC_OPS)
#include <boost/lockfree/detail/ao_primitives.hpp>
#elif defined(__APPLE__)
#include <boost/lockfree/detail/apple_primitives.hpp>
#elif defined(__GNUC__)
#include <boost/lockfree/detail/gcc_primitives.hpp>
#elif defined(BOOST_MSVC) || defined(BOOST_INTEL_WIN)
#include <boost/lockfree/detail/windows_primitives.hpp>
#endif

#endif
