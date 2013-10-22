//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_DETAIL_RESET_FUNCTION_OCT_22_2013_0854AM)
#define HPX_UTIL_DETAIL_RESET_FUNCTION_OCT_22_2013_0854AM

#include <hpx/config/function.hpp>
#include <boost/move/move.hpp>

namespace hpx { namespace util { namespace detail
{
#if defined(HPX_UTIL_FUNCTION)
    template <typename Sig, typename IArchive, typename OArchive>
    inline void reset_function(hpx::util::function<Sig, IArchive, OArchive>& f)
    {
        f.reset();
    }

    template <typename Sig>
    inline void reset_function(hpx::util::function_nonser<Sig>& f)
    {
        f.reset();
    }
#elif !defined(HPX_HAVE_CXX11_STD_FUNCTION)
    template <typename Sig>
    inline void reset_function(boost::function<Sig>& f)
    {
        f = boost::move(boost::function<Sig>());
    }
#else
    template <typename Sig>
    inline void reset_function(std::function<Sig>& f)
    {
        f = boost::move(std::function<Sig>());
    }
#endif
}}}

#endif


