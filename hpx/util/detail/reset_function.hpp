//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_DETAIL_RESET_FUNCTION_OCT_22_2013_0854AM)
#define HPX_UTIL_DETAIL_RESET_FUNCTION_OCT_22_2013_0854AM

#include <hpx/util/detail/function_template.hpp>
#include <hpx/util/detail/unique_function_template.hpp>

namespace hpx { namespace util { namespace detail
{
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

    template <typename Sig, typename IArchive, typename OArchive>
    inline void reset_function(hpx::util::unique_function<Sig, IArchive, OArchive>& f)
    {
        f.reset();
    }

    template <typename Sig>
    inline void reset_function(hpx::util::unique_function_nonser<Sig>& f)
    {
        f.reset();
    }

    template <typename Function>
    inline void reset_function(Function& f)
    {
        f = Function();
    }
}}}

#endif


