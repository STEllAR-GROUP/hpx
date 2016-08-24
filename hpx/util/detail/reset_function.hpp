//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_DETAIL_RESET_FUNCTION_HPP
#define HPX_UTIL_DETAIL_RESET_FUNCTION_HPP

#include <hpx/util/function.hpp>
#include <hpx/util/unique_function.hpp>

namespace hpx { namespace util { namespace detail
{
    template <typename Sig, bool Serializable>
    inline void reset_function(hpx::util::function<Sig, Serializable>& f)
    {
        f.reset();
    }

    template <typename Sig>
    inline void reset_function(hpx::util::function_nonser<Sig>& f)
    {
        f.reset();
    }

    template <typename Sig, bool Serializable>
    inline void reset_function(hpx::util::unique_function<Sig, Serializable>& f)
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

#endif /*HPX_UTIL_DETAIL_RESET_FUNCTION_HPP*/
