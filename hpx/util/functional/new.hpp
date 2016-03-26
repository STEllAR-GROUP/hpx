//  Copyright (c) 2015 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_FUNCTIONAL_NEW_JAN_11_2015_0535PM)
#define HPX_UTIL_FUNCTIONAL_NEW_JAN_11_2015_0535PM

#include <utility>

namespace hpx { namespace util { namespace functional
{
    template <typename T>
    struct new_
    {
        template <typename ...Ts>
        T* operator()(Ts&&... vs) const
        {
            return new T(std::forward<Ts>(vs)...);
        }
    };

    template <typename T>
    struct placement_new
    {
        template <typename ...Ts>
        T* operator()(void* p, Ts&&... vs) const
        {
            return new (p) T(std::forward<Ts>(vs)...);
        }
    };
}}}

#endif
