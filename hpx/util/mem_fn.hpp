//  Copyright (c) 2013-2015 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_MEM_FN_HPP
#define HPX_UTIL_MEM_FN_HPP

#include <hpx/config.hpp>
#include <hpx/util/invoke.hpp>
#include <hpx/util/result_of.hpp>

#include <utility>

namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        template <typename MemberPointer>
        struct mem_fn
        {
            explicit mem_fn(MemberPointer pm)
              : _pm(pm)
            {}

            mem_fn(mem_fn const& other)
              : _pm(other._pm)
            {}

            template <typename ...Ts>
            inline typename util::invoke_result<MemberPointer, Ts...>::type
            operator()(Ts&&... vs) const
            {
                return util::invoke(_pm, std::forward<Ts>(vs)...);
            }

            MemberPointer _pm;
        };
    }

    template <typename M, typename C>
    inline detail::mem_fn<M C::*>
    mem_fn(M C::*pm)
    {
        return detail::mem_fn<M C::*>(pm);
    }

    template <typename R, typename C, typename ...Ps>
    inline detail::mem_fn<R (C::*)(Ps...)>
    mem_fn(R (C::*pm)(Ps...))
    {
        return detail::mem_fn<R (C::*)(Ps...)>(pm);
    }

    template <typename R, typename C, typename ...Ps>
    inline detail::mem_fn<R (C::*)(Ps...) const>
    mem_fn(R (C::*pm)(Ps...) const)
    {
        return detail::mem_fn<R (C::*)(Ps...) const>(pm);
    }
}}

#endif
