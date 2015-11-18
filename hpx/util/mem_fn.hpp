//  Copyright (c) 2013 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_MEM_FN_HPP
#define HPX_UTIL_MEM_FN_HPP

#include <hpx/config.hpp>
#include <hpx/util/invoke.hpp>
#include <hpx/util/move.hpp>
#include <hpx/util/result_of.hpp>

#include <boost/type_traits/is_member_pointer.hpp>

#include <utility>

namespace hpx { namespace util
{
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        template <typename MemPtr>
        struct mem_fn
        {
            explicit mem_fn(MemPtr mem_ptr)
              : f(mem_ptr)
            {}

            mem_fn(mem_fn const& other)
              : f(other.f)
            {}

            mem_fn& operator=(mem_fn const& other)
            {
                f = other.f;
                return *this;
            }

            template <typename>
            struct result;

            template <typename This, typename ...Ts>
            struct result<This(Ts...)>
              : util::result_of<MemPtr(Ts...)>
            {};

            template <typename T, typename ...Ts>
            BOOST_FORCEINLINE
            typename result<mem_fn const(T, Ts...)>::type
            operator()(T&& t, Ts&&... vs) const
            {
                return util::invoke(
                    f, std::forward<T>(t), std::forward<Ts>(vs)...);
            }

            MemPtr f;
        };
    }

    template <typename MemPtr>
    detail::mem_fn<MemPtr> mem_fn(MemPtr pm)
    {
        static_assert(
            boost::is_member_pointer<MemPtr>::value,
            "boost::is_member_pointer<MemPtr>::value");

        return detail::mem_fn<MemPtr>(pm);
    }
}}

#endif
