//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/async_distributed/applier/apply_continue_fwd.hpp>
#include <hpx/async_distributed/applier/detail/apply_implementations_fwd.hpp>
#include <hpx/functional/bind.hpp>
#include <hpx/functional/invoke_result.hpp>
#include <hpx/naming_base/id_type.hpp>
#include <hpx/serialization/access.hpp>

#include <type_traits>
#include <utility>

namespace hpx { namespace actions {
    ///////////////////////////////////////////////////////////////////////////
    template <typename Cont, typename F>
    struct continuation2_impl
    {
    private:
        typedef typename std::decay<Cont>::type cont_type;
        typedef typename std::decay<F>::type function_type;

    public:
        continuation2_impl() {}

        template <typename Cont_, typename F_>
        continuation2_impl(Cont_ && cont, hpx::naming::id_type const& target,
                F_ && f)
          : cont_(std::forward<Cont_>(cont)),
            target_(target),
            f_(std::forward<F_>(f))
        {}

        virtual ~continuation2_impl() {}

        template <typename T>
        typename util::invoke_result<function_type,
            hpx::naming::id_type,
            typename util::invoke_result<
                cont_type, hpx::naming::id_type, T
            >::type
        >::type operator()(hpx::naming::id_type const& lco, T && t) const
        {
            using hpx::util::placeholders::_2;
            hpx::apply_continue(cont_, hpx::util::bind(f_, lco, _2),
                target_, std::forward<T>(t));

            // Unfortunately we need to default construct the return value,
            // this possibly imposes an additional restriction of return types.
            typedef typename util::invoke_result<function_type,
                hpx::naming::id_type,
                typename util::invoke_result<
                    cont_type, hpx::naming::id_type, T
                >::type
            >::type result_type;
            return result_type();
        }

    private:
        // serialization support
        friend class hpx::serialization::access;

        template <typename Archive>
        HPX_FORCEINLINE void serialize(Archive& ar, unsigned int const)
        {
            ar & cont_ & target_ & f_;
        }

        cont_type cont_;        // continuation type
        hpx::naming::id_type target_;
        function_type f_;
        // set_value action  (default: set_lco_value_continuation)
    };
}}

