//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/async_distributed/applier/apply.hpp>
#include <hpx/async_distributed/applier/detail/apply_implementations_fwd.hpp>
#include <hpx/functional/invoke_result.hpp>
#include <hpx/naming_base/id_type.hpp>
#include <hpx/serialization/access.hpp>

#include <type_traits>
#include <utility>

namespace hpx { namespace actions {
    ///////////////////////////////////////////////////////////////////////////
    template <typename Cont>
    struct continuation_impl
    {
    private:
        typedef typename std::decay<Cont>::type cont_type;

    public:
        continuation_impl() {}

        template <typename Cont_>
        continuation_impl(Cont_ && cont, hpx::naming::id_type const& target)
          : cont_(std::forward<Cont_>(cont)), target_(target)
        {}

        virtual ~continuation_impl() {}

        template <typename T>
        typename util::invoke_result<cont_type, hpx::naming::id_type, T>::type
        operator()(hpx::naming::id_type const& lco, T && t) const
        {
            hpx::apply_c(cont_, lco, target_, std::forward<T>(t));

            // Unfortunately we need to default construct the return value,
            // this possibly imposes an additional restriction of return types.
            typedef typename util::invoke_result<
                cont_type, hpx::naming::id_type, T
            >::type result_type;
            return result_type();
        }

    private:
        // serialization support
        friend class hpx::serialization::access;

        template <typename Archive>
        HPX_FORCEINLINE void serialize(Archive& ar, unsigned int const)
        {
            ar & cont_ & target_;
        }

        cont_type cont_;
        hpx::naming::id_type target_;
    };
}}

