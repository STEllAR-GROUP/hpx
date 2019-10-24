//  Copyright (c) 2019 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/lcos/future.hpp>
#include <hpx/runtime/serialization/detail/preprocess_futures.hpp>

namespace hpx { namespace lcos { namespace detail {

    void preprocess_future(serialization::output_archive& ar,
        hpx::lcos::detail::future_data_refcnt_base& state)
    {
        auto& handle_futures =
            ar.get_extra_data<serialization::detail::preprocess_futures>();

        handle_futures.await_future(state);
    }
}}}    // namespace hpx::lcos::detail

