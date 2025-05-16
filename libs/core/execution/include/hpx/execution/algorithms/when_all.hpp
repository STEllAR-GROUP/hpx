//  Copyright (c) 2020 ETH Zurich
//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/async_base/dataflow.hpp>

#include <hpx/execution_base/stdexec_forward.hpp>

namespace hpx::execution::experimental {

    namespace hpxexec = hpx::execution::experimental;

    template <typename F, typename Sender, typename... Senders>
    constexpr HPX_FORCEINLINE auto tag_invoke(
        hpx::detail::dataflow_t, F&& f, Sender&& sender, Senders&&... senders)
        -> decltype(hpxexec::then(hpxexec::when_all(HPX_FORWARD(Sender, sender),
                                      HPX_FORWARD(Senders, senders)...),
            HPX_FORWARD(F, f)))
    {
        return hpxexec::then(hpxexec::when_all(HPX_FORWARD(Sender, sender),
                                 HPX_FORWARD(Senders, senders)...),
            HPX_FORWARD(F, f));
    }

    template <typename F, typename Sender, typename... Senders>
    constexpr HPX_FORCEINLINE auto tag_invoke(hpx::detail::dataflow_t,
        hpx::launch, F&& f, Sender&& sender, Senders&&... senders)
        -> decltype(hpxexec::then(hpxexec::when_all(HPX_FORWARD(Sender, sender),
                                      HPX_FORWARD(Senders, senders)...),
            HPX_FORWARD(F, f)))
    {
        return hpxexec::then(hpxexec::when_all(HPX_FORWARD(Sender, sender),
                                 HPX_FORWARD(Senders, senders)...),
            HPX_FORWARD(F, f));
    }
}    // namespace hpx::execution::experimental
