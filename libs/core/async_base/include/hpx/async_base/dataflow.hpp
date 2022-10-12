//  Copyright (c) 2007-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/modules/allocator_support.hpp>
#include <hpx/modules/concepts.hpp>
#include <hpx/modules/tag_invoke.hpp>

#include <type_traits>
#include <utility>

///////////////////////////////////////////////////////////////////////////////
namespace hpx {

    namespace detail {

        // This CPO must live in hpx::detail for now as the dataflow function
        // supports being invoked in two forms: dataflow<Action>(...) and
        // dataflow(Action{}, ...). The only way to support both syntaxes (for
        // the time being, we will deprecate dataflow<Action>()) is to have a
        // real function based API that dispatches to the CPO. Once
        // dataflow<Action>(...) has been removed, this CPO can be moved to
        // namespace hpx.
        inline constexpr struct dataflow_t final
          : hpx::functional::tag<dataflow_t>
        {
        private:
            // clang-format off
            template <typename F, typename... Ts,
                HPX_CONCEPT_REQUIRES_(
                    !hpx::traits::is_allocator_v<std::decay_t<F>>
                )>
            // clang-format on
            friend constexpr HPX_FORCEINLINE auto tag_invoke(
                dataflow_t tag, F&& f, Ts&&... ts)
                -> decltype(tag(hpx::util::internal_allocator<>{},
                    HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...))
            {
                return hpx::functional::tag_invoke(tag,
                    hpx::util::internal_allocator<>{}, HPX_FORWARD(F, f),
                    HPX_FORWARD(Ts, ts)...);
            }
        } dataflow{};
    }    // namespace detail

    template <typename F, typename... Ts>
    HPX_FORCEINLINE decltype(auto) dataflow(F&& f, Ts&&... ts)
    {
        return hpx::detail::dataflow(HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...);
    }

    template <typename Allocator, typename... Ts>
    HPX_DEPRECATED_V(
        1, 9, "hpx::dataflow_alloc is deprecated, use hpx::dataflow instead")
    decltype(auto) dataflow_alloc(Allocator const& alloc, Ts&&... ts)
    {
        return hpx::detail::dataflow(alloc, HPX_FORWARD(Ts, ts)...);
    }
}    // namespace hpx

// #endif
