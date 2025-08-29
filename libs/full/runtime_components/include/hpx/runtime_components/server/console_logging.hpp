//  Copyright (c) 2007-2025 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/actions/transfer_action.hpp>
#include <hpx/actions_base/plain_action.hpp>
#include <hpx/actions_base/traits/action_does_termination_detection.hpp>
#include <hpx/async_distributed/transfer_continuation_action.hpp>
#include <hpx/components_base/component_type.hpp>
#include <hpx/datastructures/tuple.hpp>
#include <hpx/modules/logging.hpp>
#include <hpx/runtime_components/console_logging.hpp>
#include <hpx/serialization/vector.hpp>

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace hpx::components {

    using message_type =
        hpx::tuple<logging_destination, std::size_t, std::string>;

    using messages_type = std::vector<message_type>;
}    // namespace hpx::components

//////////////////////////////////////////////////////////////////////////////
namespace hpx::components::server {

    ///////////////////////////////////////////////////////////////////////////
    // console logging happens here
    void console_logging(messages_type const&);

    ///////////////////////////////////////////////////////////////////////////
    // this type is a dummy template to avoid premature instantiation of the
    // serialization support instances
    template <typename Dummy = void>
    class console_logging_action
      : public actions::direct_action<void (*)(messages_type const&),
            console_logging, console_logging_action<Dummy>>
    {
    private:
        using base_type = actions::direct_action<void (*)(messages_type const&),
            console_logging, console_logging_action>;

    public:
        console_logging_action() = default;

        // construct an action from its arguments
        explicit console_logging_action(messages_type const& msgs)
          : base_type(msgs)
        {
        }

        console_logging_action(
            threads::thread_priority, messages_type const& msgs)
          : base_type(msgs)
        {
        }

    public:
        template <typename T>
        static util::unused_type execute_function(
            naming::address_type, naming::component_type, T&& v)
        {
            try
            {
                // call the function, ignoring the return value
                console_logging(HPX_FORWARD(T, v));
            }
            // NOLINTNEXTLINE(bugprone-empty-catch)
            catch (...)
            {
                // no logging!
            }
            return util::unused;
        }
    };
}    // namespace hpx::components::server

HPX_REGISTER_ACTION_DECLARATION(
    hpx::components::server::console_logging_action<>, console_logging_action)

#if !defined(HPX_COMPUTE_DEVICE_CODE) && defined(HPX_HAVE_NETWORKING)
///////////////////////////////////////////////////////////////////////////
// Logging does not make this locality black
template <>
struct hpx::traits::action_does_termination_detection<
    hpx::components::server::console_logging_action<>>
{
    static constexpr bool call() noexcept
    {
        return true;
    }
};    // namespace hpx::traits
#endif
