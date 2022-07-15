//  Copyright (c) 2016-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/actions/transfer_action.hpp>
#include <hpx/actions_base/component_action.hpp>
#include <hpx/async_distributed/base_lco_with_value.hpp>
#include <hpx/async_distributed/transfer_continuation_action.hpp>
#include <hpx/components_base/component_type.hpp>
#include <hpx/components_base/server/component_base.hpp>
#include <hpx/components_base/traits/is_component.hpp>
#include <hpx/futures/traits/get_remote_result.hpp>
#include <hpx/futures/traits/promise_remote_result.hpp>
#include <hpx/lcos_local/channel.hpp>
#include <hpx/preprocessor/cat.hpp>
#include <hpx/preprocessor/expand.hpp>
#include <hpx/preprocessor/nargs.hpp>
#include <hpx/preprocessor/stringize.hpp>

#include <cstddef>
#include <exception>
#include <type_traits>
#include <utility>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos { namespace server {

    ///////////////////////////////////////////////////////////////////////////
    template <typename T,
        typename RemoteType = traits::promise_remote_result_t<T>>
    class channel;

    ///////////////////////////////////////////////////////////////////////////
    template <typename T, typename RemoteType>
    class channel
      : public lcos::base_lco_with_value<T, RemoteType,
            traits::detail::component_tag>
      , public components::component_base<channel<T, RemoteType>>
    {
    public:
        using base_type_holder = lcos::base_lco_with_value<T, RemoteType,
            traits::detail::component_tag>;

    private:
        using base_type = components::component_base<channel>;
        using result_type =
            std::conditional_t<std::is_void<T>::value, util::unused_type, T>;

    public:
        channel() = default;

        // disambiguate base classes
        using base_type::finalize;
        using wrapping_type = typename base_type::wrapping_type;

        static components::component_type get_component_type()
        {
            return components::get_component_type<channel>();
        }
        static void set_component_type(components::component_type type)
        {
            components::set_component_type<channel>(type);
        }

        naming::address get_current_address() const
        {
            return naming::address(
                naming::get_gid_from_locality_id(agas::get_locality_id()),
                components::get_component_type<channel>(),
                const_cast<channel*>(this));
        }

        // standard LCO action implementations

        // Push a value to the channel.
        void set_value(RemoteType&& result)
        {
            channel_.set(HPX_MOVE(result));
        }

        // Close the channel
        void set_exception(std::exception_ptr const& /*e*/)
        {
            channel_.close();
        }

        // Retrieve the next value from the channel
        result_type get_value()
        {
            return channel_.get(launch::sync);
        }
        result_type get_value(error_code& ec)
        {
            return channel_.get(launch::sync, ec);
        }

        // Additional functionality exposed by the channel component
        hpx::future<T> get_generation(std::size_t generation)
        {
            return channel_.get(generation);
        }
        HPX_DEFINE_COMPONENT_DIRECT_ACTION(channel, get_generation)

        void set_generation(RemoteType&& value, std::size_t generation)
        {
            channel_.set(HPX_MOVE(value), generation);
        }
        HPX_DEFINE_COMPONENT_DIRECT_ACTION(channel, set_generation)

        std::size_t close(bool force_delete_entries)
        {
            return channel_.close(force_delete_entries);
        }
        HPX_DEFINE_COMPONENT_ACTION(channel, close)

    private:
        lcos::local::channel<result_type> channel_;
    };
}}}    // namespace hpx::lcos::server

#define HPX_REGISTER_CHANNEL_DECLARATION(...)                                  \
    HPX_REGISTER_CHANNEL_DECLARATION_(__VA_ARGS__)                             \
/**/
#define HPX_REGISTER_CHANNEL_DECLARATION_(...)                                 \
    HPX_PP_EXPAND(HPX_PP_CAT(HPX_REGISTER_CHANNEL_DECLARATION_,                \
        HPX_PP_NARGS(__VA_ARGS__))(__VA_ARGS__))                               \
    /**/

#define HPX_REGISTER_CHANNEL_DECLARATION_1(type)                               \
    HPX_REGISTER_CHANNEL_DECLARATION_2(type, type)                             \
/**/
#define HPX_REGISTER_CHANNEL_DECLARATION_2(type, name)                         \
    using HPX_PP_CAT(__channel_, HPX_PP_CAT(type, name)) =                     \
        ::hpx::lcos::server::channel<type>;                                    \
    HPX_REGISTER_ACTION_DECLARATION(                                           \
        hpx::lcos::server::channel<type>::get_generation_action,               \
        HPX_PP_CAT(__channel_get_generation_action, HPX_PP_CAT(type, name)))   \
    HPX_REGISTER_ACTION_DECLARATION(                                           \
        hpx::lcos::server::channel<type>::set_generation_action,               \
        HPX_PP_CAT(__channel_set_generation_action, HPX_PP_CAT(type, name)))   \
    HPX_REGISTER_ACTION_DECLARATION(                                           \
        hpx::lcos::server::channel<type>::close_action,                        \
        HPX_PP_CAT(__channel_close_action, HPX_PP_CAT(type, name)))            \
    /**/

#define HPX_REGISTER_CHANNEL(...)                                              \
    HPX_REGISTER_CHANNEL_(__VA_ARGS__)                                         \
/**/
#define HPX_REGISTER_CHANNEL_(...)                                             \
    HPX_PP_EXPAND(HPX_PP_CAT(                                                  \
        HPX_REGISTER_CHANNEL_, HPX_PP_NARGS(__VA_ARGS__))(__VA_ARGS__))        \
    /**/

#define HPX_REGISTER_CHANNEL_1(type)                                           \
    HPX_REGISTER_CHANNEL_2(type, type)                                         \
/**/
#define HPX_REGISTER_CHANNEL_2(type, name)                                     \
    using HPX_PP_CAT(__channel_, HPX_PP_CAT(type, name)) =                     \
        ::hpx::lcos::server::channel<type>;                                    \
    using HPX_PP_CAT(__channel_component_, name) =                             \
        ::hpx::components::component<HPX_PP_CAT(                               \
            __channel_, HPX_PP_CAT(type, name))>;                              \
    HPX_REGISTER_DERIVED_COMPONENT_FACTORY(                                    \
        HPX_PP_CAT(__channel_component_, name),                                \
        HPX_PP_CAT(__channel_component_, name),                                \
        HPX_PP_STRINGIZE(HPX_PP_CAT(__base_lco_with_value_channel_, name)))    \
    HPX_REGISTER_ACTION(                                                       \
        hpx::lcos::server::channel<type>::get_generation_action,               \
        HPX_PP_CAT(__channel_get_generation_action, HPX_PP_CAT(type, name)))   \
    HPX_REGISTER_ACTION(                                                       \
        hpx::lcos::server::channel<type>::set_generation_action,               \
        HPX_PP_CAT(__channel_set_generation_action, HPX_PP_CAT(type, name)))   \
    HPX_REGISTER_ACTION(hpx::lcos::server::channel<type>::close_action,        \
        HPX_PP_CAT(__channel_close_action, HPX_PP_CAT(type, name)))            \
    /**/
