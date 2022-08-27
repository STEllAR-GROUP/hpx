//  Copyright (c) 2007-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/actions/transfer_action.hpp>
#include <hpx/actions_base/component_action.hpp>
#include <hpx/assert.hpp>
#include <hpx/async_distributed/base_lco_with_value.hpp>
#include <hpx/async_distributed/transfer_continuation_action.hpp>
#include <hpx/components_base/server/component.hpp>
#include <hpx/components_base/server/component_base.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/naming_base/id_type.hpp>
#include <hpx/preprocessor/cat.hpp>
#include <hpx/preprocessor/expand.hpp>
#include <hpx/preprocessor/nargs.hpp>

#include <type_traits>

namespace hpx { namespace components { namespace server {

    namespace detail {

        struct this_type
        {
        };
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    template <typename ConfigData, typename Derived = detail::this_type>
    class distributed_metadata_base
      : public hpx::components::component_base<typename std::conditional<
            std::is_same<Derived, detail::this_type>::value,
            distributed_metadata_base<ConfigData, Derived>, Derived>::type>
    {
    public:
        distributed_metadata_base()
        {
            HPX_ASSERT(false);
        }

        explicit distributed_metadata_base(ConfigData const& data)
          : data_(data)
        {
        }

        /// Retrieve the configuration data.
        ConfigData get() const
        {
            return data_;
        }

        HPX_DEFINE_COMPONENT_DIRECT_ACTION(distributed_metadata_base, get)

    private:
        ConfigData data_;
    };
}}}    // namespace hpx::components::server

#define HPX_DISTRIBUTED_METADATA_DECLARATION(...)                              \
    HPX_DISTRIBUTED_METADATA_DECLARATION_(__VA_ARGS__)                         \
    /**/
#define HPX_DISTRIBUTED_METADATA_DECLARATION_(...)                             \
    HPX_PP_EXPAND(HPX_PP_CAT(HPX_DISTRIBUTED_METADATA_DECLARATION_,            \
        HPX_PP_NARGS(__VA_ARGS__))(__VA_ARGS__))                               \
    /**/

#define HPX_DISTRIBUTED_METADATA_DECLARATION_1(config)                         \
    HPX_DISTRIBUTED_METADATA_DECLARATION_2(config, config)                     \
    /**/
#define HPX_DISTRIBUTED_METADATA_DECLARATION_2(config, name)                   \
    HPX_REGISTER_ACTION_DECLARATION(                                           \
        ::hpx::components::server::distributed_metadata_base<                  \
            config>::get_action,                                               \
        HPX_PP_CAT(__distributed_metadata_get_action_, name))                  \
    HPX_REGISTER_ACTION_DECLARATION(                                           \
        ::hpx::lcos::base_lco_with_value<config>::set_value_action,            \
        HPX_PP_CAT(__set_value_distributed_metadata_config_data_, name))       \
    /**/

#define HPX_DISTRIBUTED_METADATA(...)                                          \
    HPX_DISTRIBUTED_METADATA_(__VA_ARGS__)                                     \
    /**/
#define HPX_DISTRIBUTED_METADATA_(...)                                         \
    HPX_PP_EXPAND(HPX_PP_CAT(                                                  \
        HPX_DISTRIBUTED_METADATA_, HPX_PP_NARGS(__VA_ARGS__))(__VA_ARGS__))    \
    /**/

#define HPX_DISTRIBUTED_METADATA_1(config)                                     \
    HPX_DISTRIBUTED_METADATA_2(config, config)                                 \
    /**/
#define HPX_DISTRIBUTED_METADATA_2(config, name)                               \
    HPX_REGISTER_ACTION(::hpx::components::server::distributed_metadata_base<  \
                            config>::get_action,                               \
        HPX_PP_CAT(__distributed_metadata_get_action_, name))                  \
    HPX_REGISTER_ACTION(                                                       \
        ::hpx::lcos::base_lco_with_value<config>::set_value_action,            \
        HPX_PP_CAT(__set_value_distributed_metadata_config_data_, name))       \
    typedef ::hpx::components::component<                                      \
        ::hpx::components::server::distributed_metadata_base<config>>          \
        HPX_PP_CAT(__distributed_metadata_, name);                             \
    HPX_REGISTER_COMPONENT(HPX_PP_CAT(__distributed_metadata_, name))          \
    /**/
