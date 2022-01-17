// Copyright Vladimir Prus 2004.
//  SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/plugin/config.hpp>
#include <hpx/plugin/abstract_factory.hpp>
#include <hpx/plugin/plugin_wrapper.hpp>
#include <hpx/type_support/pack.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util { namespace plugin {

    namespace detail {

        template <typename BasePlugin, typename Concrete, typename Base,
            typename Parameter>
        struct concrete_factory_item;

        template <typename BasePlugin, typename Concrete, typename Base,
            typename... Parameters>
        struct concrete_factory_item<BasePlugin, Concrete, Base,
            hpx::util::pack<Parameters...>> : public Base
        {
            BasePlugin* create(dll_handle dll, Parameters... parameters)
            {
                return new plugin_wrapper<Concrete, Parameters...>(
                    dll, parameters...);
            }
        };
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    template <typename BasePlugin, typename Concrete>
    struct concrete_factory
      : detail::concrete_factory_item<BasePlugin, Concrete,
            abstract_factory<BasePlugin>,
            typename virtual_constructor<BasePlugin>::type>
    {
    };

}}}    // namespace hpx::util::plugin
