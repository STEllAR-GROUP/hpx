//  Copyright Vladimir Prus 2004.
//  Copyright (c) 2005-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/plugin/config.hpp>
#include <hpx/plugin/virtual_constructor.hpp>

namespace hpx::util::plugin {

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {

        struct dll_handle_holder
        {
            explicit dll_handle_holder(dll_handle dll) noexcept
              : m_dll(HPX_MOVE(dll))
            {
            }

            ~dll_handle_holder() = default;

        private:
            dll_handle m_dll;
        };
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    template <typename Wrapped, typename... Parameters>
    struct plugin_wrapper
      : public detail::dll_handle_holder
      , public Wrapped
    {
        explicit plugin_wrapper(dll_handle dll, Parameters... parameters)
          : detail::dll_handle_holder(HPX_MOVE(dll))
          , Wrapped(parameters...)
        {
        }
    };
}    // namespace hpx::util::plugin
