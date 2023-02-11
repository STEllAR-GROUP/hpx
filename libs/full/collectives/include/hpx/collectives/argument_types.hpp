//  Copyright (c) 2021-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file argument_types.hpp

#pragma once

#include <hpx/config.hpp>

#include <cstddef>

///////////////////////////////////////////////////////////////////////////////
namespace hpx::collectives {

    namespace detail {

        template <typename Tag,
            std::size_t Default = static_cast<std::size_t>(-1)>
        struct argument_type
        {
            explicit constexpr argument_type(
                std::size_t argument = Default) noexcept
              : argument_(argument)
            {
            }

            constexpr argument_type& operator=(std::size_t argument) noexcept
            {
                argument_ = argument;
                return *this;
            }

            constexpr operator std::size_t() const noexcept
            {
                return argument_;
            }

            std::size_t argument_;
        };

        struct num_sites_tag;
        struct this_site_tag;
        struct that_site_tag;
        struct generation_tag;
        struct root_site_tag;
        struct tag_tag;
        struct arity_tag;
    }    // namespace detail

    using num_sites_arg = detail::argument_type<detail::num_sites_tag>;
    using this_site_arg = detail::argument_type<detail::this_site_tag>;
    using that_site_arg = detail::argument_type<detail::that_site_tag>;
    using generation_arg = detail::argument_type<detail::generation_tag>;
    using root_site_arg = detail::argument_type<detail::root_site_tag, 0>;
    using tag_arg = detail::argument_type<detail::tag_tag, 0>;
    using arity_arg = detail::argument_type<detail::arity_tag>;
}    // namespace hpx::collectives
