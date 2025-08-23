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
                std::size_t const argument = Default) noexcept
              : argument_(argument)
            {
            }

            constexpr argument_type& operator=(
                std::size_t const argument) noexcept
            {
                argument_ = argument;
                return *this;
            }

            constexpr operator std::size_t() const noexcept
            {
                return argument_;
            }

            [[nodiscard]] constexpr bool is_default() const noexcept
            {
                return argument_ == Default;
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

    /// The number of participating sites (default: all localities)
    using num_sites_arg = detail::argument_type<detail::num_sites_tag>;

    /// The local end of the communication channel
    using this_site_arg = detail::argument_type<detail::this_site_tag>;

    /// The opposite end of the communication channel
    using that_site_arg = detail::argument_type<detail::that_site_tag>;

    /// The generational counter identifying the sequence number of the
    /// operation performed on the given base name. It needs to be supplied
    /// only if the operation on the given base name has to be performed
    /// more than once. It must be a positive number greater than zero.
    using generation_arg = detail::argument_type<detail::generation_tag>;

    /// The site that is responsible for creating the support object
    /// of the operation. It defaults to '0' (zero).
    using root_site_arg = detail::argument_type<detail::root_site_tag, 0>;

    /// The tag identifying the concrete operation
    using tag_arg = detail::argument_type<detail::tag_tag, 0>;

    /// The number of children each of the communication nodes is connected
    /// to (default: picked based on num_sites).
    using arity_arg = detail::argument_type<detail::arity_tag>;
}    // namespace hpx::collectives
