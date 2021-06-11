//  Copyright (c) 2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file argument_types.hpp

#pragma once

#include <hpx/config.hpp>

#include <cstddef>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace collectives {

    struct num_sites_arg
    {
        explicit constexpr num_sites_arg(
            std::size_t num_sites = std::size_t(-1)) noexcept
          : num_sites_(num_sites)
        {
        }

        constexpr num_sites_arg& operator=(std::size_t num_sites) noexcept
        {
            num_sites_ = num_sites;
            return *this;
        }

        constexpr operator std::size_t() const noexcept
        {
            return num_sites_;
        }

        std::size_t num_sites_;
    };

    struct this_site_arg
    {
        explicit constexpr this_site_arg(
            std::size_t this_site = std::size_t(-1)) noexcept
          : this_site_(this_site)
        {
        }

        constexpr this_site_arg& operator=(std::size_t this_site) noexcept
        {
            this_site_ = this_site;
            return *this;
        }

        constexpr operator std::size_t() const noexcept
        {
            return this_site_;
        }

        std::size_t this_site_;
    };

    struct that_site_arg
    {
        explicit constexpr that_site_arg(
            std::size_t that_site = std::size_t(-1)) noexcept
          : that_site_(that_site)
        {
        }

        constexpr that_site_arg& operator=(std::size_t that_site) noexcept
        {
            that_site_ = that_site;
            return *this;
        }

        constexpr operator std::size_t() const noexcept
        {
            return that_site_;
        }

        std::size_t that_site_;
    };

    struct generation_arg
    {
        explicit constexpr generation_arg(
            std::size_t generation = std::size_t(-1)) noexcept
          : generation_(generation)
        {
        }

        constexpr generation_arg& operator=(std::size_t generation) noexcept
        {
            generation_ = generation;
            return *this;
        }

        constexpr operator std::size_t() const noexcept
        {
            return generation_;
        }

        std::size_t generation_;
    };

    struct root_site_arg
    {
        explicit constexpr root_site_arg(
            std::size_t root_site = std::size_t(0)) noexcept
          : root_site_(root_site)
        {
        }

        constexpr root_site_arg& operator=(std::size_t root_site) noexcept
        {
            root_site_ = root_site;
            return *this;
        }

        constexpr operator std::size_t() const noexcept
        {
            return root_site_;
        }

        std::size_t root_site_;
    };
}}    // namespace hpx::collectives
