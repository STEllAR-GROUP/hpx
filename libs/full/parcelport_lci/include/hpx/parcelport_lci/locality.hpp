//  Copyright (c) 2007-2021 Hartmut Kaiser
//  Copyright (c) 2013-2014 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_NETWORKING) && defined(HPX_HAVE_PARCELPORT_LCI)
#include <hpx/modules/serialization.hpp>

#include <cstdint>

namespace hpx::parcelset::policies::lci {

    class locality
    {
    public:
        constexpr locality() noexcept
          : rank_(-1)
        {
        }

        explicit constexpr locality(std::int32_t rank) noexcept
          : rank_(rank)
        {
        }

        constexpr std::int32_t rank() const noexcept
        {
            return rank_;
        }

        static constexpr const char* type() noexcept
        {
            return "lci";
        }

        explicit constexpr operator bool() const noexcept
        {
            return rank_ != -1;
        }

        HPX_EXPORT void save(serialization::output_archive& ar) const;
        HPX_EXPORT void load(serialization::input_archive& ar);

    private:
        friend bool operator==(
            locality const& lhs, locality const& rhs) noexcept
        {
            return lhs.rank_ == rhs.rank_;
        }

        friend bool operator<(locality const& lhs, locality const& rhs) noexcept
        {
            return lhs.rank_ < rhs.rank_;
        }

        friend HPX_EXPORT std::ostream& operator<<(
            std::ostream& os, locality const& loc) noexcept;

        std::int32_t rank_;
    };
}    // namespace hpx::parcelset::policies::lci

#endif
