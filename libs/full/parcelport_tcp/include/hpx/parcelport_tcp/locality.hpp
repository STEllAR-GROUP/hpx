//  Copyright (c) 2007-2021 Hartmut Kaiser
//  Copyright (c) 2014 Thomas Heller
//  Copyright (c) 2007 Richard D Guidry Jr
//  Copyright (c) 2011 Bryce Lelbach
//  Copyright (c) 2011 Katelyn Kufahl
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_NETWORKING) && defined(HPX_HAVE_PARCELPORT_TCP)
#include <hpx/modules/serialization.hpp>

#include <cstdint>
#include <string>

namespace hpx::parcelset::policies::tcp {

    class locality
    {
    public:
        locality() noexcept
          : port_(std::uint16_t(-1))
        {
        }

        locality(std::string const& addr, std::uint16_t port)
          : address_(addr)
          , port_(port)
        {
        }

        std::string const& address() const noexcept
        {
            return address_;
        }

        std::uint16_t port() const noexcept
        {
            return port_;
        }

        static constexpr const char* type() noexcept
        {
            return "tcp";
        }

        explicit constexpr operator bool() const noexcept
        {
            return port_ != std::uint16_t(-1);
        }

        HPX_EXPORT void save(serialization::output_archive& ar) const;
        HPX_EXPORT void load(serialization::input_archive& ar);

    private:
        friend bool operator==(
            locality const& lhs, locality const& rhs) noexcept
        {
            return lhs.port_ == rhs.port_ && lhs.address_ == rhs.address_;
        }

        friend bool operator<(locality const& lhs, locality const& rhs) noexcept
        {
            return lhs.address_ < rhs.address_ ||
                (lhs.address_ == rhs.address_ && lhs.port_ < rhs.port_);
        }

        friend HPX_EXPORT std::ostream& operator<<(
            std::ostream& os, locality const& loc) noexcept;

        std::string address_;
        std::uint16_t port_;
    };
}    // namespace hpx::parcelset::policies::tcp

#endif
