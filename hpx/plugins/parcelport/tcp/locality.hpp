//  Copyright (c) 2007-2020 Hartmut Kaiser
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

#if defined(HPX_HAVE_PARCELPORT_TCP)

#include <hpx/config/asio.hpp>
#include <hpx/runtime/parcelset/locality.hpp>
#include <hpx/serialization/serialize.hpp>
#include <hpx/util/ios_flags_saver.hpp>

#include <boost/asio/ip/host_name.hpp>
#include <boost/asio/ip/tcp.hpp>

#include <cstdint>
#include <string>

namespace hpx { namespace parcelset
{
    namespace policies { namespace tcp
    {
        class locality
        {
        public:
            locality()
              : port_(std::uint16_t(-1))
            {}

            locality(std::string const& addr, std::uint16_t port)
              : address_(addr), port_(port)
            {}

            std::string const & address() const
            {
                return address_;
            }

            std::uint16_t port() const
            {
                return port_;
            }

            static const char *type()
            {
                return "tcp";
            }

            explicit operator bool() const noexcept
            {
                return port_ != std::uint16_t(-1);
            }

            void save(serialization::output_archive & ar) const
            {
                ar << address_;
                ar << port_;
            }

            void load(serialization::input_archive & ar)
            {
                ar >> address_;
                ar >> port_;
            }

        private:
            friend bool operator==(locality const & lhs, locality const & rhs)
            {
                return lhs.port_ == rhs.port_ && lhs.address_ == rhs.address_;
            }

            friend bool operator<(locality const & lhs, locality const & rhs)
            {
                return lhs.address_ < rhs.address_ ||
                    (lhs.address_ == rhs.address_ && lhs.port_ < rhs.port_);
            }

            friend std::ostream & operator<<(std::ostream & os, locality const & loc)
            {
                hpx::util::ios_flags_saver ifs(os);
                os << loc.address_ << ":" << loc.port_;

                return os;
            }

            std::string address_;
            std::uint16_t port_;
        };
    }}
}}

#endif


