//  Copyright (c) 2007-2014 Hartmut Kaiser
//  Copyright (c) 2014 Thomas Heller
//  Copyright (c) 2007 Richard D Guidry Jr
//  Copyright (c) 2011 Bryce Lelbach
//  Copyright (c) 2011 Katelyn Kufahl
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARCELSET_POLICIES_TCP_LOCALITY_HPP
#define HPX_PARCELSET_POLICIES_TCP_LOCALITY_HPP

#include <hpx/config.hpp>

#if defined(HPX_HAVE_PARCELPORT_TCP)

#include <hpx/config/asio.hpp>
#include <hpx/runtime/parcelset/locality.hpp>
#include <hpx/runtime/serialization/serialize.hpp>

#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/ip/host_name.hpp>

#include <string>

namespace hpx { namespace parcelset
{
    namespace policies { namespace tcp
    {
        class locality
        {
        public:
            locality()
              : port_(boost::uint16_t(-1))
            {}

            locality(std::string const& addr, boost::uint16_t port)
              : address_(addr), port_(port)
            {}

            std::string const & address() const
            {
                return address_;
            }

            boost::uint16_t port() const
            {
                return port_;
            }

            static const char *type()
            {
                return "tcp";
            }

            explicit operator bool() const HPX_NOEXCEPT
            {
                return port_ != boost::uint16_t(-1);
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
                boost::io::ios_flags_saver ifs(os);
                os << loc.address_ << ":" << loc.port_;

                return os;
            }

            std::string address_;
            boost::uint16_t port_;
        };
    }}
}}

#endif

#endif

