//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_RUNTIME_NAMING_ADDRESS_HPP
#define HPX_RUNTIME_NAMING_ADDRESS_HPP

#include <hpx/config.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/naming_fwd.hpp>
#include <hpx/runtime/serialization/serialization_fwd.hpp>
#include <hpx/traits/is_bitwise_serializable.hpp>

#include <cstdint>
#include <iosfwd>

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
//  address serialization format version
#define HPX_ADDRESS_VERSION 0x20

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace naming
{
    struct HPX_EXPORT address
    {
        typedef std::int32_t component_type;
        typedef std::uint64_t address_type;

        ///////////////////////////////////////////////////////////////////////
        address()
          : locality_(), type_(components::component_invalid), address_(0)
        {}

        explicit address(gid_type const& l,
                component_type t = components::component_invalid)
          : locality_(l), type_(t), address_(0)
        {}

        address(gid_type const& l, component_type t, void* lva)
          : locality_(l), type_(t),
            address_(reinterpret_cast<address_type>(lva))
        {}

        address(gid_type const& l, component_type t, address_type a)
          : locality_(l), type_(t), address_(a)
        {}

        // local only addresses
        explicit address(void* lva,
                component_type t = components::component_invalid)
          : type_(t),
            address_(reinterpret_cast<address_type>(lva))
        {}

        explicit address(address_type a)
          : type_(components::component_invalid), address_(a)
        {}

        explicit operator bool() const HPX_NOEXCEPT
        {
            return !!locality_ &&
                (components::component_invalid != type_ || 0 != address_);
        }

        friend bool operator==(address const& lhs, address const& rhs)
        {
            return lhs.type_ == rhs.type_ && lhs.address_ == rhs.address_ &&
                   lhs.locality_ == rhs.locality_;
        }

        gid_type locality_;     /// locality: ip4 address/port number
        component_type type_;   /// component type this address is referring to
        address_type address_;  /// address (sequence number)

    private:
        friend HPX_EXPORT std::ostream& operator<<(std::ostream&, address const&);

        // serialization support
        friend class hpx::serialization::access;

        template <typename Archive>
        void save(Archive& ar, const unsigned int version) const;

        template <typename Archive>
        void load(Archive& ar, const unsigned int version);

        HPX_SERIALIZATION_SPLIT_MEMBER();
    };
}}

HPX_IS_BITWISE_SERIALIZABLE(hpx::naming::address)

#include <hpx/config/warnings_suffix.hpp>

#endif /*HPX_RUNTIME_NAMING_ADDRESS_HPP*/
