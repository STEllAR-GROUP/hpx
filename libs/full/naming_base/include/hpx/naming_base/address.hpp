//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/naming_base/gid_type.hpp>
#include <hpx/naming_base/naming_base.hpp>
#include <hpx/serialization/serialization_fwd.hpp>
#include <hpx/serialization/traits/is_bitwise_serializable.hpp>

#include <cstdint>

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
//  address serialization format version
#define HPX_ADDRESS_VERSION 0x20

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace naming {

    struct address
    {
        using component_type = naming::component_type;
        using address_type = naming::address_type;

        static constexpr component_type const component_invalid = -1;

        ///////////////////////////////////////////////////////////////////////
        constexpr address() noexcept = default;

        explicit constexpr address(
            gid_type const& l, component_type t = component_invalid) noexcept
          : locality_(l)
          , type_(t)
        {
        }

        address(gid_type const& l, component_type t, void* lva) noexcept
          : locality_(l)
          , type_(t)
          , address_(reinterpret_cast<address_type>(lva))
        {
        }

        constexpr address(
            gid_type const& l, component_type t, address_type a) noexcept
          : locality_(l)
          , type_(t)
          , address_(a)
        {
        }

        // local only addresses
        explicit address(
            void* lva, component_type t = component_invalid) noexcept
          : type_(t)
          , address_(reinterpret_cast<address_type>(lva))
        {
        }

        explicit constexpr address(address_type a) noexcept
          : address_(a)
        {
        }

        explicit constexpr operator bool() const noexcept
        {
            return !!locality_ && (component_invalid != type_ || 0 != address_);
        }

        friend constexpr bool operator==(
            address const& lhs, address const& rhs) noexcept
        {
            return lhs.type_ == rhs.type_ && lhs.address_ == rhs.address_ &&
                lhs.locality_ == rhs.locality_;
        }

        gid_type locality_;
        component_type type_ = component_invalid;
        address_type address_ = 0;    /// address (local virtual address)

    private:
        // serialization support
        friend class hpx::serialization::access;

        template <typename Archive>
        void save(Archive& ar, unsigned int version) const;

        template <typename Archive>
        void load(Archive& ar, unsigned int version);

        HPX_SERIALIZATION_SPLIT_MEMBER();
    };
}}    // namespace hpx::naming

HPX_IS_BITWISE_SERIALIZABLE(hpx::naming::address)

#include <hpx/config/warnings_suffix.hpp>
