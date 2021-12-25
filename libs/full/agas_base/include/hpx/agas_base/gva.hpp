////////////////////////////////////////////////////////////////////////////////
//  Copyright (c)      2011 Bryce Adelstein-Lelbach
//  Copyright (c) 2007-2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <hpx/config.hpp>
#include <hpx/components_base/component_type.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/modules/naming.hpp>
#include <hpx/naming_base/gid_type.hpp>

#include <cstdint>
#include <iosfwd>

namespace hpx { namespace agas {

    struct gva
    {
        using component_type = std::int32_t;
        using lva_type = void*;

        constexpr gva() noexcept = default;

        constexpr explicit gva(naming::gid_type const& p,
            component_type t = components::component_invalid,
            std::uint64_t c = 1, lva_type a = nullptr,
            std::uint64_t o = 0) noexcept
          : prefix(p)
          , type(t)
          , count(c)
          , lva_(a)
          , offset(o)
        {
        }

        explicit gva(naming::gid_type const& p, component_type t,
            std::uint64_t c, std::uint64_t l, std::uint64_t o = 0) noexcept
          : prefix(p)
          , type(t)
          , count(c)
          , lva_(reinterpret_cast<lva_type>(l))
          , offset(o)
        {
        }

        constexpr explicit gva(lva_type a) noexcept
          : type(components::component_invalid)
          , lva_(a)
        {
        }

        gva& operator=(lva_type a)
        {
            prefix = naming::gid_type();
            type = components::component_invalid;
            count = 0;
            lva_ = a;
            offset = 0;
            return *this;
        }

        constexpr bool operator==(gva const& rhs) const noexcept
        {
            return type == rhs.type && count == rhs.count && lva_ == rhs.lva_ &&
                offset == rhs.offset && prefix == rhs.prefix;
        }

        constexpr bool operator!=(gva const& rhs) const noexcept
        {
            return !(*this == rhs);
        }

        void lva(lva_type a) noexcept
        {
            lva_ = a;
        }

        constexpr lva_type lva() const noexcept
        {
            return lva_;
        }

        lva_type lva(naming::gid_type const& gid,
            naming::gid_type const& gidbase) const noexcept
        {
            return static_cast<char*>(lva_) +
                (gid.get_lsb() - gidbase.get_lsb()) * offset;
        }

        gva resolve(
            naming::gid_type const& gid, naming::gid_type const& gidbase) const
        {
            gva g(*this);
            g.lva_ = g.lva(gid, gidbase);

            // This is a hack to make sure that if resolve() or lva() is called on
            // the returned GVA, an exact copy will be returned (see the last two
            // lines of lva() above.
            g.count = 1;
            return g;
        }

        naming::gid_type prefix;
        component_type type = components::component_invalid;
        std::uint64_t count = 0;

    private:
        lva_type lva_ = nullptr;

    public:
        std::uint64_t offset = 0;

    private:
        friend class hpx::serialization::access;

        template <typename Archive>
        HPX_EXPORT void save(Archive& ar, const unsigned int /*version*/) const;

        template <typename Archive>
        HPX_EXPORT void load(Archive& ar, const unsigned int version);

        HPX_SERIALIZATION_SPLIT_MEMBER()
    };

    HPX_EXPORT std::ostream& operator<<(std::ostream& os, gva const& addr);
}}    // namespace hpx::agas
