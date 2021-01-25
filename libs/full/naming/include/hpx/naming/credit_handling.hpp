//  Copyright (c) 2007-2020 Hartmut Kaiser
//  Copyright (c) 2011 Bryce Lelbach
//  Copyright (c) 2007 Richard D. Guidry Jr.
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/modules/naming_base.hpp>
#include <hpx/serialization/serialization_fwd.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <iosfwd>
#include <mutex>
#include <string>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace naming {

    ///////////////////////////////////////////////////////////////////////////
    HPX_EXPORT void decrement_refcnt(gid_type const& gid);

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {

        // We store the log2(credit) in the gid_type
        constexpr std::int16_t log2(std::int64_t val) noexcept
        {
            std::int16_t ret = -1;
            while (val != 0)
            {
                val >>= 1;
                ++ret;
            }
            return ret;
        }

        inline std::int64_t power2(std::int16_t log2credits) noexcept
        {
            HPX_ASSERT(log2credits >= 0);
            return static_cast<std::int64_t>(1) << log2credits;
        }

        ///////////////////////////////////////////////////////////////////////
        constexpr bool store_in_cache(gid_type const& id) noexcept
        {
            return (id.get_msb() & gid_type::dont_cache_mask) ? false : true;
        }

        inline void set_dont_store_in_cache(gid_type& gid) noexcept
        {
            gid.set_msb(gid.get_msb() | gid_type::dont_cache_mask);
        }

        ///////////////////////////////////////////////////////////////////////
        constexpr bool is_migratable(gid_type const& id) noexcept
        {
            return (id.get_msb() & gid_type::is_migratable) ? true : false;
        }

        inline void set_is_migratable(gid_type& gid) noexcept
        {
            gid.set_msb(gid.get_msb() | gid_type::is_migratable);
        }

        ///////////////////////////////////////////////////////////////////////
        constexpr std::int16_t get_log2credit_from_gid(
            gid_type const& id) noexcept
        {
            HPX_ASSERT(has_credits(id));
            return std::int16_t((id.get_msb() >> gid_type::credit_shift) &
                gid_type::credit_base_mask);
        }

        constexpr std::int64_t get_credit_from_gid(gid_type const& id) noexcept
        {
            return has_credits(id) ?
                detail::power2(get_log2credit_from_gid(id)) :
                0;
        }

        ///////////////////////////////////////////////////////////////////////
        inline gid_type get_stripped_gid_except_dont_cache(
            gid_type const& id) noexcept
        {
            std::uint64_t const msb =
                strip_internal_bits_except_dont_cache_from_gid(id.get_msb());
            std::uint64_t const lsb = id.get_lsb();
            return gid_type(msb, lsb);
        }

        inline std::uint64_t strip_credits_from_gid(std::uint64_t msb) noexcept
        {
            return msb & ~gid_type::credit_bits_mask;
        }

        inline gid_type& strip_credits_from_gid(gid_type& id) noexcept
        {
            id.set_msb(strip_credits_from_gid(id.get_msb()));
            return id;
        }

        ///////////////////////////////////////////////////////////////////////
        inline void set_log2credit_for_gid(
            gid_type& id, std::int16_t log2credits) noexcept
        {
            // credit should be a clean log2
            HPX_ASSERT(log2credits >= 0);
            HPX_ASSERT(0 == (log2credits & ~gid_type::credit_base_mask));

            id.set_msb((id.get_msb() & ~gid_type::credit_mask) |
                ((std::int32_t(log2credits) << gid_type::credit_shift) &
                    gid_type::credit_mask) |
                gid_type::has_credits_mask);
        }

        inline void set_credit_for_gid(
            gid_type& id, std::int64_t credits) noexcept
        {
            if (credits != 0)
            {
                std::int16_t log2credits = detail::log2(credits);
                HPX_ASSERT(detail::power2(log2credits) == credits);

                set_log2credit_for_gid(id, log2credits);
            }
            else
            {
                strip_credits_from_gid(id);
            }
        }

        ///////////////////////////////////////////////////////////////////////
        // has side effects, can't be pure
        HPX_EXPORT std::int64_t add_credit_to_gid(
            gid_type& id, std::int64_t credits);

        HPX_EXPORT std::int64_t remove_credit_from_gid(
            gid_type& id, std::int64_t debit);

        HPX_EXPORT std::int64_t fill_credit_for_gid(gid_type& id,
            std::int64_t credits = std::int64_t(HPX_GLOBALCREDIT_INITIAL));

        ///////////////////////////////////////////////////////////////////////
        HPX_EXPORT gid_type move_gid(gid_type& id);
        HPX_EXPORT gid_type move_gid_locked(
            std::unique_lock<gid_type::mutex_type> l, gid_type& gid);

        HPX_EXPORT std::int64_t replenish_credits(gid_type& id);
        HPX_EXPORT std::int64_t replenish_credits_locked(
            std::unique_lock<gid_type::mutex_type>& l, gid_type& id);

        ///////////////////////////////////////////////////////////////////////
        // splits the current credit of the given id and assigns half of it to
        // the returned copy
        HPX_EXPORT gid_type split_credits_for_gid(gid_type& id);
        HPX_EXPORT gid_type split_credits_for_gid_locked(
            std::unique_lock<gid_type::mutex_type>& l, gid_type& id);

        ///////////////////////////////////////////////////////////////////////
        HPX_EXPORT void decrement_refcnt(id_type_impl* gid);

        ///////////////////////////////////////////////////////////////////////
        // credit management (called during serialization), this function
        // has to be 'const' as save() above has to be 'const'.
        void preprocess_gid(
            id_type_impl const&, serialization::output_archive& ar);

        ///////////////////////////////////////////////////////////////////////
        // serialization
        HPX_EXPORT void save(
            serialization::output_archive& ar, id_type_impl const&, unsigned);
        HPX_EXPORT void load(
            serialization::input_archive& ar, id_type_impl&, unsigned);

        HPX_SERIALIZATION_SPLIT_FREE(id_type_impl);
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    HPX_EXPORT void save(
        serialization::output_archive& ar, id_type const&, unsigned int);
    HPX_EXPORT void load(
        serialization::input_archive& ar, id_type&, unsigned int);

    HPX_SERIALIZATION_SPLIT_FREE(id_type);
}}    // namespace hpx::naming
