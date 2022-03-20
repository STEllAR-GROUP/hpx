//  Copyright (c) 2007-2022 Hartmut Kaiser
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
namespace hpx::naming {

    ///////////////////////////////////////////////////////////////////////////
    HPX_EXPORT void decrement_refcnt(gid_type const& gid);

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {

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
        HPX_EXPORT void decrement_refcnt(id_type_impl* gid) noexcept;

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

        HPX_SERIALIZATION_SPLIT_FREE(id_type_impl)
    }    // namespace detail
}    // namespace hpx::naming

namespace hpx {

    ///////////////////////////////////////////////////////////////////////////
    HPX_EXPORT void save(
        serialization::output_archive& ar, hpx::id_type const&, unsigned int);
    HPX_EXPORT void load(
        serialization::input_archive& ar, hpx::id_type&, unsigned int);

    HPX_SERIALIZATION_SPLIT_FREE(hpx::id_type)
}    // namespace hpx
