//  Copyright (c) 2007-2019 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/modules/memory.hpp>
#include <hpx/runtime/naming_fwd.hpp>
#include <hpx/serialization/serialization_fwd.hpp>

#include <cstdint>
#include <iosfwd>
#include <utility>

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace naming
{
    namespace detail
    {
        struct HPX_EXPORT id_type_impl;
        HPX_EXPORT void intrusive_ptr_add_ref(id_type_impl* p);
        HPX_EXPORT void intrusive_ptr_release(id_type_impl* p);
    }

    ///////////////////////////////////////////////////////////////////////////
    // the local gid is actually just a wrapper around the real thing
    struct HPX_EXPORT id_type
    {
    private:
        friend struct detail::id_type_impl;

    public:
        enum management_type
        {
            unknown_deleter = -1,
            unmanaged = 0,          ///< unmanaged GID
            managed = 1,            ///< managed GID
            managed_move_credit = 2 ///< managed GID which will give up all
                                    ///< credits when sent
        };

        id_type() = default;

        id_type(std::uint64_t lsb_id, management_type t);
        id_type(gid_type const& gid, management_type t);
        id_type(std::uint64_t msb_id, std::uint64_t lsb_id, management_type t);

        id_type(id_type const& o) noexcept
          : gid_(o.gid_)
        {
        }
        id_type(id_type && o) noexcept
          : gid_(std::move(o.gid_))
        {
        }

        id_type & operator=(id_type const & o) noexcept
        {
            gid_ = o.gid_;
            return *this;
        }
        id_type & operator=(id_type && o) noexcept
        {
            gid_ = std::move(o.gid_);
            return *this;
        }

        gid_type& get_gid();
        gid_type const& get_gid() const;

        // This function is used in AGAS unit tests and application code, do not
        // remove.
        management_type get_management_type() const;

        id_type& operator++();
        id_type operator++(int);

        explicit operator bool() const;

        // comparison is required as well
        friend bool operator== (id_type const& lhs, id_type const& rhs);
        friend bool operator!= (id_type const& lhs, id_type const& rhs);

        friend bool operator< (id_type const& lhs, id_type const& rhs);
        friend bool operator<= (id_type const& lhs, id_type const& rhs);
        friend bool operator> (id_type const& lhs, id_type const& rhs);
        friend bool operator>= (id_type const& lhs, id_type const& rhs);

        // access the internal parts of the gid
        std::uint64_t get_msb() const;
        void set_msb(std::uint64_t msb);

        std::uint64_t get_lsb() const;
        void set_lsb(std::uint64_t lsb);
        void set_lsb(void* lsb);

        // Convert this id into an unmanaged one (in-place) - Use with maximum
        // care, or better, don't use this at all.
        void make_unmanaged() const;

    private:
        friend HPX_EXPORT std::ostream& operator<<(std::ostream& os,
            id_type const& id);

        friend class hpx::serialization::access;

        void save(serialization::output_archive& ar, unsigned int version) const;
        void load(serialization::input_archive& ar, unsigned int version);

        HPX_SERIALIZATION_SPLIT_MEMBER()

        hpx::intrusive_ptr<detail::id_type_impl> gid_;
    };

    ///////////////////////////////////////////////////////////////////////////
    id_type const invalid_id = id_type();
}}

#include <hpx/config/warnings_suffix.hpp>

