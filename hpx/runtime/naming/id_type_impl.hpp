//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_NAMING_ID_TYPE_IMPL_OCT_13_2013_0758PM)
#define HPX_NAMING_ID_TYPE_IMPL_OCT_13_2013_0758PM

#include <hpx/config.hpp>
#include <hpx/runtime/naming/id_type.hpp>
#include <hpx/runtime/naming/name.hpp>

#include <boost/cstdint.hpp>

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace naming
{
    ///////////////////////////////////////////////////////////////////////////
    // the local gid is actually just a wrapper around the real thing
    inline id_type::id_type(boost::uint64_t lsb_id, management_type t)
        : gid_(new detail::id_type_impl(0, lsb_id,
            static_cast<detail::id_type_management>(t)))
    {}

    inline id_type::id_type(gid_type const& gid, management_type t)
        : gid_(new detail::id_type_impl(gid,
            static_cast<detail::id_type_management>(t)))
    {
        if (t == unmanaged)
            detail::strip_internal_bits_from_gid(*gid_);
    }

    inline id_type::id_type(boost::uint64_t msb_id, boost::uint64_t lsb_id,
            management_type t)
        : gid_(new detail::id_type_impl(msb_id, lsb_id,
            static_cast<detail::id_type_management>(t)))
    {
        if (t == unmanaged)
            detail::strip_internal_bits_from_gid(*gid_);
    }

    inline gid_type& id_type::get_gid() { return *gid_; }
    inline gid_type const& id_type::get_gid() const { return *gid_; }

    // This function is used in AGAS unit tests and application code, do not
    // remove.
    inline id_type::management_type id_type::get_management_type() const
    {
        return management_type(gid_->get_management_type());
    }

    inline id_type& id_type::operator++()       // pre-increment
    {
        ++(*gid_);
        return *this;
    }
    inline id_type id_type::operator++(int)     // post-increment
    {
        return id_type((*gid_)++, unmanaged);
    }

    inline id_type::operator util::safe_bool<id_type>::result_type() const
    {
        return util::safe_bool<id_type>()(gid_ && *gid_);
    }

    // comparison is required as well
    inline bool operator== (id_type const& lhs, id_type const& rhs)
    {
        if (!lhs)
            return !rhs;
        if (!rhs)
            return !lhs;

        return *lhs.gid_ == *rhs.gid_;
    }
    inline bool operator!= (id_type const& lhs, id_type const& rhs)
    {
        return !(lhs == rhs);
    }

    inline bool operator< (id_type const& lhs, id_type const& rhs)
    {
        // LHS is null, rhs is not.
        if (!lhs && rhs)
            return true;

        // RHS is null.
        if (!rhs)
            return false;

        return *lhs.gid_ < *rhs.gid_;
    }

    inline bool operator<= (id_type const& lhs, id_type const& rhs)
    {
        // Deduced from <.
        return !(rhs < lhs);
    }

    inline bool operator> (id_type const& lhs, id_type const& rhs)
    {
        // Deduced from <.
        return rhs < lhs;
    }

    inline bool operator>= (id_type const& lhs, id_type const& rhs)
    {
        // Deduced from <.
        return !(lhs < rhs);
    }

    // access the internal parts of the gid
    inline boost::uint64_t id_type::get_msb() const
    {
        return gid_->get_msb();
    }
    inline void id_type::set_msb(boost::uint64_t msb)
    {
        gid_->set_msb(msb);
    }

    inline boost::uint64_t id_type::get_lsb() const
    {
        return gid_->get_lsb();
    }
    inline void id_type::set_lsb(boost::uint64_t lsb)
    {
        gid_->set_lsb(lsb);
    }
    inline void id_type::set_lsb(void* lsb)
    {
        gid_->set_lsb(lsb);
    }

    inline void id_type::make_unmanaged() const
    {
        gid_->set_management_type(detail::unmanaged);
    }
}}

#endif

