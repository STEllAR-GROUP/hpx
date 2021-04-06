//  Copyright (c) 2007-2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/modules/memory.hpp>
#include <hpx/naming_base/id_type.hpp>

#include <ostream>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace naming {

    ///////////////////////////////////////////////////////////////////////////
    util::internal_allocator<detail::id_type_impl> detail::id_type_impl::alloc_;

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {

        id_type_impl::deleter_type id_type_impl::get_deleter(
            id_type::management_type t) noexcept
        {
            switch (t)
            {
            case id_type::unmanaged:
                return &detail::gid_unmanaged_deleter;

            case id_type::managed:
                HPX_FALLTHROUGH;
            case id_type::managed_move_credit:
                return &detail::gid_managed_deleter;

            default:
                HPX_ASSERT(false);    // invalid management type
                return &detail::gid_unmanaged_deleter;
            }
            return nullptr;
        }

        // support functions for hpx::intrusive_ptr
        void intrusive_ptr_add_ref(id_type_impl* p)
        {
            ++p->count_;
        }

        void intrusive_ptr_release(id_type_impl* p)
        {
            if (0 == --p->count_)
            {
                id_type_impl::get_deleter(p->get_management_type())(p);
            }
        }
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    id_type& id_type::operator++()    // pre-increment
    {
        ++(*gid_);
        return *this;
    }

    id_type id_type::operator++(int)    // post-increment
    {
        return id_type((*gid_)++, unmanaged);
    }

    // comparison is required as well
    bool operator==(id_type const& lhs, id_type const& rhs)
    {
        if (!lhs)
        {
            return !rhs;
        }
        if (!rhs)
        {
            return !lhs;
        }
        return *lhs.gid_ == *rhs.gid_;
    }

    bool operator<(id_type const& lhs, id_type const& rhs)
    {
        // LHS is null, rhs is not.
        if (!lhs && rhs)
        {
            return true;
        }
        // RHS is null.
        if (!rhs)
        {
            return false;
        }
        return *lhs.gid_ < *rhs.gid_;
    }

    std::ostream& operator<<(std::ostream& os, id_type const& id)
    {
        if (!id)
        {
            os << "{invalid}";
        }
        else
        {
            os << id.get_gid();
        }
        return os;
    }

    ///////////////////////////////////////////////////////////////////////////
    constexpr char const* const management_type_names[] = {
        "unknown_deleter",       // -1
        "unmanaged",             // 0
        "managed",               // 1
        "managed_move_credit"    // 2
    };

    char const* get_management_type_name(id_type::management_type m)
    {
        if (m < id_type::unknown_deleter || m > id_type::managed_move_credit)
        {
            return "invalid";
        }
        return management_type_names[m + 1];
    }
}}    // namespace hpx::naming

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace traits {

    naming::id_type get_remote_result<naming::id_type, naming::gid_type>::call(
        naming::gid_type const& rhs)
    {
        bool has_credits = naming::detail::has_credits(rhs);
        return naming::id_type(rhs,
            has_credits ? naming::id_type::managed :
                          naming::id_type::unmanaged);
    }

    // we need to specialize this template to allow for automatic conversion of
    // the vector<naming::gid_type> to a vector<naming::id_type>
    std::vector<naming::id_type> get_remote_result<std::vector<naming::id_type>,
        std::vector<naming::gid_type>>::
        call(std::vector<naming::gid_type> const& rhs)
    {
        std::vector<naming::id_type> result;
        result.reserve(rhs.size());
        for (naming::gid_type const& r : rhs)
        {
            bool has_credits = naming::detail::has_credits(r);
            result.push_back(naming::id_type(r,
                has_credits ? naming::id_type::managed :
                              naming::id_type::unmanaged));
        }
        return result;
    }
}}    // namespace hpx::traits
