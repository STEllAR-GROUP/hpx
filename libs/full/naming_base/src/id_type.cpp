//  Copyright (c) 2007-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/modules/memory.hpp>
#include <hpx/naming_base/id_type.hpp>

#include <atomic>
#include <ostream>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace hpx {

    ///////////////////////////////////////////////////////////////////////////
    namespace naming::detail {

        void (*gid_managed_deleter)(id_type_impl const* p) noexcept = nullptr;
        void (*gid_unmanaged_deleter)(id_type_impl const* p) noexcept = nullptr;

        util::internal_allocator<id_type_impl> id_type_impl::alloc_;

        id_type_impl::deleter_type id_type_impl::get_deleter(
            hpx::id_type::management_type t) noexcept
        {
            switch (t)
            {
            case hpx::id_type::management_type::unmanaged:
                return gid_unmanaged_deleter;

            case hpx::id_type::management_type::managed:
                [[fallthrough]];
            case hpx::id_type::management_type::managed_move_credit:
                return gid_managed_deleter;

            default:
                HPX_ASSERT(false);    // invalid management type
                break;
            }
            return gid_unmanaged_deleter;
        }

        // support functions for hpx::intrusive_ptr
        void intrusive_ptr_add_ref(id_type_impl* p) noexcept
        {
            p->count_.increment();
        }

        void intrusive_ptr_release(id_type_impl* p) noexcept
        {
            if (0 == p->count_.decrement())
            {
                // The thread that decrements the reference count to zero must
                // perform an acquire to ensure that it doesn't start
                // destructing the object until all previous writes have
                // drained.
                std::atomic_thread_fence(std::memory_order_acquire);

                id_type_impl::get_deleter(p->get_management_type())(p);
            }
        }
    }    // namespace naming::detail

    ///////////////////////////////////////////////////////////////////////////
    id_type& id_type::operator++()    // pre-increment
    {
        ++(*gid_);
        return *this;
    }

    id_type id_type::operator++(int) const    // post-increment
    {
        return {(*gid_)++, management_type::unmanaged};
    }

    // comparison is required as well
    bool operator==(id_type const& lhs, id_type const& rhs) noexcept
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

    bool operator<(id_type const& lhs, id_type const& rhs) noexcept
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
}    // namespace hpx

namespace hpx::naming {

    ///////////////////////////////////////////////////////////////////////////
    inline constexpr char const* const management_type_names[] = {
        "management_type::unknown_deleter",       // -1
        "management_type::unmanaged",             // 0
        "management_type::managed",               // 1
        "management_type::managed_move_credit"    // 2
    };

    char const* get_management_type_name(
        hpx::id_type::management_type m) noexcept
    {
        if (m < hpx::id_type::management_type::unknown_deleter ||
            m > hpx::id_type::management_type::managed_move_credit)
        {
            return "invalid";
        }
        return management_type_names[static_cast<int>(m) + 1];
    }
}    // namespace hpx::naming

///////////////////////////////////////////////////////////////////////////////
namespace hpx::traits {

    hpx::id_type get_remote_result<hpx::id_type, naming::gid_type>::call(
        naming::gid_type const& rhs)
    {
        bool const has_credits = naming::detail::has_credits(rhs);
        return {rhs,
            has_credits ? hpx::id_type::management_type::managed :
                          hpx::id_type::management_type::unmanaged};
    }

    // we need to specialize this template to allow for automatic conversion of
    // the vector<naming::gid_type> to a vector<hpx::id_type>
    std::vector<hpx::id_type> get_remote_result<std::vector<hpx::id_type>,
        std::vector<naming::gid_type>>::
        call(std::vector<naming::gid_type> const& rhs)
    {
        std::vector<hpx::id_type> result;
        result.reserve(rhs.size());
        for (naming::gid_type const& r : rhs)
        {
            bool const has_credits = naming::detail::has_credits(r);
            result.emplace_back(r,
                has_credits ? hpx::id_type::management_type::managed :
                              hpx::id_type::management_type::unmanaged);
        }
        return result;
    }
}    // namespace hpx::traits
