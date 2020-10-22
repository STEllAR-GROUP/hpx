//  Copyright (c) 2007-2020 Hartmut Kaiser
//  Copyright (c) 2011 Bryce Lelbach
//  Copyright (c) 2007 Richard D. Guidry Jr.
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/allocator_support/internal_allocator.hpp>
#include <hpx/assert.hpp>
#include <hpx/execution_base/register_locks.hpp>
#include <hpx/execution_base/this_thread.hpp>
#include <hpx/concurrency/spinlock_pool.hpp>
#include <hpx/futures/traits/get_remote_result.hpp>
#include <hpx/futures/traits/promise_local_result.hpp>
#include <hpx/modules/itt_notify.hpp>
#include <hpx/runtime/naming/id_type.hpp>
#include <hpx/modules/naming.hpp>
#include <hpx/serialization/serialization_fwd.hpp>
#include <hpx/serialization/traits/is_bitwise_serializable.hpp>
#include <hpx/thread_support/atomic_count.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <iosfwd>
#include <mutex>
#include <string>
#include <vector>

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
// Version of id_type
#define HPX_IDTYPE_VERSION  0x20

namespace hpx { namespace naming
{
    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        inline void set_dont_store_in_cache(id_type& id) noexcept
        {
            id.set_msb(id.get_msb() | gid_type::dont_cache_mask);
        }

        ///////////////////////////////////////////////////////////////////////
        enum id_type_management
        {
            unknown_deleter = -1,
            unmanaged = 0,          // unmanaged GID
            managed = 1,            // managed GID
            managed_move_credit = 2 // managed GID which will give up all
                                    // credits when sent
        };

        // forward declaration
        struct HPX_EXPORT id_type_impl;

        // custom deleter for id_type_impl above
        HPX_EXPORT void gid_managed_deleter (id_type_impl* p);
        HPX_EXPORT void gid_unmanaged_deleter (id_type_impl* p);

        HPX_EXPORT void intrusive_ptr_add_ref(id_type_impl* p);
        HPX_EXPORT void intrusive_ptr_release(id_type_impl* p);

        ///////////////////////////////////////////////////////////////////////
        struct HPX_EXPORT id_type_impl : gid_type
        {
        public:
            HPX_NON_COPYABLE(id_type_impl);

        private:
            using deleter_type = void (*)(detail::id_type_impl*);
            static deleter_type get_deleter(id_type_management t) noexcept;

        public:
            // This is a tag type used to convey the information that the caller is
            // _not_ going to addref the future_data instance
            struct init_no_addref {};

            // called by serialization, needs to start off with a reference
            // count of zero
            id_type_impl() noexcept
              : count_(0), type_(unknown_deleter)
            {}

            explicit id_type_impl(init_no_addref,
                    std::uint64_t lsb_id, id_type_management t) noexcept
              : gid_type(0, lsb_id)
              , count_(1)
              , type_(t)
            {}

            explicit id_type_impl(init_no_addref, std::uint64_t msb_id,
                    std::uint64_t lsb_id, id_type_management t) noexcept
              : gid_type(msb_id, lsb_id)
              , count_(1)
              , type_(t)
            {}

            explicit id_type_impl(init_no_addref, gid_type const& gid,
                    id_type_management t) noexcept
              : gid_type(gid)
              , count_(1)
              , type_(t)
            {}

            id_type_management get_management_type() const noexcept
            {
                return type_;
            }
            void set_management_type(id_type_management type) noexcept
            {
                type_ = type;
            }

            // serialization
            void save(serialization::output_archive& ar, unsigned) const;

            void load(serialization::input_archive& ar, unsigned);

            HPX_SERIALIZATION_SPLIT_MEMBER()

            // custom allocator support
            static void* operator new(std::size_t size)
            {
                if (size != sizeof(id_type_impl))
                {
                    return ::operator new (size);
                }
                return alloc_.allocate(1);
            }

            static void operator delete(void *p, std::size_t size)
            {
                if (p == nullptr)
                {
                    return;
                }

                if (size != sizeof(id_type_impl))
                {
                    return ::operator delete (p);
                }

                return alloc_.deallocate(static_cast<id_type_impl*>(p), 1);
            }

        private:
            // credit management (called during serialization), this function
            // has to be 'const' as save() above has to be 'const'.
            void preprocess_gid(serialization::output_archive& ar) const;

            // reference counting
            friend HPX_EXPORT void intrusive_ptr_add_ref(id_type_impl* p);
            friend HPX_EXPORT void intrusive_ptr_release(id_type_impl* p);

            util::atomic_count count_;
            id_type_management type_;

            static util::internal_allocator<id_type_impl> alloc_;
        };
    }
}}

#include <hpx/runtime/naming/id_type.hpp>
#include <hpx/runtime/naming/id_type_impl.hpp>

namespace hpx { namespace naming
{
    ///////////////////////////////////////////////////////////////////////////
    HPX_EXPORT std::ostream& operator<<(std::ostream& os, id_type const& id);

    ///////////////////////////////////////////////////////////////////////
    // Handle conversion to/from locality_id
    // FIXME: these names are confusing, 'id' appears in identifiers far too
    // frequently.
    inline id_type get_id_from_locality_id(
        std::uint32_t locality_id) noexcept HPX_SUPER_PURE;

    inline id_type get_id_from_locality_id(std::uint32_t locality_id) noexcept
    {
        return id_type(
            (std::uint64_t(locality_id)+1) << gid_type::locality_id_shift,
            0, id_type::unmanaged);
    }

    inline std::uint32_t get_locality_id_from_id(
        id_type const& id) noexcept HPX_PURE;

    inline std::uint32_t get_locality_id_from_id(id_type const& id) noexcept
    {
        return std::uint32_t(id.get_msb() >> gid_type::locality_id_shift) - 1;
    }

    inline id_type get_locality_from_id(id_type const& id) noexcept
    {
        return get_id_from_locality_id(get_locality_id_from_id(id));
    }

    inline bool is_locality(id_type const& id) noexcept
    {
        return is_locality(id.get_gid());
    }

    ///////////////////////////////////////////////////////////////////////////
    HPX_EXPORT char const* get_management_type_name(id_type::management_type m);
}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace traits
{
    template <>
    struct get_remote_result<naming::id_type, naming::gid_type>
    {
        static naming::id_type call(naming::gid_type const& rhs)
        {
            bool has_credits = naming::detail::has_credits(rhs);
            return naming::id_type(rhs,
                has_credits ?
                    naming::id_type::managed :
                    naming::id_type::unmanaged);
        }
    };

    template <>
    struct promise_local_result<naming::gid_type>
    {
        typedef naming::id_type type;
    };

    // we need to specialize this template to allow for automatic conversion of
    // the vector<naming::gid_type> to a vector<naming::id_type>
    template <>
    struct get_remote_result<
        std::vector<naming::id_type>, std::vector<naming::gid_type> >
    {
        static std::vector<naming::id_type>
        call(std::vector<naming::gid_type> const& rhs)
        {
            std::vector<naming::id_type> result;
            result.reserve(rhs.size());
            for (naming::gid_type const& r : rhs)
            {
                bool has_credits = naming::detail::has_credits(r);
                result.push_back(naming::id_type(r,
                    has_credits ?
                        naming::id_type::managed :
                        naming::id_type::unmanaged));
            }
            return result;
        }
    };

    template <>
    struct promise_local_result<std::vector<naming::gid_type> >
    {
        typedef std::vector<naming::id_type> type;
    };
}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx
{
    // pull invalid id into the main namespace
    using naming::invalid_id;
}

#include <hpx/config/warnings_suffix.hpp>

