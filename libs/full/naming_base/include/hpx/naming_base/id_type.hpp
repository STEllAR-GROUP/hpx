//  Copyright (c) 2007-2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/futures/traits/get_remote_result.hpp>
#include <hpx/futures/traits/promise_local_result.hpp>
#include <hpx/modules/allocator_support.hpp>
#include <hpx/modules/memory.hpp>
#include <hpx/modules/naming_base.hpp>
#include <hpx/modules/thread_support.hpp>
#include <hpx/naming_base/naming_base.hpp>

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <utility>
#include <vector>

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace naming {

    namespace detail {

        ///////////////////////////////////////////////////////////////////////
        HPX_EXPORT void intrusive_ptr_add_ref(id_type_impl* p);
        HPX_EXPORT void intrusive_ptr_release(id_type_impl* p);

        HPX_EXPORT void gid_managed_deleter(id_type_impl* p);
        HPX_EXPORT void gid_unmanaged_deleter(id_type_impl* p);
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    HPX_EXPORT std::ostream& operator<<(std::ostream& os, id_type const& id);

    ///////////////////////////////////////////////////////////////////////////
    // the local gid is actually just a wrapper around the real thing
    struct id_type
    {
    private:
        friend struct detail::id_type_impl;

    public:
        enum management_type
        {
            unknown_deleter = -1,
            unmanaged = 0,             ///< unmanaged GID
            managed = 1,               ///< managed GID
            managed_move_credit = 2    ///< managed GID which will give up all
                                       ///< credits when sent
        };

        constexpr id_type() noexcept = default;

        id_type(std::uint64_t lsb_id, management_type t);
        id_type(gid_type const& gid, management_type t);
        id_type(std::uint64_t msb_id, std::uint64_t lsb_id, management_type t);

        id_type(id_type const& o) noexcept
          : gid_(o.gid_)
        {
        }
        id_type(id_type&& o) noexcept
          : gid_(std::move(o.gid_))
        {
        }

        id_type& operator=(id_type const& o) noexcept
        {
            gid_ = o.gid_;
            return *this;
        }
        id_type& operator=(id_type&& o) noexcept
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
        friend HPX_EXPORT bool operator==(
            id_type const& lhs, id_type const& rhs);
        friend bool operator!=(id_type const& lhs, id_type const& rhs);

        friend HPX_EXPORT bool operator<(
            id_type const& lhs, id_type const& rhs);
        friend bool operator<=(id_type const& lhs, id_type const& rhs);
        friend bool operator>(id_type const& lhs, id_type const& rhs);
        friend bool operator>=(id_type const& lhs, id_type const& rhs);

        // access the internal parts of the gid
        std::uint64_t get_msb() const;
        void set_msb(std::uint64_t msb);

        std::uint64_t get_lsb() const;
        void set_lsb(std::uint64_t lsb);
        void set_lsb(void* lsb);

        // Convert this id into an unmanaged one (in-place) - Use with maximum
        // care, or better, don't use this at all.
        void make_unmanaged() const;

        hpx::intrusive_ptr<detail::id_type_impl>& impl()
        {
            return gid_;
        }
        hpx::intrusive_ptr<detail::id_type_impl> const& impl() const
        {
            return gid_;
        }

    private:
        friend HPX_EXPORT std::ostream& operator<<(
            std::ostream& os, id_type const& id);

        hpx::intrusive_ptr<detail::id_type_impl> gid_;
    };

    ///////////////////////////////////////////////////////////////////////////
    HPX_EXPORT char const* get_management_type_name(id_type::management_type m);

    ///////////////////////////////////////////////////////////////////////////
    static id_type const invalid_id = id_type();

    ///////////////////////////////////////////////////////////////////////
    // Handle conversion to/from locality_id
    // FIXME: these names are confusing, 'id' appears in identifiers far too
    // frequently.
    inline id_type get_id_from_locality_id(std::uint32_t locality_id) noexcept
    {
        return id_type(
            (std::uint64_t(locality_id) + 1) << gid_type::locality_id_shift, 0,
            id_type::unmanaged);
    }

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
    namespace detail {

        ///////////////////////////////////////////////////////////////////////
        inline void set_dont_store_in_cache(id_type& id) noexcept
        {
            id.set_msb(id.get_msb() | gid_type::dont_cache_mask);
        }

        ///////////////////////////////////////////////////////////////////////
        struct id_type_impl : gid_type
        {
        public:
            HPX_NON_COPYABLE(id_type_impl);

        private:
            using deleter_type = void (*)(detail::id_type_impl*);
            static deleter_type get_deleter(
                id_type::management_type t) noexcept;

        public:
            // This is a tag type used to convey the information that the caller is
            // _not_ going to addref the future_data instance
            struct init_no_addref
            {
            };

            // called by serialization, needs to start off with a reference
            // count of zero
            id_type_impl() noexcept
              : count_(0)
            {
            }

            explicit id_type_impl(init_no_addref, std::uint64_t lsb_id,
                id_type::management_type t) noexcept
              : gid_type(0, lsb_id)
              , count_(1)
              , type_(t)
            {
            }

            explicit id_type_impl(init_no_addref, std::uint64_t msb_id,
                std::uint64_t lsb_id, id_type::management_type t) noexcept
              : gid_type(msb_id, lsb_id)
              , count_(1)
              , type_(t)
            {
            }

            explicit id_type_impl(init_no_addref, gid_type const& gid,
                id_type::management_type t) noexcept
              : gid_type(gid)
              , count_(1)
              , type_(t)
            {
            }

            constexpr id_type::management_type get_management_type()
                const noexcept
            {
                return type_;
            }
            constexpr void set_management_type(
                id_type::management_type type) noexcept
            {
                type_ = type;
            }

            // custom allocator support
            static void* operator new(std::size_t size)
            {
                if (size != sizeof(id_type_impl))
                {
                    return ::operator new(size);
                }
                return alloc_.allocate(1);
            }

            static void operator delete(void* p, std::size_t size)
            {
                if (p == nullptr)
                {
                    return;
                }

                if (size != sizeof(id_type_impl))
                {
                    return ::operator delete(p);
                }

                return alloc_.deallocate(static_cast<id_type_impl*>(p), 1);
            }

        private:
            // custom deleter for id_type_impl
            friend HPX_EXPORT void gid_managed_deleter(id_type_impl* p);
            friend HPX_EXPORT void gid_unmanaged_deleter(id_type_impl* p);

            // reference counting
            friend HPX_EXPORT void intrusive_ptr_add_ref(id_type_impl* p);
            friend HPX_EXPORT void intrusive_ptr_release(id_type_impl* p);

            util::atomic_count count_;
            id_type::management_type type_ = id_type::unknown_deleter;

            static util::internal_allocator<id_type_impl> alloc_;
        };
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    // the local gid is actually just a wrapper around the real thing
    inline id_type::id_type(std::uint64_t lsb_id, management_type t)
      : gid_(new detail::id_type_impl(
                 detail::id_type_impl::init_no_addref{}, 0, lsb_id, t),
            false)
    {
    }

    inline id_type::id_type(gid_type const& gid, management_type t)
      : gid_(new detail::id_type_impl(
                 detail::id_type_impl::init_no_addref{}, gid, t),
            false)
    {
        if (t == unmanaged)
        {
            detail::strip_internal_bits_except_dont_cache_from_gid(*gid_);
        }
    }

    inline id_type::id_type(
        std::uint64_t msb_id, std::uint64_t lsb_id, management_type t)
      : gid_(new detail::id_type_impl(
                 detail::id_type_impl::init_no_addref{}, msb_id, lsb_id, t),
            false)
    {
        if (t == unmanaged)
        {
            detail::strip_internal_bits_except_dont_cache_from_gid(*gid_);
        }
    }

    inline gid_type& id_type::get_gid()
    {
        return *gid_;
    }
    inline gid_type const& id_type::get_gid() const
    {
        return *gid_;
    }

    // This function is used in AGAS unit tests and application code, do not
    // remove.
    inline id_type::management_type id_type::get_management_type() const
    {
        return gid_->get_management_type();
    }

    inline id_type::operator bool() const
    {
        return gid_ && *gid_;
    }

    // comparison is required as well
    inline bool operator!=(id_type const& lhs, id_type const& rhs)
    {
        return !(lhs == rhs);
    }

    inline bool operator<=(id_type const& lhs, id_type const& rhs)
    {
        // Deduced from <.
        return !(rhs < lhs);
    }

    inline bool operator>(id_type const& lhs, id_type const& rhs)
    {
        // Deduced from <.
        return rhs < lhs;
    }

    inline bool operator>=(id_type const& lhs, id_type const& rhs)
    {
        // Deduced from <.
        return !(lhs < rhs);
    }

    // access the internal parts of the gid
    inline std::uint64_t id_type::get_msb() const
    {
        return gid_->get_msb();
    }
    inline void id_type::set_msb(std::uint64_t msb)
    {
        gid_->set_msb(msb);
    }

    inline std::uint64_t id_type::get_lsb() const
    {
        return gid_->get_lsb();
    }
    inline void id_type::set_lsb(std::uint64_t lsb)
    {
        gid_->set_lsb(lsb);
    }
    inline void id_type::set_lsb(void* lsb)
    {
        gid_->set_lsb(lsb);
    }

    inline void id_type::make_unmanaged() const
    {
        gid_->set_management_type(management_type::unmanaged);
    }
}}    // namespace hpx::naming

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace traits {

    template <>
    struct get_remote_result<naming::id_type, naming::gid_type>
    {
        HPX_EXPORT static naming::id_type call(naming::gid_type const& rhs);
    };

    template <>
    struct promise_local_result<naming::gid_type>
    {
        typedef naming::id_type type;
    };

    // we need to specialize this template to allow for automatic conversion of
    // the vector<naming::gid_type> to a vector<naming::id_type>
    template <>
    struct get_remote_result<std::vector<naming::id_type>,
        std::vector<naming::gid_type>>
    {
        HPX_EXPORT static std::vector<naming::id_type> call(
            std::vector<naming::gid_type> const& rhs);
    };

    template <>
    struct promise_local_result<std::vector<naming::gid_type>>
    {
        typedef std::vector<naming::id_type> type;
    };
}}    // namespace hpx::traits

namespace hpx {

    // Pulling important types into the main namespace
    using naming::invalid_id;
}    // namespace hpx

#include <hpx/config/warnings_suffix.hpp>
