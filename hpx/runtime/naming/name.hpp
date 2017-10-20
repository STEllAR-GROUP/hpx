//  Copyright (c) 2007-2016 Hartmut Kaiser
//  Copyright (c) 2011 Bryce Lelbach
//  Copyright (c) 2007 Richard D. Guidry Jr.
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_RUNTIME_NAMING_NAME_HPP
#define HPX_RUNTIME_NAMING_NAME_HPP

#include <hpx/config.hpp>
#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/lcos/local/spinlock_pool.hpp>
#include <hpx/runtime/naming/id_type.hpp>
#include <hpx/runtime/naming_fwd.hpp>
#include <hpx/runtime/serialization/serialization_fwd.hpp>
#include <hpx/traits/get_remote_result.hpp>
#include <hpx/traits/is_bitwise_serializable.hpp>
#include <hpx/traits/promise_local_result.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/atomic_count.hpp>
#include <hpx/util/itt_notify.hpp>
#include <hpx/util/register_locks.hpp>

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
#define HPX_GIDTYPE_VERSION 0x10

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace naming
{
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        // forward declaration
        inline std::uint64_t strip_internal_bits_from_gid(std::uint64_t msb)
            HPX_SUPER_PURE;

        inline std::uint64_t strip_lock_from_gid(std::uint64_t msb)
            HPX_SUPER_PURE;

        inline std::uint64_t get_internal_bits(std::uint64_t msb)
            HPX_SUPER_PURE;

        inline std::uint64_t strip_internal_bits_and_locality_from_gid(
                std::uint64_t msb) HPX_SUPER_PURE;

        inline bool is_locked(gid_type const& gid);
    }

    ///////////////////////////////////////////////////////////////////////////
    /// Global identifier for components across the HPX system.
    struct HPX_EXPORT gid_type
    {
        struct tag {};

        // These typedefs are for Boost.ICL.
        typedef gid_type size_type;
        typedef gid_type difference_type;

        static std::uint64_t const credit_base_mask = 0x1full;
        static std::uint16_t const credit_shift = 24;

        static std::uint64_t const credit_mask = credit_base_mask << credit_shift;
        static std::uint64_t const was_split_mask = 0x80000000ull; //-V112
        static std::uint64_t const has_credits_mask = 0x40000000ull; //-V112
        static std::uint64_t const is_locked_mask = 0x20000000ull; //-V112

        static std::uint64_t const locality_id_mask = 0xffffffff00000000ull;
        static std::uint16_t const locality_id_shift = 32; //-V112

        static std::uint64_t const virtual_memory_mask = 0x7fffffull;

        // don't cache this id in the AGAS caches
        static std::uint64_t const dont_cache_mask = 0x800000ull; //-V112

        static std::uint64_t const credit_bits_mask =
            credit_mask | was_split_mask | has_credits_mask;
        static std::uint64_t const internal_bits_mask =
            credit_bits_mask | is_locked_mask | dont_cache_mask;
        static std::uint64_t const special_bits_mask =
            locality_id_mask | internal_bits_mask;

        explicit gid_type (std::uint64_t lsb_id = 0)
          : id_msb_(0), id_lsb_(lsb_id)
        {}

        explicit gid_type (std::uint64_t msb_id, std::uint64_t lsb_id)
          : id_msb_(naming::detail::strip_lock_from_gid(msb_id)),
            id_lsb_(lsb_id)
        {
        }

        gid_type (gid_type const& rhs)
          : id_msb_(naming::detail::strip_lock_from_gid(rhs.get_msb())),
            id_lsb_(rhs.get_lsb())
        {
        }
        gid_type (gid_type && rhs)
          : id_msb_(naming::detail::strip_lock_from_gid(rhs.get_msb())),
            id_lsb_(rhs.get_lsb())
        {
            rhs.id_lsb_ = rhs.id_msb_ = 0;
        }

        ~gid_type()
        {
            HPX_ASSERT(!is_locked());
        }

        gid_type& operator=(std::uint64_t lsb_id)
        {
            HPX_ASSERT(!is_locked());
            id_msb_ = 0;
            id_lsb_ = lsb_id;
            return *this;
        }

        gid_type& operator=(gid_type const& rhs)
        {
            if (this != &rhs)
            {
                HPX_ASSERT(!is_locked());
                id_msb_ = naming::detail::strip_lock_from_gid(rhs.get_msb());
                id_lsb_ = rhs.get_lsb();
            }
            return *this;
        }
        gid_type& operator=(gid_type && rhs)
        {
            if (this != &rhs)
            {
                HPX_ASSERT(!is_locked());
                id_msb_ = naming::detail::strip_lock_from_gid(rhs.get_msb());
                id_lsb_ = rhs.get_lsb();

                rhs.id_lsb_ = rhs.id_msb_ = 0;
            }
            return *this;
        }

        explicit operator bool() const noexcept
        {
            return 0 != id_lsb_ || 0 != id_msb_;
        }

        // We support increment, decrement, addition and subtraction
        gid_type& operator++()       // pre-increment
        {
            *this += 1;
            return *this;
        }
        gid_type operator++(int)     // post-increment
        {
            gid_type t(*this);
            ++(*this);
            return t;
        }

        gid_type& operator--()       // pre-decrement
        {
            *this -= 1;
            return *this;
        }
        gid_type operator--(int)     // post-decrement
        {
            gid_type t(*this);
            --(*this);
            return t;
        }

        // GID + GID
        friend HPX_EXPORT gid_type operator+ (
            gid_type const& lhs, gid_type const& rhs);
        gid_type operator+= (gid_type const& rhs)
        { return (*this = *this + rhs); }

        // GID + std::uint64_t
        friend gid_type operator+ (gid_type const& lhs, std::uint64_t rhs)
        { return lhs + gid_type(0, rhs); }
        gid_type operator+= (std::uint64_t rhs)
        { return (*this = *this + rhs); }

        // GID - GID
        friend HPX_EXPORT gid_type operator- (gid_type const& lhs,
            gid_type const& rhs);
        gid_type operator-= (gid_type const& rhs)
        { return (*this = *this - rhs); }

        // GID - std::uint64_t
        friend gid_type operator- (gid_type const& lhs, std::uint64_t rhs)
        { return lhs - gid_type(0, rhs); }
        gid_type operator-= (std::uint64_t rhs)
        { return (*this = *this - rhs); }

        friend gid_type operator& (gid_type const& lhs, std::uint64_t rhs)
        {
            return gid_type(lhs.id_msb_, lhs.id_lsb_ & rhs);
        }

        // comparison is required as well
        friend bool operator== (gid_type const& lhs, gid_type const& rhs)
        {
            return (detail::strip_internal_bits_from_gid(lhs.id_msb_) ==
                    detail::strip_internal_bits_from_gid(rhs.id_msb_)) &&
                (lhs.id_lsb_ == rhs.id_lsb_);
        }
        friend bool operator!= (gid_type const& lhs, gid_type const& rhs)
        {
            return !(lhs == rhs);
        }

        friend bool operator< (gid_type const& lhs, gid_type const& rhs)
        {
            if (detail::strip_internal_bits_from_gid(lhs.id_msb_) <
                detail::strip_internal_bits_from_gid(rhs.id_msb_))
            {
                return true;
            }
            if (detail::strip_internal_bits_from_gid(lhs.id_msb_) >
                detail::strip_internal_bits_from_gid(rhs.id_msb_))
            {
                return false;
            }
            return lhs.id_lsb_ < rhs.id_lsb_;
        }

        friend bool operator<= (gid_type const& lhs, gid_type const& rhs)
        {
            if (detail::strip_internal_bits_from_gid(lhs.id_msb_) <
                detail::strip_internal_bits_from_gid(rhs.id_msb_))
            {
                return true;
            }
            if (detail::strip_internal_bits_from_gid(lhs.id_msb_) >
                detail::strip_internal_bits_from_gid(rhs.id_msb_))
            {
                return false;
            }
            return lhs.id_lsb_ <= rhs.id_lsb_;
        }

        friend bool operator> (gid_type const& lhs, gid_type const& rhs)
        {
            return !(lhs <= rhs);
        }

        friend bool operator>= (gid_type const& lhs, gid_type const& rhs)
        {
            return !(lhs < rhs);
        }

        std::uint64_t get_msb() const
        {
            return id_msb_;
        }
        void set_msb(std::uint64_t msb)
        {
            id_msb_ = msb;
        }
        std::uint64_t get_lsb() const
        {
            return id_lsb_;
        }
        void set_lsb(std::uint64_t lsb)
        {
            id_lsb_ = lsb;
        }
        void set_lsb(void* lsb)
        {
            id_lsb_ = reinterpret_cast<std::uint64_t>(lsb);
        }

        std::string to_string() const;

        // this type is at the same time its own mutex type
        typedef gid_type mutex_type;

        // Note: we deliberately don't register this lock with the lock
        //       tracking to avoid false positives. We know that gid_types need
        //       to be locked while suspension.
        void lock()
        {
            HPX_ITT_SYNC_PREPARE(this);

            for (std::size_t k = 0; !acquire_lock(); ++k)
            {
                lcos::local::spinlock::yield(k);
            }

            util::register_lock(this);

            HPX_ITT_SYNC_ACQUIRED(this);
        }

        bool try_lock()
        {
            HPX_ITT_SYNC_PREPARE(this);

            if (acquire_lock())
            {
                HPX_ITT_SYNC_ACQUIRED(this);
                util::register_lock(this);
                return true;
            }

            HPX_ITT_SYNC_CANCEL(this);
            return false;
        }

        void unlock()
        {
            HPX_ITT_SYNC_RELEASING(this);

            relinquish_lock();
            util::unregister_lock(this);

            HPX_ITT_SYNC_RELEASED(this);
        }

        mutex_type& get_mutex() const { return const_cast<mutex_type&>(*this); }

    private:
        friend HPX_EXPORT std::ostream& operator<<(std::ostream& os,
            gid_type const& id);

        friend class hpx::serialization::access;

        void save(serialization::output_archive& ar, unsigned int version) const;

        void load(serialization::input_archive& ar, unsigned int version);

        HPX_SERIALIZATION_SPLIT_MEMBER()

        // lock implementation
        typedef lcos::local::spinlock_pool<tag> internal_mutex_type;

        // returns whether lock has been acquired
        bool acquire_lock()
        {
            internal_mutex_type::scoped_lock l(this);
            bool was_locked = (id_msb_ & is_locked_mask) ? true : false;
            if (!was_locked)
            {
                id_msb_ |= is_locked_mask;
                return true;
            }
            return false;
        }

        void relinquish_lock()
        {
            util::ignore_lock(this);
            internal_mutex_type::scoped_lock l(this);
            util::reset_ignored(this);

            id_msb_ &= ~is_locked_mask;
        }

        // this is used for assertions only, no need to acquire the lock
        bool is_locked() const
        {
            return (id_msb_ & is_locked_mask) ? true : false;
        }

        friend bool detail::is_locked(gid_type const& gid);

        // actual gid
        std::uint64_t id_msb_;
        std::uint64_t id_lsb_;
    };
}}

///////////////////////////////////////////////////////////////////////////////
// we know that we can serialize a gid as a byte sequence
HPX_IS_BITWISE_SERIALIZABLE(hpx::naming::gid_type)

namespace hpx { namespace naming
{
    ///////////////////////////////////////////////////////////////////////////
    //  Handle conversion to/from locality_id
    inline gid_type get_gid_from_locality_id(std::uint32_t locality_id)
        HPX_SUPER_PURE;

    inline gid_type get_gid_from_locality_id(std::uint32_t locality_id)
    {
        return gid_type(
            std::uint64_t(locality_id+1) << gid_type::locality_id_shift,
            0);
    }

    inline std::uint32_t get_locality_id_from_gid(std::uint64_t msb) HPX_PURE;

    inline std::uint32_t get_locality_id_from_gid(std::uint64_t msb)
    {
        return std::uint32_t(msb >> gid_type::locality_id_shift) - 1;
    }

    inline std::uint32_t get_locality_id_from_gid(gid_type const& id) HPX_PURE;

    inline std::uint32_t get_locality_id_from_gid(gid_type const& id)
    {
        return get_locality_id_from_gid(id.get_msb());
    }

    inline gid_type get_locality_from_gid(gid_type const& id)
    {
        return get_gid_from_locality_id(get_locality_id_from_gid(id));
    }

    inline bool is_locality(gid_type const& gid)
    {
        return get_locality_from_gid(gid) == gid;
    }

    inline std::uint64_t replace_locality_id(std::uint64_t msb,
        std::uint32_t locality_id) HPX_PURE;

    inline std::uint64_t replace_locality_id(std::uint64_t msb,
        std::uint32_t locality_id)
    {
        msb &= ~gid_type::locality_id_mask;
        return msb | get_gid_from_locality_id(locality_id).get_msb();
    }

    inline gid_type replace_locality_id(gid_type const& gid,
        std::uint32_t locality_id) HPX_PURE;

    inline gid_type replace_locality_id(gid_type const& gid,
        std::uint32_t locality_id)
    {
        std::uint64_t msb = gid.get_msb() & ~gid_type::locality_id_mask;
        msb |= get_gid_from_locality_id(locality_id).get_msb();
        return gid_type(msb, gid.get_lsb());
    }

    ///////////////////////////////////////////////////////////////////////////
    inline bool refers_to_virtual_memory(gid_type const& gid)
    {
        return !(gid.get_msb() & gid_type::virtual_memory_mask);
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        // We store the log2(credit) in the gid_type
        inline std::int16_t log2(std::int64_t val)
        {
            std::int16_t ret = -1;
            while (val != 0)
            {
                val >>= 1;
                ++ret;
            }
            return ret;
        }

        inline std::int64_t power2(std::int16_t log2credits)
        {
            HPX_ASSERT(log2credits >= 0);
            return static_cast<std::int64_t>(1) << log2credits;
        }

        ///////////////////////////////////////////////////////////////////////
        inline bool has_credits(gid_type const& id)
        {
            return (id.get_msb() & gid_type::has_credits_mask) ? true : false;
        }

        inline bool gid_was_split(gid_type const& id)
        {
            return (id.get_msb() & gid_type::was_split_mask) ? true : false;
        }

        inline void set_credit_split_mask_for_gid(gid_type& id)
        {
            id.set_msb(id.get_msb() | gid_type::was_split_mask);
        }

        ///////////////////////////////////////////////////////////////////////
        inline bool store_in_cache(gid_type const& id)
        {
            return (id.get_msb() & gid_type::dont_cache_mask) ? false : true;
        }

        inline void set_dont_store_in_cache(gid_type& gid)
        {
            gid.set_msb(gid.get_msb() | gid_type::dont_cache_mask);
        }

        inline void set_dont_store_in_cache(id_type& id)
        {
            id.set_msb(id.get_msb() | gid_type::dont_cache_mask);
        }

        ///////////////////////////////////////////////////////////////////////
        inline std::int64_t get_credit_from_gid(gid_type const& id) HPX_PURE;

        inline std::int16_t get_log2credit_from_gid(gid_type const& id)
        {
            HPX_ASSERT(has_credits(id));
            return std::int16_t((id.get_msb() >> gid_type::credit_shift) &
                    gid_type::credit_base_mask);
        }

        inline std::int64_t get_credit_from_gid(gid_type const& id)
        {
            return has_credits(id) ? detail::power2(get_log2credit_from_gid(id)) : 0;
        }

        ///////////////////////////////////////////////////////////////////////
        inline std::uint64_t strip_internal_bits_from_gid(std::uint64_t msb)
        {
            return msb & ~gid_type::internal_bits_mask;
        }

        inline gid_type& strip_internal_bits_from_gid(gid_type& id)
        {
            id.set_msb(strip_internal_bits_from_gid(id.get_msb()));
            return id;
        }

        inline std::uint64_t strip_internal_bits_except_dont_cache_from_gid(
            std::uint64_t msb)
        {
            return msb & ~(gid_type::credit_bits_mask | gid_type::is_locked_mask);
        }

        inline gid_type& strip_internal_bits_except_dont_cache_from_gid(
            gid_type& id)
        {
            id.set_msb(
                strip_internal_bits_except_dont_cache_from_gid(id.get_msb()));
            return id;
        }

        inline std::uint64_t get_internal_bits(std::uint64_t msb)
        {
            return msb & gid_type::internal_bits_mask;
        }

        inline std::uint64_t strip_internal_bits_and_locality_from_gid(
                std::uint64_t msb)
        {
            return msb & ~gid_type::special_bits_mask;
        }

        ///////////////////////////////////////////////////////////////////////
        inline std::uint64_t strip_lock_from_gid(std::uint64_t msb)
        {
            return msb & ~gid_type::is_locked_mask;
        }

        inline gid_type& strip_lock_from_gid(gid_type& gid)
        {
            gid.set_msb(strip_lock_from_gid(gid.get_msb()));
            return gid;
        }

        inline bool is_locked(gid_type const& gid)
        {
            return gid.is_locked();
        }

        ///////////////////////////////////////////////////////////////////////
        inline gid_type get_stripped_gid(gid_type const& id) HPX_PURE;

        inline gid_type get_stripped_gid(gid_type const& id)
        {
            std::uint64_t const msb = strip_internal_bits_from_gid(id.get_msb());
            std::uint64_t const lsb = id.get_lsb();
            return gid_type(msb, lsb);
        }

        inline gid_type get_stripped_gid_except_dont_cache(gid_type const& id)
            HPX_PURE;

        inline gid_type get_stripped_gid_except_dont_cache(gid_type const& id)
        {
            std::uint64_t const msb =
                strip_internal_bits_except_dont_cache_from_gid(id.get_msb());
            std::uint64_t const lsb = id.get_lsb();
            return gid_type(msb, lsb);
        }

        inline std::uint64_t strip_credits_from_gid(std::uint64_t msb)
        {
            return msb & ~gid_type::credit_bits_mask;
        }

        inline gid_type& strip_credits_from_gid(gid_type& id)
        {
            id.set_msb(strip_credits_from_gid(id.get_msb()));
            return id;
        }

        ///////////////////////////////////////////////////////////////////////
        inline void set_log2credit_for_gid(gid_type& id, std::int16_t log2credits)
        {
            // credit should be a clean log2
            HPX_ASSERT(log2credits >= 0);
            HPX_ASSERT(0 == (log2credits & ~gid_type::credit_base_mask));

            id.set_msb((id.get_msb() & ~gid_type::credit_mask) |
                ((std::int32_t(log2credits) << gid_type::credit_shift)
                    & gid_type::credit_mask) |
                gid_type::has_credits_mask);
        }

        inline void set_credit_for_gid(gid_type& id, std::int64_t credits)
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
        HPX_EXPORT std::int64_t add_credit_to_gid(gid_type& id,
            std::int64_t credits);

        HPX_EXPORT std::int64_t remove_credit_from_gid(gid_type& id,
            std::int64_t debit);

        HPX_EXPORT std::int64_t fill_credit_for_gid(gid_type& id,
            std::int64_t credits = std::int64_t(HPX_GLOBALCREDIT_INITIAL));

        ///////////////////////////////////////////////////////////////////////
        HPX_EXPORT gid_type move_gid(gid_type& id);
        HPX_EXPORT gid_type move_gid_locked(std::unique_lock<gid_type::mutex_type> l,
            gid_type& gid);

        HPX_EXPORT std::int64_t replenish_credits(gid_type& id);
        HPX_EXPORT std::int64_t replenish_credits_locked(
            std::unique_lock<gid_type::mutex_type>& l, gid_type& id);

        ///////////////////////////////////////////////////////////////////////
        // splits the current credit of the given id and assigns half of it to
        // the returned copy
        HPX_EXPORT gid_type split_credits_for_gid(gid_type& id);
        HPX_EXPORT gid_type split_credits_for_gid_locked(
            std::unique_lock<gid_type::mutex_type>& l, gid_type& id);
    }

    HPX_EXPORT gid_type operator+ (gid_type const& lhs, gid_type const& rhs);
    HPX_EXPORT gid_type operator- (gid_type const& lhs, gid_type const& rhs);

    ///////////////////////////////////////////////////////////////////////////
    gid_type const invalid_gid = gid_type();

    ///////////////////////////////////////////////////////////////////////////
    HPX_EXPORT std::ostream& operator<<(std::ostream& os, gid_type const& id);

    namespace detail
    {
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
            typedef void (*deleter_type)(detail::id_type_impl*);
            static deleter_type get_deleter(id_type_management t);

        public:
            id_type_impl()
              : count_(0), type_(unknown_deleter)
            {}

            explicit id_type_impl (std::uint64_t lsb_id, id_type_management t)
              : gid_type(0, lsb_id), count_(0), type_(t)
            {}

            explicit id_type_impl (std::uint64_t msb_id, std::uint64_t lsb_id,
                    id_type_management t)
              : gid_type(msb_id, lsb_id), count_(0), type_(t)
            {}

            explicit id_type_impl (gid_type const& gid, id_type_management t)
              : gid_type(gid), count_(0), type_(t)
            {}

            id_type_management get_management_type() const
            {
                return type_;
            }
            void set_management_type(id_type_management type)
            {
                type_ = type;
            }

            // serialization
            void save(serialization::output_archive& ar, unsigned) const;

            void load(serialization::input_archive& ar, unsigned);

            HPX_SERIALIZATION_SPLIT_MEMBER()

        private:
            // credit management (called during serialization), this function
            // has to be 'const' as save() above has to be 'const'.
            void preprocess_gid(serialization::output_archive& ar) const;

            // reference counting
            friend HPX_EXPORT void intrusive_ptr_add_ref(id_type_impl* p);
            friend HPX_EXPORT void intrusive_ptr_release(id_type_impl* p);

            util::atomic_count count_;
            id_type_management type_;
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    HPX_API_EXPORT gid_type get_parcel_dest_gid(id_type const& id);
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
    inline id_type get_id_from_locality_id(std::uint32_t locality_id) HPX_SUPER_PURE;

    inline id_type get_id_from_locality_id(std::uint32_t locality_id)
    {
        return id_type(
            std::uint64_t(locality_id+1) << gid_type::locality_id_shift,
            0, id_type::unmanaged);
    }

    inline std::uint32_t get_locality_id_from_id(id_type const& id) HPX_PURE;

    inline std::uint32_t get_locality_id_from_id(id_type const& id)
    {
        return std::uint32_t(id.get_msb() >> gid_type::locality_id_shift) - 1;
    }

    inline id_type get_locality_from_id(id_type const& id)
    {
        return get_id_from_locality_id(get_locality_id_from_id(id));
    }

    inline bool is_locality(id_type const& id)
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

///////////////////////////////////////////////////////////////////////////////
namespace std
{
    // specialize std::hash for hpx::naming::gid_type
    template <>
    struct hash<hpx::naming::gid_type>
    {
        typedef hpx::naming::gid_type argument_type;
        typedef std::size_t result_type;

        result_type operator()(argument_type const& gid) const
        {
            result_type const h1 (std::hash<std::uint64_t>()(gid.get_lsb()));
            result_type const h2 (std::hash<std::uint64_t>()(
                hpx::naming::detail::strip_internal_bits_from_gid(gid.get_msb())));
            return h1 ^ (h2 << 1);
        }
    };
}

#include <hpx/config/warnings_suffix.hpp>

#endif /*HPX_RUNTIME_NAMING_NAME_HPP*/
