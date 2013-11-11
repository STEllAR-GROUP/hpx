//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2011 Bryce Lelbach
//  Copyright (c) 2007 Richard D. Guidry Jr.
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_NAMING_NAME_MAR_24_2008_0942AM)
#define HPX_NAMING_NAME_MAR_24_2008_0942AM

#include <ios>
#include <iomanip>
#include <iostream>

#include <boost/foreach.hpp>
#include <boost/io/ios_state.hpp>
#include <boost/cstdint.hpp>
#include <boost/serialization/version.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/intrusive_ptr.hpp>
#include <boost/detail/atomic_count.hpp>

#include <hpx/config.hpp>
#include <hpx/exception.hpp>
#include <hpx/util/safe_bool.hpp>
#include <hpx/util/spinlock_pool.hpp>
#include <hpx/util/serialize_intrusive_ptr.hpp>
#include <hpx/util/stringstream.hpp>
#include <hpx/runtime/naming/address.hpp>
#include <hpx/traits/promise_remote_result.hpp>
#include <hpx/traits/promise_local_result.hpp>

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
        ///////////////////////////////////////////////////////////////////////////
        // forward declaration
        inline boost::uint64_t strip_credit_from_gid(boost::uint64_t msb) HPX_SUPER_PURE;
    }

    ///////////////////////////////////////////////////////////////////////////
    /// Global identifier for components across the HPX system.
    struct HPX_EXPORT gid_type
    {
        // These typedefs are for Boost.ICL.
        typedef gid_type size_type;
        typedef gid_type difference_type;

        static boost::uint64_t const credit_base_mask = 0x7ffful;
        static boost::uint64_t const credit_mask = credit_base_mask << 16;
        static boost::uint64_t const was_split_mask = 0x80000000ul; //-V112
        static boost::uint64_t const locality_id_mask = 0xffffffff00000000ull;

        explicit gid_type (boost::uint64_t lsb_id = 0)
          : id_msb_(0), id_lsb_(lsb_id)
        {}

        explicit gid_type (boost::uint64_t msb_id, boost::uint64_t lsb_id)
          : id_msb_(msb_id), id_lsb_(lsb_id)
        {}

        gid_type& operator=(boost::uint64_t lsb_id)
        {
            id_msb_ = 0;
            id_lsb_ = lsb_id;
            return *this;
        }

        operator util::safe_bool<gid_type>::result_type() const
        {
            return util::safe_bool<gid_type>()(0 != id_lsb_ || 0 != id_msb_);
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
        friend gid_type operator+ (gid_type const& lhs, gid_type const& rhs)
        {
            boost::uint64_t lsb = lhs.id_lsb_ + rhs.id_lsb_;
            boost::uint64_t msb = lhs.id_msb_ + rhs.id_msb_;
            if (lsb < lhs.id_lsb_ || lsb < rhs.id_lsb_)
                ++msb;
            return gid_type(msb, lsb);
        }
        gid_type operator+= (gid_type const& rhs)
        { return (*this = *this + rhs); }

        // GID + boost::uint64_t
        friend gid_type operator+ (gid_type const& lhs, boost::uint64_t rhs)
        { return lhs + gid_type(0, rhs); }
        gid_type operator+= (boost::uint64_t rhs)
        { return (*this = *this + rhs); }

        // GID - GID
        friend gid_type operator- (gid_type const& lhs, gid_type const& rhs)
        {
            boost::uint64_t lsb = lhs.id_lsb_ - rhs.id_lsb_;
            boost::uint64_t msb = lhs.id_msb_ - rhs.id_msb_;
            if (lsb > lhs.id_lsb_)
                --msb;
            return gid_type(msb, lsb);
        }
        gid_type operator-= (gid_type const& rhs)
        { return (*this = *this - rhs); }

        // GID - boost::uint64_t
        friend gid_type operator- (gid_type const& lhs, boost::uint64_t rhs)
        { return lhs - gid_type(0, rhs); }
        gid_type operator-= (boost::uint64_t rhs)
        { return (*this = *this - rhs); }

        friend gid_type operator& (gid_type const& lhs, boost::uint64_t rhs)
        {
            return gid_type(lhs.id_msb_, lhs.id_lsb_ & rhs);
        }

        // comparison is required as well
        friend bool operator== (gid_type const& lhs, gid_type const& rhs)
        {
            return (detail::strip_credit_from_gid(lhs.id_msb_) ==
                    detail::strip_credit_from_gid(rhs.id_msb_)) &&
                (lhs.id_lsb_ == rhs.id_lsb_);
        }
        friend bool operator!= (gid_type const& lhs, gid_type const& rhs)
        {
            return !(lhs == rhs);
        }

        friend bool operator< (gid_type const& lhs, gid_type const& rhs)
        {
            if (lhs.id_msb_ < rhs.id_msb_)
                return true;
            if (lhs.id_msb_ > rhs.id_msb_)
                return false;
            return lhs.id_lsb_ < rhs.id_lsb_;
        }

        friend bool operator<= (gid_type const& lhs, gid_type const& rhs)
        {
            if (lhs.id_msb_ < rhs.id_msb_)
                return true;
            if (lhs.id_msb_ > rhs.id_msb_)
                return false;
            return lhs.id_lsb_ <= rhs.id_lsb_;
        }

        friend bool operator> (gid_type const& lhs, gid_type const& rhs)
        {
            if (lhs.id_msb_ > rhs.id_msb_)
                return true;
            if (lhs.id_msb_ < rhs.id_msb_)
                return false;
            return lhs.id_lsb_ > rhs.id_lsb_;
        }

        friend bool operator>= (gid_type const& lhs, gid_type const& rhs)
        {
            if (lhs.id_msb_ > rhs.id_msb_)
                return true;
            if (lhs.id_msb_ < rhs.id_msb_)
                return false;
            return lhs.id_lsb_ >= rhs.id_lsb_;
        }

        boost::uint64_t get_msb() const
        {
            return id_msb_;
        }
        void set_msb(boost::uint64_t msb)
        {
            id_msb_ = msb;
        }
        boost::uint64_t get_lsb() const
        {
            return id_lsb_;
        }
        void set_lsb(boost::uint64_t lsb)
        {
            id_lsb_ = lsb;
        }
        void set_lsb(void* lsb)
        {
            id_lsb_ = reinterpret_cast<boost::uint64_t>(lsb);
        }

        std::string to_string() const
        {
            hpx::util::osstream out;
            out << std::hex
                << std::right << std::setfill('0') << std::setw(16) << id_msb_
                << std::right << std::setfill('0') << std::setw(16) << id_lsb_;
            return hpx::util::osstream_get_string(out);
        }

        struct tag {};
        typedef hpx::util::spinlock_pool<tag> mutex_type;

    private:
        friend std::ostream& operator<< (std::ostream& os, gid_type const& id);

        friend class boost::serialization::access;

        template <typename Archive>
        void save(Archive& ar, const unsigned int /*version*/) const
        {
            if(ar.flags() & util::disable_array_optimization)
                ar << id_msb_ << id_lsb_;
            else
                ar.save(*this);
        }

        template <typename Archive>
        void load(Archive& ar, const unsigned int /*version*/)
        {
            if(ar.flags() & util::disable_array_optimization)
                ar >> id_msb_ >> id_lsb_;
            else
                ar.load(*this);
        }

        BOOST_SERIALIZATION_SPLIT_MEMBER()

        // actual gid
        boost::uint64_t id_msb_;
        boost::uint64_t id_lsb_;
    };
}}

namespace boost { namespace serialization
{
    ///////////////////////////////////////////////////////////////////////////
    // we know that we can serialize a gid as a byte sequence
    template <>
    struct is_bitwise_serializable<hpx::naming::gid_type>
       : boost::mpl::true_
    {};
}}

namespace hpx { namespace naming
{
    ///////////////////////////////////////////////////////////////////////////
    inline std::ostream& operator<< (std::ostream& os, gid_type const& id)
    {
        boost::io::ios_flags_saver ifs(os);
        os << std::hex
           << "{" << std::right << std::setfill('0') << std::setw(16)
                  << id.id_msb_ << ", "
                  << std::right << std::setfill('0') << std::setw(16)
                  << id.id_lsb_ << "}";
        return os;
    }

    ///////////////////////////////////////////////////////////////////////////
    //  Handle conversion to/from locality_id
    inline gid_type get_gid_from_locality_id(boost::uint32_t locality_id) HPX_SUPER_PURE;

    inline gid_type get_gid_from_locality_id(boost::uint32_t locality_id)
    {
        return gid_type(boost::uint64_t(locality_id+1) << 32, 0); //-V112
    }

    inline boost::uint32_t get_locality_id_from_gid(gid_type const& id) HPX_PURE;

    inline boost::uint32_t get_locality_id_from_gid(gid_type const& id)
    {
        return boost::uint32_t(id.get_msb() >> 32)-1; //-V112
    }

    inline gid_type get_locality_from_gid(gid_type const& id)
    {
        return get_gid_from_locality_id(get_locality_id_from_gid(id));
    }

    inline bool is_locality(gid_type const& gid)
    {
        return get_locality_from_gid(gid) == gid;
    }

    inline boost::uint64_t replace_locality_id(boost::uint64_t msb,
        boost::uint32_t locality_id) HPX_PURE;

    inline boost::uint64_t replace_locality_id(boost::uint64_t msb,
        boost::uint32_t locality_id)
    {
        msb &= ~gid_type::locality_id_mask;
        return msb | get_gid_from_locality_id(locality_id).get_msb();
    }

    inline gid_type replace_locality_id(gid_type const& gid,
        boost::uint32_t locality_id) HPX_PURE;

    inline gid_type replace_locality_id(gid_type const& gid,
        boost::uint32_t locality_id)
    {
        boost::uint64_t msb = gid.get_msb() & ~gid_type::locality_id_mask;
        msb |= get_gid_from_locality_id(locality_id).get_msb();
        return gid_type(msb, gid.get_lsb());
    }

    boost::uint32_t const invalid_locality_id = ~0U;

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        inline boost::int16_t get_credit_from_gid(gid_type const& id) HPX_PURE;

        inline boost::int16_t get_credit_from_gid(gid_type const& id)
        {
            return boost::int16_t((id.get_msb() >> 16) & gid_type::credit_base_mask);
        }

        inline void set_credit_for_gid(gid_type& id, boost::int16_t credit)
        {
            BOOST_ASSERT(0 == (credit & ~gid_type::credit_base_mask));
            id.set_msb((id.get_msb() & ~gid_type::credit_mask) |
                (boost::int32_t(credit << 16) & gid_type::credit_mask));
        }

        // has side effects, can't be pure
        inline boost::int16_t add_credit_to_gid(gid_type& id, boost::int16_t credit)
        {
            BOOST_ASSERT(credit < (std::numeric_limits<boost::int16_t>::max)());

            boost::int32_t c = get_credit_from_gid(id);

            c += credit;
            BOOST_ASSERT(0 == (c & ~gid_type::credit_base_mask));

            set_credit_for_gid(id, c);

            BOOST_ASSERT(c < (std::numeric_limits<boost::int16_t>::max)());
            return static_cast<boost::uint16_t>(c);
        }

        inline boost::int16_t remove_credit_from_gid(gid_type& id, boost::int16_t debit)
        {
            BOOST_ASSERT(debit < (std::numeric_limits<boost::int16_t>::max)());

            boost::int32_t c = get_credit_from_gid(id);
            BOOST_ASSERT(c > debit);

            c = static_cast<boost::int32_t>(c - debit);
            BOOST_ASSERT(0 == (c & ~gid_type::credit_base_mask));

            set_credit_for_gid(id, c);

            BOOST_ASSERT(c < (std::numeric_limits<boost::uint16_t>::max)());
            return static_cast<boost::uint16_t>(c);
        }

        inline boost::uint64_t strip_credit_from_gid(boost::uint64_t msb)
        {
            return msb & ~(gid_type::credit_mask | gid_type::was_split_mask);
        }

        inline gid_type& strip_credit_from_gid(gid_type& id)
        {
            id.set_msb(strip_credit_from_gid(id.get_msb()));
            return id;
        }

        inline gid_type strip_credit_from_gid(gid_type const& id) HPX_PURE;

        inline gid_type get_stripped_gid(gid_type const& id)
        {
            boost::uint64_t const msb = strip_credit_from_gid(id.get_msb());
            boost::uint64_t const lsb = id.get_lsb();
            return gid_type(msb, lsb);
        }

        inline void set_credit_split_mask_for_gid(gid_type& id)
        {
            id.set_msb(id.get_msb() | gid_type::was_split_mask);
        }

        // splits the current credit of the given id and assigns half of it to the
        // returned copy
        inline gid_type split_credits_for_gid(gid_type& id, int fraction = 2)
        {
            boost::uint64_t msb = id.get_msb();
            boost::int32_t credits = get_credit_from_gid(id);

            BOOST_ASSERT(fraction > 0);
            boost::int32_t newcredits = static_cast<boost::int32_t>(credits / fraction);

            msb &= ~gid_type::credit_mask;
            id.set_msb(msb |
                (boost::int32_t((credits - newcredits) << 16) & gid_type::credit_mask) |
                gid_type::was_split_mask);

            return gid_type(msb |
                    (boost::int32_t(newcredits << 16) & gid_type::credit_mask) |
                    gid_type::was_split_mask,
                id.get_lsb());
        }

        inline bool gid_was_split(gid_type const& id)
        {
            return (id.get_msb() & gid_type::was_split_mask) != 0;
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    gid_type const invalid_gid = gid_type();

    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        enum id_type_management
        {
            unknown_deleter = -1,
            unmanaged = 0,          // unmanaged GID
            managed = 1             // managed GID
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
        private:
            typedef void (*deleter_type)(detail::id_type_impl*);
            static deleter_type get_deleter(id_type_management t);

        public:
            id_type_impl()
              : count_(0), type_(unknown_deleter)
            {}

            explicit id_type_impl (boost::uint64_t lsb_id, id_type_management t)
              : gid_type(0, lsb_id), count_(0), type_(t)
            {}

            explicit id_type_impl (boost::uint64_t msb_id, boost::uint64_t lsb_id,
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

            // serialization
            template <typename Archive>
            void save(Archive& ar) const;

            template <typename Archive>
            void load(Archive& ar);

        private:
            // credit management (called during serialization), this function
            // has to be 'const' as save() above has to be 'const'.
            naming::gid_type preprocess_gid() const;

            void postprocess_gid();

            // reference counting
            friend HPX_EXPORT void intrusive_ptr_add_ref(id_type_impl* p);
            friend HPX_EXPORT void intrusive_ptr_release(id_type_impl* p);

            boost::detail::atomic_count count_;
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
    inline std::ostream& operator<< (std::ostream& os, id_type const& id)
    {
        os << id.get_gid();
        return os;
    }

    ///////////////////////////////////////////////////////////////////////
    // Handle conversion to/from locality_id
    // FIXME: these names are confusing, 'id' appears in identifiers far too
    // frequently.
    inline id_type get_id_from_locality_id(boost::uint32_t locality_id) HPX_SUPER_PURE;

    inline id_type get_id_from_locality_id(boost::uint32_t locality_id)
    {
        return id_type(boost::uint64_t(locality_id+1) << 32, 0, id_type::unmanaged); //-V112
    }

    inline boost::uint32_t get_locality_id_from_id(id_type const& id) HPX_PURE;

    inline boost::uint32_t get_locality_id_from_id(id_type const& id)
    {
        return boost::uint32_t(id.get_msb() >> 32) - 1; //-V112
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
            return naming::id_type(rhs, naming::id_type::managed);
        }
    };

    template <>
    struct promise_local_result<naming::gid_type>
      : boost::mpl::identity<naming::id_type>
    {};

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
            BOOST_FOREACH(naming::gid_type const& r, rhs)
            {
                result.push_back(naming::id_type(r, naming::id_type::managed));
            }
            return result;
        }
    };

    template <>
    struct promise_local_result<std::vector<naming::gid_type> >
      : boost::mpl::identity<std::vector<naming::id_type> >
    {};
}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx
{
    // pull invalid id into the mainnamespace
    using naming::invalid_id;
}

///////////////////////////////////////////////////////////////////////////////
// this is the current version of the id_type serialization format
#if defined(__GNUG__) && !defined(__INTEL_COMPILER)
#if defined(HPX_GCC_DIAGNOSTIC_PRAGMA_CONTEXTS)
#pragma GCC diagnostic push
#endif
#pragma GCC diagnostic ignored "-Wold-style-cast"
#endif

BOOST_CLASS_VERSION(hpx::naming::gid_type, HPX_GIDTYPE_VERSION)
BOOST_CLASS_TRACKING(hpx::naming::gid_type, boost::serialization::track_never)
BOOST_CLASS_VERSION(hpx::naming::id_type, HPX_IDTYPE_VERSION)
BOOST_CLASS_TRACKING(hpx::naming::id_type, boost::serialization::track_never)
BOOST_SERIALIZATION_INTRUSIVE_PTR(hpx::naming::detail::id_type_impl)

#if defined(__GNUG__) && !defined(__INTEL_COMPILER)
#if defined(HPX_GCC_DIAGNOSTIC_PRAGMA_CONTEXTS)
#pragma GCC diagnostic pop
#endif
#endif

#include <hpx/config/warnings_suffix.hpp>

#endif

