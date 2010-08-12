//  Copyright (c) 2007-2010 Hartmut Kaiser
//  Copyright (c) 2007 Richard D. Guidry Jr.
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_NAMING_NAME_MAR_24_2008_0942AM)
#define HPX_NAMING_NAME_MAR_24_2008_0942AM

#include <boost/cstdint.hpp>
#include <boost/serialization/version.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/enable_shared_from_this.hpp>

#include <hpx/util/safe_bool.hpp>
#include <hpx/util/spinlock_pool.hpp>
#include <hpx/runtime/naming/address.hpp>

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
// Version of id_type
#define HPX_IDTYPE_VERSION  0x10
#define HPX_GIDTYPE_VERSION 0x10

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace naming
{
    ///////////////////////////////////////////////////////////////////////////
    /// Global identifier for components across the PX system
    struct HPX_EXPORT gid_type
    {
        static boost::uint64_t const credit_mask = 0xffff0000ul;

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

        gid_type& operator++()       // pre-increment
        {
            if (boost::uint64_t(~0x0) == id_lsb_) 
                ++id_msb_;
            ++id_lsb_;
            return *this;
        }
        gid_type operator++(int)     // post-increment
        {
            gid_type t(*this);
            ++(*this);
            return t;
        }

        operator util::safe_bool<gid_type>::result_type() const 
        { 
            return util::safe_bool<gid_type>()(0 != id_lsb_ || 0 != id_msb_); 
        }

        // we support increment and addition as operators
        friend gid_type operator+ (gid_type const& lhs, gid_type const& rhs)
        {
            boost::uint64_t lsb = lhs.id_lsb_ + rhs.id_lsb_;
            boost::uint64_t msb = lhs.id_msb_ + rhs.id_msb_;
            if (lsb < lhs.id_lsb_ || lsb < rhs.id_lsb_)
                ++msb;
            return gid_type(msb, lsb);
        }
        gid_type operator+= (gid_type const& rhs)
        {
            boost::uint64_t lsb = id_lsb_ + rhs.id_lsb_;
            id_msb_ += rhs.id_msb_;
            if (lsb < id_lsb_ || lsb < rhs.id_lsb_)
                ++id_msb_;
            id_lsb_ = lsb;
            return *this;
        }

        friend gid_type operator+ (gid_type const& lhs, boost::uint64_t rhs)
        {
            boost::uint64_t lsb = lhs.id_lsb_ + rhs;
            boost::uint64_t msb = lhs.id_msb_;
            if (lsb < lhs.id_lsb_ || lsb < rhs)
                ++msb;
            return gid_type(msb, lsb);
        }
        gid_type operator+= (boost::uint64_t rhs)
        {
            boost::uint64_t lsb = id_lsb_ + rhs;
            if (lsb < id_lsb_ || lsb < rhs)
                ++id_msb_;
            id_lsb_ = lsb;
            return *this;
        }

        friend gid_type operator& (gid_type const& lhs, boost::uint64_t rhs)
        {
            return gid_type(lhs.id_msb_, lhs.id_lsb_ & rhs);
        }

        // comparison is required as well
        friend bool operator== (gid_type const& lhs, gid_type const& rhs)
        {
            return (lhs.id_msb_ == rhs.id_msb_) && (lhs.id_lsb_ == rhs.id_lsb_);
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

        struct tag {};
        typedef hpx::util::spinlock_pool<tag> mutex_type;

    private:
        friend std::ostream& operator<< (std::ostream& os, gid_type const& id);

        friend class boost::serialization::access;

        template<class Archive>
        void serialize(Archive & ar, const unsigned int version) 
        {
            ar & id_msb_;
            ar & id_lsb_;
        }

        // actual gid
        boost::uint64_t id_msb_;
        boost::uint64_t id_lsb_;
    };

    ///////////////////////////////////////////////////////////////////////////
    inline std::ostream& operator<< (std::ostream& os, gid_type const& id)
    {
        os << "{" << std::hex << id.id_msb_ << ", " 
                  << std::hex << id.id_lsb_ << "}";
        return os;
    }

    ///////////////////////////////////////////////////////////////////////////
    //  Handle conversion to/from prefix
    inline gid_type get_gid_from_prefix(boost::uint32_t prefix)
    {
        return gid_type(boost::uint64_t(prefix) << 32, 0);
    }

    inline boost::uint32_t get_prefix_from_gid(gid_type const& id)
    {
        return boost::uint32_t(id.get_msb() >> 32);
    }

    ///////////////////////////////////////////////////////////////////////////
    inline boost::uint16_t get_credit_from_gid(gid_type const& id)
    {
        return boost::uint16_t((id.get_msb() & gid_type::credit_mask) >> 16);
    }

    inline boost::uint16_t add_credit_to_gid(gid_type& id, boost::uint16_t credit)
    {
        boost::uint64_t msb = id.get_msb();
        boost::uint32_t c = 
            boost::uint16_t((msb & gid_type::credit_mask) >> 16) + credit;

        BOOST_ASSERT(0 == (c & ~0xffff));
        id.set_msb((msb & ~gid_type::credit_mask) | ((c & 0xffff) << 16));
        return c;
    }

    inline boost::uint64_t strip_credit_from_gid(boost::uint64_t msb)
    {
        return msb & ~gid_type::credit_mask;
    }
    inline void strip_credit_from_gid(gid_type& id)
    {
        id.set_msb(strip_credit_from_gid(id.get_msb()));
    }

    inline void set_credit_for_gid(gid_type& id, boost::uint16_t credit)
    {
        BOOST_ASSERT(0 == (credit & ~0xffff));
        id.set_msb((id.get_msb() & ~gid_type::credit_mask) | ((credit & 0xffff) << 16));
    }

    inline gid_type split_credits_for_gid(gid_type& id, int fraction = 2)
    {
        boost::uint64_t msb = id.get_msb();
        boost::uint16_t credits = boost::uint16_t((msb & gid_type::credit_mask) >> 16);
        boost::uint16_t newcredits = credits / fraction;

        msb &= ~gid_type::credit_mask;
        id.set_msb(msb | ((credits - newcredits) << 16));

        return gid_type(msb | (newcredits << 16), id.get_lsb());
    }

    ///////////////////////////////////////////////////////////////////////////
    inline bool is_local_address(gid_type const& gid, gid_type const& prefix)
    {
        return strip_credit_from_gid(gid.get_msb()) == prefix.get_msb();
    }

    ///////////////////////////////////////////////////////////////////////////
    gid_type const invalid_gid = gid_type();

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        struct HPX_EXPORT id_type_impl
          : public gid_type
        {
            explicit id_type_impl (boost::uint64_t lsb_id = 0) 
            {}

            explicit id_type_impl (boost::uint64_t msb_id, boost::uint64_t lsb_id) 
              : gid_type(msb_id, lsb_id)
            {}

            explicit id_type_impl (gid_type const& gid) 
              : gid_type(gid)
            {}

            explicit id_type_impl (boost::uint64_t msb_id, boost::uint64_t lsb_id,
                locality const& l, naming::address::component_type type, 
                naming::address::address_type a) 
              : gid_type(msb_id, lsb_id), address_(l, type, a)
            {}

            bool is_local_cached();
            bool is_cached() const;
            bool is_local();
            bool resolve(naming::address& addr);
            bool is_resolved() const { return address_; }
            bool get_local_address(naming::address& addr) 
            {
                if (!is_local_cached() && !resolve())
                    return false;
                addr = address_;
                return true;
            }

            bool get_address_cached(naming::address& addr) const
            {
                if (!is_cached())
                    return false;
                addr = address_;
                return true;
            }

            // cached resolved address
            naming::address address_;

        protected:
            bool resolve();
        };

        // custom deleter for id_type_impl above
        void gid_managed_deleter (id_type_impl* p);
        void gid_unmanaged_deleter (id_type_impl* p);
    }

    ///////////////////////////////////////////////////////////////////////////
    // the local gid is actually just a wrapper around the real thing
    struct HPX_EXPORT id_type
    {
        enum type { managed, unmanaged };

    private:
        typedef void (*deleter_type)(detail::id_type_impl*);

        static deleter_type get_deleter(type t)
        {
            return (t == managed) ? &detail::gid_managed_deleter : 
                &detail::gid_unmanaged_deleter;
        }

    public:
        id_type() {}

        explicit id_type(boost::uint64_t lsb_id, type t/* = unmanaged*/) 
          : gid_(new detail::id_type_impl(0, lsb_id), get_deleter(t))
        {}

        explicit id_type(gid_type const& gid, type t/* = unmanaged*/) 
          : gid_(new detail::id_type_impl(gid), get_deleter(t))
        {
            BOOST_ASSERT(get_credit_from_gid(*gid_) || t == unmanaged);
        }

        explicit id_type(boost::uint64_t msb_id, boost::uint64_t lsb_id, type t/* = unmanaged*/) 
          : gid_(new detail::id_type_impl(msb_id, lsb_id), get_deleter(t))
        {
            BOOST_ASSERT(get_credit_from_gid(*gid_) || t == unmanaged);
        }

        explicit id_type(boost::uint64_t msb_id, boost::uint64_t lsb_id, 
              locality const& l, naming::address::component_type type_, 
              naming::address::address_type a, type t/* = unmanaged*/) 
          : gid_(new detail::id_type_impl(msb_id, lsb_id, l, type_, a), 
                      get_deleter(t))
        {
            BOOST_ASSERT(get_credit_from_gid(*gid_) || t == unmanaged);
        }

//         explicit id_type(::gid const& rhs) 
//           : gid_(reinterpret_cast<detail::id_type_impl*>(rhs.gid_)->shared_from_this())
//         {}

        gid_type& get_gid() { return *gid_; }
        gid_type const& get_gid() const { return *gid_; }

//         id_type& operator=(::gid const& rhs)
//         {
//             gid_ = reinterpret_cast<detail::id_type_impl*>(rhs.gid_)->shared_from_this();
//             return *this;
//         }

        id_type& operator++()       // pre-increment
        {
            ++(*gid_);
            return *this;
        }
        id_type operator++(int)     // post-increment
        {
            (*gid_)++;
            return *this;
        }

        operator util::safe_bool<id_type>::result_type() const 
        { 
            return util::safe_bool<id_type>()(gid_); 
        }

        // comparison is required as well
        friend bool operator== (id_type const& lhs, id_type const& rhs)
        {
            return lhs.gid_.get() == rhs.gid_.get();
        }
        friend bool operator!= (id_type const& lhs, id_type const& rhs)
        {
            return !(lhs == rhs);
        }

        friend bool operator< (id_type const& lhs, id_type const& rhs)
        {
            return lhs.gid_.get() < rhs.gid_.get();
        }

        friend bool operator<= (id_type const& lhs, id_type const& rhs)
        {
            return lhs.gid_.get() <= rhs.gid_.get();
        }

        friend bool operator> (id_type const& lhs, id_type const& rhs)
        {
            return lhs.gid_.get() > rhs.gid_.get();
        }

        friend bool operator>= (id_type const& lhs, id_type const& rhs)
        {
            return lhs.gid_.get() >= rhs.gid_.get();
        }

        boost::uint64_t get_msb() const
        {
            return gid_->get_msb();
        }
        void set_msb(boost::uint64_t msb) 
        {
            gid_->set_msb(msb);
        }
        boost::uint64_t get_lsb() const
        {
            return gid_->get_lsb();
        }
        void set_lsb(boost::uint64_t lsb) 
        {
            gid_->set_lsb(lsb);
        }
        void set_lsb(void* lsb) 
        {
            gid_->set_lsb(lsb);
        }

        // functions for credit management
        boost::uint16_t get_credit() const
        {
            gid_type::mutex_type::scoped_lock l(gid_.get());
            return get_credit_from_gid(*gid_);
        }
        void strip_credit() const
        {
            gid_type::mutex_type::scoped_lock l(gid_.get());
            strip_credit_from_gid(*gid_);
        }
        boost::uint16_t add_credit(boost::uint16_t credit) const
        {
            gid_type::mutex_type::scoped_lock l(gid_.get());
            return add_credit_to_gid(*gid_, credit);
        }
        void set_credit(boost::uint16_t credit) const
        {
            gid_type::mutex_type::scoped_lock l(gid_.get());
            set_credit_for_gid(*gid_, credit);
        }
        id_type split_credits(int fraction = 2) const
        {
            gid_type::mutex_type::scoped_lock l(gid_.get());
            return id_type(split_credits_for_gid(*gid_, fraction), unmanaged);
        }

        ///////////////////////////////////////////////////////////////////////
        bool is_local_cached() const
        {
            return gid_->is_local_cached();
        }
        bool is_local() const
        {
            return gid_->is_local();
        }
        bool get_local_address(naming::address& addr) const
        {
            return gid_->get_local_address(addr);
        }
        bool get_address_cached(naming::address& addr) const
        {
            return gid_->get_address_cached(addr);
        }

        bool resolve(address& addr)
        {
            return gid_->resolve(addr);
        }
        bool is_resolved() const
        {
            return gid_->is_resolved();
        }

    private:
        friend std::ostream& operator<< (std::ostream& os, id_type const& id);

        friend class boost::serialization::access;

        template <class Archive>
        void save(Archive & ar, const unsigned int version) const;

        template <class Archive>
        void load(Archive & ar, const unsigned int version);

        BOOST_SERIALIZATION_SPLIT_MEMBER()

        boost::shared_ptr<detail::id_type_impl> gid_;
    };

    ///////////////////////////////////////////////////////////////////////////
    inline std::ostream& operator<< (std::ostream& os, id_type const& id)
    {
        os << id.get_gid();
        return os;
    }

    ///////////////////////////////////////////////////////////////////////
    //  Handle conversion to/from prefix
    inline id_type get_id_from_prefix(boost::uint32_t prefix)
    {
        return id_type(boost::uint64_t(prefix) << 32, 0, id_type::unmanaged);
    }

    inline boost::uint32_t get_prefix_from_id(id_type const& id)
    {
        return boost::uint32_t(id.get_msb() >> 32);
    }

    ///////////////////////////////////////////////////////////////////////
    inline bool is_local_address(id_type const& gid, id_type const& prefix)
    {
        return strip_credit_from_gid(gid.get_msb()) == prefix.get_msb();
    }

    ///////////////////////////////////////////////////////////////////////
    id_type const invalid_id = id_type();

///////////////////////////////////////////////////////////////////////////////
}}

///////////////////////////////////////////////////////////////////////////////
// this is the current version of the id_type serialization format
BOOST_CLASS_VERSION(hpx::naming::gid_type, HPX_GIDTYPE_VERSION)
BOOST_CLASS_TRACKING(hpx::naming::gid_type, boost::serialization::track_never)
BOOST_CLASS_VERSION(hpx::naming::id_type, HPX_IDTYPE_VERSION)
BOOST_CLASS_TRACKING(hpx::naming::id_type, boost::serialization::track_never)

#include <hpx/config/warnings_suffix.hpp>

#endif 
