//  Copyright (c) 2007 Richard D. Guidry Jr.
//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_NAMING_NAME_MAR_24_2008_0942AM)
#define HPX_NAMING_NAME_MAR_24_2008_0942AM

#include <boost/cstdint.hpp>
#include <boost/serialization/version.hpp>
#include <boost/serialization/serialization.hpp>
#include <hpx/util/safe_bool.hpp>

///////////////////////////////////////////////////////////////////////////////
// Version of id_type
#define HPX_IDTYPE_VERSION 0x10

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace naming
{
    /// Global identifier for components across the PX system
    struct id_type : public util::safe_bool<id_type>
    {
        explicit id_type (boost::uint64_t lsb_id = 0) 
          : id_msb_(0), id_lsb_(lsb_id)
        {}

        explicit id_type (boost::uint64_t msb_id, boost::uint64_t lsb_id) 
          : id_msb_(msb_id), id_lsb_(lsb_id)
        {}

        id_type& operator++()       // pre-increment
        {
            if (boost::uint64_t(~0x0) == id_lsb_) 
                ++id_msb_;
            ++id_lsb_;
            return *this;
        }
        id_type operator++(int)     // post-increment
        {
            id_type t(*this);
            ++(*this);
            return t;
        }

        // this get's called from the safe_bool base class 
        bool operator_bool() const { return 0 != id_lsb_ || 0 != id_msb_; }

        // we support increment and addition as operators
        friend id_type operator+ (id_type const& lhs, id_type const& rhs)
        {
            boost::uint64_t lsb = lhs.id_lsb_ + rhs.id_lsb_;
            boost::uint64_t msb = lhs.id_msb_ + rhs.id_msb_;
            if (lsb < lhs.id_lsb_ && lsb < rhs.id_lsb_)
                ++msb;
            return id_type(msb, lsb);
        }

        friend id_type operator+ (id_type const& lhs, boost::uint64_t rhs)
        {
            boost::uint64_t lsb = lhs.id_lsb_ + rhs;
            boost::uint64_t msb = lhs.id_msb_;
            if (lsb < lhs.id_lsb_ && lsb < rhs)
                ++msb;
            return id_type(msb, lsb);
        }

        // comparison is required as well
        friend bool operator== (id_type const& lhs, id_type const& rhs)
        {
            return (lhs.id_msb_ == rhs.id_msb_) && (lhs.id_lsb_ == rhs.id_lsb_);
        }
        friend bool operator!= (id_type const& lhs, id_type const& rhs)
        {
            return !(lhs == rhs);
        }

        friend bool operator< (id_type const& lhs, id_type const& rhs)
        {
            if (lhs.id_msb_ < rhs.id_msb_)
                return true;
            if (lhs.id_msb_ > rhs.id_msb_)
                return false;
            return lhs.id_lsb_ < rhs.id_lsb_;
        }

        friend bool operator<= (id_type const& lhs, id_type const& rhs)
        {
            if (lhs.id_msb_ < rhs.id_msb_)
                return true;
            if (lhs.id_msb_ > rhs.id_msb_)
                return false;
            return lhs.id_lsb_ <= rhs.id_lsb_;
        }

        friend bool operator> (id_type const& lhs, id_type const& rhs)
        {
            if (lhs.id_msb_ > rhs.id_msb_)
                return true;
            if (lhs.id_msb_ < rhs.id_msb_)
                return false;
            return lhs.id_lsb_ > rhs.id_lsb_;
        }

        friend bool operator>= (id_type const& lhs, id_type const& rhs)
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
        boost::uint64_t get_lsb() const
        {
            return id_lsb_;
        }

    private:
        boost::uint64_t id_msb_;    // Type that we use for global IDs
        boost::uint64_t id_lsb_;

    private:
        friend std::ostream& operator<< (std::ostream& os, id_type const& id);

        friend class boost::serialization::access;
        template<class Archive>
        void serialize(Archive &ar, const unsigned int version) 
        {
            ar & id_msb_;
            ar & id_lsb_;
        }
    };

    inline std::ostream& operator<< (std::ostream& os, id_type const& id)
    {
        os << "(" << std::hex << id.id_msb_ << ", " 
                  << std::hex << id.id_lsb_ << ")";
        return os;
    }

    ///////////////////////////////////////////////////////////////////////////
    //  Handle conversion to/from prefix
    inline id_type get_id_from_prefix(boost::uint32_t prefix)
    {
        return id_type(boost::uint64_t(prefix) << 32, 0);
    }

    inline boost::uint32_t get_prefix_from_id(id_type const& id)
    {
        return boost::uint32_t(id.get_msb() >> 32);
    }

    inline bool is_prefix_only(id_type const& id)
    {
        return (id.get_msb() & 0xFFFFFFFFFFFFLL) ? false : true;
    }

    ///////////////////////////////////////////////////////////////////////////
    // By convention the runtime_support has a gid identical to the prefix of the 
    // locality the runtime_support is responsible for
    inline id_type get_runtime_support_id(id_type const& id)
    {
        return id_type(id.get_msb() & ~0xFFFFFFFFFFFFLL, 0);
    }

    ///////////////////////////////////////////////////////////////////////////
    id_type const invalid_id = id_type();

///////////////////////////////////////////////////////////////////////////////
}}

///////////////////////////////////////////////////////////////////////////////
// this is the current version of the id_type serialization format
BOOST_CLASS_VERSION(hpx::naming::id_type, HPX_IDTYPE_VERSION)
BOOST_CLASS_TRACKING(hpx::naming::id_type, boost::serialization::track_never)

#endif 
