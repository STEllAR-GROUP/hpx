//  Copyright (c) 2007-2010 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_NAMING_FULL_ADDRESS_FEB_03_2009_1124AM)
#define HPX_NAMING_FULL_ADDRESS_FEB_03_2009_1124AM

#include <hpx/runtime/naming/address.hpp>
#include <hpx/runtime/naming/name.hpp>

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
//  address serialization format version
#define HPX_FULL_ADDRESS_VERSION 0x10

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace naming
{
    class full_address 
    {
    public:
        full_address() 
        {}

        explicit full_address(id_type const& gid)
          : gid_(gid)
        {}

        full_address(id_type const& gid, address const& addr)
          : gid_(gid), address_(addr)
        {}

        full_address(boost::uint64_t msb_id, boost::uint64_t lsb_id, 
                locality const& l, components::component_type t, 
                naming::address::address_type a)
          : gid_(msb_id, lsb_id, id_type::unmanaged), address_(l, t, a)
        {}

        // local only full_address
        explicit full_address(address const& addr)
          : address_(addr)
        {}

        /// \brief Make sure the stored gid has been resolved
        bool resolve();

        operator util::safe_bool<full_address>::result_type() const 
        { 
            return util::safe_bool<full_address>()(address_); 
        }

        naming::id_type& gid() { return gid_; }
        naming::id_type const& cgid() const { return gid_; }

        naming::address& addr() { return address_; }
        naming::address const& caddr() const { return address_; }

    private:
        naming::id_type gid_;
        naming::address address_;

    private:
        friend std::ostream& operator<< (std::ostream&, full_address const&);

        // serialization support
        friend class boost::serialization::access;

        template<class Archive>
        void save(Archive & ar, const unsigned int version) const
        {
            ar << gid_ << address_; 
        }

        template<class Archive>
        void load(Archive & ar, const unsigned int version)
        {
            if (version > HPX_FULL_ADDRESS_VERSION) {
                throw exception(version_too_new, 
                    "trying to load full address with unknown version");
            }
            ar >> gid_ >> address_; 
        }

        BOOST_SERIALIZATION_SPLIT_MEMBER()
    };

    inline std::ostream& operator<< (std::ostream& os, full_address const& fa)
    {
        os << fa.gid_ << "[" << fa.address_ << "]"; 
        return os;
    }

///////////////////////////////////////////////////////////////////////////////
}}

///////////////////////////////////////////////////////////////////////////////
// this is the current version of the address serialization format
// this definition needs to be in the global namespace
BOOST_CLASS_VERSION(hpx::naming::full_address, HPX_FULL_ADDRESS_VERSION)
BOOST_CLASS_TRACKING(hpx::naming::full_address, boost::serialization::track_never)

#include <hpx/config/warnings_suffix.hpp>

#endif
