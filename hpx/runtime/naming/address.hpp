//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_NAMING_ADDRESS_MAR_24_2008_0949AM)
#define HPX_NAMING_ADDRESS_MAR_24_2008_0949AM

#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/serialization/serialize.hpp>

#include <boost/io/ios_state.hpp>
#include <boost/cstdint.hpp>

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
//  address serialization format version
#define HPX_ADDRESS_VERSION 0x20

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace naming
{
    struct HPX_EXPORT address
    {
        typedef boost::int32_t component_type;
        typedef boost::uint64_t address_type;

        ///////////////////////////////////////////////////////////////////////
        address()
          : locality_(), type_(components::component_invalid), address_(0), offset_(0)
        {}

        address(gid_type const& l,
                component_type t = components::component_invalid)
          : locality_(l), type_(t), address_(0), offset_(0)
        {}

        address(gid_type const& l, component_type t, void* lva,
                boost::uint64_t offset = 0)
          : locality_(l), type_(t),
            address_(reinterpret_cast<address_type>(lva)),
            offset_(offset)
        {}

        address(gid_type const& l, component_type t, address_type a)
          : locality_(l), type_(t), address_(a), offset_(0)
        {}

        // local only addresses
        explicit address(void* lva,
                component_type t = components::component_invalid)
          : type_(t),
            address_(reinterpret_cast<address_type>(lva)), offset_(0)
        {}

        explicit address(address_type a)
          : type_(components::component_invalid), address_(a), offset_(0)
        {}

        explicit operator bool() const HPX_NOEXCEPT
        {
            return !!locality_ &&
                (components::component_invalid != type_ || 0 != address_);
        }

        friend bool operator==(address const& lhs, address const& rhs)
        {
            return lhs.type_ == rhs.type_ && lhs.address_ == rhs.address_ &&
                   lhs.locality_ == rhs.locality_;
        }

        address_type lva(naming::gid_type const& gid,
                     naming::gid_type const& gidbase) const
        {
            address_type l = address_;
            l += (gid.get_lsb() - gidbase.get_lsb()) * offset_;
            return l;
        }

        address resolve(naming::gid_type const& gid,
                    naming::gid_type const& gidbase) const
        {
            address a;
            a.locality_ = locality_;
            a.type_     = type_;
            a.address_  = a.lva(gid, gidbase);
            a.offset_   = 0;

            return a;
        }

        gid_type locality_;     /// locality: ip4 address/port number
        component_type type_;   /// component type this address is referring to
        address_type address_;  /// address (sequence number)
        address_type offset_;   /// offset

    private:
        friend std::ostream& operator<< (std::ostream&, address const&);

        // serialization support
        friend class hpx::serialization::access;

        template <typename Archive>
        void save(Archive& ar, const unsigned int version) const;

        template <typename Archive>
        void load(Archive& ar, const unsigned int version);

        HPX_SERIALIZATION_SPLIT_MEMBER();
    };

    inline std::ostream& operator<< (std::ostream& os, address const& addr)
    {
        boost::io::ios_flags_saver ifs(os);
        os << "(" << addr.locality_ << ":"
           << components::get_component_type_name(addr.type_)
           << ":" << std::showbase << std::hex << addr.address_ << ")";
        return os;
    }

///////////////////////////////////////////////////////////////////////////////
}}

HPX_IS_BITWISE_SERIALIZABLE(hpx::naming::address)

#include <hpx/config/warnings_suffix.hpp>

#endif

