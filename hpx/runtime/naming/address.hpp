//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_NAMING_ADDRESS_MAR_24_2008_0949AM)
#define HPX_NAMING_ADDRESS_MAR_24_2008_0949AM

#include <boost/io/ios_state.hpp>
#include <boost/cstdint.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/version.hpp>

#include <hpx/runtime/naming/locality.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/util/safe_bool.hpp>

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
          : locality_(), type_(components::component_invalid), address_(0)
        {}

        address(locality const& l,
                component_type t = components::component_invalid)
          : locality_(l), type_(t), address_(0)
        {}

        address(locality const& l, component_type t, void* lva)
          : locality_(l), type_(t),
            address_(reinterpret_cast<address_type>(lva))
        {}

        address(locality const& l, component_type t, address_type a)
          : locality_(l), type_(t), address_(a)
        {}

        // local only addresses
        address(void* lva)
          : locality_(), type_(components::component_invalid),
            address_(reinterpret_cast<address_type>(lva))
        {}

        address(address_type a)
          : locality_(), type_(components::component_invalid), address_(a)
        {}

        // safe operator bool()
        operator util::safe_bool<address>::result_type() const
        {
            return util::safe_bool<address>()(!!locality_ &&
                (components::component_invalid != type_ || 0 != address_));
        }

        friend bool operator==(address const& lhs, address const& rhs)
        {
            return lhs.type_ == rhs.type_ && lhs.address_ == rhs.address_ &&
                   lhs.locality_ == rhs.locality_;
        }

        locality locality_;     /// locality: ip4 address/port number
        component_type type_;   /// component type this address is referring to
        address_type address_;  /// address (sequence number)

        parcelset::connection_type get_connection_type() const
        {
            return locality_.get_type();
        }

    private:
        friend std::ostream& operator<< (std::ostream&, address const&);

        // serialization support
        friend class boost::serialization::access;

        template <typename Archive>
        void save(Archive& ar, const unsigned int version) const;

        template <typename Archive>
        void load(Archive& ar, const unsigned int version);

        BOOST_SERIALIZATION_SPLIT_MEMBER()
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

///////////////////////////////////////////////////////////////////////////////
// this is the current version of the address serialization format
// this definition needs to be in the global namespace
#if defined(__GNUG__) && !defined(__INTEL_COMPILER)
#if defined(HPX_GCC_DIAGNOSTIC_PRAGMA_CONTEXTS)
#pragma GCC diagnostic push
#endif
#pragma GCC diagnostic ignored "-Wold-style-cast"
#endif
BOOST_CLASS_VERSION(hpx::naming::address, HPX_ADDRESS_VERSION)
BOOST_CLASS_TRACKING(hpx::naming::address, boost::serialization::track_never)
#if defined(__GNUG__) && !defined(__INTEL_COMPILER)
#if defined(HPX_GCC_DIAGNOSTIC_PRAGMA_CONTEXTS)
#pragma GCC diagnostic pop
#endif
#endif

#include <hpx/config/warnings_suffix.hpp>

#endif

