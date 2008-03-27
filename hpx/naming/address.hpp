//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_NAMING_ADDRESS_MAR_24_2008_0949AM)
#define HPX_NAMING_ADDRESS_MAR_24_2008_0949AM

#include <boost/cstdint.hpp>

#include <hpx/naming/locality.hpp>
#include <hpx/util/safe_bool.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace naming
{
    struct address : public util::safe_bool<address>
    {
        typedef boost::uint64_t component_type;
        typedef boost::uint64_t address_type;
        
        ///////////////////////////////////////////////////////////////////////
        address()
          : type_(0), address_(0)
        {}

        address(locality l, component_type t, address_type a)
          : locality_(l), type_(t), address_(a) 
        {}
        
        // this get's called from the safe_bool base class 
        bool operator_bool() const { return 0 != address_; }
        
        friend bool operator==(address const& lhs, address const& rhs)
        {
            return lhs.locality_ == rhs.locality_ &&
                   lhs.type_ == rhs.type_ && lhs.address_ == rhs.address_;
        }
        
        locality locality_;     /// locality: ip4 address/port number
        component_type type_;   /// component type this address is referring to
        address_type address_;  /// address (sequence number)
    };

///////////////////////////////////////////////////////////////////////////////
}}

#endif
