//  Copyright (c) 2007-2010 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_bh_MAY_18_2008_0822AM)
#define HPX_COMPONENTS_bh_MAY_18_2008_0822AM

#include <hpx/runtime.hpp>
#include <hpx/runtime/components/client_base.hpp>

#include "stubs/bh.hpp"

namespace hpx { namespace components 
{
    ///////////////////////////////////////////////////////////////////////////
    /// The \a bh class is the client side representation of a 
    /// specific \a server#bh component
    class bh 
      : public client_base<bh, stubs::bh>
    {
        typedef client_base<bh, stubs::bh> base_type;

    public:
        bh() 
        {}

        /// Create a client side representation for the existing
        /// \a server#bh instance with the given global id \a gid.
        bh(naming::id_type gid) 
          : base_type(gid)
        {}

        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        /// Initialize the bh value
        void init() 
        {
            BOOST_ASSERT(gid_);
            this->base_type::init(gid_);
        }

        /// Add the given number to the bh
        void add (unsigned long arg) 
        {
            BOOST_ASSERT(gid_);
            this->base_type::add(gid_, arg);
        }

        /// Print the current value of the bh
        void print() 
        {
            BOOST_ASSERT(gid_);
            this->base_type::print(gid_);
        }

        /// Query the current value of the bh
        unsigned long query() 
        {
            BOOST_ASSERT(gid_);
            return this->base_type::query(gid_);
        }

        /// Asynchronously query the current value of the bh
        lcos::future_value<unsigned long> query_async() 
        {
            return this->base_type::query_async(gid_);
        }
    };
    
}}

#endif
