//  Copyright (c) 2011 Matt Anderson
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_CONTACT_JUN_06_2011_1123AM)
#define HPX_COMPONENTS_CONTACT_JUN_06_2011_1123AM

#include <hpx/runtime.hpp>
#include <hpx/runtime/components/client_base.hpp>

#include "stubs/contact.hpp"

namespace hpx { namespace components 
{
    ///////////////////////////////////////////////////////////////////////////
    /// The \a contact class is the client side representation of a 
    /// specific \a server#contact component
    class contact 
      : public client_base<contact, stubs::contact>
    {
        typedef 
            client_base<contact, stubs::contact> 
        base_type;

    public:
        contact() 
        {}

        /// Create a client side representation for the existing
        /// \a server#contact instance with the given global id \a gid.
        contact(naming::id_type gid) 
          : base_type(gid)
        {}

        ~contact() 
        {}

        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        /// Initialize the contact value
        void init(int i) 
        {
            BOOST_ASSERT(gid_);
            this->base_type::init(gid_,i);
        }

        /// Add the given number to the contact
        void contactsearch () 
        {
            BOOST_ASSERT(gid_);
            this->base_type::contactsearch(gid_);
        }

        hpx::lcos::promise<void> contactsearch_async () 
        {
            BOOST_ASSERT(gid_);
            return(this->base_type::contactsearch_async(gid_));
        }

        /// Print the current value of the contact
        void contactenforce() 
        {
            BOOST_ASSERT(gid_);
            this->base_type::contactenforce(gid_);
        }
        /// Asynchronously query the current value of the contact
        hpx::lcos::promise<void> contactenforce_async () 
        {
            BOOST_ASSERT(gid_);
            return(this->base_type::contactenforce_async(gid_));
        }

        /// Query the current value of the contact
        int query() 
        {
            BOOST_ASSERT(gid_);
            return this->base_type::query(gid_);
        }

        /// Asynchronously query the current value of the contact
        lcos::promise<int> query_async() 
        {
            return this->base_type::query_async(gid_);
        }
    };
    
}}

#endif
