//  Copyright (c) 2011 Matt Anderson
//  Copyright (c) 2011 Bryce Lelbach
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_SERVER_CONTACT_JUN_06_2011_1154AM)
#define HPX_COMPONENTS_SERVER_CONTACT_JUN_06_2011_1154AM

#include <sstream>
#include <iostream>

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/simple_component_base.hpp>
#include <boost/thread/locks.hpp>

namespace hpx { namespace components { namespace server 
{
    ///////////////////////////////////////////////////////////////////////////
    /// \class contact contact.hpp hpx/components/contact.hpp
    class contact 
      : public simple_component_base<contact>
    {
    public:
        // parcel action code: the action to be performed on the destination 
        // object (the accumulator)
        enum actions
        {
            contact_init = 0,
            contact_contactsearch = 1,
            contact_query_value = 2,
            contact_contactenforce = 3
        };

        // constructor: initialize contact value
        contact()
          : arg_(0), prefix_(0)
        {
            applier::applier* appl = applier::get_applier_ptr();
            if (appl)
                prefix_ = naming::get_prefix_from_gid(appl->get_prefix());
        }

        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        /// Initialize the accumulator
        void init(int i) 
        {
            hpx::lcos::mutex::scoped_lock l(mtx_);

            //std::ostringstream oss;
            //oss << "[L" << prefix_ << "/" << this << "]"
            //    << " Initializing count to " << i << "\n";
            //std::cout << oss.str() << std::flush;

            arg_ = i;

            // normally we would get the vertex info from a mesh
            // we just use a random number here

            srand(i);
            // initialize a vertex
            posx = rand();
            posy = rand();
            velx = rand();
            vely = rand();
        }

        // Search if contact has happened
        void contactsearch() 
        {
            hpx::lcos::mutex::scoped_lock l(mtx_);

            //std::ostringstream oss;
            //oss << "[L" << prefix_ << "/" << this << "]"
            //    << " Incrementing count from " << arg_
            //    << " to " << (arg_ + 1) << "\n";
            //std::cout << oss.str() << std::flush;

            arg_ += 1;
        }

        /// Return the current value to the caller
        int query() 
        {
            hpx::lcos::mutex::scoped_lock l(mtx_);

            std::ostringstream oss;
            oss << "[L" << prefix_ << "/" << this << "]"
                << " Querying count, current value is " << arg_ << "\n"; 
            std::cout << oss.str() << std::flush;

            return arg_;
        }

        /// Print the current value of the accumulator
        void contactenforce() 
        {
            hpx::lcos::mutex::scoped_lock l(mtx_);

            //std::ostringstream oss;
            //oss << "[L" << prefix_ << "/" << this << "]"
            //    << ", final count is " << arg_ << "\n";
            //std::cout << oss.str() << std::flush;
        }

        ///////////////////////////////////////////////////////////////////////
        // Each of the exposed functions needs to be encapsulated into an action
        // type, allowing to generate all required boilerplate code for threads,
        // serialization, etc.
        typedef hpx::actions::action1<
            contact, contact_init, int, 
            &contact::init
        > init_action;

        typedef hpx::actions::action0<
            contact, contact_contactsearch, 
            &contact::contactsearch
        > contactsearch_action;

        typedef hpx::actions::result_action0<
            contact, int, contact_query_value, 
            &contact::query
        > query_action;

        typedef hpx::actions::action0<
            contact, contact_contactenforce, 
            &contact::contactenforce
        > contactenforce_action;

    private:
        int arg_;
        double posx,posy,velx,vely;
        boost::uint32_t prefix_;
        hpx::lcos::mutex mtx_;
    };

}}}

#endif

