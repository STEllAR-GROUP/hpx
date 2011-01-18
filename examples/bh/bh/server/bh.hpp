//  Copyright (c) 2007-2010 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_SERVER_bh_MAY_17_2008_0731PM)
#define HPX_COMPONENTS_SERVER_bh_MAY_17_2008_0731PM

#include <iostream>

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/actions/component_action.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace server 
{
    ///////////////////////////////////////////////////////////////////////////
    /// The bh is a very simple example of a HPX component. 
    ///
    /// The bh class is a small example components demonstrating the
    /// main principles of writing your own components. It exposes 4 different
    /// actions: init, add, query, and print, showing how to used and implement
    /// functionality in a way conformant with the HPX runtime system. 
    ///
    /// Note that the implementation of the bh does not require any 
    /// special data members or virtual functions.
    ///
    class bh 
      : public components::detail::managed_component_base<bh> 
    {
    public:
        // parcel action code: the action to be performed on the destination 
        // object (the bh)
        enum actions
        {
            bh_init = 0,
            bh_add = 1,
            bh_query_value = 2,
            bh_print = 3
        };

        // constructor: initialize bh value
        bh()
          : arg_(0)
        {}

        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        /// Initialize the bh
        void init() 
        {
            arg_ = 0;
        }

        /// Add the given number to the bh
        void add (unsigned long arg) 
        {
            arg_ += arg;
        }

        /// Return the current value to the caller
        unsigned long query() 
        {
            return arg_;
        }

        /// Print the current value of the bh
        void print() 
        {
            applier::applier& appl = applier::get_applier();
            std::cout << appl.get_runtime_support_gid() << "> " 
                      << arg_ << std::flush << std::endl;
        }

        ///////////////////////////////////////////////////////////////////////
        // Each of the exposed functions needs to be encapsulated into an action
        // type, allowing to generate all required boilerplate code for threads,
        // serialization, etc.
        typedef hpx::actions::action0<
            bh, bh_init, &bh::init
        > init_action;

        typedef hpx::actions::action1<
            bh, bh_add, unsigned long, &bh::add
        > add_action;

        typedef hpx::actions::result_action0<
            bh, unsigned long, bh_query_value, &bh::query
        > query_action;

        typedef hpx::actions::action0<
            bh, bh_print, &bh::print
        > print_action;

    private:
        unsigned long arg_;
    };

}}}

#endif
