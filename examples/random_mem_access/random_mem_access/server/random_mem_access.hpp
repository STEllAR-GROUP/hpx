//  Copyright (c) 2011 Matt Anderson
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_SERVER_RANDOM_JUN_06_2011_1154AM)
#define HPX_COMPONENTS_SERVER_RANDOM_JUN_06_2011_1154AM

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
    /// \class random_mem_access random_mem_access.hpp hpx/components/random_mem_access.hpp
    ///
    /// The random_mem_access is a small example components demonstrating 
    /// the main principles of writing your own components. It exposes 4 
    /// different actions: init, add, query, and print, showing how to used and 
    /// implement functionality in a way conformant with the HPX runtime system. 
    /// The random_mem_access is a very simple example of an HPX component. 
    ///
    /// Note that the implementation of the random_mem_access does not require 
    /// special data members or virtual functions. All specifics are embedded 
    /// in the random_mem_access_base class the random_mem_access is derived 
    /// from.
    ///
    class random_mem_access 
      : public simple_component_base<random_mem_access>
    {
    public:
        // parcel action code: the action to be performed on the destination 
        // object (the accumulator)
        enum actions
        {
            random_mem_access_init = 0,
            random_mem_access_add = 1,
            random_mem_access_query_value = 2,
            random_mem_access_print = 3
        };

        // constructor: initialize random_mem_access value
        random_mem_access()
          : arg_(0)
        {}

        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        /// Initialize the accumulator
        void init(int i) 
        {
            hpx::lcos::mutex::scoped_lock l(mtx_);
            arg_ = i;
            arg_init_ = i;
        }

        /// Add the given number to the accumulator
        void add() 
        {
            hpx::lcos::mutex::scoped_lock l(mtx_);
            arg_ += 1;
        }

        /// Return the current value to the caller
        int query() 
        {
            hpx::lcos::mutex::scoped_lock l(mtx_);
            return arg_;
        }

        /// Print the current value of the accumulator
        void print() 
        {
            hpx::lcos::mutex::scoped_lock l(mtx_);
            std::cout << " I started as " << arg_init_ << " and I finished as " << arg_ << std::flush << std::endl;
        }

        ///////////////////////////////////////////////////////////////////////
        // Each of the exposed functions needs to be encapsulated into an action
        // type, allowing to generate all required boilerplate code for threads,
        // serialization, etc.
        typedef hpx::actions::action1<
            random_mem_access, random_mem_access_init,int, 
            &random_mem_access::init
        > init_action;

        typedef hpx::actions::action0<
            random_mem_access, random_mem_access_add, 
            &random_mem_access::add
        > add_action;

        typedef hpx::actions::result_action0<
            random_mem_access, int, random_mem_access_query_value, 
            &random_mem_access::query
        > query_action;

        typedef hpx::actions::action0<
            random_mem_access, random_mem_access_print, 
            &random_mem_access::print
        > print_action;

    private:
        int arg_;
        int arg_init_;
        hpx::lcos::mutex mtx_;
    };

}}}

#endif
