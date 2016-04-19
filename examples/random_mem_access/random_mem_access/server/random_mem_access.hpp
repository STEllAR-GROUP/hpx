//  Copyright (c) 2011 Matt Anderson
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_SERVER_RANDOM_JUN_06_2011_1154AM)
#define HPX_COMPONENTS_SERVER_RANDOM_JUN_06_2011_1154AM

#include <hpx/hpx.hpp>
#include <hpx/include/components.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/threads/thread_data.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/runtime/components/component_type.hpp>

#include <iostream>
#include <mutex>
#include <sstream>

namespace hpx { namespace components { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    /// \class random_mem_access random_mem_access.hpp
    /// hpx/components/random_mem_access.hpp
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
      : public component_base<random_mem_access>
    {
    public:
        // constructor: initialize random_mem_access value
        random_mem_access()
          : arg_(0), arg_init_(0), prefix_(0)
        {
            applier::applier* appl = applier::get_applier_ptr();
            if (appl)
                prefix_ = appl->get_locality_id();
        }

        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        /// Initialize the accumulator
        void init(int i)
        {
            std::lock_guard<hpx::lcos::local::mutex> l(mtx_);

            std::ostringstream oss;
            oss << "[L" << prefix_ << "/" << this << "]"
                << " Initializing count to " << i << "\n";
            std::cout << oss.str() << std::flush;

            arg_ = i;
            arg_init_ = i;
        }

        /// Add the given number to the accumulator
        void add()
        {
            std::lock_guard<hpx::lcos::local::mutex> l(mtx_);

            std::ostringstream oss;
            oss << "[L" << prefix_ << "/" << this << "]"
                << " Incrementing count from " << arg_
                << " to " << (arg_ + 1) << "\n";
            std::cout << oss.str() << std::flush;

            arg_ += 1;
        }

        /// Return the current value to the caller
        int query()
        {
            std::lock_guard<hpx::lcos::local::mutex> l(mtx_);

            std::ostringstream oss;
            oss << "[L" << prefix_ << "/" << this << "]"
                << " Querying count, current value is " << arg_ << "\n";
            std::cout << oss.str() << std::flush;

            return arg_;
        }

        /// Print the current value of the accumulator
        void print()
        {
            std::lock_guard<hpx::lcos::local::mutex> l(mtx_);

            std::ostringstream oss;
            oss << "[L" << prefix_ << "/" << this << "]"
                << " Initial count was " << arg_init_
                << ", final count is " << arg_ << "\n";
            std::cout << oss.str() << std::flush;
        }

        ///////////////////////////////////////////////////////////////////////
        // Each of the exposed functions needs to be encapsulated into an action
        // type, allowing to generate all required boilerplate code for threads,
        // serialization, etc.
        HPX_DEFINE_COMPONENT_ACTION(random_mem_access, init);
        HPX_DEFINE_COMPONENT_ACTION(random_mem_access, add);
        HPX_DEFINE_COMPONENT_ACTION(random_mem_access, query);
        HPX_DEFINE_COMPONENT_ACTION(random_mem_access, print);

    private:
        int arg_;
        int arg_init_;
        boost::uint32_t prefix_;
        hpx::lcos::local::mutex mtx_;
    };

}}}

HPX_REGISTER_ACTION_DECLARATION(
    hpx::components::server::random_mem_access::init_action,
    random_mem_access_init_action);
HPX_REGISTER_ACTION_DECLARATION(
    hpx::components::server::random_mem_access::add_action,
    random_mem_access_add_action);
HPX_REGISTER_ACTION_DECLARATION(
    hpx::components::server::random_mem_access::query_action,
    random_mem_access_query_action);
HPX_REGISTER_ACTION_DECLARATION(
    hpx::components::server::random_mem_access::print_action,
    random_mem_access_print_action);

#endif

