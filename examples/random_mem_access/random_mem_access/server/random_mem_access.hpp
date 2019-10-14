//  Copyright (c) 2011 Matt Anderson
//  Copyright (c) 2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_SERVER_RANDOM_JUN_06_2011_1154AM)
#define HPX_COMPONENTS_SERVER_RANDOM_JUN_06_2011_1154AM

#include <hpx/hpx.hpp>

#include <cstddef>
#include <cstdint>
#include <iostream>
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
      : public hpx::components::locking_hook<
            hpx::components::component_base<random_mem_access> >
    {
    public:
        // constructor: initialize random_mem_access value
        random_mem_access()
          : arg_(0), arg_init_(0), prefix_(hpx::get_locality_id())
        {}

        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        /// Initialize the accumulator
        void init(std::size_t i)
        {
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
            std::ostringstream oss;
            oss << "[L" << prefix_ << "/" << this << "]"
                << " Incrementing count from " << arg_
                << " to " << (arg_ + 1) << "\n";
            std::cout << oss.str() << std::flush;

            arg_ += 1;
        }

        /// Return the current value to the caller
        std::size_t query()
        {
            std::ostringstream oss;
            oss << "[L" << prefix_ << "/" << this << "]"
                << " Querying count, current value is " << arg_ << "\n";
            std::cout << oss.str() << std::flush;

            return arg_;
        }

        /// Print the current value of the accumulator
        void print()
        {
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
        std::size_t arg_;
        std::size_t arg_init_;
        std::uint32_t prefix_;
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

