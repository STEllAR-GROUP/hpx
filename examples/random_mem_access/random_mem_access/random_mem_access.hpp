//  Copyright (c) 2011 Matt Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_RANDOM_ACCESS_JUN_06_2011_1123AM)
#define HPX_COMPONENTS_RANDOM_ACCESS_JUN_06_2011_1123AM

#include <hpx/runtime.hpp>
#include <hpx/runtime/components/client_base.hpp>

#include "stubs/random_mem_access.hpp"

namespace hpx { namespace components
{
    ///////////////////////////////////////////////////////////////////////////
    /// The \a random_mem_access class is the client side representation of a
    /// specific \a server#random_mem_access component
    class random_mem_access
      : public client_base<random_mem_access, stubs::random_mem_access>
    {
        typedef
            client_base<random_mem_access, stubs::random_mem_access>
        base_type;

    public:
        random_mem_access()
        {}

        /// Create a client side representation for the existing
        /// \a server#random_mem_access instance with the given global id \a gid.
        random_mem_access(naming::id_type gid)
          : base_type(gid)
        {}

        ~random_mem_access()
        {}

        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        /// Initialize the random_mem_access value
        void init(int i)
        {
            BOOST_ASSERT(gid_);
            this->base_type::init(gid_,i);
        }

        /// Add the given number to the random_mem_access
        void add ()
        {
            BOOST_ASSERT(gid_);
            this->base_type::add(gid_);
        }

        hpx::lcos::future<void> add_async ()
        {
            BOOST_ASSERT(gid_);
            return(this->base_type::add_async(gid_));
        }

        /// Print the current value of the random_mem_access
        void print()
        {
            BOOST_ASSERT(gid_);
            this->base_type::print(gid_);
        }
        /// Asynchronously query the current value of the random_mem_access
        hpx::lcos::future<void> print_async ()
        {
            BOOST_ASSERT(gid_);
            return(this->base_type::print_async(gid_));
        }

        /// Query the current value of the random_mem_access
        int query()
        {
            BOOST_ASSERT(gid_);
            return this->base_type::query(gid_);
        }

        /// Asynchronously query the current value of the random_mem_access
        lcos::future<int> query_async()
        {
            return this->base_type::query_async(gid_);
        }
    };

}}

#endif
