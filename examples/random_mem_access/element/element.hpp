//  Copyright (c) 2011 Matt Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_1E3EEA8A_9573_4362_A823_615537B5E9E6)
#define HPX_1E3EEA8A_9573_4362_A823_615537B5E9E6

#include <hpx/runtime/components/client_base.hpp>

#include "stubs/element.hpp"

namespace random_mem_access
{
    ///////////////////////////////////////////////////////////////////////////
    /// The client side representation of a
    /// \a random_mem_access::server::element components.
    class element
      : public hpx::components::client_base<element, stubs::element>
    {
        typedef
            hpx::components::client_base<element, stubs::element>
        base_type;

    public:
        element()
        {}

        /// Create a client side representation for the existing
        /// \a random_mem_access::server::element instance with the given GID.
        element(hpx::naming::id_type const& gid)
          : base_type(gid)
        {}

        ///////////////////////////////////////////////////////////////////////
        // Exposed functionality of this component.

        /// Initialize the element. Fire-and-forget semantics.
        void init(std::size_t i)
        {
            BOOST_ASSERT(gid_);
            this->base_type::init(gid_,i);
        }

        /// Increment the element's value. 
        void add()
        {
            BOOST_ASSERT(gid_);
            this->base_type::add(gid_);
        }

        /// Asynchronously increment the element's value.
        hpx::lcos::promise<void> add_async()
        {
            BOOST_ASSERT(gid_);
            return (this->base_type::add_async(gid_));
        }

        /// Print the current value of the element.
        void print()
        {
            BOOST_ASSERT(gid_);
            this->base_type::print(gid_);
        }

        /// Asynchronously print the current value of the element.
        hpx::lcos::promise<void> print_async()
        {
            BOOST_ASSERT(gid_);
            return (this->base_type::print_async(gid_));
        }
    };
}

#endif

