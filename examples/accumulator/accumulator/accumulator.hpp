//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_ACCUMULATOR_MAY_18_2008_0822AM)
#define HPX_COMPONENTS_ACCUMULATOR_MAY_18_2008_0822AM

#include <hpx/runtime.hpp>
#include <hpx/runtime/components/client_base.hpp>

#include "stubs/accumulator.hpp"

namespace hpx { namespace components
{
    ///////////////////////////////////////////////////////////////////////////
    /// The \a accumulator class is the client side representation of a
    /// specific \a server#accumulator component
    class accumulator
      : public client_base<accumulator, stubs::accumulator>
    {
        typedef client_base<accumulator, stubs::accumulator> base_type;

    public:
        accumulator()
        {}

        /// Create a client side representation for the existing
        /// \a server#accumulator instance with the given global id \a gid.
        accumulator(naming::id_type gid)
          : base_type(gid)
        {}

        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        /// Initialize the accumulator value
        void init()
        {
            BOOST_ASSERT(gid_);
            this->base_type::init(gid_);
        }

        /// Add the given number to the accumulator
        void add (unsigned long arg)
        {
            BOOST_ASSERT(gid_);
            this->base_type::add(gid_, arg);
        }

        /// Print the current value of the accumulator
        void print()
        {
            BOOST_ASSERT(gid_);
            this->base_type::print(gid_);
        }

        /// Query the current value of the accumulator
        unsigned long query()
        {
            BOOST_ASSERT(gid_);
            return this->base_type::query(gid_);
        }

        /// Asynchronously query the current value of the accumulator
        lcos::promise<unsigned long> query_async()
        {
            return this->base_type::query_async(gid_);
        }
    };

}}

#endif
