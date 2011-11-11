//  Copyright (c) 2007-2010 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_SIMPLE_ACCUMULATOR_JUL_18_2008_1123AM)
#define HPX_COMPONENTS_SIMPLE_ACCUMULATOR_JUL_18_2008_1123AM

#include <hpx/runtime.hpp>
#include <hpx/runtime/components/client_base.hpp>

#include "stubs/simple_accumulator.hpp"

namespace hpx { namespace components
{
    ///////////////////////////////////////////////////////////////////////////
    /// The \a simple_accumulator class is the client side representation of a
    /// specific \a server#simple_accumulator component
    class simple_accumulator
      : public client_base<simple_accumulator, stubs::simple_accumulator>
    {
        typedef
            client_base<simple_accumulator, stubs::simple_accumulator>
        base_type;

    public:
        simple_accumulator()
        {}

        /// Create a client side representation for the existing
        /// \a server#simple_accumulator instance with the given global id \a gid.
        simple_accumulator(naming::id_type gid)
          : base_type(gid)
        {}

        ~simple_accumulator()
        {}

        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        /// Initialize the simple_accumulator value
        void init()
        {
            BOOST_ASSERT(gid_);
            this->base_type::init(gid_);
        }

        /// Add the given number to the simple_accumulator
        void add (double arg)
        {
            BOOST_ASSERT(gid_);
            this->base_type::add(gid_, arg);
        }

        /// Print the current value of the simple_accumulator
        void print()
        {
            BOOST_ASSERT(gid_);
            this->base_type::print(gid_);
        }

        /// Query the current value of the simple_accumulator
        double query()
        {
            BOOST_ASSERT(gid_);
            return this->base_type::query(gid_);
        }

        /// Asynchronously query the current value of the simple_accumulator
        lcos::promise<double> query_async()
        {
            return this->base_type::query_async(gid_);
        }
    };

}}

#endif
