//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_SIMPLE_ACCUMULATOR_JUL_18_2008_1123AM)
#define HPX_COMPONENTS_SIMPLE_ACCUMULATOR_JUL_18_2008_1123AM

#include <hpx/runtime/runtime.hpp>
#include <hpx/components/stubs/simple_accumulator.hpp>

namespace hpx { namespace components 
{
    ///////////////////////////////////////////////////////////////////////////
    /// The \a simple_accumulator class is the client side representation of a 
    /// specific \a server#simple_accumulator component
    class simple_accumulator : public stubs::simple_accumulator
    {
        typedef stubs::simple_accumulator base_type;
        
    public:
        /// Create a client side representation for the existing
        /// \a server#simple_accumulator instance with the given global id \a gid.
        simple_accumulator(applier::applier& appl, naming::id_type gid) 
          : base_type(appl), gid_(gid)
        {}

        ~simple_accumulator() 
        {}

        /// Create a new instance of an simple_accumulator on the locality as given by
        /// the parameter \a targetgid
        static simple_accumulator 
        create(threads::thread_self& self, applier::applier& appl, 
            naming::id_type const& targetgid)
        {
            return simple_accumulator(appl, base_type::create(self, appl, targetgid));
        }

        void free(naming::id_type const& targetgid)
        {
            stubs::simple_accumulator::free(app_, targetgid, gid_);
            gid_ = naming::invalid_id;
        }

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
        double query(threads::thread_self& self) 
        {
            BOOST_ASSERT(gid_);
            return this->base_type::query(self, gid_);
        }

        /// Asynchronously query the current value of the simple_accumulator
        lcos::simple_future<double> query_async() 
        {
            return this->base_type::query_async(gid_);
        }

    private:
        naming::id_type gid_;
    };
    
}}

#endif
