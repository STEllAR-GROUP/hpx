//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_ACCUMULATOR_MAY_18_2008_0822AM)
#define HPX_COMPONENTS_ACCUMULATOR_MAY_18_2008_0822AM

#include <hpx/runtime/runtime.hpp>
#include <hpx/components/accumulator/stubs/accumulator.hpp>

namespace hpx { namespace components 
{
    ///////////////////////////////////////////////////////////////////////////
    /// The \a accumulator class is the client side representation of a 
    /// specific \a server#accumulator component
    class accumulator : public stubs::accumulator
    {
        typedef stubs::accumulator base_type;
        
    public:
        /// Create a client side representation for the existing
        /// \a server#accumulator instance with the given global id \a gid.
        accumulator(applier::applier& appl, naming::id_type gid, bool freeonexit = true) 
          : base_type(appl), gid_(gid), freeonexit_(freeonexit)
        {}

        ~accumulator() 
        {
            if (freeonexit_)
                stubs::accumulator::free(gid_);
        }

        /// Create a new instance of an accumulator on the locality as given by
        /// the parameter \a targetgid
        static accumulator 
        create(threads::thread_self& self, applier::applier& appl, 
            naming::id_type const& targetgid, bool freeonexit = true)
        {
            return accumulator(appl, base_type::create(self, appl, targetgid), 
                freeonexit);
        }

        void free()
        {
            stubs::accumulator::free(gid_);
            gid_ = naming::invalid_id;
        }

        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        /// Initialize the accumulator value
        void init() 
        {
            BOOST_ASSERT(gid_);
            this->base_type::init(gid_);
        }

        /// Add the given number to the accumulator
        void add (double arg) 
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
        double query(threads::thread_self& self) 
        {
            BOOST_ASSERT(gid_);
            return this->base_type::query(self, gid_);
        }

        /// Asynchronously query the current value of the accumulator
        lcos::simple_future<double> query_async() 
        {
            return this->base_type::query_async(gid_);
        }

    private:
        naming::id_type gid_;
        bool freeonexit_;
    };
    
}}

#endif
