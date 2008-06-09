//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_ACCUMULATOR_MAY_18_2008_0822AM)
#define HPX_COMPONENTS_ACCUMULATOR_MAY_18_2008_0822AM

#include <hpx/naming/name.hpp>
#include <hpx/applier/applier.hpp>
#include <hpx/components/stubs/accumulator.hpp>

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
        accumulator(applier::applier& app, naming::id_type gid) 
          : stubs::accumulator(app), gid_(gid)
        {}
        
        ~accumulator() 
        {}
        
        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component
        
        /// Initialize the accumulator value
        void init() 
        {
            this->base_type::init(gid_);
        }
        
        /// Add the given number to the accumulator
        void add (double arg) 
        {
            this->base_type::add(gid_, arg);
        }
        
        /// Print the current value of the accumulator
        void print() 
        {
            this->base_type::print(gid_);
        }
        
    private:
        naming::id_type gid_;
    };
    
}}

#endif
