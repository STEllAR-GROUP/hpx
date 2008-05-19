//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_ACCUMULATOR_MAY_18_2008_0822AM)
#define HPX_COMPONENTS_ACCUMULATOR_MAY_18_2008_0822AM

#include <hpx/naming/name.hpp>
#include <hpx/components/server/accumulator.hpp>

namespace hpx { namespace components 
{
    ///////////////////////////////////////////////////////////////////////////
    // The \a accumulator class is the client side representation of a 
    // \a server#accumulator component
    class accumulator
    {
    public:
        /// Create a client side representation for the existing
        /// \a server#accumulator instance with the given global id \a gid.
        accumulator(applier& app, naming::id_type gid) 
          : app_(app), gid_(gid)
        {
            
        }
        ~accumulator() {}
        
        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component
        
        /// Initialize the accumulator value
        void init() 
        {
            app_.apply<server::accumulator::init_action>(gid_);
        }
        
        /// Add the given number to the accumulator
        void add (double arg) 
        {
            app_.apply<server::accumulator::add_action>(gid_, arg);
        }
        
        /// Print the current value of the accumulator
        void print() 
        {
            app_.apply<server::accumulator::print_action>(gid_);
        }
        
    private:
        naming::id_type gid_;
        applier& app_;
    };
    
}}

#endif
