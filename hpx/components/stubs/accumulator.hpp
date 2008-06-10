//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_STUBS_ACCUMULATOR_JUN_09_2008_0458PM)
#define HPX_COMPONENTS_STUBS_ACCUMULATOR_JUN_09_2008_0458PM

#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/components/server/accumulator.hpp>

namespace hpx { namespace components { namespace stubs
{
    ///////////////////////////////////////////////////////////////////////////
    /// The \a stubs#accumulator class is the client side representation of all
    /// \a server#accumulator components
    class accumulator
    {
    public:
        /// Create a client side representation for any existing
        /// \a server#accumulator instance.
        accumulator(applier::applier& appl) 
          : app_(appl)
        {}
        
        ~accumulator() 
        {}
        
        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component
        
        /// Initialize the accumulator value of the server#accumulator instance 
        /// with the give \a gid
        void init(naming::id_type gid) 
        {
            app_.apply<server::accumulator::init_action>(gid);
        }
        
        /// Add the given number to the server#accumulator instance 
        /// with the give \a gid
        void add (naming::id_type gid, double arg) 
        {
            app_.apply<server::accumulator::add_action>(gid, arg);
        }
        
        /// Print the current value of the server#accumulator instance 
        /// with the give \a gid
        void print(naming::id_type gid) 
        {
            app_.apply<server::accumulator::print_action>(gid);
        }
        
    private:
        applier::applier& app_;
    };
    
}}}

#endif
