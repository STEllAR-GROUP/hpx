//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_AMR_STENCIL_OCT_17_2008_0847AM)
#define HPX_COMPONENTS_AMR_STENCIL_OCT_17_2008_0847AM

#include <hpx/components/amr/server/functional_component.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace amr 
{
    /// This class implements the time step evolution functionality. It has to
    /// expose two functions: \a eval() and \a is_last_timestep(). The function
    /// eval computes the value for the current time step based on the values
    /// as computed by the previous time step. The functions is_last_timestep()
    /// decides whether the current time step is the last one (the computation
    /// has reached the final time step).
    class HPX_COMPONENT_EXPORT stencil 
      : public amr::server::functional_component
    {
    private:
        typedef amr::server::functional_component base_type;

    public:
        typedef stencil wrapped_type;
        typedef stencil wrapping_type;

        stencil(applier::applier& appl)
          : base_type(appl)
        {}

        /// This is the function implementing the actual time step functionality
        /// It takes the values as calculated during the previous time step 
        /// and needs to return the current calculated value.
        ///
        /// The name of this function must be eval(), it must return whether
        /// the current time step is the last or past the last. The parameter
        /// \a result passes in the gid of the memory block where the result
        /// of the current time step has to be stored. The parameter \a gids
        /// is a vector of gids referencing the memory blocks of the results of
        /// previous time step.
        threads::thread_state eval(threads::thread_self&, applier::applier&, 
            bool* islast, naming::id_type const& result, 
            std::vector<naming::id_type> const& gids);

        /// The init function is supposed to create a new memory block instance 
        /// suitable for storing all data needed for a single time step.
        threads::thread_state init(threads::thread_self&, applier::applier&, 
            naming::id_type* result);

        /// The free function releases the memory allocated by init
        threads::thread_state free(threads::thread_self&, applier::applier&, 
            naming::id_type const&);
    };

}}}

#endif
