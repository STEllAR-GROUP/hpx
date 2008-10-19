//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_AMR_STENCIL_OCT_17_2008_0847AM)
#define HPX_COMPONENTS_AMR_STENCIL_OCT_17_2008_0847AM

#include <hpx/components/amr/server/stencil_value.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace amr { namespace detail
{
    /// This class implements the time step evolution functionality. It has to
    /// expose two functions: \a eval() and \a is_last_timestep(). The function
    /// eval computes the value for the current time step based on the values
    /// as computed by the previous time step. The functions is_last_timestep()
    /// decides whether the current time step is the last one (the computation
    /// has reached the final time step).
    class stencil 
      : public server::detail::stencil_value<stencil, double, 3>
    {
    private:
        typedef server::detail::stencil_value<stencil, double, 3> base_type;

    public:
        stencil(applier::applier& appl)
          : base_type(appl), timestep_(0)
        {}

        /// This is the function implementing the actual time step functionality
        /// It takes the values as calculated during the previous time step 
        /// and needs to return the current calculated value.
        ///
        /// The name of this function must be eval(), the number of parameters 
        /// must match the degree of the stencil, and the types of the return
        /// value and the parameters must match the types of the grid functions
        /// this evolution is computing.
        double eval(double, double, double);

        ///
        bool is_last_timestep() const;

    private:
        int timestep_;    // count evaluated time steps
    };

}}}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace amr 
{
    /// This class defines the overall component type. It is needed to build
    /// the proper infrastructure for the HPX component architecture
    struct stencil 
      : public managed_component_base<amr::detail::stencil, stencil>
    {
        typedef amr::detail::stencil wrapped_type;
        typedef managed_component_base<wrapped_type, stencil> base_type;

        stencil(applier::applier& appl) 
          : base_type(new wrapped_type(appl)) 
        {}
    };

}}}

#endif
