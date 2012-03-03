//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2009-2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_STENCIL_STENCIL_OCT_17_2011_0847AM)
#define HPX_COMPONENTS_STENCIL_STENCIL_OCT_17_2011_0847AM

#include "../dataflow/server/functional_component.hpp"
#include "stencil_data.hpp"
#include "../dataflow/dataflow_stencil.hpp"

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace adaptive1d
{
    /// This class implements the time step evolution functionality. It has to
    /// expose several functions: \a eval, \a alloc_data and \a free_data.
    /// The function \a eval computes the value for the current time step based
    /// on the values as computed by the previous time step. The functions
    /// \a alloc_data is used to allocate the data needed to store one
    /// datapoint, while the function \a free_data is used to free the memory
    /// allocated using alloc_data.
    class HPX_COMPONENT_EXPORT stencil
      : public adaptive1d::server::functional_component
    {
    private:
        typedef adaptive1d::server::functional_component base_type;

    public:
        typedef stencil wrapped_type;
        typedef stencil wrapping_type;

        stencil();
        ~stencil() {}

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
        int eval(naming::id_type const& result,
            std::vector<naming::id_type> const& gids, std::size_t row, std::size_t column,
            double cycle_time,parameter const& par);

        /// The alloc function is supposed to create a new memory block instance
        /// suitable for storing all data needed for a single time step.
        /// Additionally it fills the memory with initial data for the data
        /// item given by the parameter \a item (if item != -1).
        naming::id_type alloc_data(int item, int maxitems, int row,
                                   std::vector<naming::id_type> const& interp_src_data,
                                   double time,
                                   parameter const& par);

        double interp_linear(double y1, double y2,
                             double x, double x1, double x2);

        void interpolate(double x, double minx,double h,
                              access_memory_block<stencil_data> &val,
                              nodedata &result, parameter const& par);

        /// The init function initializes this stencil point
        void init(std::size_t, naming::id_type const&);

    private:
        std::size_t numsteps_;
        naming::id_type log_;
    };

}}}

#endif
