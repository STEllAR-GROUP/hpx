//  Copyright (c) 2012 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_A07C7784_8AD2_4A12_B5BA_174DFBE03222)
#define HPX_A07C7784_8AD2_4A12_B5BA_174DFBE03222

#include <vector>
#include <queue>

#include <hpx/include/components.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/local_lcos.hpp>
#include <hpx/include/util.hpp>

#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/util/unlock_lock.hpp>
#include <boost/serialization/complex.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace nekbone { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    class HPX_COMPONENT_EXPORT partition
      : public hpx::components::managed_component_base<partition>
    {
    public:
        partition()
        {}

        ///////////////////////////////////////////////////////////////////////
        // Exposed functionality of this component.

        void loop_wrapper(std::size_t numberpe,std::size_t mype,
                   std::vector<hpx::naming::id_type> const& point_components);

        void broadcast_int_parameters(int *integer_params);

        void set_int_params(std::size_t which,
                           std::size_t generation,
                           int intparams);

        void double_mpi_allreduce(double *x, double *w,int *n,int *ierr);
        void set_double_mpi_allreduce(std::size_t which,
                               std::size_t generation,
                               std::vector<double> const& data);

        // Each of the exposed functions needs to be encapsulated into an
        // action type, generating all required boilerplate code for threads,
        // serialization, etc.
        HPX_DEFINE_COMPONENT_ACTION(partition, loop_wrapper, loop_action);
        HPX_DEFINE_COMPONENT_ACTION(partition, set_int_params, set_int_params_action);
        HPX_DEFINE_COMPONENT_ACTION(partition, set_double_mpi_allreduce, set_double_mpi_allreduce_action);

    private:
        typedef hpx::lcos::local::spinlock mutex_type;
        typedef hpx::lcos::local::base_and_gate<> and_gate_type;

        std::size_t item_;
        std::vector<hpx::naming::id_type> components_;
        std::vector<hpx::naming::id_type> all_but_root_;
        std::vector<double> mpi_allreduce_data_;
        mutable mutex_type mtx_;
        int intparams_;
        and_gate_type broadcast_gate_;
        and_gate_type double_allreduce_gate_;     
    };
}}

// Declaration of serialization support for the actions
HPX_ACTION_USES_HUGE_STACK(nekbone::server::partition::loop_action);
HPX_REGISTER_ACTION_DECLARATION(
    nekbone::server::partition::loop_action,
    nekbone_point_loop_action);

HPX_REGISTER_ACTION_DECLARATION(
    nekbone::server::partition::set_int_params_action,
    nekbone_point_set_int_params_action);

HPX_REGISTER_ACTION_DECLARATION(
    nekbone::server::partition::set_double_mpi_allreduce_action,
    nekbone_point_set_double_mpi_allreduce_action);


#endif

