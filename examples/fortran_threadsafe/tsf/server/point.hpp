//  Copyright (c) 2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_6F5C49B6_A033_4BC9_9E0F_CAFEA7C8A2B9)
#define HPX_6F5C49B6_A033_4BC9_9E0F_CAFEA7C8A2B9

#include <vector>
#include <queue>

#include <hpx/include/components.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/local_lcos.hpp>
#include <hpx/include/util.hpp>

#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/util/unlock_lock.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace tsf { namespace server
{

    ///////////////////////////////////////////////////////////////////////////
    class HPX_COMPONENT_EXPORT point
      : public hpx::components::managed_component_base<point>
    {
    public:
        point()
        {}

        ///////////////////////////////////////////////////////////////////////
        // Exposed functionality of this component.

        void setup_wrapper(std::size_t numberpe,std::size_t mype,
                   std::vector<hpx::naming::id_type> const& point_components);
        void broadcast_parameters(int *integer_params,double *real_params,
                                  int *n_integers,int *n_reals);

        void set_params(std::size_t which,
                        std::size_t generation,
                        std::vector<int> const& intparams,
                        std::vector<double> const& realparams);

        // Each of the exposed functions needs to be encapsulated into an
        // action type, generating all required boilerplate code for threads,
        // serialization, etc.
        HPX_DEFINE_COMPONENT_ACTION(point, setup_wrapper, setup_action);
        HPX_DEFINE_COMPONENT_ACTION(point, set_params, set_params_action);

    private:
        typedef hpx::lcos::local::spinlock mutex_type;
        std::size_t item_;
        std::vector<hpx::naming::id_type> toroidal_comm_,partd_comm_;
        std::size_t left_pe_,right_pe_;
        hpx::lcos::local::and_gate gate_; // synchronization gate
        std::vector<hpx::naming::id_type> components_;
        std::size_t generation_;
        mutable mutex_type mtx_;
        std::vector<int> intparams_;
        std::vector<double> realparams_;
        std::vector<double> dnireceive_;
        std::size_t in_toroidal_,in_particle_;
    };
}}

// Declaration of serialization support for the actions
HPX_REGISTER_ACTION_DECLARATION(
    tsf::server::point::setup_action,
    tsf_point_setup_action);

HPX_REGISTER_ACTION_DECLARATION(
    tsf::server::point::set_params_action,
    tsf_point_set_params_action);

#endif

