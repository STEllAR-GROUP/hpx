//  Copyright (c) 2011 Matthew Anderson
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
namespace sendrecv { namespace server
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
        void toroidal_sndrecv(double *csend,int* csend_size,double *creceive,
                                 int *creceive_size,int* dest);
        void set_sndrecv_data(std::size_t which,
                           std::size_t generation,
                           std::vector<double> const& send);

        // Each of the exposed functions needs to be encapsulated into an
        // action type, generating all required boilerplate code for threads,
        // serialization, etc.
        HPX_DEFINE_COMPONENT_ACTION(partition, loop_wrapper, loop_action);
        HPX_DEFINE_COMPONENT_ACTION(partition, set_sndrecv_data, set_sndrecv_data_action);

    private:
        typedef hpx::lcos::local::spinlock mutex_type;
        typedef hpx::lcos::local::base_and_gate<> and_gate_type;

        std::size_t item_;
        std::vector<int> t_comm_,p_comm_;
        std::size_t left_pe_,right_pe_;

        hpx::lcos::local::trigger sndrecv_gate_;

        std::vector<hpx::naming::id_type> components_;
        mutable mutex_type mtx_;
        std::vector<double> sndrecv_;
        std::size_t myrank_toroidal_;
        std::size_t myrank_partd_;
    };
}}

// Declaration of serialization support for the actions
HPX_ACTION_USES_HUGE_STACK(sendrecv::server::partition::loop_action);
HPX_REGISTER_ACTION_DECLARATION(
    sendrecv::server::partition::loop_action,
    sendrecv_point_loop_action);

HPX_REGISTER_ACTION_DECLARATION(
    sendrecv::server::partition::set_sndrecv_data_action,
    sendrecv_point_set_sndrecv_data_action);

#endif

