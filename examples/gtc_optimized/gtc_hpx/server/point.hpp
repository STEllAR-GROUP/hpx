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

///////////////////////////////////////////////////////////////////////////////
namespace gtc { namespace server
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

        std::size_t setup_wrapper(std::size_t numberpe,std::size_t mype,
                   std::vector<hpx::naming::id_type> const& point_components);
        void chargei_wrapper();
        void partd_allreduce(double *dnitmp,double *densityi, int* mgrid, int *mzetap1);
        void broadcast_parameters(int *integer_params,double *real_params,
                                  int *n_integers,int *n_reals);
        void set_data(std::size_t item, std::size_t generation,
                              std::vector<double> const& data);

        void set_tdata(std::size_t item, std::size_t generation,
                              std::vector<double> const& data);

        void set_params(std::size_t which,
                        std::size_t generation,
                        std::vector<int> const& intparams,
                        std::vector<double> const& realparams);

        void toroidal_sndleft(double *csend, int* mgrid);
        void toroidal_rcvright(double *creceive);
        void toroidal_sndright(double *csend, int* mgrid);
        void toroidal_rcvleft(double *creceive);

        void toroidal_allreduce(double *input,double *output, int* size);

        void set_sendleft_data(std::size_t which,
                          std::size_t generation,
                          std::vector<double> const& send);

        void set_sendright_data(std::size_t which,
                          std::size_t generation,
                          std::vector<double> const& send);

        void timeloop(std::size_t, std::size_t);
        void toroidal_gather(double *csend, int *size,int *dst);
        void toroidal_gather_receive(double *creceive, int *dst);
        void set_toroidal_gather_data(std::size_t which,
                          std::size_t generation,
                          std::vector<double> const& send);
        void toroidal_scatter(double *csend, int *size,int *src);
        void toroidal_scatter_receive(double *creceive, int *src);
        void set_toroidal_scatter_data(std::size_t which,
                          std::size_t generation,
                          std::vector<double> const& send);

        void comm_allreduce(double *in,double *out, int* msize);
        void int_comm_allreduce(int *in,int *out, int* msize);
        void set_comm_allreduce_data(std::size_t which,
                std::size_t generation, std::vector<double> const& data);
        void set_int_comm_allreduce_data(std::size_t which,
                std::size_t generation, std::vector<int> const& data);

        void int_toroidal_sndright(int *csend, int* mgrid);
        void int_toroidal_rcvleft(int *creceive);
        void set_int_sendright_data(std::size_t which,
                          std::size_t generation,
                          std::vector<int> const& send);
        void set_int_sendleft_data(std::size_t which,
                          std::size_t generation,
                          std::vector<int> const& send);
        void int_toroidal_sndleft(int *csend, int* mgrid);
        void int_toroidal_rcvright(int *creceive);

        // Each of the exposed functions needs to be encapsulated into an
        // action type, generating all required boilerplate code for threads,
        // serialization, etc.
        HPX_DEFINE_COMPONENT_ACTION(point, setup_wrapper, setup_action);
        HPX_DEFINE_COMPONENT_ACTION(point, timeloop, timeloop_action);
        HPX_DEFINE_COMPONENT_ACTION(point, chargei_wrapper, chargei_action);
        HPX_DEFINE_COMPONENT_ACTION(point, set_data, set_data_action);
        HPX_DEFINE_COMPONENT_ACTION(point, set_tdata, set_tdata_action);
        HPX_DEFINE_COMPONENT_ACTION(point, set_params, set_params_action);
        HPX_DEFINE_COMPONENT_ACTION(point, set_sendleft_data, set_sendleft_data_action);
        HPX_DEFINE_COMPONENT_ACTION(point, set_sendright_data, set_sendright_data_action);
        HPX_DEFINE_COMPONENT_ACTION(point, set_toroidal_gather_data, set_toroidal_gather_data_action);
        HPX_DEFINE_COMPONENT_ACTION(point, set_toroidal_scatter_data, set_toroidal_scatter_data_action);
        HPX_DEFINE_COMPONENT_ACTION(point, set_comm_allreduce_data, set_comm_allreduce_data_action);
        HPX_DEFINE_COMPONENT_ACTION(point, set_int_comm_allreduce_data, set_int_comm_allreduce_data_action);
        HPX_DEFINE_COMPONENT_ACTION(point, set_int_sendright_data, set_int_sendright_data_action);
        HPX_DEFINE_COMPONENT_ACTION(point, set_int_sendleft_data, set_int_sendleft_data_action);

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
        std::vector<double> treceive_;
        std::size_t in_toroidal_,in_particle_;
        std::vector<double> sendleft_receive_;
        std::vector<double> sendright_receive_;
        std::vector<double> toroidal_gather_receive_;
        std::vector<double> toroidal_scatter_receive_;
        std::vector<double> comm_allreduce_receive_;
        std::vector<int> int_comm_allreduce_receive_;
        std::vector<int> int_sendright_receive_;
        std::vector<int> int_sendleft_receive_;
    };
}}

// Declaration of serialization support for the actions
HPX_REGISTER_ACTION_DECLARATION_EX(
    gtc::server::point::setup_action,
    gtc_point_setup_action);

// ensure sufficient stack space for chargei_action
HPX_ACTION_USES_LARGE_STACK(gtc::server::point::timeloop_action);
HPX_REGISTER_ACTION_DECLARATION_EX(
    gtc::server::point::timeloop_action,
    gtc_point_timeloop_action);

// ensure sufficient stack space for timeloop_action
HPX_ACTION_USES_LARGE_STACK(gtc::server::point::chargei_action);
HPX_REGISTER_ACTION_DECLARATION_EX(
    gtc::server::point::chargei_action,
    gtc_point_chargei_action);

HPX_REGISTER_ACTION_DECLARATION_EX(
    gtc::server::point::set_data_action,
    gtc_point_set_data_action);

HPX_REGISTER_ACTION_DECLARATION_EX(
    gtc::server::point::set_tdata_action,
    gtc_point_set_tdata_action);

HPX_REGISTER_ACTION_DECLARATION_EX(
    gtc::server::point::set_params_action,
    gtc_point_set_params_action);

HPX_REGISTER_ACTION_DECLARATION_EX(
    gtc::server::point::set_sendleft_data_action,
    gtc_point_set_sendleft_data_action);

HPX_REGISTER_ACTION_DECLARATION_EX(
    gtc::server::point::set_sendright_data_action,
    gtc_point_set_sendright_data_action);

HPX_REGISTER_ACTION_DECLARATION_EX(
    gtc::server::point::set_toroidal_gather_data_action,
    gtc_point_set_toroidal_gather_data_action);

HPX_REGISTER_ACTION_DECLARATION_EX(
    gtc::server::point::set_toroidal_scatter_data_action,
    gtc_point_set_toroidal_scatter_data_action);

HPX_REGISTER_ACTION_DECLARATION_EX(
    gtc::server::point::set_comm_allreduce_data_action,
    gtc_point_set_comm_allreduce_data_action);

HPX_REGISTER_ACTION_DECLARATION_EX(
    gtc::server::point::set_int_comm_allreduce_data_action,
    gtc_point_set_int_comm_allreduce_data_action);

HPX_REGISTER_ACTION_DECLARATION_EX(
    gtc::server::point::set_int_sendright_data_action,
    gtc_point_set_int_sendright_data_action);

HPX_REGISTER_ACTION_DECLARATION_EX(
    gtc::server::point::set_int_sendleft_data_action,
    gtc_point_set_int_sendleft_data_action);

#endif

