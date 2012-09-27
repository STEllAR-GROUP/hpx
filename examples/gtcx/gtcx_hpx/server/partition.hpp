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
namespace gtcx { namespace server
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

        void set_toroidal_cmm(int *send,int*length,int *myrank_toroidal);
        void set_partd_cmm(int *send,int*length,int *myrank_partd);
        void loop_wrapper(std::size_t numberpe,std::size_t mype,
                   std::vector<hpx::naming::id_type> const& point_components);
        void broadcast_parameters(int *integer_params,double *real_params,
                                  int *n_integers,int *n_reals);
        void set_params(std::size_t which,
                        std::size_t generation,
                        std::vector<int> const& intparams,
                        std::vector<double> const& realparams);

        void partd_allreduce(double *dnitmp,double *densityi, int* mgrid, int *mzetap1);
        void toroidal_allreduce(double *input,double *output, int* size);
        void set_data(std::size_t item, std::size_t generation,
                              std::vector<double> const& data);

        void set_tdata(std::size_t item, std::size_t generation,
                              std::vector<double> const& data);

        void comm_allreduce(double *in,double *out, int* msize);
        void int_comm_allreduce(int *in,int *out, int* msize);
        void set_comm_allreduce_data(std::size_t which,
                std::size_t generation, std::vector<double> const& data);
        void set_int_comm_allreduce_data(std::size_t which,
                std::size_t generation, std::vector<int> const& data);

        void set_int_comm_allgather_data(std::size_t which,
                std::size_t generation, std::vector<int> const& data);
        void int_comm_allgather(int *in,int *out, int* msize);
        void toroidal_sndrecv(double *csend,int* csend_size,double *creceive,
                                 int *creceive_size,int* dest);
        void set_sndrecv_data(std::size_t which,
                           std::size_t generation,
                           std::vector<double> const& send);

        void ntoroidal_gather(double *csend, int *csize,double *creceive,int *tdst);
        void set_ntoroidal_gather_data(std::size_t which,
                          std::size_t generation,
                          std::vector<double> const& send);
        void ntoroidal_scatter(double *csend, int *csize,double *creceive,int *tsrc);
        void set_ntoroidal_scatter_data(std::size_t which,
                          std::size_t generation,
                          std::vector<double> const& send);
        void complex_ntoroidal_gather(std::complex<double> *csend, int *csize,std::complex<double> *creceive,int *tdst);
        void set_complex_ntoroidal_gather_data(std::size_t which,
                           std::size_t generation,
                           std::vector<std::complex<double> > const& send);
        void toroidal_reduce(double *input,double *output, int* size,int *dest);
        void set_toroidal_reduce_data(std::size_t item, std::size_t generation,
                              std::vector<double> const& data);
        void p2p_send(double *csend, int *csize,int *tdst);
        void p2p_receive(double *creceive, int *csize,int *tdst);
        void set_p2p_sendreceive_data(std::size_t which,
                           std::size_t generation,
                           std::vector<double> const& send);
        void broadcast_int_parameters(int *integer_params, int *n_integers);
        void set_int_params(std::size_t which,
                           std::size_t generation,
                           std::vector<int> const& intparams);
        void comm_reduce(double *input,double *output, int* size,int *tdest);
        void set_comm_reduce_data(std::size_t which,
                std::size_t generation, std::vector<double> const& data);

        void int_toroidal_sndrecv(int *csend,int* csend_size,int *creceive,int *creceive_size,int* dest);
        void set_int_sndrecv_data(std::size_t which,
                           std::size_t generation,
                           std::vector<int> const& send);

        void set_real_params(std::size_t which,
                           std::size_t generation,
                           std::vector<double> const& realparams);
        void broadcast_real_parameters(double *real_params,int *n_reals);
        void partd_allgather(double *in,double *out, int* size);
        void set_partd_allgather_data(std::size_t which,
                std::size_t generation, std::vector<double> const& data);

        // Each of the exposed functions needs to be encapsulated into an
        // action type, generating all required boilerplate code for threads,
        // serialization, etc.
        HPX_DEFINE_COMPONENT_ACTION(partition, loop_wrapper, loop_action);
        HPX_DEFINE_COMPONENT_ACTION(partition, set_data, set_data_action);
        HPX_DEFINE_COMPONENT_ACTION(partition, set_tdata, set_tdata_action);
        HPX_DEFINE_COMPONENT_ACTION(partition, set_params, set_params_action);
        HPX_DEFINE_COMPONENT_ACTION(partition, set_comm_allreduce_data, set_comm_allreduce_data_action);
        HPX_DEFINE_COMPONENT_ACTION(partition, set_int_comm_allreduce_data, set_int_comm_allreduce_data_action);
        HPX_DEFINE_COMPONENT_ACTION(partition, set_int_comm_allgather_data, set_int_comm_allgather_data_action);
        HPX_DEFINE_COMPONENT_ACTION(partition, set_sndrecv_data, set_sndrecv_data_action);
        HPX_DEFINE_COMPONENT_ACTION(partition, set_ntoroidal_gather_data, set_ntoroidal_gather_data_action);
        HPX_DEFINE_COMPONENT_ACTION(partition, set_ntoroidal_scatter_data, set_ntoroidal_scatter_data_action);
        HPX_DEFINE_COMPONENT_ACTION(partition, set_complex_ntoroidal_gather_data, set_complex_ntoroidal_gather_data_action);
        HPX_DEFINE_COMPONENT_ACTION(partition, set_toroidal_reduce_data, set_toroidal_reduce_data_action);
        HPX_DEFINE_COMPONENT_ACTION(partition, set_p2p_sendreceive_data, set_p2p_sendreceive_data_action);
        HPX_DEFINE_COMPONENT_ACTION(partition, set_int_params, set_int_params_action);
        HPX_DEFINE_COMPONENT_ACTION(partition, set_comm_reduce_data, set_comm_reduce_data_action);
        HPX_DEFINE_COMPONENT_ACTION(partition, set_int_sndrecv_data, set_int_sndrecv_data_action);
        HPX_DEFINE_COMPONENT_ACTION(partition, set_real_params, set_real_params_action);
        HPX_DEFINE_COMPONENT_ACTION(partition, set_partd_allgather_data, set_partd_allgather_data_action);

    private:
        typedef hpx::lcos::local::spinlock mutex_type;
        typedef hpx::lcos::local::base_and_gate<> and_gate_type;

        std::size_t item_;
        std::vector<int> t_comm_,p_comm_;
        std::size_t left_pe_,right_pe_;

        and_gate_type allreduce_gate_;     // synchronization gates
        and_gate_type allgather_gate_;
//        hpx::lcos::local::trigger sndleft_gate_;
//        hpx::future<void> sndleft_future_;
//        hpx::lcos::local::trigger sndright_gate_;
//        hpx::future<void> sndright_future_;
        and_gate_type gather_gate_;
//        hpx::future<void> gather_future_;
        hpx::lcos::local::trigger scatter_gate_;
//        hpx::future<void> scatter_future_;
        and_gate_type broadcast_gate_;
//        hpx::future<void> sndrecv_future_;
        hpx::lcos::local::trigger sndrecv_gate_;
        and_gate_type toroidal_allreduce_gate_; 
        and_gate_type toroidal_reduce_gate_; 
        and_gate_type comm_reduce_gate_; 
        hpx::future<void> p2p_sendreceive_future_;
        hpx::lcos::local::trigger p2p_sendreceive_gate_;
//        hpx::future<void> int_sndrecv_future_;
        hpx::lcos::local::trigger int_sndrecv_gate_;
        and_gate_type partd_allgather_gate_;

        std::vector<hpx::naming::id_type> components_;
        mutable mutex_type mtx_;
        std::vector<int> intparams_;
        std::vector<double> realparams_;
        std::vector<double> dnireceive_;
        std::vector<double> treceive_;
        std::vector<double> comm_allreduce_receive_;
        std::vector<int> int_comm_allreduce_receive_;
        std::vector<int> int_comm_allgather_receive_;
        std::vector<double> sndrecv_;
        std::vector<double> ntoroidal_gather_receive_;
        std::vector<double> ntoroidal_scatter_receive_;
        std::vector<double> toroidal_reduce_;
        std::vector<std::complex<double> > complex_ntoroidal_gather_receive_;
        std::size_t myrank_toroidal_;
        std::size_t myrank_partd_;
        std::vector<double> p2p_sendreceive_;
        std::vector<double> comm_reduce_;
        std::vector<int> int_sndrecv_;
        std::vector<double> partd_allgather_;
    };
}}

// Declaration of serialization support for the actions
HPX_ACTION_USES_HUGE_STACK(gtcx::server::partition::loop_action);
HPX_REGISTER_ACTION_DECLARATION_EX(
    gtcx::server::partition::loop_action,
    gtcx_point_loop_action);

HPX_REGISTER_ACTION_DECLARATION_EX(
    gtcx::server::partition::set_data_action,
    gtcx_point_set_data_action);

HPX_REGISTER_ACTION_DECLARATION_EX(
    gtcx::server::partition::set_tdata_action,
    gtcx_point_set_tdata_action);

HPX_REGISTER_ACTION_DECLARATION_EX(
    gtcx::server::partition::set_params_action,
    gtcx_point_set_params_action);

HPX_REGISTER_ACTION_DECLARATION_EX(
    gtcx::server::partition::set_comm_allreduce_data_action,
    gtcx_point_set_comm_allreduce_data_action);

HPX_REGISTER_ACTION_DECLARATION_EX(
    gtcx::server::partition::set_int_comm_allreduce_data_action,
    gtcx_point_set_int_comm_allreduce_data_action);

HPX_REGISTER_ACTION_DECLARATION_EX(
    gtcx::server::partition::set_sndrecv_data_action,
    gtcx_point_set_sndrecv_data_action);

HPX_REGISTER_ACTION_DECLARATION_EX(
    gtcx::server::partition::set_ntoroidal_gather_data_action,
    gtcx_point_set_ntoroidal_gather_data_action);

HPX_REGISTER_ACTION_DECLARATION_EX(
    gtcx::server::partition::set_ntoroidal_scatter_data_action,
    gtcx_point_set_ntoroidal_scatter_data_action);

HPX_REGISTER_ACTION_DECLARATION_EX(
    gtcx::server::partition::set_complex_ntoroidal_gather_data_action,
    gtcx_point_set_complex_ntoroidal_gather_data_action);

HPX_REGISTER_ACTION_DECLARATION_EX(
    gtcx::server::partition::set_toroidal_reduce_data_action,
    gtcx_point_set_toroidal_reduce_data_action);

HPX_REGISTER_ACTION_DECLARATION_EX(
    gtcx::server::partition::set_p2p_sendreceive_data_action,
    gtcx_point_set_p2p_sendreceive_data_action);

HPX_REGISTER_ACTION_DECLARATION_EX(
    gtcx::server::partition::set_int_params_action,
    gtcx_point_set_int_params_action);

HPX_REGISTER_ACTION_DECLARATION_EX(
    gtcx::server::partition::set_comm_reduce_data_action,
    gtcx_point_set_comm_reduce_data_action);

HPX_REGISTER_ACTION_DECLARATION_EX(
    gtcx::server::partition::set_int_sndrecv_data_action,
    gtcx_point_set_int_sndrecv_data_action);

HPX_REGISTER_ACTION_DECLARATION_EX(
    gtcx::server::partition::set_real_params_action,
    gtcx_point_set_real_params_action);

HPX_REGISTER_ACTION_DECLARATION_EX(
    gtcx::server::partition::set_partd_allgather_data_action,
    gtcx_point_set_partd_allgather_data_action);

#endif

