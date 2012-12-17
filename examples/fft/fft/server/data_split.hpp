#if !defined(HPX_JSAkwkF4IeYspGV6Psl21TtfPwEFYdJyJjaHZCML)
#define HPX_JSAkwkF4IeYspGV6Psl21TtfPwEFYdJyJjaHZCML

//#define HPX_LIMIT 6

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/runtime/components/server/simple_component_base.hpp>
#include <hpx/components/dataflow/dataflow.hpp>

#include <fstream>
#include <iostream>
#include <vector>
#include <utility>
#include "fft_common.hpp"
#include "fft.hpp"

namespace fft {namespace server{

    using hpx::lcos::dataflow;
    using hpx::lcos::dataflow_base;

    class HPX_COMPONENT_EXPORT distribute
        : public hpx::components::simple_component_base<distribute>
    {
    public:
        distribute();
        ~distribute();

        enum actions 
        {
            d_init_config = 0
        };
        
        typedef std::vector<fft::complex_type> complex_vec;
        //initialize distribute, filename, etc
        void init_config(std::string const& data_filename
            , std::string const& symbolic_name_base
            , std::size_t const& num_workers
            , std::size_t const& num_localities
            , hpx::naming::id_type const& comp_id
            , fft::comp_rank_vec_type const& comp_rank);
        
        void split_init_data();
        complex_vec read_file_data(std::string const& data_filename);
        complex_vec split_data_oe(complex_vec const& initial_vec, 
            std::size_t const& cardinality, std::size_t const& num_localities);
                
        void dist_transform();
        void dist_transform_dataflow();
        complex_vec fetch_remote(std::size_t previous_level);
        complex_vec dataflow_fetch_remote();
        complex_vec get_result();
        void dsend_remote(hpx::lcos::dataflow_base<complex_vec> result_vec);

        //complex_vec this_identity(complex_vec init_value);
        template <typename T>
        T this_identity(T init_value);

        complex_vec remote_xform(std::size_t dlevel);

        HPX_DEFINE_COMPONENT_ACTION(distribute, init_config, init_config_action);
        HPX_DEFINE_COMPONENT_ACTION(distribute, split_init_data,
            split_init_data_action);
        HPX_DEFINE_COMPONENT_ACTION(distribute, read_file_data, read_data_action);
        HPX_DEFINE_COMPONENT_ACTION(distribute, split_data_oe, split_fn_action);
        HPX_DEFINE_COMPONENT_ACTION(distribute, dist_transform, dist_transform_action);
        HPX_DEFINE_COMPONENT_ACTION(distribute, dist_transform_dataflow
            , dist_transform_dataflow_action);
        HPX_DEFINE_COMPONENT_ACTION(distribute, fetch_remote, fetch_remote_action);
        HPX_DEFINE_COMPONENT_ACTION(distribute, dataflow_fetch_remote
            , dataflow_fetch_remote_action);
        HPX_DEFINE_COMPONENT_ACTION(distribute, get_result, get_result_action);
        HPX_DEFINE_COMPONENT_ACTION(distribute, dsend_remote, dsend_remote_action);
        //HPX_DEFINE_COMPONENT_ACTION(distribute, this_identity, dflow_init_action);
        HPX_DEFINE_COMPONENT_ACTION(distribute, remote_xform, remote_xform_action);

        template <typename T>
        struct dflow_init_action
            : hpx::actions::make_action<T (distribute::*)(T), 
                &distribute::template this_identity<T>, dflow_init_action<T> >
        {};
        
    private:

        std::size_t get_prev_level();

        fft::config_data data_;
        complex_vec local_vec_;
        complex_vec result_vec_;
        complex_vec remote_vec_;
        hpx::lcos::dataflow_base<complex_vec> dlocal_vec_, dresult_vec_, dremote_vec_;
        std::size_t level_, level_previous_;
    };
}}

//////////////////////////////////////////////////////////////////////////////

HPX_REGISTER_ACTION_DECLARATION(fft::server::distribute::init_config_action
    , fft_distribute_init_config_action);
HPX_REGISTER_ACTION_DECLARATION(fft::server::distribute::split_init_data_action
    , fft_distribute_split_init_data_action);                                          
HPX_REGISTER_ACTION_DECLARATION(fft::server::distribute::split_fn_action
    , fft_distribute_split_f_action);
HPX_REGISTER_ACTION_DECLARATION(fft::server::distribute::read_data_action
    , fft_distribute_read_data_action);
HPX_REGISTER_ACTION_DECLARATION(
    fft::server::distribute::fetch_remote_action
    , fft_distribute_fetch_remote_action);
HPX_REGISTER_ACTION_DECLARATION(
    fft::server::distribute::dataflow_fetch_remote_action
    , fft_distribute_dataflow_fetch_remote_action);
HPX_REGISTER_ACTION_DECLARATION(
    fft::server::distribute::get_result_action
    , fft_distribute_get_result_action);
//HPX_REGISTER_ACTION_DECLARATION(fft::server::distribute::dremote_action,
//    fft_distribute_dsend_remote_action);
HPX_REGISTER_ACTION_DECLARATION(fft::server::distribute::remote_xform_action,
    fft_distribute_remote_xform_action);
HPX_REGISTER_ACTION_DECLARATION(fft::server::distribute::dist_transform_action,
    fft_distribute_dist_transform_action);
HPX_REGISTER_ACTION_DECLARATION(
    fft::server::distribute::dist_transform_dataflow_action,
    fft_distribute_dist_transform_dataflow_action);
//HPX_REGISTER_ACTION_DECLARATION(fft::server::distribute::dflow_init_action,
//    fft_distribute_dflow_init_action);
HPX_REGISTER_ACTION_DECLARATION_TEMPLATE(
    (template <typename T>),
    (fft::server::distribute::dflow_init_action<T>)
)
#endif //HPX_JSAkwkF4IeYspGV6Psl21TtfPwEFYdJyJjaHZCML
