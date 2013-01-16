
#include <hpx/hpx.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/serialization.hpp>
#include <hpx/components/distributing_factory/server/distributing_factory.hpp>
#include <hpx/util/locality_result.hpp>


#include <fstream>
#include <iostream>
#include <vector>
#include <utility>

#include "./server/fft_d_factory.hpp"
#include "./server/data_split.hpp"
#include "fft_distrib.hpp"

#define default_size 1024

namespace fft
{
    fft_distrib::fft_distrib()
    {}
    fft_distrib::~fft_distrib()
    {}

    fft_distrib::fft_distrib(std::string const& datafilename
        , std::string const& symbolic_name_base, std::size_t const& num_workers
        , bool const& use_dataflow)
        :datafilename_(datafilename), sn_base_(symbolic_name_base)
        , num_workers_(num_workers), use_dataflow_(use_dataflow)
    {
		//std::cout << "Filename(distrib-obj creation), fft_distrib:" <<
		//datafilename_ << std::endl;
	}
    /////////////////////////////////////////////////////////////////////////////
    // create instance of component
    void fft_distrib::instantiate()
    {
        /// component type to be distributed across localities
        hpx::components::component_type config_type = 
            hpx::components::get_component_type<server::distribute>();

        hpx::naming::id_type this_prefix = hpx::find_here();

        BOOST_ASSERT(num_workers_ != 0);
        //std::size_t count = num_workers_;
        
        /// get list of localities that support the given component_type
        std::vector<hpx::naming::id_type> localities = 
            hpx::find_all_localities(config_type);
        
        std::size_t num_localities = localities.size();
        fft::server::d_factory::create_components_structured_action 
            action_type;
        /// distribute the number of components 
        fft_distrib::async_create_result_type results;

        results = hpx::async(action_type, this_prefix, config_type,  num_workers_);

        init(datafilename_, sn_base_, num_workers_, num_localities, results);

        //bool component_instantiated = true;
    }

    void fft_distrib::init(std::string const& datafilename
        , std::string const& symbolic_name_base, std::size_t const& num_workers
        ,  std::size_t const& num_localities
        , fft_distrib::async_create_result_type& futures)
    {

        typedef fft::server::distribute::init_config_action 
            init_action_type;
        fft_distrib::result_type results = futures.get();
        //fft_distrib::result_type result_idtype;
        std::size_t available_locs = results.size();
        std::size_t cardinality = 0;
	
		//std::cout << "Filename inside init, fft-distrib:" << datafilename 
		//<< std::endl;

        /// pair< locality, cardinality> 
        ///fft::comp_rank_vec_type comp_rank_pair;
        
        std::vector<hpx::lcos::future<void > > f;

        BOOST_FOREACH(locality_result_type& res_t, results)
        {
            //hpx::lcos::future<void> f;
            //localities_.push_back(res_t.prefix_);
            BOOST_FOREACH(hpx::naming::id_type& comp_id, res_t.gids_)
            {
                comp_rank_vec_.push_back(std::make_pair(comp_id, cardinality));
                if(cardinality == 0)
                    loc_zero_comp_id_ = comp_id;     //location zero comp_id
                ++cardinality;
            }
            
        }
        
        //comp_rank_ = comp_rank;
        BOOST_FOREACH(fft::comp_rank_pair_type& cr, comp_rank_vec_)
        {
            f.push_back(hpx::async<init_config_action_type>(cr.first, 
                datafilename, symbolic_name_base, num_workers, num_localities,
                cr.first, comp_rank_vec_));
        }
        
        //--cardinality;
        //BOOST_ASSERT((cardinality - 1) == available_locs);

        BOOST_FOREACH(hpx::lcos::future<void> &future_ret, f)
        {
            future_ret.get();
        }
    }

    fft::comp_rank_vec_type fft_distrib::get_comp_rank_vec()
    {
        return comp_rank_vec_;
    }

    void fft_distrib::read_split_data()
    {
        std::vector<hpx::lcos::future<void> > f; 

        fft::comp_rank_vec_type comp_rank_vec = fft_distrib::get_comp_rank_vec();
        BOOST_FOREACH(fft::comp_rank_pair_type& cr, comp_rank_vec)
        {
            f.push_back(hpx::async<split_init_action_type>(cr.first));
        }
        
        BOOST_FOREACH(hpx::lcos::future<void> &future_ret, f)
        {
            future_ret.get();
        }
    }

    fft_distrib::complex_vec fft_distrib::transform()
    {
        typedef fft::server::distribute::dist_transform_action 
            transform_action_type;
        typedef fft::server::distribute::dist_transform_dataflow_action
            transform_dataflow_action_type;
        typedef fft::server::distribute::get_result_action 
            get_result_action_type;
        typedef fft::server::distribute::dataflow_fetch_remote_action 
            dflow_fetch_remote_action_type;
        
        fft::comp_rank_vec_type comp_rank_vec = get_comp_rank_vec();
        
        std::vector<hpx::lcos::future<void> > transform_;

        if(!use_dataflow_)
        {
            BOOST_FOREACH(fft::comp_rank_pair_type& cr, comp_rank_vec)
            {
                transform_.push_back(hpx::async<
                    transform_action_type>(cr.first));
            }
        }
        else
        {
            BOOST_FOREACH(fft::comp_rank_pair_type& cr, comp_rank_vec)
            {
                transform_.push_back(hpx::async<
                    transform_dataflow_action_type>(cr.first));
            }
        }

        BOOST_FOREACH(hpx::lcos::future<void>& txfm, transform_)
        {
            txfm.get();
        }

        hpx::lcos::future<fft::fft_distrib::complex_vec> final_result;
        
        if(!use_dataflow_)
        {
            final_result = 
                hpx::async<get_result_action_type>(get_local_comp_id());
        }
        else
        {
            final_result = 
                hpx::async<dflow_fetch_remote_action_type>(get_local_comp_id());
        }
        return final_result.get();
    }

    hpx::naming::id_type fft_distrib::get_local_comp_id()
    {
        return loc_zero_comp_id_;
    }
}
