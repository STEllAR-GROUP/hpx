#include <hpx/hpx_fwd.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/serialization.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>
#include <hpx/lcos/wait_any.hpp>

#include <vector>

#include <boost/thread.hpp>

#include "data_split.hpp"
//#include "../server/fft.hpp"
#include "../stubs/fft.hpp"
//#define default_size 1024;          // remove this

namespace fft { namespace server
{
    using hpx::lcos::dataflow;
    using hpx::lcos::dataflow_base;

    typedef fft::server::distribute::fetch_remote_action fetch_remote_action_type;
    typedef std::pair<hpx::naming::id_type, std::size_t> pair_type;
    typedef server::fourier_xform::r2ditfft_action r2ditfft_action_type;
    typedef server::fourier_xform::r2ditfft_args_action r2ditfft_args_action_type;  
    
    typedef distribute::dsend_remote_action dsend_remote_action;
    typedef distribute::remote_xform_action remote_action;
    typedef distribute::complex_vec complex_vec;
    
    distribute::distribute() : local_vec_(NULL)
    {}
    distribute::~distribute(){}

    void distribute::init_config(std::string const& data_filename
        , std::string const& symbolic_name_base
        , std::size_t const& num_workers 
        , std::size_t const& num_localities
        , hpx::naming::id_type const& comp_id
        , fft::comp_rank_vec_type const& comp_rank)
    {
        data_.data_filename_ = data_filename; 
        data_.symbolic_name_ = symbolic_name_base;
        data_.num_workers_= num_workers;
        data_.num_localities_ = num_localities;
        data_.comp_rank_vec_ = comp_rank;
        data_.comp_id_ = comp_id;
        data_.valid_ = true;
        
        BOOST_FOREACH(fft::comp_rank_pair_type& cr, data_.comp_rank_vec_)
        {
            if(cr.first == data_.comp_id_)
                data_.comp_cardinality_ = cr.second;
        }
        std::cout << "My locality rank(cardinality):" << data_.comp_cardinality_ << std::endl;
    }

    void distribute::split_init_data()
    {
      typedef fft::server::distribute::read_data_action read_action_type;
      typedef fft::server::distribute::split_fn_action split_action_type;

      hpx::lcos::future<complex_vec> file_data = 
          hpx::async<read_action_type>(data_.comp_id_, data_.data_filename_);
          
      hpx::lcos::future<complex_vec> split_data = 
          hpx::async<split_action_type>(data_.comp_id_
          , file_data.get(),data_.comp_cardinality_
          , data_.num_localities_ * data_.num_workers_);

      local_vec_ = split_data.get();
    }

    distribute::complex_vec 
    distribute::read_file_data(std::string const& data_filename)
    {
        distribute::complex_vec x;                                                       
        distribute::complex_vec::iterator itr; 
        std::size_t fft_size;

        std::ifstream fin(data_filename);                                          
        if(fin.is_open())                                                          
        {                                                                          
            while(!fin.eof())                                                           
            {                                                                        
                fft::complex_type temp(0,0);                                         
                fin >> temp.re >> temp.im;                                       
                x.push_back(temp);                                               
            }
            fft_size = x.size() - 1;
            x.resize(fft_size);
            std::cout << " Read from file complete!! Size of read array:" <<         
                x.size() << std::endl;                                               
                                                                                    
            fin.close();
        }
        else                                                                         
        {                                                                            
            HPX_THROW_EXCEPTION(hpx::bad_parameter,
                "fft::distribute::read_file_data",
                "unable to open file");
        } 
        return x;
    }

    //Split vector into odd/even part, and return either odd/even part
    distribute::complex_vec 
    distribute::split_data_oe(distribute::complex_vec const& 
        input_vec, std::size_t const& cardinality, std::size_t const& num_components)
    {
        int cardinality_ = cardinality;
        BOOST_ASSERT(local_vec_.size() == 0);
        BOOST_ASSERT(data_.num_localities_ != 0);
        
        std::size_t level = num_components;
        complex_vec split_odd, split_even;
        complex_vec temp = input_vec;
        
        while(level != 1)          // data level
        {
            std::size_t n = temp.size();
            std::vector<fft::complex_type>::iterator itr = temp.begin();
            std::size_t k = 0;
            while(k < n/2)
            {
                split_even.push_back(*itr);
                ++itr;
                split_odd.push_back(*itr);
                ++itr;
                ++k;
            }

            level = level/2;                // half of locality

            if(level >= cardinality_ + 1)
            {
                temp = split_even;
            }
            else if(level < cardinality_ + 1)
            {
                cardinality_ = cardinality_ - level;
                temp = split_odd;
            }
            split_odd.resize(0);
            split_even.resize(0);
        }
        return temp;
    }
    
    void distribute::dist_transform()
    {   
        std::size_t my_cardinality, my_prev_cardinality;
        std::size_t my_remote_cardinality
            , total_components = data_.num_localities_ * data_.num_workers_;

        level_ = level_previous_ = 1;
        data_.current_level_ = level_;

        hpx::naming::id_type remote_gid;

        hpx::components::component_type fft_type =
            hpx::components::get_component_type<server::fourier_xform>();   

        hpx::naming::id_type this_prefix = hpx::find_here();
        
        hpx::lcos::future<hpx::naming::id_type> fft_gid = 
        hpx::components::stubs::runtime_support::create_component_async<fft::server::fourier_xform>(this_prefix);

        hpx::naming::id_type fft_gid_get = fft_gid.get();
        my_cardinality = my_prev_cardinality = data_.comp_cardinality_;
        std::size_t num_components = data_.num_localities_;

        result_vec_.resize(0);
        complex_vec temp;
        while(level_ <= total_components)
        {
            if(level_ == 1)
            {      
                hpx::lcos::future<complex_vec> result_vec = 
                    fft::stubs::fourier_xform::r2ditfft_async(fft_gid_get, local_vec_);    
                result_vec_ = result_vec.get();
                local_vec_ = result_vec_;

                //if(my_cardinality != 0)
                my_cardinality = my_cardinality >> 1;

                // update data_level for next level
                level_ = level_ << 1;
                data_.current_level_ = level_;
            }
            else
            {   
                if(my_prev_cardinality%2 == 0 && data_.valid_ == true)
                {
                    result_vec_.resize(0);

                    my_remote_cardinality = data_.comp_cardinality_ 
                        + level_previous_;
                    BOOST_FOREACH(comp_rank_pair_type p, data_.comp_rank_vec_)
                    {
                        if(my_remote_cardinality == p.second)
                            remote_gid = p.first;                                
                    }
                    hpx::lcos::future<complex_vec> remote_data = 
                            hpx::async<fetch_remote_action_type>(remote_gid
                                , level_previous_);
                          
                    remote_vec_ = remote_data.get();
                            
                    BOOST_ASSERT(remote_vec_.size() == local_vec_.size());
                    //my local component id
                    //compute on this locality
                    hpx::lcos::future<complex_vec> result_vec = 
                        fft::stubs::fourier_xform::r2ditfft_args_async
                        (fft_gid_get, local_vec_, remote_vec_);
                          
                    result_vec_ = result_vec.get();
                    local_vec_ = result_vec_;
                    //data_.comp_gid);
                    my_prev_cardinality = my_cardinality;
                    my_cardinality = my_cardinality >> 1;
                    
                    level_previous_ = level_;
                    level_ = level_ << 1;
                }
                else
                {
                    data_.valid_ = false;
                    level_ = level_ << 1;
                }
            }
        }
    }

    void distribute::dist_transform_dataflow()
    {         
        int my_cardinality, my_prev_cardinality;
        std::size_t my_remote_cardinality
            , total_components = data_.num_localities_ * data_.num_workers_;
        
        level_ = level_previous_ = data_.current_level_ = 1;

        hpx::naming::id_type here = hpx::find_here();

        hpx::naming::id_type remote_gid;

        hpx::components::component_type fft_type =
            hpx::components::get_component_type<server::fourier_xform>();   

        hpx::naming::id_type this_prefix = hpx::find_here();
        
        hpx::lcos::future<hpx::naming::id_type> fft_gid = 
            hpx::components::stubs::runtime_support::create_component_async<
            fft::server::fourier_xform>(this_prefix);

        hpx::naming::id_type fft_gid_get = fft_gid.get();
        my_cardinality = my_prev_cardinality = data_.comp_cardinality_;
        std::size_t num_components = data_.num_localities_;

        dlocal_vec_ = dataflow<fft::server::distribute::dflow_init_action<
                            complex_vec> >(here, local_vec_);
        
        while(level_ <= total_components)
        {
            if(level_ == 1)
            {
                if(my_cardinality%2 == 0)
                {   
                    dlocal_vec_ = dataflow<r2ditfft_action_type>(fft_gid_get, dlocal_vec_);
                    
                    if(total_components > 1)
                    {
                        my_remote_cardinality = data_.comp_cardinality_ + level_;
                        // BOOST_ASSERT
                        BOOST_FOREACH(comp_rank_pair_type p, data_.comp_rank_vec_)
                        {
                            if(my_remote_cardinality == p.second)
                                remote_gid = p.first;                                
                        }

                        dataflow_base<std::size_t> dlevel = 
                            dataflow<fft::server::distribute::dflow_init_action<
                                std::size_t> >(here, level_);
                        dremote_vec_ = dataflow<remote_action>(remote_gid, dlevel);
                    }
                                                       
                    my_cardinality = my_cardinality >> 1;
                    
                    // update data_level for next level
                    level_ = level_ << 1; 
                    data_.current_level_ = level_;
                }
                else
                {   
                    my_cardinality = my_cardinality >> 1;
                    level_ = level_ << 1;
                    data_.current_level_ = level_;
                    data_.valid_ = false;
                }
            }
            else
            {   
                if(my_cardinality%2 == 0 && data_.valid_ == true)
                {
                    dlocal_vec_ = dataflow<r2ditfft_args_action_type>(
                        fft_gid_get, dlocal_vec_, dremote_vec_);
                    
                    if(level_ != total_components)
                    {
                        my_remote_cardinality = data_.comp_cardinality_ + level_;
                        // BOOST_ASSERT
                        BOOST_FOREACH(comp_rank_pair_type p, data_.comp_rank_vec_)
                        {
                            if(my_remote_cardinality == p.second)
                                remote_gid = p.first;                                
                        }
                        dataflow_base<std::size_t> dlevel = 
                            dataflow<fft::server::distribute::dflow_init_action<
                            std::size_t> >(here, level_);
                        dremote_vec_ = dataflow<remote_action>(remote_gid, dlevel);
                    }
                    
                    my_prev_cardinality = my_cardinality;
                    my_cardinality = my_cardinality >> 1;

                    level_ = level_ << 1;
                    data_.current_level_ = level_;
                    level_previous_ = level_previous_ << 1;
                }
                else
                {
                    level_ = level_ << 1;
                    data_.current_level_ = level_;
                    data_.valid_ = false;
                }
            }
        }
        
        if(data_.comp_cardinality_ == 0)
        {
            //while(local_vec_.size() != 1024)
            //{
                local_vec_ = dlocal_vec_.get_future().get();
                //hpx::this_thread::suspend(boost::posix_time::microseconds(50));
            //}
        }
    }

    //use stubs?
    distribute::complex_vec distribute::fetch_remote(std::size_t remote_prev_level)
    {
        while(result_vec_.size() == 0 && (remote_prev_level != get_prev_level()))
        {
            hpx::this_thread::suspend(boost::posix_time::microseconds(50));
        }
        return this->local_vec_;
    }
    
    std::size_t distribute::get_prev_level()
    {
        return this->level_previous_;
    }

    distribute::complex_vec distribute::dataflow_fetch_remote()
    {
        data_.valid_ = false;
        return this->local_vec_;
    }
    
    distribute::complex_vec distribute::get_result()
    {
        return this->local_vec_;
    }

   complex_vec distribute::remote_xform(std::size_t remote_level)
    {
        //BOOST_ASSERT(remote_level == data_.current_level_);
        distribute::complex_vec temp;
        hpx::components::component_type fft_type =
            hpx::components::get_component_type<server::fourier_xform>();   

        hpx::naming::id_type this_prefix = hpx::find_here();
        
        hpx::lcos::future<hpx::naming::id_type> fft_gid = 
            hpx::components::stubs::runtime_support::create_component_async<
            fft::server::fourier_xform>(this_prefix);

        dataflow_base<complex_vec> temp_result;

        hpx::naming::id_type fft_gid_get = fft_gid.get();

        if(remote_level > 1)
        {
            temp_result = dataflow<r2ditfft_args_action_type>(
                fft_gid_get, dlocal_vec_, dremote_vec_);
        }
        else
        {
            temp_result = dataflow<r2ditfft_action_type>(
                fft_gid_get, dlocal_vec_);
        }
        temp = temp_result.get_future().get();
        return temp;
    }
 
    template<typename T>
    T distribute::this_identity(T init_value)
    {
        return init_value;
    }
}}

//////////////////////////////////////////////////////////////////////////////
typedef hpx::components::simple_component<
    fft::server::distribute
> distribute_component_type;

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(                                          
    distribute_component_type, distribute_type);

HPX_REGISTER_ACTION(fft::server::distribute::init_config_action,  
    fft_distribute_init_config_action);
HPX_REGISTER_ACTION(fft::server::distribute::split_init_data_action,  
    fft_distribute_split_init_data_action);                                          
HPX_REGISTER_ACTION(fft::server::distribute::split_fn_action,
    fft_distribute_split_fn_action);      
HPX_REGISTER_ACTION(fft::server::distribute::read_data_action,
    fft_distribute_read_data_action);
HPX_REGISTER_ACTION(fft::server::distribute::dist_transform_action, 
    fft_distribute_dist_transform_action);
HPX_REGISTER_ACTION(fft::server::distribute::dist_transform_dataflow_action, 
    fft_distribute_dist_transform_dataflow_action);
HPX_REGISTER_ACTION(fft::server::distribute::fetch_remote_action,
    fft_distribute_fetch_remote_action);
HPX_REGISTER_ACTION(
    fft::server::distribute::dataflow_fetch_remote_action
    , fft_distribute_dataflow_fetch_remote_action);
HPX_REGISTER_ACTION(fft::server::distribute::get_result_action
    , fft_distribute_get_result_action);
HPX_REGISTER_ACTION(fft::server::distribute::remote_xform_action,
    fft_distribute_remote_xform_action);
