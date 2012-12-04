#if !defined(HPX_rHj6TWiv7uC8pDbVJipH3xZR5uUKrSPqpzDK6zJ4)
#define HPX_rHj6TWiv7uC8pDbVJipH3xZR5uUKrSPqpzDK6zJ4

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/components/distributing_factory/distributing_factory.hpp>

#include <vector>

#include "stubs/fft_common.hpp"
#include "stubs/fft.hpp"
#include "stubs/data_split.hpp"

#include "server/fft_common.hpp"
#include "server/data_split.hpp"
#include "server/fft.hpp"

//////////////////////////////////////////////////////////////////////////////
namespace fft
{
    class HPX_COMPONENT_EXPORT fft_distrib
    {
        ////////////////////////////////////////////////////////////////////////
        //typedef std::vector<hpx::util::remote_locality_result> remote_result_type;
        //typedef hpx::util::remote_locality_result remote_locality_result_type;
        
        

        ////////////////////////////////////////////////////////////////////////
    public:
        fft_distrib();
        fft_distrib(std::string const& datafilename
        , std::string const& symbolic_name_base, std::size_t const& num_workers
        , bool const& use_dataflow);

        ~fft_distrib();
        
        typedef std::vector<fft::complex_type> complex_vec;

        typedef std::vector<hpx::util::locality_result> result_type;
        typedef hpx::util::locality_result locality_result_type;
        typedef hpx::lcos::future<result_type>
        //typedef fft::server::d_factory::async_create_result_type   
            async_create_result_type;

        typedef server::distribute::init_config_action init_config_action_type;

        //typedef fft::server::distribute::read_data_action read_action_type;
        //tyepdef fft::server::distribute::init_data_action init_data_actioin_type;
        typedef fft::server::distribute::split_init_data_action split_init_action_type;
        
        
        
        //Create Distributed FFT data object and initialize it
        void instantiate();

        //Distribute Data across localities and threads
        //void distribute(std::size_t loc_rank);
        //Perform Fourier transform on all localities
        fft_distrib::complex_vec transform();

        void read_split_data();
        fft::comp_rank_vec_type get_comp_rank_vec();
        hpx::naming::id_type get_local_comp_id();
    private:
        void init(std::string const& datafilename
            , std::string const& symbolic_name_base, std::size_t const& num_workers
            , std::size_t const& num_localities, async_create_result_type& results);

        /// returns component_rank_vector maintained by this object
        

        //complex_vec read_data_file(std::string const& data_filename);
    private:
        std::vector<hpx::naming::id_type> localities_;
        // components and their ranks
        fft::comp_rank_vec_type comp_rank_vec_;
        hpx::naming::id_type loc_zero_comp_id_;
        result_type result_type_;
        std::string datafilename_;
        std::string sn_base_; 
        std::size_t num_workers_;
        bool use_dataflow_;
    };
}

#endif //HPX_rHj6TWiv7uC8pDbVJipH3xZR5uUKrSPqpzDK6zJ4
