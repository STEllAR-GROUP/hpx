#if !defined(HPX_ksl6hRsPmUi084tJmYkaLibE1l7XSJR6zKo0jV8K)
#define HPX_ksl6hRsPmUi084tJmYkaLibE1l7XSJR6zKo0jV8K
#include <hpx/hpx_fwd.hpp>
//#include <hpx/lcos/future.hpp>
//#include <hpx/runtime/components/stubs/stub_base.hpp>
#include "../server/data_split.hpp"

#include <vector>

/*namespace fft { namespace stubs{
    ///////////////////////////////////////////////////////////////////////////
    struct distribute
        : hpx::components::stubs::stub_base<fft::server::distribute>
    {
        typedef std::vector<fft::complex_type> complex_vector;
        //////////////////////////////////////////////////////////////////////
        static hpx::lcos::future<void>
            init_config_async(hpx::naming::id_type const& gid
            , std::string const& data_filename, std::string const& 
            symbolic_name_base, std::size_t const& num_workers, std::size_t const& 
            num_localities)
        {
            typedef fft::server::distribute::init_data_action action_type;
            return hpx::async<action_type>(gid, data_filename,  symbolic_name_base
                , num_workers, num_localities);
        }

        static void init_config(hpx::naming::id_type const& gid
            , std::string const& data_filename
            , std::string const& symbolic_name_base
            , std::size_t const& num_workers, std::size_t const& 
            num_localities)
        {
            init_config_async(gid, data_filename, symbolic_name_base
                , num_workers, num_localities).get();
        }
        //////////////////////////////////////////////////////////////////////
        static hpx::lcos::future<void>
        init_data_async(hpx::naming::id_type const& gid
            , complex_vector const& c_vec)
        {
            typedef fft::server::distribute::init_data_action action_type;
            return hpx::async<action_type>(gid, c_vec);
        }

        static void init_data(hpx::naming::id_type const& gid
            , complex_vector const& c_vec)
        {
            init_data_async(gid, c_vec).get();
        }
        ////////////////////////////////////////////////////////////////////////
        static hpx::lcos::future<void>
        split_fn_async(hpx::naming::id_type const& gid)
        {
            typedef fft::server::distribute::split_fn_action action_type;
            return hpx::async<action_type>(gid);
        }

        static void split_fn(hpx::naming::id_type const& gid)
        {
            split_fn_async(gid).get();
        }
       
    };
}}*/



#endif //HPX_ksl6hRsPmUi084tJmYkaLibE1l7XSJR6zKo0jV8K
