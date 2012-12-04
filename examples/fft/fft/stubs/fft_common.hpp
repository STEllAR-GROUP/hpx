#if !defined(HPX_S3Akvs8z5OMdLum3wXPJhXY0GvAp3AAISgvHcP1h)
#define HPX_S3Akvs8z5OMdLum3wXPJhXY0GvAp3AAISgvHcP1h

#include "../server/fft_common.hpp"

/*namespace fft { namespace stubs
{
    struct fft_common
        : hpx::components::stubs::stub_base<server::fft_common>
    {
        //////////////////////////////////////////////////////////////////////
        static hpx::lcos::future<void>
        init_async(hpx::naming::id_type const& gid, std::string const& filename
            , std::size_t const& num_workers, std::size_t num_localities)
        {
            typedef fft::server::fft_common::init_action action_type;
            hpx::async<action_type>(gid, filename, num_workers, num_localities);
        }

        static void init(hpx::naming::id_type const& gid, std::string const& 
            filename, std::size_t const& num_workers, std::size_t num_localities)
        {
            init_async(gid, filename, num_workers, num_localities);
        }
        ///////////////////////////////////////////////////////////////////////
        static hpx::lcos::future<fft::config_data>
        config_read_async(hpx::naming::id_type const& gid)
        {
            typedef fft::server::fft_common::config_read_action action_type;
            return hpx::async<action_type>(gid);
        }

        static fft::config_data config_read(hpx::naming::id_type const& gid)
        {
            return config_read_async(gid).get(); 
        }
    };

}}*/

#endif //HPX_S3Akvs8z5OMdLum3wXPJhXY0GvAp3AAISgvHcP1h
