

#if !defined(HPX_QS8bOEkaaAXoeu7EuXAR5ECiGiXXqYTEOsv7oa1h)
#define HPX_QS8bOEkaaAXoeu7EuXAR5ECiGiXXqYTEOsv7oa1h

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/promise.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>

#include "../server/datastructure.hpp"

namespace distributed { namespace stubs
{
    struct datastructure
        : hpx::components::stubs::stub_base<distributed::server::datastructure>
    {
        ////////////////////////
        static hpx::lcos::promise<void>
        data_init_async(hpx::naming::id_type const& gid
            ,std::string const& symbolic_name, std::size_t num_instances
            ,std::size_t my_cardinality, std::size_t init_length,
            std::size_t init_value)
        {
            // init_async --> promise
            typedef distributed::server::datastructure::init_action action_type;
            return hpx::lcos::eager_future<action_type>(
                gid, symbolic_name, num_instances, my_cardinality, init_length
                , init_value);
        }
    
        static void data_init(hpx::naming::id_type const& gid
            , std::string const& symbolic_name, std::size_t num_instances
            , std::size_t my_cardinality, std::size_t init_length
            , std::size_t init_value)
        {
            data_init_async(gid, symbolic_name, num_instances, my_cardinality
            , init_length, init_value).get();
        }
        ////////////////////////
    };
    
}}

#endif //HPX_QS8bOEkaaAXoeu7EuXAR5ECiGiXXqYTEOsv7oa1h

