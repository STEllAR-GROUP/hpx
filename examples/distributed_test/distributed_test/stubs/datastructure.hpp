//  Copyright (c) 2012 Vinay C Amatya
//
//  Distributed under the Boost Software License, Version 1.0. (Seec accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_QS8bOEkaaAXoeu7EuXAR5ECiGiXXqYTEOsv7oa1h)
#define HPX_QS8bOEkaaAXoeu7EuXAR5ECiGiXXqYTEOsv7oa1h

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>

#include "../server/datastructure.hpp"

namespace distributed { namespace stubs
{
    struct datastructure
        : hpx::components::stub_base<distributed::server::datastructure>
    {
        typedef std::vector<std::size_t> client_data_type;
        /////////////////////////////////////////////////////////////////////
        static hpx::lcos::future<void>
        data_init_async(hpx::naming::id_type const& gid
            ,std::string const& symbolic_name, std::size_t num_instances
            ,std::size_t my_cardinality, std::size_t init_length,
            std::size_t init_value)
        {
            // init_async --> future
            typedef distributed::server::datastructure::init_action action_type;
            return hpx::async<action_type>(
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
        ////////////////////////////////////////////////////////////////////
        static hpx::lcos::future<void>
        data_write_async(hpx::naming::id_type const& gid
            , std::string const& symbolic_name, std::size_t num_instances
            , std::size_t my_cardinality, client_data_type client_data)
        {
            typedef distributed::server::datastructure::write_action action_type;
            return hpx::async<action_type>(
                gid, symbolic_name, num_instances, my_cardinality, client_data);
        }

        static void data_write(hpx::naming::id_type const& gid
            , std::string const& symbolic_name, std::size_t num_instances
            , std::size_t my_cardinality, client_data_type client_data)
        {
            data_write_async(gid, symbolic_name, num_instances, my_cardinality
                , client_data);
        }
        ////////////////////////////////////////////////////////////////////
        static hpx::lcos::future<distributed::config_comp>
        get_config_info_async (hpx::naming::id_type const& gid)
        {
            typedef distributed::server::datastructure::get_config_action 
                action_type;
            return hpx::async<action_type>(gid);
        }

        static distributed::config_comp get_config_info(
            hpx::naming::id_type const& gid)
        {
            return get_config_info_async(gid).get();
        }
        ////////////////////////////////////////////////////////////////////
        static hpx::lcos::future<std::vector<std::size_t>>
        get_data_async (hpx::naming::id_type const& gid)
        {
            typedef distributed::server::datastructure::get_data_action action_type;
            return hpx::async<action_type>(gid);
        }

        static std::vector<std::size_t> get_data( hpx::naming::id_type const& gid)
        {
            return get_data_async(gid).get();
        }
        ////////////////////////////////////////////////////////////////////
        static hpx::lcos::future<std::size_t>
        get_data_at_async(hpx::naming::id_type const& gid, std::size_t pos)
        {
            typedef distributed::server::datastructure::get_data_at_action
                action_type;
            return hpx::async<action_type>(gid, pos);
        }

        static std::size_t get_data_at(hpx::naming::id_type const& gid
            , std::size_t pos)
        {
            return get_data_at_async(gid, pos).get();
        }
        ////////////////////////////////////////////////////////////////////
    };
    
}}

#endif //HPX_QS8bOEkaaAXoeu7EuXAR5ECiGiXXqYTEOsv7oa1h

