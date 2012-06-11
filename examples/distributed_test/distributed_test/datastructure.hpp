//  Copyright (c) 2012 Vinay C Amatya
//
//  Distributed under the Boost Software License, Version 1.0. (Seec accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_xFq1eUILdlLmLfIiA5xbaFTHnuEhtkSQTOvbZtzx)
#define HPX_xFq1eUILdlLmLfIiA5xbaFTHnuEhtkSQTOvbZtzx

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/runtime/components/client_base.hpp>

#include "stubs/datastructure.hpp"

namespace distributed
{
    ////////////////////////////////////////////////////////////////////////////
    class datastructure
        : public hpx::components::client_base<
            datastructure, distributed::stubs::datastructure>
    {
    private:
        typedef hpx::components::client_base<
            datastructure, distributed::stubs::datastructure> base_type;
    
    public:
        // create new data and initialize in sync
        // can be initialized with gid or w/o gid!!
        datastructure(std::string const& symbolic_name, std::size_t num_instances
            , std::size_t my_cardinality, std::size_t initial_length
            , std::size_t initial_value)
            : base_type(distributed::stubs::datastructure::create_sync(hpx::find_here()))
        {
            data_init(symbolic_name, num_instances, my_cardinality
                , initial_length, initial_value);    
        }
        datastructure(hpx::naming::id_type gid, std::string const& symbolic_name
                , std::size_t num_instances, std::size_t my_cardinality
                , std::size_t initial_length, std::size_t initial_value)
            : base_type(distributed::stubs::datastructure::create_sync(gid))
        {
            data_init(symbolic_name, num_instances, my_cardinality
                , initial_length, initial_value);
        }
        datastructure(hpx::naming::id_type gid)
            : base_type(gid)
        {}
        datastructure()
        {}
        //////////////////////////////////////////////////////////////////////
        hpx::lcos::future<void>
        data_init_async(std::string const& symbolic_name, std::size_t num_instances
            , std::size_t my_cardinality, std::size_t initial_length
            , std::size_t initial_value)
        {
            return stubs::datastructure::data_init_async(this->gid_,
                symbolic_name, num_instances, my_cardinality, initial_length
                , initial_value);
        }
        
        void data_init(std::string const& symbolic_name, std::size_t num_instances
            , std::size_t my_cardinality, std::size_t initial_length
            , std::size_t initial_value)
        {
            stubs::datastructure::data_init(this->gid_, symbolic_name,
                num_instances, my_cardinality, initial_length, initial_value);
        }
        //////////////////////////////////////////////////////////////////////
        hpx::lcos::future<void>
        data_write_async(std::string const& symbolic_name, std::size_t num_instances
            , std::size_t my_cardinality, std::vector<std::size_t> client_data)
        {
            return stubs::datastructure::data_write_async(this->gid_,
                symbolic_name, num_instances, my_cardinality, client_data);
        }
        
        void data_write(std::string const& symbolic_name, std::size_t num_instances
            , std::size_t my_cardinality, std::vector<std::size_t> client_data)
        {
            stubs::datastructure::data_write(this->gid_, symbolic_name,
                num_instances, my_cardinality, client_data);
        }
        //////////////////////////////////////////////////////////////////////
        hpx::lcos::future<distributed::config_comp>
        get_config_info_async()
        {
            return stubs::datastructure::get_config_info_async(this->gid_);
        }

        distributed::config_comp get_config_info()
        {
            return stubs::datastructure::get_config_info(this->gid_);
        }
        //////////////////////////////////////////////////////////////////////
        hpx::lcos::future<std::vector<std::size_t>>
        get_data_async()
        {
            return stubs::datastructure::get_data_async(this->gid_);
        }

        std::vector<std::size_t> get_data()
        {
            return stubs::datastructure::get_data(this->gid_);
        }
        //////////////////////////////////////////////////////////////////////
        hpx::lcos::future<std::size_t>
        get_data_at_async(std::size_t pos)
        {
            return stubs::datastructure::get_data_at_async(this->gid_, pos);
        }

        std::size_t get_data_at(std::size_t pos)
        {
            return stubs::datastructure::get_data_at(this->gid_, pos);
        }
//////////////////////////////////////////////////////////////////////////////
    };
}


#endif //HPX_xFq1eUILdlLmLfIiA5xbaFTHnuEhtkSQTOvbZtzx
