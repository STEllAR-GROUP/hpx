//  Copyright (c) 2012 Vinay C Amatya
//
//  Distributed under the Boost Software License, Version 1.0. (Seec accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/runtime/agas/interface.hpp>
#include <hpx/runtime/components/component_factory_base.hpp>
#include <hpx/components/distributing_factory/distributing_factory.hpp>

#include <boost/foreach.hpp>
#include <boost/assert.hpp>

#include "distribution.hpp"

HPX_REGISTER_COMPONENT_MODULE(); //entry point for component factory

typedef distributed::datastructure datastructure_client_type;
HPX_DEFINE_GET_COMPONENT_TYPE(datastructure_client_type);

//----------------------------------------------------

namespace distributed
{
    distribution::distribution():comp_created_(false)
    {}
    distribution::~distribution(){}

    void distribution::create(std::string const& symbolic_name_base
        , std::size_t num_instances, std::vector<std::size_t> data_received)
    {
        std::size_t init_length = 1, init_value = 0;
        hpx::components::component_type type =
            hpx::components::get_component_type<server::datastructure>();

        typedef hpx::components::distributing_factory distributing_factory;

        distributing_factory factory(
            distributing_factory::create_sync(hpx::find_here()));
        //asyncronously create comonents, which will be distributed across
        //all available localities
        distributing_factory::async_create_result_type result =
            factory.create_components_async(type, num_instances);

        //initialize locality mappings: Total Component instances
        comp_instances_.reserve(num_instances);

        //wait for the components to be created
        distributing_factory::result_type results = result.get();
        distributing_factory::iterator_range_type parts =
            hpx::components::server::locality_results(results);

        std::size_t cardinality = 0;
        //Also define cardinality here: TO DO
        BOOST_FOREACH(hpx::naming::id_type id, parts){
            comp_instances_.push_back(id);
        }

        //Initialize all attached component objects
        std::size_t num_comps = comp_instances_.size();
        BOOST_ASSERT( 0 != num_comps);
        BOOST_ASSERT( 0 != num_instances);
        std::vector<hpx::naming::id_type> prefixes = hpx::find_all_localities();

        std::vector<hpx::lcos::future<void> > result_future;

        std::vector<hpx::naming::id_type>::iterator loc_itr = comp_instances_.begin();
        while(loc_itr != comp_instances_.end())
        {
            result_future.push_back(stubs::datastructure::data_init_async(*loc_itr
                , symbolic_name_base, num_comps, cardinality
                , init_length, init_value));
            ++cardinality;
            ++loc_itr;
        }
        hpx::lcos::wait(result_future);

        typedef std::vector<std::vector<std::size_t> > client_data_type;

        client_data_type dd_vector;

        split_client_data(num_instances, data_received, dd_vector);
        //loc_itr = comp_instances_.begin();
        //result_future.resize(0);
        std::vector<hpx::lcos::future<void> > result_future2;
        client_data_type::iterator dd_itr;
        dd_itr = dd_vector.begin();
        loc_itr = comp_instances_.begin();
        cardinality = 0;
        while(loc_itr != comp_instances_.end())
        {
            BOOST_ASSERT(dd_itr <= dd_vector.end());
            result_future2.push_back(stubs::datastructure::data_write_async(
                *loc_itr, symbolic_name_base, num_comps, cardinality
                , *dd_itr));
            ++cardinality;
            if(dd_itr < dd_vector.end())
                ++dd_itr;
            if(loc_itr < comp_instances_.end())
                ++loc_itr;
        }
        hpx::lcos::wait(result_future2);
        //Create component object locally.
        loc_itr = comp_instances_.begin();
        hpx::lcos::future<distributed::config_comp> config_result =
            stubs::datastructure::get_config_info_async(*loc_itr);
        distributed::config_comp config_data = config_result.get();
        hpx::lcos::future<std::vector<std::size_t>> data_fraction;
        data_fraction = stubs::datastructure::get_data_async(*(++loc_itr));
        std::vector<std::size_t> temp_vector = data_fraction.get();
                //hpx::lcos::wait(data_fraction);
        comp_created_ = true;
    }

    void distribution::split_client_data(
        std::size_t num_instances, std::vector<std::size_t> &data_received
        , std::vector<std::vector<std::size_t>> &dd_vector)
    {

        dd_vector.resize(0);
        std::size_t client_vec_length = data_received.size();
        std::size_t quotient = client_vec_length / num_instances;
        std::size_t rem = client_vec_length%num_instances;
        std::vector<std::size_t>::iterator itr;
        itr = data_received.begin();
        std::vector<std::size_t> temp;
        if(rem == 0)
        {
            max_comp_size_ = quotient;  //total component size
            for(std::size_t i = 0; i<num_instances; ++i)
            {
                temp.resize(0);
                if(itr <= (data_received.end() - quotient))
                {
                    temp.assign(itr, itr + quotient);
                    dd_vector.push_back(temp);
                    itr+=quotient+1;
                }
            }
        }
        else
        {
            max_comp_size_ = quotient + 1;  //total component size
            for(std::size_t i = 0; i<num_instances - 1; ++ i)
            {
                temp.resize(0);
                if(itr <= (data_received.end() - quotient))
                {
                    temp.assign(itr, itr+(max_comp_size_));
                    dd_vector.push_back(temp);
                    itr+=max_comp_size_;
                }
            }

            temp.resize(0);
            temp.assign(itr, data_received.end());
            dd_vector.push_back(temp);
        }
    }

    std::size_t distribution::get_data_at(std::size_t n)
    {
        // n_pos is ordinal position
        std::size_t find_component_no, temp = 0, n_pos = n+1, quo, rem;
        std::size_t max_elements = comp_instances_.size() * max_comp_size_;

        std::vector<hpx::naming::id_type>::iterator itr;
        if(n_pos <= max_elements)
        {
            if(n_pos <= max_comp_size_)
            {
                itr = comp_instances_.begin();
            }
            else
            {
                itr = comp_instances_.begin();
                temp+= max_comp_size_;
                while( temp <= n_pos )
                {
                    ++itr;
                    temp+= max_comp_size_;
                }
            }

            rem = n_pos%max_comp_size_;
            quo = n_pos/max_comp_size_;

            if(rem == 0)
            {
                //use max_comp_size_ (element_ordinal_position)

                //return hpx::apply<distributed::server::datastructure::get_data_at_action>
                //return hpx::apply<distributed::server::datastructure::get_data_at_action>
                //    (*itr, max_comp_size_ - 1);
                hpx::lcos::future<std::size_t> value_at =
                    stubs::datastructure::get_data_at_async(*itr, max_comp_size_ - 1 );
                return value_at.get();
            }
            else
            {
                //use rem (element_ordinal_position)

                //return hpx::applier::apply<distributed::server::datastructure::get_data_at_action>
                //return hpx::lcos::apply<distributed::server::datastructure::get_data_at_action>
                //    (*itr, rem - 1);
                hpx::lcos::future<std::size_t> value_at =
                    stubs::datastructure::get_data_at_async(*itr, rem - 1);
                return value_at.get();
            }
        }
        else
        {
            return 0;
        }
    }
}
