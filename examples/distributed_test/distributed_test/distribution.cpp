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
        , std::size_t num_instances, std::size_t my_cardinality
        , std::size_t init_length, std::size_t init_value)
    {
        hpx::components::component_type type = 
            hpx::components::get_component_type<server::datastructure>();
        
        typedef hpx::components::distributing_factory distributing_factory;
    
        distributing_factory factory(
            distributing_factory::create_sync(hpx::find_here()));

        //hpx::naming::id_type temp_myid = hpx::find_here();

        //asyncronously create comonents, which will be distributed across
        //all available localities
        
        distributing_factory::async_create_result_type result = 
            factory.create_components_async(type, num_instances);

        //initialize locality mappings
        localities_.reserve(num_instances);

        //wait for the components to be created
        distributing_factory::result_type results = result.get();
        distributing_factory::iterator_range_type parts = 
            hpx::components::server::locality_results(results);

        std::size_t cardinality = 0;
        //Also define cardinality here
        BOOST_FOREACH(hpx::naming::id_type id, parts){
            localities_.push_back(id);
        }

        //Initialize all attached component objects
        std::size_t num_localities = localities_.size();
        BOOST_ASSERT( 0 != num_localities);

        std::vector<hpx::naming::id_type> prefixes = hpx::find_all_localities();
       
        std::vector<hpx::lcos::promise<void> > result_future;

        std::vector<hpx::naming::id_type>::iterator loc_itr = localities_.begin();
        while(loc_itr != localities_.end())
        {
            result_future.push_back(stubs::datastructure::data_init_async(*loc_itr
                , symbolic_name_base, num_localities, cardinality
                , init_length, init_value));
            ++cardinality;
            ++loc_itr;
        }

        hpx::lcos::wait(result_future);
        //Create component object locally. 

        comp_created_ = true;
        
    }
}
