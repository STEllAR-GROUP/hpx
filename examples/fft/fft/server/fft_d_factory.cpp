

#include <hpx/hpx.hpp>                                                           
#include <hpx/exception.hpp>                                                     
#include <hpx/include/serialization.hpp>                                         
//#include <hpx/components/distributing_factory/server/distributing_factory.hpp>   

#include <boost/serialization/vector.hpp>                                        
#include <boost/move/move.hpp>                                                   
#include <vector>   

#include "fft_d_factory.hpp"

//////////////////////////////////////////////////////////////////////////////
namespace fft { namespace server {
    //////////////////////////////////////////////////////////////////////////
    struct lazy_result
    {
        lazy_result(hpx::naming::gid_type const& locality_id)
            : locality_(locality_id)
        {}

        hpx::naming::gid_type locality_;
        hpx::lcos::future<std::vector<hpx::naming::gid_type > > gids_;
    };

    //////////////////////////////////////////////////////////////////////////
    //create new component
    d_factory::remote_result_type 
    d_factory::create_components_structured(hpx::components::component_type type
        , std::size_t num_workers) const
    {
        // make sure we get localities for derieved component type
        // to do 

        // list of localities supporting given type
        std::vector<hpx::naming::id_type> localities = 
            hpx::find_all_localities(type);

        typedef std::vector<lazy_result> futures_type;
        typedef 
            hpx::components::server::runtime_support::bulk_create_components_action
            action_type;

        futures_type v;

        BOOST_FOREACH(hpx::naming::id_type const& fact, localities)
        {
            std::size_t numcreate = num_workers;
            //std::size_t numcreate = count_on_locality;

            // create components for each locality in one go
            v.push_back(futures_type::value_type(fact.get_gid()));
            hpx::lcos::packaged_action<action_type
                , std::vector<hpx::naming::gid_type> > p;
            p.apply(fact, type, numcreate);
            v.back().gids_ = p.get_future();    // gids for created components
        }

        //wait for the results
        //hpx::lcos::future<result_type, remote_result_type> 
        //    async_create_result_type;

        remote_result_type results;

        BOOST_FOREACH(lazy_result& lr, v)
        {
            results.push_back(remote_result_type::value_type(lr.locality_
                , type));
            //gids created in a locality
            results.back().gids_ = boost::move(lr.gids_.move());
        }

        return results;
    }

}}

//////////////////////////////////////////////////////////////////////////////
typedef hpx::components::simple_component<
    fft::server::d_factory
> d_factory_component_type;

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(                                          
    d_factory_component_type, d_factory_type);

HPX_REGISTER_ACTION(fft::server::d_factory::create_components_structured_action
    , d_factory_create_components_structured_action)
;
