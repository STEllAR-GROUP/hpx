#include <hpx/hpx_fwd.hpp>                                                           
#include <hpx/runtime/agas/interface.hpp>                                        
#include <hpx/runtime/components/component_factory_base.hpp>                     
//#include <hpx/runtime/components/component_startup_shutdown.hpp>                 
#include <hpx/components/distributing_factory/distributing_factory.hpp>          
                                                                                 
#include <hpx/include/async.hpp>                                                 
#include <hpx/lcos/future_wait.hpp>                                              
#include <hpx/lcos/local/packaged_task.hpp>                                      
                                                                                 
#include <boost/foreach.hpp>                                                     
#include <boost/assert.hpp>                                                      
#include <boost/make_shared.hpp>                                                 
#include <boost/move/move.hpp>

#include <utility>
#include <vector>

#include "server/tictactoe.hpp"
#include "dist_factory.hpp"

// factory registration functionality.
HPX_REGISTER_COMPONENT_MODULE();  // Create entry point for component factory

//typedef game::server::tictactoe tictactoe_client_type;
//HPX_DEFINE_GET_COMPONENT_TYPE(tictactoe_client_type);

namespace game 
{
	using hpx::lcos::dataflow;
	using hpx::lcos::dataflow_base;

    dist_factory::dist_factory()
    {}
    dist_factory::~dist_factory()
    {}

    char dist_factory::create()
    {
        //num_instances of components to be created: 2
        std::size_t num_instances;
        num_instances = 2;
        // component type of t-t-t
        hpx::components::component_type type =
            hpx::components::get_component_type<server::tictactoe>();

        typedef hpx::components::distributing_factory distributing_factory;
        distributing_factory factory;
        factory.create(hpx::find_here());
        distributing_factory::async_create_result_type result = 
            factory.create_components_async(type, num_instances);

        loc_comps_.reserve(num_instances);
        
        distributing_factory::result_type results = result.get();
        distributing_factory::iterator_range_type parts = 
            hpx::util::locality_results(results);

        BOOST_FOREACH(hpx::naming::id_type id, parts)
        {
            loc_comps_.push_back(id);
        }

        // toss, who is going to be x
        std::default_random_engine generator;
        std::uniform_int_distribution<std::size_t> distribution(0,1); 
        std::vector<std::pair<char, hpx::naming::id_type> > char_id_pair_vec;

        std::size_t toss = distribution(generator);
        std::size_t count = 0;

		std::vector<hpx::lcos::future<void> > init_future_vec;
        
        BOOST_FOREACH(hpx::naming::id_type id, loc_comps_)
        {
            if(toss == count)
            {
                char_id_pair_vec.push_back(std::make_pair('x', id));
            }
            else
            {
                char_id_pair_vec.push_back(std::make_pair('o', id));
            }
            ++count;
        }
        
        std::cout<< "Client Created" << std::endl; 

        //initialize remote components:
        typedef game::server::tictactoe::init_action init_action; 
        BOOST_FOREACH(hpx::naming::id_type id, loc_comps_)
        {
            init_future_vec.push_back(hpx::async<init_action>(id
				, char_id_pair_vec, id));
        }

		BOOST_FOREACH(hpx::lcos::future<void> fut, init_future_vec)
		{
			fut.get();
		}
        std::cout << "Remote Initialized!" << std::endl;

        hpx::naming::id_type loc1, loc2;
        typedef std::pair<char, hpx::naming::id_type> pair_type;

        typedef game::server::tictactoe::start_action startgame_action;

        //start play on the locality which has x
        BOOST_FOREACH(pair_type pair, char_id_pair_vec)
        {
            if(pair.first == 'x')
            {
                //hpx::lcos::future<void> start_fut = 
				//	hpx::async<startgame_action>(pair.second);
				dataflow_base<void> df_start =
					dataflow<startgame_action>(pair.second);
				//start_fut.get();
				df_start.get_future().get();
                loc1 = pair.second;
            }
            else
            {
                loc2 = pair.second;
            }
        }
        
        std::cout << "Game Started!" << std::endl;
       
        typedef game::server::tictactoe::get_count_value_action countvalue_action;
        std::size_t count1, count2;
        count1 = count2 = 0;
        //query if game is complete or not
        while( (count1 != 9) && (count2 != 9))
        {
            //hpx::lcos::future<std::size_t> temp1 = hpx::async<countvalue_action>(loc1);
            //hpx::lcos::future<std::size_t> temp2 = hpx::async<countvalue_action>(loc2);
			dataflow_base<std::size_t> temp1 = dataflow<countvalue_action>(loc1);
			dataflow_base<std::size_t> temp2 = dataflow<countvalue_action>(loc2);
            count1 = temp2.get_future().get();//temp1.get();
            count2 = temp2.get_future().get();//temp2.get();
        }
        
		//query winner
		typedef game::server::tictactoe::get_winner_action 
			get_winner_action;
        std::cout << "Querying the winner" << std::endl; 
        //hpx::lcos::future<char> winner = hpx::async<get_winner_action>(loc1);
        //char winner_value = winner.get();
		dataflow_base<char> winner = dataflow<get_winner_action>(loc1);
        char winner_value = winner.get_future().get();//'x';
		//char winner_value = 'x';
        return winner_value;
    } 

}
