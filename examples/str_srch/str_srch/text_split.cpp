#include <hpx/hpx_fwd.hpp>
//#include <hpx/runtime/agas/interface.hpp>
#include <hpx/runtime/components/component_factory_base.hpp>
#include <hpx/components/distributing_factory/distributing_factory.hpp>

//#include <hpx/include/async.hpp>
//#include <hpx/lcos/future.hpp>
//#include <hpx/lcos/future_wait.hpp>
//#include <hpx/lcos/local/packaged_task.hpp>

//#include <boost/foreach.hpp>
//#include <boost/assert.hpp>

//#include <utility>
//#include <cstring>
//#include <vector>

#include "./server/str_search.hpp"
#include "text_split.hpp"

// factory registration functionality
HPX_REGISTER_COMPONENT_MODULE();

namespace text
{
	using hpx::lcos::dataflow;
	using hpx::lcos::dataflow_base;

	text_split::text_split(){}
    text_split::~text_split(){}

    void text_split::create()
    {
        std::size_t num_instances;
        num_instances = 4;      // default no. of components to be created;
        hpx::components::component_type type =
            hpx::components::get_component_type<server::text_process>();

        typedef hpx::components::distributing_factory distributing_factory;
        distributing_factory factory;
        factory.create(hpx::find_here());
        distributing_factory::async_create_result_type result = 
            factory.create_components_async(type, num_instances);

        distributing_factory::result_type results = result.get();
        distributing_factory::iterator_range_type parts = 
            hpx::util::locality_results(results);

        BOOST_FOREACH(hpx::naming::id_type id, parts)
        {
            loc_comps_.push_back(id);
        }

        std::size_t cardinality = 0;

        card_id_vec_.reserve(loc_comps_.size());
        BOOST_FOREACH(hpx::naming::id_type id, loc_comps_)
        {
            card_id_vec_.push_back(std::make_pair(cardinality, id));
            ++cardinality;
        }
        // initialize remote components
        typedef text::server::text_process::init_action init_action;
        typedef std::pair<std::size_t, hpx::naming::id_type> card_id_pair_type;

        //std::vector<card_id_pair_type> card_id_pair_vec;
		std::vector<hpx::lcos::future<void> > init_futures;

        BOOST_FOREACH(card_id_pair_type id_pair, card_id_vec_)
        {
			init_futures.push_back(hpx::async<init_action>(id_pair.second
				, id_pair));
        }

		BOOST_FOREACH(hpx::lcos::future<void> fut, init_futures)
		{
			fut.get();
		}
        std::cout << "Object Created!!" << std::endl;
    }

    std::string text_split::process(char character, std::string str)
    {
        std::cout << "Split Process Started: " << std::endl;
		
        typedef text::server::text_process::remove_char_action remove_action;
		typedef text::server::text_process::replace_char_action replace_action;
        typedef std::pair<std::size_t, std::string> card_string_pair_type;
        typedef std::pair<std::size_t, hpx::naming::id_type> card_id_pair_type;
        std::vector<std::string> str_vec_in;
        std::vector<std::string>::iterator str_itr;
		
        std::string final_result = "";

        std::vector<card_string_pair_type> card_string_pair_vec;
        std::vector<card_string_pair_type>::iterator itr_csp;

        std::vector<hpx::lcos::future<card_string_pair_type> > futures;
        std::vector<hpx::lcos::future<card_string_pair_type> >::iterator itr_fut;
		std::vector<dataflow_base<card_string_pair_type> > dstr_vec_out;
		std::vector<dataflow_base<card_string_pair_type> >::iterator ditr_vec;

		//std::vector<dataflow_base<

        //split input string
        std::size_t step; 
        std::size_t comp_count = card_id_vec_.size();

        str_vec_in.reserve(comp_count);
        std::size_t pos_begin, pos_end, str_size;
        str_size = str.length() - 1;    //strlen(), discarding the null char 
        BOOST_ASSERT(str_size != 0);
       
        if(str_size%comp_count == 0) 
            step = str_size/comp_count;
        else
            step = str_size/comp_count + 1;

        for(std::size_t i = 0; i <= str_size; i+= step)
        {
            if(i+step > str_size)
            {
                str_vec_in.push_back(str.substr(i));
				//dstr_vec_in.push_back(dataflow<text::server::this_identity
            }
            else
            {
                str_vec_in.push_back(str.substr(i, step));
            }
        }
        
        //std::cout << "inside split process1: " << std::endl;

        str_itr = str_vec_in.begin();

		//replace instead of remove 
        BOOST_FOREACH(card_id_pair_type id_pair, card_id_vec_)
        {
            futures.push_back(hpx::async<replace_action>(id_pair.second, character
                , *str_itr));
			dstr_vec_out.push_back(dataflow<replace_action>(id_pair.second
				, character, *str_itr));
            ++str_itr;
        }

        //itr_fut = futures.begin();
        //std::cout << "inside split process2: " << std::endl;

        //BOOST_FOREACH(card_id_pair_type id_pair, card_id_vec_)
        //{
        //    card_string_pair_vec.push_back(itr_fut->get());
        //    ++itr_fut;
        //}
        //BOOST_FOREACH(hpx::lcos::future<card_string_pair_type> csp, futures)
        //{
        //    card_string_pair_vec.push_back(csp.get());
        //}

		BOOST_FOREACH(dataflow_base<card_string_pair_type> df_csp, dstr_vec_out)
		{
			card_string_pair_vec.push_back(df_csp.get_future().get());
		}
        
        //std::cout << "inside split process3:" << std::endl;
        std::sort(card_string_pair_vec.begin(), card_string_pair_vec.end()
            , compare_by_first());
        //c++11 compliant compiler required
        //std::sort(card_string_pair_vec.begin(), card_string_pair_vec.end()
        //    , [](const std::pair<std::size_t, std::string>& first, 
        //    const std::pair<std::size_t, std::string>& second)
        //    { return first.first <  second.first;  }
        //);
        //std::cout << "inside split process4:" << std::endl;
        itr_csp = card_string_pair_vec.begin();
       
        BOOST_FOREACH(card_string_pair_type id_pair, card_string_pair_vec) 
        {
            final_result+=id_pair.second; 
        }

        std::cout << "Split and caclulate process finished" << std::endl;
        return final_result; 
    }
} 
