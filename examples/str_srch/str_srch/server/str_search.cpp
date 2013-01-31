#include <hpx/hpx_fwd.hpp>
#include <hpx/include/serialization.hpp>
#include <hpx/runtime/components/component_factory.hpp>

#include <utility>
#include <string>
#include <algorithm>

#include "str_search.hpp"

namespace text { namespace server
{
    typedef std::pair<std::size_t, hpx::naming::id_type> card_id_pair_type;
    typedef std::pair<std::size_t, std::string> card_string_pair_type;
    ////////////////////////////////////////////////////////////////////////////
	template <typename T>
	T this_identity(T init_value)
	{
		return init_value;
	}
	////////////////////////////////////////////////////////////////////////////
    void text_process::init(card_id_pair_type card_id_pair)
    {
        card_id_pair_ = card_id_pair;
        my_cardinality_ = card_id_pair_.first;
        my_id_ = card_id_pair_.second;

        //std::cout << "Init text_process, started " << std::endl;
		std::cout << "My cardinality:" << my_cardinality_ << std::endl;
    }
    
    // instead of remove, replace, or do sth else;
    card_string_pair_type text_process::remove_char(char char_in, std::string string_in)
    {
        //std::cout << "Remove char started " <<std::endl;
        std::string temp(string_in);
        //temp = string_in;

        temp.erase(std::remove(temp.begin(), temp.end(), char_in), temp.end());
    
        card_string_pair_type return_pair = std::make_pair(my_cardinality_, temp);
        return return_pair;
    }

	// replace given character with @ in the input string.
	card_string_pair_type text_process::replace_char(char char_in, std::string string_in)
	{
		std::string temp(string_in);
		//temp.replace(std::replace(temp.begin(), temp.end(), char_in, '@')
		//, temp.end());
		std::replace(temp.begin(), temp.end(), char_in, '@');
		card_string_pair_type return_pair = std::make_pair(my_cardinality_, temp);
		return return_pair;
	}
}}

typedef text::server::text_process text_process_type;
HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(
    hpx::components::managed_component<text_process_type>
    , text_text_process_type);

HPX_REGISTER_ACTION(
    text::server::text_process::init_action, text_process_init_action);
HPX_REGISTER_ACTION(
    text::server::text_process::remove_char_action, text_process_remove_char_action);
HPX_REGISTER_ACTION(
	text::server::text_process::replace_char_action, text_process_replace_char_action);

