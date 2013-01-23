#if !defined(HPX_SeaF18t5rAEjYuzobcwR4i6hG5NOpSCRKz1smLWQ)
#define HPX_SeaF18t5rAEjYuzobcwR4i6hG5NOpSCRKz1smLWQ

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/runtime/components/component_factory.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>

#include <string>
#include <vector>
#include <iostream>
#include <utility>


//File to remove all the encountered "\a character" in the given string

namespace text { namespace server
{

    typedef std::pair<std::size_t, hpx::naming::id_type> card_id_pair_type;
    typedef std::pair<std::size_t, std::string> card_string_pair_type;

	template <typename T>
	T this_identity(T init_value);

    class HPX_COMPONENT_EXPORT text_process
        : public hpx::components::managed_component_base<text_process>
    {
    public:
        void init(card_id_pair_type card_id_pair);
        card_string_pair_type remove_char(char character, std::string string_m);
		card_string_pair_type replace_char(char character, std::string string_m);

        HPX_DEFINE_COMPONENT_ACTION(text_process, init, init_action);
        HPX_DEFINE_COMPONENT_ACTION(text_process, remove_char
			, remove_char_action);
		HPX_DEFINE_COMPONENT_ACTION(text_process, replace_char
			, replace_char_action);
    private:
        card_id_pair_type card_id_pair_;
        std::size_t my_cardinality_;
        hpx::naming::id_type my_id_;
    };
}}

HPX_REGISTER_ACTION_DECLARATION(
    text::server::text_process::init_action, text_process_init_action);
HPX_REGISTER_ACTION_DECLARATION(
    text::server::text_process::remove_char_action
	, text_process_remove_char_action);
HPX_REGISTER_ACTION_DECLARATION(
    text::server::text_process::replace_char_action
	, text_process_replace_char_action);
#endif //HPX_SeaF18t5rAEjYuzobcwR4i6hG5NOpSCRKz1smLWQ
