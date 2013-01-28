
#if !defined(HPX_MURX8kN3vjuLyTLj9l19t48OB5Z2zVH7AU4471Lw)
#define HPX_MURX8kN3vjuLyTLj9l19t48OB5Z2zVH7AU4471Lw

/// a tic-tac-toe game, using dataflow

#include <hpx/hpx_fwd.hpp>
//#include <hpx/hpx_init.hpp>a
#include <hpx/runtime/components/component_factory.hpp>
#include <hpx/lcos/future.hpp>
//#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
//#include <hpx/runtime/components/server/simple_component_base.hpp>
//#include <hpx/components/dataflow/dataflow.hpp>
//#include <hpx/runtime/components/plain_component_factory.hpp>
#include <hpx/components/dataflow/dataflow.hpp>

//#include <boost/program_options.hpp>
#include <boost/lexical_cast.hpp>
//#include <boost/serialization/serialization.hpp>
//#include <boost/serialization/export.hpp>

#include <iostream>
#include <cstring>
#include <utility>
#include <vector>
#include <random>

namespace game { namespace server
{

    struct play_chart
    {
        std::vector<char> chart;
        std::size_t count;
        char winner;
    };

    typedef std::vector<std::pair<char, hpx::naming::id_type> > char_id_pair_type;
    
    class HPX_COMPONENT_EXPORT tictactoe 
        : public hpx::components::managed_component_base<tictactoe>
    {
    public:
        void init(char_id_pair_type char_id_pair, hpx::naming::id_type console_id);
        //void init(char_id_pair_type char_id_pair);
        bool win_position(std::vector<char>& chart, char& x_o);
        //void switch_player_turn(std::vector<char> board, std::size_t next_player, 
        //    bool i_win);
        void play(game::server::play_chart board);
        // get start trigger from console:
        void start();
        // return count value
        std::size_t get_count_value();
        // return winner
        char get_winner();

        HPX_DEFINE_COMPONENT_ACTION(tictactoe, init, init_action);
        HPX_DEFINE_COMPONENT_ACTION(tictactoe, win_position, win_position_action);
        HPX_DEFINE_COMPONENT_ACTION(tictactoe, play, play_action);
        HPX_DEFINE_COMPONENT_ACTION(tictactoe, start, start_action);
        HPX_DEFINE_COMPONENT_ACTION(tictactoe, get_count_value, get_count_value_action);
        HPX_DEFINE_COMPONENT_ACTION(tictactoe, get_winner, get_winner_action);
        
    private:
        play_chart board_;
        char my_char_;
        bool i_win_;
        bool i_played_;
        //std::default_random_engine generator_;
        //std::uniform_int_distribution<std::size_t> distribution_(0,8);
        //std::size_t my_play_;
        hpx::naming::id_type console_id_;
        hpx::naming::id_type my_id_, my_opponent_id_;
        char_id_pair_type char_id_pair_;

        //friend class boost::serialization::access;
        //template<class Archive>
        //void serialize(Archive & ar, const unsigned int version){
        //    my_char_
        //}
    };
}} 
//////////////////////////////////////////////////////////////////////////////
// Non-intrusive serialization
namespace boost { namespace serialization
{
    template <typename Archive>
    void serialize(Archive&, game::server::play_chart&, unsigned int const);
}}

HPX_REGISTER_ACTION_DECLARATION(
    game::server::tictactoe::init_action, tictactoe_init_action);
//HPX_REGISTER_ACTION_DECLARATION(
//    game::server::tictactoe::win_position_action, tictactoe_win_position_action);
HPX_REGISTER_ACTION_DECLARATION(
    game::server::tictactoe::play_action, tictactoe_play_action);
HPX_REGISTER_ACTION_DECLARATION(
    game::server::tictactoe::start_action, tictactoe_start_action);
HPX_REGISTER_ACTION_DECLARATION(
    game::server::tictactoe::get_count_value_action
    , tictactoe_get_count_value_action);
HPX_REGISTER_ACTION_DECLARATION(
    game::server::tictactoe::get_winner_action, tictactoe_get_winner_action);

#endif //HPX_MURX8kN3vjuLyTLj9l19t48OB5Z2zVH7AU4471Lw

