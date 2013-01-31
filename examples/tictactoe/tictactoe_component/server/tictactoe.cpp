#include <hpx/hpx_fwd.hpp> 
#include <hpx/include/serialization.hpp>

#include <boost/lexical_cast.hpp>

#include "tictactoe.hpp"


namespace game { namespace server
{
	using hpx::lcos::dataflow;
	using hpx::lcos::dataflow_base;

    typedef std::pair<char, hpx::naming::id_type> pair_type;
    typedef std::vector<pair_type> char_id_pair_type;
///////////////////////////////////////////////////////////////////////////////
    void tictactoe::init(game::server::char_id_pair_type char_id_pair
        , hpx::naming::id_type this_comp)
    {
        char_id_pair_ = char_id_pair;
        my_id_ = this_comp;
        
        BOOST_FOREACH(pair_type ptype, char_id_pair_)
        {
            //if(ptype.second == hpx::this_prefix())
            if(ptype.second == my_id_)
            {
                my_char_ = ptype.first;
                //my_id_ = ptype.second;
            }
            else if(ptype.second != my_id_)
            {
                my_opponent_id_ = ptype.second;
            } 
        }
        board_.chart.resize(9);
        board_.count = 0;
        i_win_ = false;
        board_.winner ='$';
		//distribution_(0,8);
        std::cout << "initialized! for:" << my_id_ << std::endl;
    }
    bool tictactoe::win_position(std::vector<char>& chart, char& x_o)
    {
        /// winning positions(row matrix): {0,1,2}, {0,3,6}, {0,4,8}, {1,4,7},
        /// {2,4,6}, {2,5,8}, {3,4,5}, {6,7,8}
        if(chart.at(0) == x_o && chart.at(1) == x_o && chart.at(2) == x_o)
            return true;
        else if(chart.at(0) == x_o && chart.at(3) == x_o && chart.at(6) == x_o)
            return true;
        else if(chart.at(0) == x_o && chart.at(4) == x_o && chart.at(8) == x_o)
            return true;
        else if(chart.at(1) == x_o && chart.at(4) == x_o && chart.at(7) == x_o)
            return true;
        else if(chart.at(2) == x_o && chart.at(4) == x_o && chart.at(6) == x_o)
            return true;
        else if(chart.at(2) == x_o && chart.at(5) == x_o && chart.at(8) == x_o)
            return true;
        else if(chart.at(3) == x_o && chart.at(4) == x_o && chart.at(5) == x_o)
            return true;
        else if(chart.at(6) == x_o && chart.at(7) == x_o && chart.at(8) == x_o)
            return true;
        else 
            return false;
    }
    
    char tictactoe::get_winner()
    {
        return board_.winner;
    }
    
    std::size_t tictactoe::get_count_value()
    {
        return board_.count;
    }
    void tictactoe::start()
    {
        play(board_);
    }
  
    void tictactoe::play(game::server::play_chart board)
    {
        game::server::play_chart temp_board = board;
        board_ = temp_board;
		std::size_t my_play_;
        //i_played_ = false;
		dataflow_base<void> return_obj;
        if(board_.count < 9)
        {
			std::default_random_engine generator_;                                   
			std::uniform_int_distribution<std::size_t> distribution_(0,8);
            do
            {
                i_played_ = false;
                //make a play
                
                my_play_ = 
					boost::lexical_cast<std::size_t>(distribution_(generator_));

				std::cout<< "my play:" << my_play_ << ", my_id_: " << my_id_ << std::endl;

                if(board_.chart.at(my_play_) != 'x' && board_.chart.at(my_play_) != 'o')
                {
                    board_.chart.at(my_play_) = my_char_;
                    board_.count+=1;
                    i_played_ = true;
                    //board_ = temp_board;
                }
        
                i_win_ = win_position(board_.chart, my_char_);
            
                if(i_win_ == true)
                {
                    board_.count = 9;
                    board_.winner = my_char_;
                    
					//update remote that I've won, update console that I've won
                    //hpx::lcos::future<void> future = 
					//	hpx::async<play_action>(my_opponent_id_, board_);
					//future.get();

					//using dataflow
					return_obj = dataflow<play_action>(my_opponent_id_, board_);
                }
                else if(i_played_ == true)
                {
                    //next player's turn
                    //hpx::lcos::future<void> future1 = 
					//	hpx::async<play_action>(my_opponent_id_, board_);
					//future1.get();

					//using dataflow
					return_obj = dataflow<play_action>(my_opponent_id_, board_);
				}

            }while(i_played_ != true);
			return_obj.get_future().get();
        }
    }
}}

///////////////////////////////////////////////////////////////////////////////
namespace boost { namespace serialization
{
    template <typename Archive>
    void serialize(Archive& ar, game::server::play_chart& p_c, 
        unsigned int const)
    {
        ar & p_c.chart;
        ar & p_c.count;
        ar & p_c.winner;
    }
}}
///////////////////////////////////////////////////////////////////////////////
typedef game::server::tictactoe tictactoe_type;
HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(
    hpx::components::managed_component<tictactoe_type>
    , game_tictactoe_type);

// Serialize actions
HPX_REGISTER_ACTION(                                                    
    game::server::tictactoe::init_action, tictactoe_init_action);                   
//HPX_REGISTER_ACTION(                                                    
//    game::server::tictactoe::win_position_action, tictactoe_win_position_action);
HPX_REGISTER_ACTION(                                                 
    game::server::tictactoe::play_action, tictactoe_play_action);                
HPX_REGISTER_ACTION(                                                 
    game::server::tictactoe::start_action, tictactoe_start_action);              
HPX_REGISTER_ACTION(
    game::server::tictactoe::get_count_value_action
    , tictactoe_get_count_value_action);
HPX_REGISTER_ACTION(
    game::server::tictactoe::get_winner_action, tictactoe_get_winner_action);
