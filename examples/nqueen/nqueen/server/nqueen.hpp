/*
 * nqueen.hpp
 *      Author: vamatya
 */

#include <iostream>

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/actions/component_action.hpp>

#define N 8

unsigned int c_soln = 0;

typedef std::vector<int> list_t;

namespace hpx { namespace components { namespace server
{

	class Board
		: public components::detail::managed_component_base<Board>
	{
	private:
		list_t list_;
		int level_;
		unsigned int size_;

		friend class boost::serialization::access;
		template<class Archive>
		void serialize(Archive & ar, const unsigned int version){
			ar & size_;
			ar & list_;
			ar & level_;
		}

	public:
		enum actions
		{
			board_init = 0,
			board_update = 1,
			board_access = 2,
			board_check = 3,
			board_solve = 4
		};

		Board():list_(0), size_(0), level_(0)
		{}

		Board(list_t list, unsigned int size, int level):list_(list), level_(level), size_(size)
		{}

		~Board(){}

		void initBoard(unsigned int size, int level)
				{//list_ = list;
				 	 Board::size_ = size; Board::level_ = level;
				 	 unsigned int i = 0;
				 	 while(i!=Board::size_)
				 	 {
				 		 Board::list_.push_back(0);
				 		 i++;
					 }
				}

		bool checkBoard(list_t list, int level)
		{
			list_t list_ = list;
			int level_= level;
				for(int i=0;i<level_;i++){
					if((list_.at(i)==list_.at(level_)) || (list_.at(level_)-list_.at(i)==level_-i)
							|| (list_.at(i)-list_.at(level_)==level_-i))
						return 0;
				}
				return 1;
		}

		//unsigned int getSize()	{return size_;}

		//int getLevel() {return level_;}

		list_t accessBoard() const { return list_;}

		void updateBoard(int level, int pos){int level_ = level, pos_= pos;list_.at(level_)=pos_;}

		void solveNqueen(Board board, unsigned int size,int level){

			int size_ = size, level_ = level;
			//Board board_(board.accessBoard(),level_);
			Board board_(board.accessBoard(), size_, level_);
			//std::cout << "I'm here again" << std::endl;
			if(level_== size_){
				std::cout << std::endl << "Solution "<<(++c_soln)<<":"<< std::endl;
				for(int i=0; i < size_; i++){
					for(int j=0; j < size_; j++){
						if(board_.accessBoard().at(i)==j)
							std::cout << "X";
						else
							std::cout << "-";
					}
					std::cout << std::endl;
				}
				std::cout << "reached here" << std::endl;
			}
			else{
				for(int i = 0; i < size_; i++){
					board_.updateBoard(level_,i);
					if(board_.checkBoard(board_.accessBoard(),level_)){
						//soln_cnt = Queens(board,size,level+1);
						solveNqueen(board_,size_,level_+1);
					}
				}
			}
			//delete &board;
			//return c_soln;
		}

		typedef hpx::actions::action2<
				Board, board_init, unsigned int, int, &Board::initBoard
			> init_action;

		typedef hpx::actions::result_action0<
				Board const, list_t, board_access, &Board::accessBoard
			> access_action;

		typedef hpx::actions::action2<
				Board, board_update, int, int, &Board::updateBoard
			>update_action;

		typedef hpx::actions::result_action2<
				Board, bool, board_check, list_t, int, &Board::checkBoard
			>check_action;

		typedef hpx::actions::action3<
				Board, board_solve, Board, unsigned int, int, &Board::solveNqueen
			>solve_action;
	};

}}}
