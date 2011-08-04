//  Copyright (c) 2011 Vinay C Amatya
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_9FEC203D_0AAB_4213_BA36_456BE578ED3D)
#define HPX_9FEC203D_0AAB_4213_BA36_456BE578ED3D

#include <iostream>

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/actions/component_action.hpp>

#define DS 8         //default size

unsigned int c_soln = 0;

typedef std::vector<std::size_t> list_t;

namespace hpx { namespace components { namespace server
{

    class Board
        : public components::detail::managed_component_base<Board>
    {
    private:
        list_t list_;
        std::size_t level_;
        std::size_t size_;

        // here Board is a component

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
            board_solve = 4,
            board_print = 5,
            board_size = 6,
            board_level = 7,
            board_test = 8,
            board_clear = 9
        };

        Board():list_(0), level_(0), size_(0)
        {}

        Board(list_t list, std::size_t size, std::size_t level):list_(list), level_(level), size_(size)
        {}

        ~Board(){}

        void initBoard(std::size_t size, std::size_t level)
                {//list_ = list;
                      size_ = size;
                      level_ = level;
                      std::size_t i = 0;
                      while(i!=size_)
                      {
                          list_.push_back(size_);
                          i++;
                     }
                }

        void printBoard()
        {
          std::cout << "Total Solution Count:" << c_soln << std::endl;
          std::cout << "Printing Contents" << std::endl;
          std::cout << "Size:" << size_ << "   "
            " Level:" << level_ << std::endl;
          std::cout << "List contents " << std::endl;
          std::size_t i = 0;
          while(i!=size_)
          {
            std::cout << list_.at(i) << std::endl;
            i++;
          };
        }

        bool checkBoard(list_t list, std::size_t level)
        {
            list_t tmp_list = list;
            std::size_t tmp_level = level;
            for(std::size_t i=0;i<tmp_level;i++){
                if((tmp_list.at(i)==tmp_list.at(tmp_level)) || (tmp_list.at(tmp_level)-tmp_list.at(i)==tmp_level-i)
                        || (tmp_list.at(i)-tmp_list.at(tmp_level)==tmp_level-i))
                    return 0;
            }
            return 1;
        }

        std::size_t getSize()    {return size_;}

        std::size_t getLevel() {return level_;}

        list_t accessBoard() { return list_;}

        void updateBoard(std::size_t pos, std::size_t val){std::size_t pos_ = pos, val_= val;list_.at(pos_)=val_;}

        void clearBoard(){ Board::list_.clear();}


        void solveNqueen(list_t list, std::size_t size, std::size_t level){

            list_t tmp_list = list;
            std::size_t tmp_size = size;
            std::size_t tmp_level = level;

            Board board_(tmp_list, tmp_size, tmp_level);

            if(tmp_level== tmp_size){
            	++c_soln;
                //std::cout << std::endl << "Solution "<<(++c_soln)<<":"<< std::endl;
                //for(std::size_t i=0; i < tmp_size; i++){
                    //for(std::size_t j=0; j < tmp_size; j++){
                        //if(board_.accessBoard().at(i)==j)
                            //std::cout << "X";
                        //else
                            //std::cout << "-";
                    //}
                    //std::cout << std::endl;
                //}
            }
            else{
                for(std::size_t i = 0; i < tmp_size; i++){
                    board_.updateBoard(tmp_level,i);
                    if(board_.checkBoard(board_.accessBoard(),tmp_level)){
                        solveNqueen(board_.accessBoard(),tmp_size,tmp_level+1);
                    }
                }
            }
        }

        ////////////////////////////////test function////////////////////////////
        // FIXME: level is unused
        void testBoard(list_t list, std::size_t size, std::size_t level){

            Board board_(list, size, 0);
            if(board_.accessBoard().at(0) == 5){
                board_.initBoard(5,0);
                board_.printBoard();
                }
            else{
            }


        }

        typedef hpx::actions::action2<
                Board, board_init, std::size_t, std::size_t, &Board::initBoard
            > init_action;

        typedef hpx::actions::action0<
                Board, board_print, &Board::printBoard
            > print_action;
                

        typedef hpx::actions::result_action0<
                Board, list_t, board_access, &Board::accessBoard
            > access_action;

        typedef hpx::actions::result_action0<
                Board, std::size_t, board_size, &Board::getSize
            > size_action;

        typedef hpx::actions::result_action0<
                Board, std::size_t, board_level, &Board::getLevel
            > level_action;

        typedef hpx::actions::action2<
                Board, board_update, std::size_t, std::size_t, &Board::updateBoard
            >update_action;

        typedef hpx::actions::result_action2<
                Board, bool, board_check, list_t, std::size_t, &Board::checkBoard
            >check_action;

        typedef hpx::actions::action3<
                Board, board_solve, list_t, std::size_t, std::size_t, &Board::solveNqueen
            >solve_action;

        typedef hpx::actions::action3<
                Board, board_test, list_t, std::size_t, std::size_t, &Board::testBoard
            >test_action;

        typedef hpx::actions::action0<
                Board, board_clear, &Board::clearBoard
            >clear_action;
    };

}}}

#endif // HPX_9FEC203D_0AAB_4213_BA36_456BE578ED3D

