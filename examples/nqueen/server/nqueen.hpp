//  Copyright (c) 2011 Vinay C Amatya
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_9FEC203D_0AAB_4213_BA36_456BE578ED3D)
#define HPX_9FEC203D_0AAB_4213_BA36_456BE578ED3D

#include <iostream>

#include <hpx/hpx_fwd.hpp>
#include <hpx/include/components.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/threads/thread_data.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/runtime/serialization/serialize.hpp>

namespace nqueen
{
    typedef std::vector<std::size_t> list_type;

namespace server
{
    class board
        : public hpx::components::component_base<board>
    {
    private:
        list_type list_;
        std::size_t level_;
        std::size_t size_;
        std::size_t count_;

        // here board is a component

        friend class hpx::serialization::access;
        template<class Archive>
        void serialize(Archive & ar, const unsigned int version){
            ar & size_;
            ar & list_;
            ar & level_;
            ar & count_;
        }

    public:

        board():list_(0), level_(0), size_(0), count_(0)
        {}

        board(list_type const& list, std::size_t size, std::size_t level)
            : list_(list), level_(level), size_(size), count_(0)
        {}

        ~board(){}



        void init_board(std::size_t size)
        {
            std::size_t i = 0;
            while(i!=size)
            {
                list_.push_back(size);
                ++i;
             }
         }

        bool check_board(list_type const& list, std::size_t level)
        {
            for(std::size_t i = 0 ;i < level; ++i){
                if((list.at(i) == list.at(level))
                    || (list.at(level) - list.at(i) == level - i)
                    || (list.at(i) - list.at(level) == level - i))
                    return false;
            }
            return true;
        }

        list_type access_board()
        {
            return list_;
        }

        void update_board(std::size_t pos, std::size_t val)
        {
            list_.at(pos) = val;
        }

        void clear_board()
        {
            board::list_.clear();
        }

        std::size_t solve_board(list_type const& list, std::size_t size,
            std::size_t level, std::size_t col)
        {

            board b(list, size, level);

            if(level == size){
                return 1;
            }
            else if(level == 0)
            {
                b.update_board(level, col);
                if(b.check_board( b.access_board(), level))
                {
                   b.count_+= solve_board( b.access_board(),
                                            size, level + 1, col);
                }
            }
            else
            {
                for(std::size_t i = 0; i < size; ++i)
                {
                    b.update_board(level,i);
                    if(b.check_board( b.access_board(), level))
                    {
                       b.count_+=  solve_board( b.access_board(),
                                                size, level+1, col);
                    }
                }
            }
            return b.count_;
        }

        HPX_DEFINE_COMPONENT_ACTION(board, init_board, init_action);
        HPX_DEFINE_COMPONENT_ACTION(board, access_board, access_action);
        HPX_DEFINE_COMPONENT_ACTION(board, update_board, update_action);
        HPX_DEFINE_COMPONENT_ACTION(board, check_board, check_action);
        HPX_DEFINE_COMPONENT_ACTION(board, solve_board, solve_action);
        HPX_DEFINE_COMPONENT_ACTION(board, clear_board, clear_action);
    };

}}

// Declaration of serialization support for the board actions

HPX_REGISTER_ACTION_DECLARATION(
    nqueen::server::board::init_action,
    board_init_action);

HPX_REGISTER_ACTION_DECLARATION(
    nqueen::server::board::check_action,
    board_check_action);

HPX_REGISTER_ACTION_DECLARATION(
    nqueen::server::board::access_action,
    board_access_action);

HPX_REGISTER_ACTION_DECLARATION(
    nqueen::server::board::update_action,
    board_update_action);

HPX_REGISTER_ACTION_DECLARATION(
    nqueen::server::board::solve_action,
    board_solve_action);

HPX_REGISTER_ACTION_DECLARATION(
    nqueen::server::board::clear_action,
    board_clear_action);

HPX_REGISTER_ACTION_DECLARATION(
    hpx::lcos::base_lco_with_value<nqueen::list_type>::set_value_action,
    set_value_action_vector_std_size_t);

HPX_REGISTER_ACTION_DECLARATION(
    hpx::lcos::base_lco_with_value<nqueen::list_type>::get_value_action,
    get_value_action_vector_std_size_t);


#endif // HPX_9FEC203D_0AAB_4213_BA36_456BE578ED3D

