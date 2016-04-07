//  Copyright (c) 2016 Satyaki Upadhyay
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_SUDOKU_EXAMPLE_SERVER)
#define HPX_SUDOKU_EXAMPLE_SERVER

#include <hpx/include/async.hpp>
#include <hpx/include/util.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/hpx_fwd.hpp>
#include <hpx/include/components.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/threads/thread_data.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/runtime/serialization/serialize.hpp>
#include <boost/cstdint.hpp>

namespace sudoku
{
    class cancellation_token{

        public:
        bool cancel;
        cancellation_token(){
            cancel = false;
        }

        friend class hpx::serialization::access;
        template<class Archive>
        void serialize(Archive & ar, const unsigned int version){
            ar & cancel;
        }
    };

    typedef std::vector<boost::uint8_t> board_type;

namespace server
{
    class HPX_COMPONENT_EXPORT board
        : public hpx::components::component_base<board>
    {
    private:
        board_type board_config;
        std::size_t level_;
        std::size_t size_;
        std::size_t count_;

        // here board is a component

        friend class hpx::serialization::access;
        template<class Archive>
        void serialize(Archive & ar, const unsigned int version){
            ar & size_;
            ar & board_config;
            ar & level_;
            ar & count_;
        }

    public:

        board():board_config(0), level_(0), size_(0)
        {}

        board(board_type const& board_config, std::size_t size, std::size_t level)
            : board_config(board_config), level_(level), size_(size)
        {}

        ~board(){}

        void init_board(std::size_t size);

        bool check_board(std::size_t level, boost::uint8_t value);

        board_type access_board();

        void update_board(std::size_t pos, boost::uint8_t val);

        std::vector<boost::uint8_t> solve_board(std::size_t size, std::size_t level,
                                        cancellation_token ct);

        HPX_DEFINE_COMPONENT_ACTION(board, init_board, init_action);
        HPX_DEFINE_COMPONENT_ACTION(board, access_board, access_action);
        HPX_DEFINE_COMPONENT_ACTION(board, update_board, update_action);
        HPX_DEFINE_COMPONENT_ACTION(board, check_board, check_action);
        HPX_DEFINE_COMPONENT_ACTION(board, solve_board, solve_action);
    };
}}

// Declaration of serialization support for the board actions

HPX_REGISTER_ACTION_DECLARATION(
    sudoku::server::board::init_action,
    board_init_action);

HPX_REGISTER_ACTION_DECLARATION(
    sudoku::server::board::check_action,
    board_check_action);

HPX_REGISTER_ACTION_DECLARATION(
    sudoku::server::board::access_action,
    board_access_action);

HPX_REGISTER_ACTION_DECLARATION(
    sudoku::server::board::update_action,
    board_update_action);

HPX_REGISTER_ACTION_DECLARATION(
    sudoku::server::board::solve_action,
    board_solve_action);

#endif // HPX_SUDOKU_EXAMPLE_SERVER

