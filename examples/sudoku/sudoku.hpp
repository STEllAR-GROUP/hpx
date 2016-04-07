//  Copyright (c) 2016 Satyaki Upadhyay
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_SUDOKU_EXAMPLE)
#define HPX_SUDOKU_EXAMPLE

#include <hpx/runtime.hpp>
#include <hpx/include/client.hpp>

#include <examples/sudoku/server/sudoku.hpp>

namespace sudoku
{
    class board
        : public hpx::components::client_base<board, server::board>
    {
        //[board_base_type
        typedef hpx::components::client_base<
            board, server::board
        > base_type;
        //]

    public:
        /// Default construct an empty client side representation (not
        /// connected to any existing component).
        board()
        {}

        /// Create a client side representation for the existing
        /// \a server::board instance with the given GID.
        board(hpx::future<hpx::id_type> && gid)
          : base_type(std::move(gid))
        {}

        void init_board(std::size_t size){

            HPX_ASSERT(this->get_id());

            typedef server::board::init_action action_type;
            hpx::apply<action_type>(this->get_id(), size);
        }

        //-------------------------------------------------------

        hpx::lcos::future<board_type>
        access_board_async(hpx::naming::id_type const& gid)
        {
            HPX_ASSERT(gid);

            typedef server::board::access_action action_type;
            return hpx::async<action_type>(gid);
        }

        board_type access_board(){

            HPX_ASSERT(this->get_id());

            return access_board_async(this->get_id()).get();
        }

        //------------------------------------------------------

        void update_board(std::size_t level, boost::uint8_t pos){

            HPX_ASSERT(this->get_id());

            typedef server::board::update_action action_type;
            hpx::apply<action_type>(this->get_id(), level, pos);
        }

        //-----------------------------------------------------

        hpx::lcos::future<bool>
        check_board_async(hpx::naming::id_type const& gid,
            std::size_t level, boost::uint8_t value)
        {
            HPX_ASSERT(gid);

            typedef server::board::check_action action_type;
            return hpx::async<action_type>(gid, level, value);
        }

        bool check_board(std::size_t level, boost::uint8_t value){

            HPX_ASSERT(this->get_id());

            return check_board_async(this->get_id(), level, value).get();
        }

        //---------------------------------------------------------

        hpx::lcos::future<std::vector<boost::uint8_t>> solve_board_async(
            hpx::naming::id_type const& gid, std::size_t size, std::size_t level)
        {
            HPX_ASSERT(gid);

            cancellation_token ct;
            typedef server::board::solve_action action_type;
            return hpx::async<action_type>(gid, size, level, std::ref(ct));
        }

        std::vector<boost::uint8_t> solve_board(std::size_t size, std::size_t level)
        {
            HPX_ASSERT(this->get_id());

            return solve_board_async(this->get_id(), size, level).get();
        }
    };
}

#endif // HPX_SUDOKU_EXAMPLE
