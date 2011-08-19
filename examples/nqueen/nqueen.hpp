//  Copyright (c) 2011 Vinay C Amatya
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_527D225B_F1EC_4BC5_9245_3A69C6AE5304)
#define HPX_527D225B_F1EC_4BC5_9245_3A69C6AE5304

#include <hpx/runtime.hpp>
#include <hpx/runtime/components/client_base.hpp>

#include <examples/nqueen/stubs/nqueen.hpp>

namespace hpx { namespace components
{
    class board
        : public client_base<board, stubs::board>
    {
        typedef client_base<board, stubs::board> base_type;

    public:
        board()
        {}
        board(naming::id_type gid)
            : base_type(gid)
        {}

        void init_board(std::size_t size ){
            BOOST_ASSERT(gid_);
            return this->base_type::init_board(gid_, size);
        }
        //-------------------------------------------------------

        list_t access_board(){
            BOOST_ASSERT(gid_);
            return this->base_type::access_board(gid_);
        }

        lcos::future_value<list_t> access_board_async(){
            return this->base_type::access_board_async(gid_);
        }
        //------------------------------------------------------

        void update_board(std::size_t level, std::size_t pos){
            BOOST_ASSERT(gid_);
            return this->base_type::update_board(gid_, level, pos);
        }
        //-----------------------------------------------------

        bool check_board(list_t list, std::size_t level){
            BOOST_ASSERT(gid_);
            return this->base_type::check_board(gid_, list, level);
        }

        lcos::future_value<bool> check_board_async(list_t list, 
            std::size_t level)
        {
            return this->base_type::check_board_async(gid_, list, level);
        }
        //---------------------------------------------------------

        std::size_t solve_board(list_t list, std::size_t size, 
            std::size_t level, std::size_t col)
        {
            BOOST_ASSERT(gid_);
            return this->base_type::solve_board(gid_, list, size, level, col);
        }
        
        lcos::future_value<std::size_t> solve_board_async(list_t list, 
            std::size_t size, std::size_t level, std::size_t col)
        {
            return this->base_type::solve_board_async(gid_, list, size, level,
            col);
        }
        //---------------------------------------------------------

        void clear_board(){
            BOOST_ASSERT(gid_);
            return this->base_type::clear_board(gid_);
        }
    };

}}

#endif // HPX_527D225B_F1EC_4BC5_9245_3A69C6AE5304

