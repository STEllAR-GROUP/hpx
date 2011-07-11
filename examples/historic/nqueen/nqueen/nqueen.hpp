//  Copyright (c) 2011 Vinay C Amatya
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_527D225B_F1EC_4BC5_9245_3A69C6AE5304)
#define HPX_527D225B_F1EC_4BC5_9245_3A69C6AE5304

#include <hpx/runtime.hpp>
#include <hpx/runtime/components/client_base.hpp>

#include <examples/nqueen/nqueen/stubs/nqueen.hpp>

namespace hpx { namespace components
{
    class Board
        : public client_base<Board, stubs::Board>
    {
        typedef client_base<Board, stubs::Board> base_type;

    public:
        Board()
        {}
        Board(naming::id_type gid)
            : base_type(gid)
        {}

        void initBoard(std::size_t size, std::size_t level){
            BOOST_ASSERT(gid_);
            return this->base_type::initBoard(gid_, size, level);
        }

        void printBoard()
        {
            BOOST_ASSERT(gid_);
            return this->base_type::printBoard(gid_);
        }

        list_t accessBoard(){
            BOOST_ASSERT(gid_);
            return this->base_type::accessBoard(gid_);
        }

        lcos::future_value<list_t> accessBoard_async(){
            return this->base_type::accessBoard_async(gid_);
        }

        std::size_t getSize(){
            BOOST_ASSERT(gid_);
            return this->base_type::getSize(gid_);
        }

        lcos::future_value<std::size_t> getSize_async() {
            return this->base_type::getSize_async(gid_);
        }

        std::size_t getLevel(){
            BOOST_ASSERT(gid_);
            return this->base_type::getLevel(gid_);
        }

        lcos::future_value<std::size_t> getLevel_async() {
            return this->base_type::getLevel_async(gid_);
        }

        void updateBoard(std::size_t level, std::size_t pos){
            BOOST_ASSERT(gid_);
            return this->base_type::updateBoard(gid_, level, pos);
        }

        bool checkBoard(list_t list, std::size_t level){
            BOOST_ASSERT(gid_);
            return this->base_type::checkBoard(gid_, list, level);
        }

        lcos::future_value<bool> checkBoard_async(list_t list, std::size_t level){
            return this->base_type::checkBoard_async(gid_, list, level);
        }

        void solveNqueen(list_t list, std::size_t size, std::size_t level){
            BOOST_ASSERT(gid_);
            return this->base_type::solveNqueen(gid_, list, size, level);
        }

        void clearBoard(){
            BOOST_ASSERT(gid_);
            return this->base_type::clearBoard(gid_);
        }
        void testBoard(list_t list, std::size_t size, std::size_t level){
            BOOST_ASSERT(gid_);
            return this->base_type::testBoard(gid_, list, size, level);
        }

    };

}}

#endif // HPX_527D225B_F1EC_4BC5_9245_3A69C6AE5304

