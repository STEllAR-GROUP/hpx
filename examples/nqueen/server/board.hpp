//  Copyright (c) 2011 Vinay C Amatya
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_9FEC203D_0AAB_4213_BA36_456BE578ED3D)
#define HPX_9FEC203D_0AAB_4213_BA36_456BE578ED3D

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/constructor_argument.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/actions/component_action.hpp>

namespace hpx { namespace nqueen { namespace server
{

    class board
        : public components::detail::managed_component_base<board>
    {
    private:
        std::vector<std::size_t> list_;
        std::size_t level_;
        std::size_t size_;

        friend class boost::serialization::access;
        template<class Archive>
        void serialize(Archive& ar, const unsigned int version)
        {
            ar & size_;
            ar & list_;
            ar & level_;
        }

    public:
        enum actions
        {
            board_update,
            board_access,
            board_check,
            board_solve,
            board_print,
            board_clear
        };

        board() : level_(0), size_(0) {}

        board(components::constructor_argument const& s)
        {
            initialize(boost::get<std::size_t>(s), 0);
        }

        board(std::vector<std::size_t> const& list, std::size_t size,
            std::size_t level)
            : list_(list), level_(level), size_(size) {}

        void initialize(std::size_t size, std::size_t level)
        {
            size_ = size;
            level_ = level;

            for (std::size_t i = 0; i != size_; ++i)
                list_.push_back(size_);
        }

        void print()
        {
            std::cout << "Size: " << size_ << std::endl
                      << "Level: " << level_ << std::endl
                      << "List contents: " << std::endl;

            for (std::size_t i = 0; i != size_; ++i)
                std::cout << "  " << list_.at(i) << std::endl;
        }

        bool check(std::vector<std::size_t> const& list, std::size_t level)
        {
            for (std::size_t i = 0; i < level; ++i)
            {
                if ((list.at(i) == list.at(level))
                 || (list.at(level) - list.at(i) == level - i) 
                 || (list.at(i) - list.at(level) == level - i))
                    return false;
            }

            return true;
        }

        std::vector<std::size_t> access()
        {
            return list_;
        }

        void update(std::size_t pos, std::size_t val)
        {
            list_.at(pos) = val;
        }

        void clear()
        {
            list_.clear();
        }

        void solve(std::vector<std::size_t> const& list, std::size_t size,
            std::size_t level)
        {

            board board_(list, size, level);

            if (level != size)
            {
                for (std::size_t i = 0; i < size; ++i)
                {
                    board_.update(level, i);

                    if (board_.check(board_.access(), level))
                        solve(board_.access(), size, level + 1);
                }
            }
        }

        typedef hpx::actions::action0<
            board,
            board_print,
            &board::print
        > print_action;
                
        typedef hpx::actions::result_action0<
            board,
            std::vector<std::size_t>,
            board_access,
            &board::access
        > access_action;

        typedef hpx::actions::action2<
            board,
            board_update,
            std::size_t,
            std::size_t,
            &board::update
        > update_action;

        typedef hpx::actions::result_action2<
            board,
            bool,
            board_check,
            std::vector<std::size_t> const&,    
            std::size_t,
            &board::check
        > check_action;

        typedef hpx::actions::action3<
            board,
            board_solve,
            std::vector<std::size_t> const&,
            std::size_t,
            std::size_t,
            &board::solve
        > solve_action;

        typedef hpx::actions::action0<
            board,
            board_clear,
            &board::clear
        > clear_action;
    };

}}}

#endif // HPX_9FEC203D_0AAB_4213_BA36_456BE578ED3D

