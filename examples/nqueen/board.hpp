//  Copyright (c) 2011 Vinay C Amatya
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_527D225B_F1EC_4BC5_9245_3A69C6AE5304)
#define HPX_527D225B_F1EC_4BC5_9245_3A69C6AE5304

#include <hpx/runtime.hpp>
#include <hpx/runtime/components/client_base.hpp>

#include <nqueen/stubs/board.hpp>

namespace hpx { namespace nqueen
{

    class board
        : public components::client_base<board, stubs::board>
    {
        typedef components::client_base<board, stubs::board> base_type;

    public:
        board(naming::id_type gid = naming::invalid_id)
            : base_type(gid)
        {}

        ///////////////////////////////////////////////////////////////////////
        void print()
        {
            return this->base_type::print(gid_);
        }

        ///////////////////////////////////////////////////////////////////////
        std::vector<std::size_t> access()
        {
            return this->base_type::access(gid_);
        }

        lcos::future_value<std::vector<std::size_t> > access_async()
        {
            return this->base_type::access_async(gid_);
        }

        ///////////////////////////////////////////////////////////////////////
        void update(std::size_t level, std::size_t pos)
        {
            return this->base_type::update(gid_, level, pos);
        }

        ///////////////////////////////////////////////////////////////////////
        bool check(std::vector<std::size_t> const& list, std::size_t level)
        {
            return this->base_type::check(gid_, list, level);
        }

        lcos::future_value<bool>
        check_async(std::vector<std::size_t> const& list, std::size_t level)
        {
            return this->base_type::check_async(gid_, list, level);
        }

        ///////////////////////////////////////////////////////////////////////
        void solve(std::vector<std::size_t> const& list, std::size_t size,
            std::size_t level)
        {
            return this->base_type::solve(gid_, list, size, level);
        }

        ///////////////////////////////////////////////////////////////////////
        void clear()
        {
            return this->base_type::clear(gid_);
        }
    };

}}

#endif // HPX_527D225B_F1EC_4BC5_9245_3A69C6AE5304

