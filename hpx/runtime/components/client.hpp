//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_CLIENT_JUN_25_2015_0145PM)
#define HPX_COMPONENTS_CLIENT_JUN_25_2015_0145PM

#include <hpx/config.hpp>
#include <hpx/runtime/components/client.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/util/move.hpp>

namespace hpx { namespace components
{
    template <typename Component>
    class client : client<client<Component>, Component>
    {
        typedef client<client<Component>, Component> base_type;
        typedef typename base_type::future_type future_type;

    public:
        // construction
        client() {}

        explicit client(naming::id_type const& gid)
          : base_type(lcos::make_ready_future(gid))
        {}
        explicit client(naming::id_type && gid)
          : base_type(lcos::make_ready_future(std::move(gid)))
        {}

        explicit client(future_type const& gid)
          : base_type(gid)
        {}
        explicit client(future_type && gid)
          : base_type(std::move(gid))
        {}
        explicit client(future<naming::id_type> && gid)
          : base_type(gid.share())
        {}

        explicit client(client const& rhs)
          : base_type(rhs.gid_)
        {}
        explicit client(client && rhs)
          : base_type(std::move(rhs.gid_))
        {}

        // copy assignment and move assignment
        client& operator=(naming::id_type const & gid)
        {
            base_type::operator=(lcos::make_ready_future(gid));
            return *this;
        }
        client& operator=(naming::id_type && gid)
        {
            base_type::operator=(lcos::make_ready_future(std::move(gid)));
            return *this;
        }

        client& operator=(future_type const & gid)
        {
            base_type::operator=(gid);
            return *this;
        }
        client& operator=(future_type && gid)
        {
            base_type::operator=(std::move(gid));
            return *this;
        }
        client& operator=(future<naming::id_type> && gid)
        {
            base_type::operator=(gid);
            return *this;
        }

        client& operator=(client const & rhs)
        {
            base_type::operator=(rhs);
            return *this;
        }
        client& operator=(client && rhs)
        {
            base_type::operator=(std::move(rhs));
            return *this;
        }
    };
}}

#endif


