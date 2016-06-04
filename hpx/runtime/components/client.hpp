//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_CLIENT_JUN_25_2015_0145PM)
#define HPX_COMPONENTS_CLIENT_JUN_25_2015_0145PM

#include <hpx/config.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/runtime/components/client_base.hpp>

#include <utility>

namespace hpx { namespace components
{
    template <typename Component>
    class client : public client_base<client<Component>, Component>
    {
        typedef client_base<client<Component>, Component> base_type;
        typedef typename base_type::future_type future_type;

    public:
        // construction
        client() {}

        explicit client(naming::id_type const& id)
          : base_type(id)
        {}
        explicit client(naming::id_type && id)
          : base_type(std::move(id))
        {}

        explicit client(future_type const& f)
          : base_type(f)
        {}
        explicit client(future_type && f)
          : base_type(std::move(f))
        {}
        client(future<naming::id_type> && f)
          : base_type(std::move(f))
        {}

        client(client const& rhs)
          : base_type(rhs.shared_state_)
        {}
        client(client && rhs)
          : base_type(std::move(rhs.shared_state_))
        {}

        // copy assignment and move assignment
        client& operator=(naming::id_type const& id)
        {
            base_type::operator=(id);
            return *this;
        }
        client& operator=(naming::id_type && id)
        {
            base_type::operator=(std::move(id));
            return *this;
        }

        client& operator=(future_type const& f)
        {
            base_type::operator=(f);
            return *this;
        }
        client& operator=(future_type && f)
        {
            base_type::operator=(std::move(f));
            return *this;
        }
        client& operator=(future<naming::id_type> && f)
        {
            base_type::operator=(std::move(f));
            return *this;
        }

        client& operator=(client const& rhs)
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


