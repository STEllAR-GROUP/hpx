//  Copyright (c) 2007-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/components/client_base.hpp>
#include <hpx/futures/future.hpp>

#include <utility>

namespace hpx { namespace components {

    template <typename Component>
    class client : public client_base<client<Component>, Component>
    {
        using base_type = client_base<client<Component>, Component>;
        using future_type = typename base_type::future_type;

    public:
        // construction
        client() = default;

        explicit client(naming::id_type const& id)
          : base_type(id)
        {
        }
        explicit client(naming::id_type&& id)
          : base_type(std::move(id))
        {
        }

        explicit client(future_type const& f) noexcept
          : base_type(f)
        {
        }
        explicit client(future_type&& f) noexcept
          : base_type(std::move(f))
        {
        }
        client(future<naming::id_type>&& f) noexcept
          : base_type(std::move(f))
        {
        }
        client(future<client>&& c)
          : base_type(std::move(c))
        {
        }

        client(client const& rhs) noexcept
          : base_type(rhs.shared_state_)
        {
        }
        client(client&& rhs) noexcept
          : base_type(std::move(rhs.shared_state_))
        {
        }

        // copy assignment and move assignment
        client& operator=(naming::id_type const& id)
        {
            base_type::operator=(id);
            return *this;
        }
        client& operator=(naming::id_type&& id)
        {
            base_type::operator=(std::move(id));
            return *this;
        }

        client& operator=(future_type const& f) noexcept
        {
            base_type::operator=(f);
            return *this;
        }
        client& operator=(future_type&& f) noexcept
        {
            base_type::operator=(std::move(f));
            return *this;
        }
        client& operator=(future<naming::id_type>&& f) noexcept
        {
            base_type::operator=(std::move(f));
            return *this;
        }

        client& operator=(client const& rhs) noexcept
        {
            base_type::operator=(rhs);
            return *this;
        }
        client& operator=(client&& rhs) noexcept
        {
            base_type::operator=(std::move(rhs));
            return *this;
        }
    };
}}    // namespace hpx::components
