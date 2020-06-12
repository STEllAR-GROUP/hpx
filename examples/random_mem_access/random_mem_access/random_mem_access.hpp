//  Copyright (c) 2011 Matt Anderson
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/include/runtime.hpp>
#include <hpx/include/client.hpp>

#include "server/random_mem_access.hpp"

#include <cstddef>
#include <utility>

namespace hpx { namespace components
{
    ///////////////////////////////////////////////////////////////////////////
    /// The \a random_mem_access class is the client side representation of a
    /// specific \a server#random_mem_access component
    class random_mem_access
      : public client_base<random_mem_access, server::random_mem_access>
    {
        typedef
            client_base<random_mem_access, server::random_mem_access>
        base_type;

    public:
        random_mem_access() = default;

        /// Create a client side representation for the existing
        /// \a server#random_mem_access instance with the given global \a id.
        random_mem_access(id_type id)
          : base_type(std::move(id))
        {}

        ~random_mem_access() = default;

        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        /// Initialize the random_mem_access value
        void init(std::size_t i)
        {
            typedef server::random_mem_access::init_action init_action;
            hpx::apply<init_action>(this->get_id(), i);
        }

        /// Add the given number to the random_mem_access
        void add()
        {
            add_async().get();
        }

        hpx::lcos::future<void> add_async()
        {
            typedef server::random_mem_access::add_action action_type;
            return hpx::async<action_type>(this->get_id());
        }

        /// Print the current value of the random_mem_access
        void print()
        {
            print_async().get();
        }
        /// Asynchronously query the current value of the random_mem_access
        hpx::lcos::future<void> print_async ()
        {
            typedef server::random_mem_access::print_action action_type;
            return hpx::async<action_type>(this->get_id());
        }

        /// Query the current value of the random_mem_access
        std::size_t query()
        {
            return query_async().get();
        }

        /// Asynchronously query the current value of the random_mem_access
        lcos::future<std::size_t> query_async()
        {
            typedef server::random_mem_access::query_action action_type;
            return hpx::async<action_type>(this->get_id());
        }
    };
}}

