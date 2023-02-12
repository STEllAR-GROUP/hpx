//  Copyright (c) 2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if !defined(HPX_COMPUTE_DEVICE_CODE)

#include <hpx/actions_base/component_action.hpp>
#include <hpx/components_base/server/component_base.hpp>
#include <hpx/datastructures/any.hpp>
#include <hpx/lcos_local/and_gate.hpp>
#include <hpx/modules/futures.hpp>
#include <hpx/naming_base/id_type.hpp>
#include <hpx/synchronization/spinlock.hpp>
#include <hpx/thread_support/assert_owns_lock.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx::traits {

    // This type will be specialized for a particular collective operation
    template <typename Communicator, typename Operation>
    struct communication_operation;
}    // namespace hpx::traits

namespace hpx::collectives::detail {

    ///////////////////////////////////////////////////////////////////////////
    HPX_EXPORT std::size_t calculate_connected_node(
        std::size_t site, std::size_t arity) noexcept;
    HPX_EXPORT std::size_t calculate_num_connected(
        std::size_t num_sites, std::size_t site, std::size_t arity) noexcept;
    HPX_EXPORT std::string get_local_communication_node_name(char const* name);
    HPX_EXPORT hpx::id_type resolve_local_communication_set_name(
        std::string basename, std::size_t site);

    ///////////////////////////////////////////////////////////////////////////
    constexpr std::size_t next_power_of_two(std::uint64_t v) noexcept
    {
        --v;
        v |= v >> 1;
        v |= v >> 2;
        v |= v >> 4;
        v |= v >> 8;
        v |= v >> 16;
        v |= v >> 32;    //-V112
        return ++v;
    }

    ///////////////////////////////////////////////////////////////////////////
    class communication_set_node
      : public hpx::components::component_base<communication_set_node>
    {
        using mutex_type = hpx::spinlock;

    public:
        HPX_EXPORT communication_set_node();

        HPX_EXPORT communication_set_node(std::size_t num_sites,
            std::string const& name, std::size_t site, std::size_t arity);

        ///////////////////////////////////////////////////////////////////////
        // generic get action, dispatches to proper operation
        template <typename Operation, typename Result, typename... Args>
        Result get_result(
            std::size_t which, std::size_t generation, Args... args)
        {
            using collective_operation =
                traits::communication_operation<communication_set_node,
                    Operation>;

            return collective_operation::template get<Result>(
                *this, which, generation, HPX_MOVE(args)...);
        }

        template <typename Operation, typename Result, typename... Args>
        struct communication_set_get_action
          : hpx::actions::make_action<Result (communication_set_node::*)(
                                          std::size_t, std::size_t, Args...),
                &communication_set_node::get_result<Operation, Result, Args...>,
                communication_set_get_action<Operation, Result, Args...>>::type
        {
        };

        template <typename Operation, typename Result, typename... Args>
        Result set_result(
            std::size_t which, std::size_t generation, Args... args)
        {
            using collective_operation =
                traits::communication_operation<communication_set_node,
                    Operation>;

            return collective_operation::template set<Result>(
                *this, which, generation, HPX_MOVE(args)...);
        }

        template <typename Operation, typename Result, typename... Args>
        struct communication_set_set_action
          : hpx::actions::make_action<Result (communication_set_node::*)(
                                          std::size_t, std::size_t, Args...),
                &communication_set_node::set_result<Operation, Result, Args...>,
                communication_set_set_action<Operation, Result, Args...>>::type
        {
        };

    private:
        // re-initialize data
        template <typename T, typename Lock>
        void reinitialize_data(Lock& l)
        {
            HPX_ASSERT_OWNS_LOCK(l);
            if (needs_initialization_)
            {
                needs_initialization_ = false;
                data_ = std::vector<T>(num_sites_);
            }
        }

        template <typename T, typename Lock>
        std::vector<T>& access_data(Lock& l)
        {
            HPX_ASSERT_OWNS_LOCK(l);
            reinitialize_data<T>(l);
            return hpx::any_cast<std::vector<T>&>(data_);
        }

        template <typename Lock>
        void invalidate_data(Lock& l)
        {
            HPX_ASSERT_OWNS_LOCK(l);
            if (!needs_initialization_)
            {
                needs_initialization_ = true;
                data_.reset();
            }
        }

    private:
        template <typename Communicator, typename Operation>
        friend struct hpx::traits::communication_operation;

    private:
        mutex_type mtx_;
        std::size_t const arity_;
        std::size_t const num_connected_;    // number of connecting nodes
        std::size_t const num_sites_;        // number of participants
        std::size_t const site_;
        std::size_t const connect_to_;
        std::size_t which_;
        bool needs_initialization_;
        hpx::unique_any_nonser data_;
        lcos::local::and_gate gate_;

        // valid if site_ != connect_to_
        hpx::shared_future<id_type> connected_node_;
    };

    ///////////////////////////////////////////////////////////////////////////
    HPX_EXPORT hpx::future<hpx::id_type> register_communication_set_name(
        hpx::future<hpx::id_type>&& f, std::string basename, std::size_t site);

    HPX_EXPORT hpx::future<hpx::id_type> create_communication_set_node(
        std::string basename, std::size_t num_sites, std::size_t this_site,
        std::size_t arity);

    HPX_EXPORT hpx::id_type create_local_communication_set_node(
        char const* basename, std::size_t num_sites, std::size_t this_site);
}    // namespace hpx::collectives::detail

#endif    // COMPUTE_HOST_CODE
