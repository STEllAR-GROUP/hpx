//  Copyright (c) 2020-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if !defined(HPX_COMPUTE_DEVICE_CODE)

#include <hpx/actions_base/component_action.hpp>
#include <hpx/async_base/launch_policy.hpp>
#include <hpx/components_base/server/component_base.hpp>
#include <hpx/datastructures/any.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/lcos_local/and_gate.hpp>
#include <hpx/synchronization/spinlock.hpp>
#include <hpx/thread_support/assert_owns_lock.hpp>
#include <hpx/type_support/unused.hpp>

#include <cstddef>
#include <mutex>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx::traits {

    // This type can be specialized for a particular collective operation
    template <typename Communicator, typename Operation>
    struct communication_operation;
}    // namespace hpx::traits

namespace hpx::collectives::detail {

    ///////////////////////////////////////////////////////////////////////////
    class communicator_server
      : public hpx::components::component_base<communicator_server>
    {
        using mutex_type = hpx::spinlock;

    public:
        HPX_EXPORT communicator_server() noexcept;

        HPX_EXPORT explicit communicator_server(std::size_t num_sites) noexcept;

        ///////////////////////////////////////////////////////////////////////
        // generic get action, dispatches to proper operation
        template <typename Operation, typename Result, typename... Args>
        Result get_result(
            std::size_t which, std::size_t generation, Args... args)
        {
            using collective_operation =
                traits::communication_operation<communicator_server, Operation>;

            return collective_operation::template get<Result>(
                *this, which, generation, HPX_MOVE(args)...);
        }

        template <typename Operation, typename Result, typename... Args>
        struct communication_get_action
          : hpx::actions::make_action<Result (communicator_server::*)(
                                          std::size_t, std::size_t, Args...),
                &communicator_server::get_result<Operation, Result, Args...>,
                communication_get_action<Operation, Result, Args...>>::type
        {
        };

        template <typename Operation, typename Result, typename... Args>
        Result set_result(
            std::size_t which, std::size_t generation, Args... args)
        {
            using collective_operation =
                traits::communication_operation<communicator_server, Operation>;

            return collective_operation::template set<Result>(
                *this, which, generation, HPX_MOVE(args)...);
        }

        template <typename Operation, typename Result, typename... Args>
        struct communication_set_action
          : hpx::actions::make_action<Result (communicator_server::*)(
                                          std::size_t, std::size_t, Args...),
                &communicator_server::set_result<Operation, Result, Args...>,
                communication_set_action<Operation, Result, Args...>>::type
        {
        };

    private:
        // re-initialize data
        template <typename T, typename Lock>
        void reinitialize_data(Lock& l, std::size_t num_values)
        {
            HPX_ASSERT_OWNS_LOCK(l);
            if (needs_initialization_)
            {
                needs_initialization_ = false;
                data_available_ = false;
                data_ = std::vector<T>(
                    num_values == static_cast<std::size_t>(-1) ? num_sites_ :
                                                                 num_values);
            }
        }

        template <typename T, typename Lock>
        std::vector<T>& access_data(
            Lock& l, std::size_t num_values = static_cast<std::size_t>(-1))
        {
            HPX_ASSERT_OWNS_LOCK(l);
            reinitialize_data<T>(l, num_values);
            return hpx::any_cast<std::vector<T>&>(data_);
        }

        template <typename Lock>
        void invalidate_data(Lock& l) noexcept
        {
            HPX_ASSERT_OWNS_LOCK(l);
            if (!needs_initialization_)
            {
                needs_initialization_ = true;
                data_available_ = false;
                data_.reset();
            }
        }

        template <typename Lock>
        void set_and_invalidate_data(
            std::size_t which, std::size_t generation, Lock l) noexcept
        {
            HPX_ASSERT_OWNS_LOCK(l);

            // set() consumes lock
            if (gate_.set(which, HPX_MOVE(l)))
            {
                l = Lock(mtx_);
                if (generation != static_cast<std::size_t>(-1))
                {
                    gate_.next_generation(l, generation);
                }
                invalidate_data(l);
            }
        }

        template <typename F, typename Lock>
        auto get_future_and_synchronize(std::size_t generation, F&& f, Lock& l)
        {
            HPX_ASSERT_OWNS_LOCK(l);
            auto fut = gate_.get_shared_future(l).then(
                hpx::launch::sync, HPX_FORWARD(F, f));

            gate_.synchronize(generation == static_cast<std::size_t>(-1) ?
                    gate_.generation(l) :
                    generation,
                l);

            return fut;
        }

        // Step will be invoked under lock for each site that checks in (either
        // set or get).
        //
        // Finalizer will be invoked under lock after all sites have checked in.
        template <typename Data, typename Step, typename Finalizer>
        auto handle_data(std::size_t which, std::size_t generation,
            [[maybe_unused]] Step&& step, Finalizer&& finalizer,
            std::size_t num_values = static_cast<std::size_t>(-1))
        {
            auto on_ready = [this, num_values,
                                finalizer = HPX_FORWARD(Finalizer, finalizer)](
                                shared_future<void>&& f) mutable {
                f.get();    // propagate any exceptions

                if constexpr (!std::is_same_v<std::nullptr_t,
                                  std::decay_t<Finalizer>>)
                {
                    std::unique_lock l(mtx_);
                    [[maybe_unused]] util::ignore_while_checking il(&l);

                    // call provided finalizer
                    return HPX_FORWARD(Finalizer, finalizer)(
                        access_data<Data>(l, num_values), data_available_);
                }
                else
                {
                    HPX_UNUSED(this);
                    HPX_UNUSED(num_values);
                    HPX_UNUSED(finalizer);
                }
            };

            std::unique_lock l(mtx_);
            [[maybe_unused]] util::ignore_while_checking il(&l);

            auto f =
                get_future_and_synchronize(generation, HPX_MOVE(on_ready), l);

            if constexpr (!std::is_same_v<std::nullptr_t, std::decay_t<Step>>)
            {
                // call provided step function for each invocation site
                HPX_FORWARD(Step, step)(access_data<Data>(l, num_values));
            }

            set_and_invalidate_data(which, generation, HPX_MOVE(l));

            return f;
        }

        // protect against vector<bool> idiosyncrasies
        template <typename ValueType, typename Data>
        static decltype(auto) handle_bool(Data&& data)
        {
            if constexpr (std::is_same_v<ValueType, bool>)
            {
                return static_cast<bool>(data);
            }
            else
            {
                return HPX_FORWARD(Data, data);
            }
        }

        template <typename Communicator, typename Operation>
        friend struct hpx::traits::communication_operation;

        mutex_type mtx_;
        hpx::unique_any_nonser data_;
        lcos::local::and_gate gate_;
        std::size_t const num_sites_;
        bool needs_initialization_;
        bool data_available_;
    };
}    // namespace hpx::collectives::detail

#endif    // COMPUTE_HOST_CODE
