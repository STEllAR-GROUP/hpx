//  Copyright (c) 2014-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file target_distribution_policy.hpp

#pragma once

#include <hpx/config.hpp>
#include <hpx/actions_base/actions_base_support.hpp>
#include <hpx/actions_base/traits/extract_action.hpp>
#include <hpx/actions_base/traits/is_distribution_policy.hpp>
#include <hpx/async_base/launch_policy.hpp>
#include <hpx/async_distributed/applier/detail/apply_implementations_fwd.hpp>
#include <hpx/async_distributed/dataflow.hpp>
#include <hpx/async_distributed/detail/async_implementations_fwd.hpp>
#include <hpx/async_distributed/packaged_action.hpp>
#include <hpx/components_base/agas_interface.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/futures/traits/promise_local_result.hpp>
#include <hpx/naming_base/id_type.hpp>
#include <hpx/runtime_components/create_component_helpers.hpp>
#include <hpx/serialization/serialization_fwd.hpp>

#include <algorithm>
#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace components {

    /// This class specifies the parameters for a simple distribution policy
    /// to use for creating (and evenly distributing) a given number of items
    /// on a given set of localities.
    struct target_distribution_policy
    {
    public:
        /// Default-construct a new instance of a \a target_distribution_policy.
        /// This policy will represent one locality (the local locality).
        target_distribution_policy() = default;

        /// Create a new \a target_distribution_policy representing the given
        /// locality
        ///
        /// \param loc     [in] The locality the new instance should
        ///                 represent
        target_distribution_policy operator()(id_type const& id) const
        {
            return target_distribution_policy(id);
        }

        /// Create one object on one of the localities associated by
        /// this policy instance
        ///
        /// \param vs  [in] The arguments which will be forwarded to the
        ///            constructor of the new object.
        ///
        /// \note This function is part of the placement policy implemented by
        ///       this class
        ///
        /// \returns A future holding the global address which represents
        ///          the newly created object
        ///
        template <typename Component, typename... Ts>
        hpx::future<hpx::id_type> create(Ts&&... vs) const
        {
            // by default the object will be created on the current
            // locality
            return components::create_async<Component>(
                get_next_target(), HPX_FORWARD(Ts, vs)...);
        }

        /// \cond NOINTERNAL
        typedef std::pair<hpx::id_type, std::vector<hpx::id_type>>
            bulk_locality_result;
        /// \endcond

        /// Create multiple objects on the localities associated by
        /// this policy instance
        ///
        /// \param count [in] The number of objects to create
        /// \param vs   [in] The arguments which will be forwarded to the
        ///             constructors of the new objects.
        ///
        /// \note This function is part of the placement policy implemented by
        ///       this class
        ///
        /// \returns A future holding the list of global addresses which
        ///          represent the newly created objects
        ///
        template <typename Component, typename... Ts>
        hpx::future<std::vector<bulk_locality_result>> bulk_create(
            std::size_t count, Ts&&... vs) const
        {
            // by default the object will be created on the current
            // locality
            hpx::id_type id = get_next_target();
            hpx::future<std::vector<hpx::id_type>> f =
                components::bulk_create_async<Component>(
                    id, count, HPX_FORWARD(Ts, vs)...);

            return f.then(hpx::launch::sync,
                [id = HPX_MOVE(id)](hpx::future<std::vector<hpx::id_type>>&& f)
                    -> std::vector<bulk_locality_result> {
                    std::vector<bulk_locality_result> result;
                    result.emplace_back(id, f.get());
                    return result;
                });
        }

        /// \note This function is part of the invocation policy implemented by
        ///       this class
        ///
        template <typename Action>
        struct async_result
        {
            using type = hpx::future<
                typename traits::promise_local_result<typename hpx::traits::
                        extract_action<Action>::remote_result_type>::type>;
        };

        template <typename Action, typename... Ts>
        HPX_FORCEINLINE typename async_result<Action>::type async(
            launch policy, Ts&&... vs) const
        {
            return hpx::detail::async_impl<Action>(
                policy, get_next_target(), HPX_FORWARD(Ts, vs)...);
        }

        /// \note This function is part of the invocation policy implemented by
        ///       this class
        ///
        template <typename Action, typename Callback, typename... Ts>
        HPX_FORCEINLINE typename async_result<Action>::type async_cb(
            launch policy, Callback&& cb, Ts&&... vs) const
        {
            return hpx::detail::async_cb_impl<Action>(policy, get_next_target(),
                HPX_FORWARD(Callback, cb), HPX_FORWARD(Ts, vs)...);
        }

        /// \note This function is part of the invocation policy implemented by
        ///       this class
        ///
        template <typename Action, typename Continuation, typename... Ts>
        bool apply(Continuation&& c, threads::thread_priority priority,
            Ts&&... vs) const
        {
            return hpx::detail::apply_impl<Action>(HPX_FORWARD(Continuation, c),
                get_next_target(), priority, HPX_FORWARD(Ts, vs)...);
        }

        template <typename Action, typename... Ts>
        bool apply(threads::thread_priority priority, Ts&&... vs) const
        {
            return hpx::detail::apply_impl<Action>(
                get_next_target(), priority, HPX_FORWARD(Ts, vs)...);
        }

        /// \note This function is part of the invocation policy implemented by
        ///       this class
        ///
        template <typename Action, typename Continuation, typename Callback,
            typename... Ts>
        bool apply_cb(Continuation&& c, threads::thread_priority priority,
            Callback&& cb, Ts&&... vs) const
        {
            return hpx::detail::apply_cb_impl<Action>(
                HPX_FORWARD(Continuation, c), get_next_target(), priority,
                HPX_FORWARD(Callback, cb), HPX_FORWARD(Ts, vs)...);
        }

        template <typename Action, typename Callback, typename... Ts>
        bool apply_cb(
            threads::thread_priority priority, Callback&& cb, Ts&&... vs) const
        {
            return hpx::detail::apply_cb_impl<Action>(get_next_target(),
                priority, HPX_FORWARD(Callback, cb), HPX_FORWARD(Ts, vs)...);
        }

        /// Returns the number of associated localities for this distribution
        /// policy
        ///
        /// \note This function is part of the creation policy implemented by
        ///       this class
        ///
        std::size_t get_num_localities() const
        {
            return 1;
        }

        /// Returns the locality which is anticipated to be used for the next
        /// async operation
        hpx::id_type get_next_target() const
        {
            return id_ ?
                id_ :
                naming::get_id_from_locality_id(agas::get_locality_id());
        }

    protected:
        /// \cond NOINTERNAL
        explicit target_distribution_policy(hpx::id_type const& id)
          : id_(id)
        {
        }

        friend class hpx::serialization::access;

        template <typename Archive>
        void serialize(Archive& ar, unsigned int const)
        {
            // clang-format off
            ar & id_;
            // clang-format on
        }

        hpx::id_type id_;    // locality to encapsulate
        /// \endcond
    };

    /// A predefined instance of the \a target_distribution_policy. It will
    /// represent the local locality and will place all items to create here.
    static target_distribution_policy const target{};
}}    // namespace hpx::components

/// \cond NOINTERNAL
namespace hpx {

    using hpx::components::target;
    using hpx::components::target_distribution_policy;

    namespace traits {
        template <>
        struct is_distribution_policy<components::target_distribution_policy>
          : std::true_type
        {
        };
    }    // namespace traits
}    // namespace hpx
/// \endcond
