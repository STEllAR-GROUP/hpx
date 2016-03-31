//  Copyright (c) 2014-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file target_distribution_policy.hpp

#if !defined(HPX_COMPONENTS_TARGET_DISTRIBUTION_POLICY_APR_12_2015_1245PM)
#define HPX_COMPONENTS_TARGET_DISTRIBUTION_POLICY_APR_12_2015_1245PM

#include <hpx/config.hpp>
#include <hpx/traits/is_distribution_policy.hpp>
#include <hpx/traits/component_type_is_compatible.hpp>
#include <hpx/traits/action_is_target_valid.hpp>
#include <hpx/runtime/agas/interface.hpp>
#include <hpx/runtime/applier/detail/apply_implementations_fwd.hpp>
#include <hpx/runtime/actions/action_support.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/naming/id_type.hpp>
#include <hpx/runtime/launch_policy.hpp>
#include <hpx/runtime/serialization/serialization_fwd.hpp>
#include <hpx/lcos/packaged_action.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/dataflow.hpp>
#include <hpx/lcos/detail/async_implementations_fwd.hpp>

#include <algorithm>
#include <vector>

namespace hpx { namespace components
{
    /// This class specifies the parameters for a simple distribution policy
    /// to use for creating (and evenly distributing) a given number of items
    /// on a given set of localities.
    struct targeting_distribution_policy
    {
    public:
        /// Default-construct a new instance of a \a targeting_distribution_policy.
        /// This policy will represent one locality (the local locality).
        targeting_distribution_policy()
        {}

        /// Create a new \a targeting_distribution_policy representing the given
        /// locality
        ///
        /// \param loc     [in] The locality the new instance should
        ///                 represent
        targeting_distribution_policy operator()(id_type const& id) const
        {
            return targeting_distribution_policy(id);
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
        template <typename Component, typename ...Ts>
        hpx::future<hpx::id_type> create(Ts&&... vs) const
        {
            // by default the object will be created on the current
            // locality
            return components::stub_base<Component>::create_async(
                get_next_target(), std::forward<Ts>(vs)...);
        }

        /// \cond NOINTERNAL
        typedef std::pair<hpx::id_type, std::vector<hpx::id_type> >
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
        template <typename Component, typename ...Ts>
        hpx::future<std::vector<bulk_locality_result> >
        bulk_create(std::size_t count, Ts&&... vs) const
        {
            // by default the object will be created on the current
            // locality
            hpx::id_type id = get_next_target();
            hpx::future<std::vector<hpx::id_type> > f =
                components::stub_base<Component>::bulk_create_async(
                    id, count, std::forward<Ts>(vs)...);

            return f.then(hpx::launch::sync,
                [id](hpx::future<std::vector<hpx::id_type> > && f)
                    -> std::vector<bulk_locality_result>
                {
                    std::vector<bulk_locality_result> result;
#if !defined(HPX_GCC_VERSION) || HPX_GCC_VERSION >= 408000
                    result.emplace_back(id, f.get());
#else
                    result.push_back(std::make_pair(id, f.get()));
#endif
                    return result;
                });
        }

        /// \note This function is part of the invocation policy implemented by
        ///       this class
        ///
        template <typename Action, typename ...Ts>
        HPX_FORCEINLINE hpx::future<
            typename traits::promise_local_result<
                typename hpx::actions::extract_action<Action>::remote_result_type
            >::type>
        async(launch policy, Ts&&... vs) const
        {
            return hpx::detail::async_impl<Action>(policy,
                get_next_target(), std::forward<Ts>(vs)...);
        }

        /// \note This function is part of the invocation policy implemented by
        ///       this class
        ///
        template <typename Action, typename Callback, typename ...Ts>
        HPX_FORCEINLINE hpx::future<
            typename traits::promise_local_result<
                typename hpx::actions::extract_action<Action>::remote_result_type
            >::type>
        async_cb(launch policy, Callback&& cb, Ts&&... vs) const
        {
            return hpx::detail::async_cb_impl<Action>(policy,
                get_next_target(), std::forward<Callback>(cb),
                std::forward<Ts>(vs)...);
        }

        /// \note This function is part of the invocation policy implemented by
        ///       this class
        ///
        template <typename Action, typename Continuation, typename ...Ts>
        bool apply(Continuation && c,
            threads::thread_priority priority, Ts&&... vs) const
        {
            return hpx::detail::apply_impl<Action>(std::forward<Continuation>(c),
                get_next_target(), priority, std::forward<Ts>(vs)...);
        }

        template <typename Action, typename ...Ts>
        bool apply(
            threads::thread_priority priority, Ts&&... vs) const
        {
            return hpx::detail::apply_impl<Action>(
                get_next_target(), priority, std::forward<Ts>(vs)...);
        }

        /// \note This function is part of the invocation policy implemented by
        ///       this class
        ///
        template <typename Action, typename Continuation, typename Callback,
            typename ...Ts>
        bool apply_cb(Continuation && c,
            threads::thread_priority priority, Callback&& cb, Ts&&... vs) const
        {
            return hpx::detail::apply_cb_impl<Action>(std::forward<Continuation>(c),
                get_next_target(), priority,
                std::forward<Callback>(cb), std::forward<Ts>(vs)...);
        }

        template <typename Action, typename Callback, typename ...Ts>
        bool apply_cb(
            threads::thread_priority priority, Callback&& cb, Ts&&... vs) const
        {
            return hpx::detail::apply_cb_impl<Action>(
                get_next_target(), priority,
                std::forward<Callback>(cb), std::forward<Ts>(vs)...);
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
            return id_ ? id_ : hpx::find_here();
        }

    protected:
        /// \cond NOINTERNAL
        targeting_distribution_policy(hpx::id_type const& id)
          : id_(id)
        {}

        friend class hpx::serialization::access;

        template <typename Archive>
        void serialize(Archive& ar, unsigned int const)
        {
            ar & id_;
        }

        hpx::id_type id_;   // locality to encapsulate
        /// \endcond
    };

    /// A predefined instance of the \a targeting_distribution_policy. It will
    /// represent the local locality and will place all items to create here.
    static targeting_distribution_policy const target;
}}

/// \cond NOINTERNAL
namespace hpx
{
    using hpx::components::targeting_distribution_policy;
    using hpx::components::target;

    namespace traits
    {
        template <>
        struct is_distribution_policy<components::targeting_distribution_policy>
          : std::true_type
        {};
    }
}
/// \endcond

#endif
