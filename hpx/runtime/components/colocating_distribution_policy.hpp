//  Copyright (c) 2014-2018 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file colocating_distribution_policy.hpp

#if !defined(HPX_COMPONENTS_COLOCATING_DISTRIBUTION_POLICY_APR_10_2015_0227PM)
#define HPX_COMPONENTS_COLOCATING_DISTRIBUTION_POLICY_APR_10_2015_0227PM

#include <hpx/config.hpp>
#include <hpx/lcos/detail/async_colocated.hpp>
#include <hpx/lcos/detail/async_colocated_callback.hpp>
#include <hpx/lcos/detail/async_implementations.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/runtime/applier/detail/apply_colocated_callback_fwd.hpp>
#include <hpx/runtime/applier/detail/apply_colocated_fwd.hpp>
#include <hpx/runtime/applier/detail/apply_implementations.hpp>
#include <hpx/runtime/components/client_base.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>
#include <hpx/runtime/launch_policy.hpp>
#include <hpx/runtime/find_here.hpp>
#include <hpx/runtime/naming/id_type.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/serialization/serialization_fwd.hpp>
#include <hpx/traits/extract_action.hpp>
#include <hpx/traits/is_distribution_policy.hpp>
#include <hpx/traits/promise_local_result.hpp>

#include <algorithm>
#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace components
{
    /// This class specifies the parameters for a distribution policy to use
    /// for creating a given number of items on the locality where a given
    /// object is currently placed.
    struct colocating_distribution_policy
    {
    public:
        /// Default-construct a new instance of a \a colocating_distribution_policy.
        /// This policy will represent the local locality.
        colocating_distribution_policy()
        {}

        /// Create a new \a colocating_distribution_policy representing the
        /// locality where the given object os current located
        ///
        /// \param id     [in] The global address of the object with which
        ///                the new instances should be colocated on
        ///
        colocating_distribution_policy operator()(id_type const& id) const
        {
            return colocating_distribution_policy(id);
        }

        /// Create a new \a colocating_distribution_policy representing the
        /// locality where the given object os current located
        ///
        /// \param client  [in] The client side representation of the object
        ///                with which the new instances should be colocated on
        ///
        template <typename Client, typename Stub>
        colocating_distribution_policy operator()(
            client_base<Client, Stub> const& client) const
        {
            return colocating_distribution_policy(client.get_id());
        }

        /// Create one object on the locality of the object this distribution
        /// policy instance is associated with
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
            if (!id_)
            {
                return components::stub_base<Component>::create_async(
                    hpx::find_here(), std::forward<Ts>(vs)...);
            }
            return components::stub_base<Component>::create_colocated_async(
                id_, std::forward<Ts>(vs)...);
        }

        /// \cond NOINTERNAL
        typedef std::pair<hpx::id_type, std::vector<hpx::id_type> >
            bulk_locality_result;
        /// \endcond

        /// Create multiple objects colocated with the object represented
        /// by this policy instance
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
            using components::stub_base;

            hpx::id_type id;
            hpx::future<std::vector<hpx::id_type> > f;
            if (!id_)
            {
                id = hpx::find_here();
                f = stub_base<Component>::bulk_create_async(
                        id, count, std::forward<Ts>(vs)...);
            }
            else
            {
                id = id_;
                f = stub_base<Component>::bulk_create_colocated_async(
                        id, count, std::forward<Ts>(vs)...);
            }

            return f.then(hpx::launch::sync,
                [HPX_CAPTURE_MOVE(id)](
                    hpx::future<std::vector<hpx::id_type> > && f
                ) -> std::vector<bulk_locality_result>
                {
                    std::vector<bulk_locality_result> result;
                    result.emplace_back(id, f.get());
                    return result;
                });
        }

        /// \note This function is part of the invocation policy implemented by
        ///       this class
        ///
        template <typename Action, typename ...Ts>
        hpx::future<
            typename traits::promise_local_result<
                typename hpx::traits::extract_action<Action>::remote_result_type
            >::type>
        async(launch policy, Ts&&... vs) const
        {
            if (!id_)
            {
                return hpx::detail::async_impl<Action>(
                    policy, hpx::find_here(), std::forward<Ts>(vs)...);
            }
            return hpx::detail::async_colocated<Action>(
                id_, std::forward<Ts>(vs)...);
        }

        /// \note This function is part of the invocation policy implemented by
        ///       this class
        ///
        template <typename Action, typename Callback, typename ...Ts>
        hpx::future<
            typename traits::promise_local_result<
                typename hpx::traits::extract_action<Action>::remote_result_type
            >::type>
        async_cb(launch policy, Callback&& cb, Ts&&... vs) const
        {
            if (!id_)
            {
                return hpx::detail::async_cb_impl<Action>(
                    policy, hpx::find_here(),
                    std::forward<Callback>(cb), std::forward<Ts>(vs)...);
            }
            return hpx::detail::async_colocated_cb<Action>(id_,
                std::forward<Callback>(cb), std::forward<Ts>(vs)...);
        }

        /// \note This function is part of the invocation policy implemented by
        ///       this class
        ///
        template <typename Action, typename Continuation, typename ...Ts>
        bool apply(Continuation && c,
            threads::thread_priority priority, Ts&&... vs) const
        {
            if (!id_)
            {
                return hpx::detail::apply_impl<Action>(
                    std::forward<Continuation>(c),
                    hpx::find_here(), priority, std::forward<Ts>(vs)...);
            }
            return hpx::detail::apply_colocated<Action>(
                std::forward<Continuation>(c), id_, std::forward<Ts>(vs)...);
        }

        template <typename Action, typename ...Ts>
        bool apply(threads::thread_priority priority, Ts&&... vs) const
        {
            if (!id_)
            {
                return hpx::detail::apply_impl<Action>(
                    hpx::find_here(), priority, std::forward<Ts>(vs)...);
            }
            return hpx::detail::apply_colocated<Action>(
                id_, std::forward<Ts>(vs)...);
        }

        /// \note This function is part of the invocation policy implemented by
        ///       this class
        ///
        template <typename Action, typename Continuation, typename Callback,
            typename ...Ts>
        bool apply_cb(Continuation && c,
            threads::thread_priority priority, Callback&& cb, Ts&&... vs) const
        {
            if (!id_)
            {
                return hpx::detail::apply_cb_impl<Action>(
                    std::forward<Continuation>(c),
                    hpx::find_here(), priority, std::forward<Callback>(cb),
                    std::forward<Ts>(vs)...);
            }
            return hpx::detail::apply_colocated_cb<Action>(
                std::forward<Continuation>(c),
                id_, std::forward<Callback>(cb),
                std::forward<Ts>(vs)...);
        }

        template <typename Action, typename Callback, typename ...Ts>
        bool apply_cb(
            threads::thread_priority priority, Callback&& cb, Ts&&... vs) const
        {
            if (!id_)
            {
                return hpx::detail::apply_cb_impl<Action>(
                    hpx::find_here(), priority, std::forward<Callback>(cb),
                    std::forward<Ts>(vs)...);
            }
            return hpx::detail::apply_colocated_cb<Action>(
                id_, std::forward<Callback>(cb), std::forward<Ts>(vs)...);
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
        colocating_distribution_policy(id_type const& id)
          : id_(id)
        {}

        friend class hpx::serialization::access;

        template <typename Archive>
        void serialize(Archive& ar, unsigned int const)
        {
            ar & id_;
        }

        hpx::id_type id_;   // the global address of the object with which the
                            // new objects will be colocated
        /// \endcond
    };

    /// A predefined instance of the co-locating \a distribution_policy. It
    /// will represent the local locality and will place all items to create
    /// here.
    static colocating_distribution_policy const colocated;
}}

/// \cond NOINTERNAL
namespace hpx
{
    using hpx::components::colocating_distribution_policy;
    using hpx::components::colocated;

    namespace traits
    {
        template <>
        struct is_distribution_policy<components::colocating_distribution_policy>
          : std::true_type
        {};
    }
}
/// \endcond

#endif
