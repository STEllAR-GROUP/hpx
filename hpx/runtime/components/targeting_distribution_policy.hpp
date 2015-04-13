//  Copyright (c) 2014-2015 Hartmut Kaiser
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
#include <hpx/runtime/actions/action_support.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/naming/id_type.hpp>
#include <hpx/lcos/packaged_action.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/lcos/local/dataflow.hpp>
#include <hpx/util/move.hpp>

#include <algorithm>
#include <vector>

namespace hpx { namespace detail
{
    /// \cond NOINTERNAL
    BOOST_FORCEINLINE bool has_async_policy(BOOST_SCOPED_ENUM(launch) policy)
    {
        return (static_cast<int>(policy) &
            static_cast<int>(launch::async_policies)) ? true : false;
    }

    template <typename Action, typename Result>
    struct sync_local_invoke
    {
        template <typename ...Ts>
        BOOST_FORCEINLINE static lcos::future<Result> call(
            naming::id_type const& gid, naming::address const&,
            Ts&&... vs)
        {
            lcos::packaged_action<Action, Result> p;
            p.apply(launch::sync, gid, std::forward<Ts>(vs)...);
            return p.get_future();
        }
    };

    template <typename Action, typename R>
    struct sync_local_invoke<Action, lcos::future<R> >
    {
        template <typename ...Ts>
        BOOST_FORCEINLINE static lcos::future<R> call(
            boost::mpl::true_, naming::id_type const&,
            naming::address const& addr, Ts&&... vs)
        {
            HPX_ASSERT(traits::component_type_is_compatible<
                typename Action::component_type>::call(addr));
            return Action::execute_function(addr.address_,
                std::forward<Ts>(vs)...);
        }
    };

    ///////////////////////////////////////////////////////////////////////
    template <typename Action, typename Result>
    struct sync_local_invoke_cb
    {
        template <typename Callback, typename ...Ts>
        BOOST_FORCEINLINE static lcos::future<Result> call(
            naming::id_type const& gid, naming::address const&,
            Callback&& cb, Ts&&... vs)
        {
            lcos::packaged_action<Action, Result> p;
            p.apply_cb(launch::sync, gid, std::forward<Callback>(cb),
                std::forward<Ts>(vs)...);
            return p.get_future();
        }
    };

    template <typename Action, typename R>
    struct sync_local_invoke_cb<Action, lcos::future<R> >
    {
        template <typename Callback, typename ...Ts>
        BOOST_FORCEINLINE static lcos::future<R> call(
            boost::mpl::true_, naming::id_type const&,
            naming::address const& addr, Callback&& cb, Ts&&... vs)
        {
            HPX_ASSERT(traits::component_type_is_compatible<
                typename Action::component_type>::call(addr));
            lcos::future<R> f = Action::execute_function(addr.address_,
                std::forward<Ts>(vs)...);

            // invoke callback
            cb(boost::system::error_code(), parcelset::parcel());

            return f;
        }
    };

    ///////////////////////////////////////////////////////////////////////
    struct keep_id_alive
    {
        explicit keep_id_alive(naming::id_type const& gid)
            : gid_(gid)
        {}

        void operator()() const {}

        naming::id_type gid_;
    };
    /// \endcond
}}

namespace hpx { namespace applier { namespace detail
{
    // forward declaration only
    template <typename Action, typename ...Ts>
    bool apply_r_p(naming::address&& addr, actions::continuation* c,
        naming::id_type const& id, threads::thread_priority priority,
        Ts&&... vs);

    template <typename Action, typename Callback, typename ...Ts>
    bool apply_r_p_cb(naming::address&& addr, actions::continuation* c,
        naming::id_type const& id, threads::thread_priority priority,
        Callback && cb, Ts&&... vs);
}}}

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
            if (id_)
            {
                return components::stub_base<Component>::create_async(
                    id_, std::forward<Ts>(vs)...);
            }

            // by default the object will be created on the current
            // locality
            return components::stub_base<Component>::create_async(
                hpx::find_here(), std::forward<Ts>(vs)...);
        }

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
        hpx::future<std::vector<hpx::id_type> >
        bulk_create(std::size_t count, Ts&&... vs) const
        {
            // handle special cases
            if (id_)
            {
                return components::stub_base<Component>::bulk_create_async(
                    id_, count, std::forward<Ts>(vs)...);
            }
            return components::stub_base<Component>::bulk_create_async(
                hpx::find_here(), count, std::forward<Ts>(vs)...);
        }

        /// \note This function is part of the invocation policy implemented by
        ///       this class
        ///
        template <typename Action, typename ...Ts>
        hpx::future<
            typename traits::promise_local_result<
                typename hpx::actions::extract_action<Action>::remote_result_type
            >::type>
        async(BOOST_SCOPED_ENUM(launch) policy, Ts&&... vs) const
        {
            typedef typename hpx::actions::extract_action<Action>::type action_type;
            typedef typename traits::promise_local_result<
                typename action_type::remote_result_type
            >::type result_type;

            hpx::id_type id = id_ ? id_ : hpx::find_here();

            naming::address addr;
            if (agas::is_local_address_cached(id, addr) && policy == launch::sync)
            {
                return hpx::detail::sync_local_invoke<action_type, result_type>::
                    call(id, addr, std::forward<Ts>(vs)...);
            }

            lcos::packaged_action<action_type, result_type> p;

            bool target_is_managed = false;
            if (policy == launch::sync || hpx::detail::has_async_policy(policy))
            {
                if (addr) {
                    p.apply(policy, std::move(addr), id,
                        std::forward<Ts>(vs)...);
                }
                else if (id.get_management_type() == naming::id_type::managed) {
                    p.apply(policy,
                        naming::id_type(id.get_gid(), naming::id_type::unmanaged),
                        std::forward<Ts>(vs)...);
                    target_is_managed = true;
                }
                else {
                    p.apply(policy, id, std::forward<Ts>(vs)...);
                }
            }

            // keep id alive, if needed - this allows to send the destination as an
            // unmanaged id
            future<result_type> f = p.get_future();

            if (target_is_managed)
            {
                typedef typename lcos::detail::shared_state_ptr_for<
                    future<result_type>
                >::type shared_state_ptr;

                shared_state_ptr const& state = lcos::detail::get_shared_state(f);
                state->set_on_completed(hpx::detail::keep_id_alive(id));
            }

            return f;
        }

        /// \note This function is part of the invocation policy implemented by
        ///       this class
        ///
        template <typename Action, typename Callback, typename ...Ts>
        hpx::future<
            typename traits::promise_local_result<
                typename hpx::actions::extract_action<Action>::remote_result_type
            >::type>
        async_cb(BOOST_SCOPED_ENUM(launch) policy, Callback&& cb, Ts&&... vs) const
        {
            typedef typename hpx::actions::extract_action<Action>::type action_type;
            typedef typename traits::promise_local_result<
                typename action_type::remote_result_type
            >::type result_type;

            hpx::id_type id = id_ ? id_ : hpx::find_here();

            naming::address addr;
            if (agas::is_local_address_cached(id, addr) && policy == launch::sync)
            {
                return hpx::detail::sync_local_invoke_cb<action_type, result_type>::
                    call(gid, addr, std::forward<Callback>(cb),
                        std::forward<Ts>(vs)...);
            }

            lcos::packaged_action<action_type, result_type> p;

            bool target_is_managed = false;
            if (policy == launch::sync || hpx::detail::has_async_policy(policy))
            {
                if (addr) {
                    p.apply_cb(policy, std::move(addr), id,
                        std::forward<Callback>(cb), std::forward<Ts>(vs)...);
                }
                else if (gid.get_management_type() == naming::id_type::managed) {
                    p.apply_cb(policy,
                        naming::id_type(id.get_gid(), naming::id_type::unmanaged),
                        std::forward<Callback>(cb), std::forward<Ts>(vs)...);
                    target_is_managed = true;
                }
                else {
                    p.apply_cb(policy, id, std::forward<Callback>(cb),
                        std::forward<Ts>(vs)...);
                }
            }

            // keep id alive, if needed - this allows to send the destination
            // as an unmanaged id
            future<result_type> f = p.get_future();

            if (target_is_managed)
            {
                typedef typename lcos::detail::shared_state_ptr_for<
                    future<result_type>
                >::type shared_state_ptr;

                shared_state_ptr const& state = lcos::detail::get_shared_state(f);
                state->set_on_completed(detail::keep_id_alive(id));
            }

            return f;
        }

        /// \note This function is part of the invocation policy implemented by
        ///       this class
        ///
        template <typename Action, typename ...Ts>
        bool apply(actions::continuation* c, threads::thread_priority priority,
            Ts&&... vs) const
        {
            hpx::id_type id = id_ ? id_ : hpx::find_here();

            if (!traits::action_is_target_valid<Action>::call(id)) {
                HPX_THROW_EXCEPTION(bad_parameter,
                    "targeting_distribution_policy::apply",
                    boost::str(boost::format(
                        "the target (destination) does not match the action type (%s)"
                    ) % hpx::actions::detail::get_action_name<Action>()));
                return false;
            }

            // Determine whether the id is local or remote
            naming::address addr;
            if (agas::is_local_address_cached(id, addr)) {
                return applier::detail::apply_l_p<Action>(
                    c, id, std::move(addr), priority, std::forward<Ts>(vs)...);
            }

            // apply remotely
            return applier::detail::apply_r_p<Action>(std::move(addr), c, id,
                priority, std::forward<Ts>(vs)...);
        }

        /// \note This function is part of the invocation policy implemented by
        ///       this class
        ///
        template <typename Action, typename Callback, typename ...Ts>
        bool apply_cb(actions::continuation* c,
            threads::thread_priority priority, Callback&& cb, Ts&&... vs) const
        {
            hpx::id_type id = id_ ? id_ : hpx::find_here();

            if (!traits::action_is_target_valid<Action>::call(id)) {
                HPX_THROW_EXCEPTION(bad_parameter,
                    "targeting_distribution_polcy::apply_cb",
                    boost::str(boost::format(
                        "the target (destination) does not match the action type (%s)"
                    ) % hpx::actions::detail::get_action_name<Action>()));
                return false;
            }

            // Determine whether the gid is local or remote
            naming::address addr;
            if (agas::is_local_address_cached(id, addr)) {
                // apply locally
                bool result = applier::detail::apply_l_p<Action>(
                    c, id, std::move(addr), priority, std::forward<Ts>(vs)...);

                // invoke callback
                cb(boost::system::error_code(), parcelset::parcel());
                return result;
            }

            // apply remotely
            return applier::detail::apply_r_p_cb<Action>(std::move(addr), c, id,
                priority, std::forward<Callback>(cb), std::forward<Ts>(vs)...);
        }

        /// \cond NOINTERNAL
        // FIXME: this can be removed once the vector<>::create() functions
        //        have been adapted
        std::size_t
        get_num_items(std::size_t items, hpx::id_type const& loc) const
        {
            // make sure the given id is known to this distribution policy
            HPX_ASSERT(!id_ || loc == hpx::find_here());

            // this distribution policy places all items onto the given locality
            return items;
        }
        /// \endcond

    protected:
        /// \cond NOINTERNAL
        targeting_distribution_policy(hpx::id_type const& id)
          : id_(id)
        {}

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
