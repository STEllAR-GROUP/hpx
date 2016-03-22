//  Copyright (c) 2014-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file default_distribution_policy.hpp

#if !defined(HPX_COMPONENTS_DISTRIBUTION_POLICY_APR_07_2015_1246PM)
#define HPX_COMPONENTS_DISTRIBUTION_POLICY_APR_07_2015_1246PM

#include <hpx/config.hpp>
#include <hpx/traits/is_distribution_policy.hpp>
#include <hpx/traits/component_type_is_compatible.hpp>
#include <hpx/runtime/actions/action_support.hpp>
#include <hpx/runtime/applier/apply.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/naming/id_type.hpp>
#include <hpx/runtime/launch_policy.hpp>
#include <hpx/runtime/serialization/serialization_fwd.hpp>
#include <hpx/runtime/serialization/vector.hpp>
#include <hpx/lcos/packaged_action.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/dataflow.hpp>
#include <hpx/util/move.hpp>

#include <algorithm>
#include <vector>

namespace hpx { namespace components
{
    ///////////////////////////////////////////////////////////////////////////
    /// \cond NOINTERNAL
    namespace detail
    {
        HPX_FORCEINLINE std::size_t
        round_to_multiple(std::size_t n1, std::size_t n2, std::size_t n3)
        {
            return (n1 / n2) * n3;
        }
    }
    /// \endcond

    /// This class specifies the parameters for a simple distribution policy
    /// to use for creating (and evenly distributing) a given number of items
    /// on a given set of localities.
    struct default_distribution_policy
    {
    public:
        /// Default-construct a new instance of a \a default_distribution_policy.
        /// This policy will represent one locality (the local locality).
        default_distribution_policy()
        {}

        /// Create a new \a default_distribution policy representing the given
        /// set of localities.
        ///
        /// \param locs     [in] The list of localities the new instance should
        ///                 represent
        default_distribution_policy operator()(
            std::vector<id_type> const& locs) const
        {
            return default_distribution_policy(locs);
        }

        default_distribution_policy operator()(
            std::vector<id_type> && locs) const
        {
            return default_distribution_policy(std::move(locs));
        }

        /// Create a new \a default_distribution policy representing the given
        /// locality
        ///
        /// \param loc     [in] The locality the new instance should
        ///                 represent
        default_distribution_policy operator()(id_type const& loc) const
        {
            return default_distribution_policy(loc);
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
            using components::stub_base;

            for (hpx::id_type const& loc: localities_)
            {
                if (get_num_items(1, loc) != 0)
                {
                    return stub_base<Component>::create_async(
                        loc, std::forward<Ts>(vs)...);
                }
            }

            // by default the object will be created on the current
            // locality
            return stub_base<Component>::create_async(
                hpx::find_here(), std::forward<Ts>(vs)...);
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
            using components::stub_base;

            if (localities_.size() > 1)
            {
                // schedule creation of all objects across given localities
                std::vector<hpx::future<std::vector<hpx::id_type> > > objs;
                objs.reserve(localities_.size());
                for (hpx::id_type const& loc: localities_)
                {
                    objs.push_back(stub_base<Component>::bulk_create_async(
                        loc, get_num_items(count, loc), vs...));
                }

                // consolidate all results
                return hpx::dataflow(hpx::launch::sync,
                    [=](std::vector<hpx::future<std::vector<hpx::id_type> > > && v)
                        mutable -> std::vector<bulk_locality_result>
                    {
                        HPX_ASSERT(localities_.size() == v.size());

                        std::vector<bulk_locality_result> result;
                        result.reserve(v.size());

                        for (std::size_t i = 0; i != v.size(); ++i)
                        {
#if !defined(HPX_GCC_VERSION) || HPX_GCC_VERSION >= 408000
                            result.emplace_back(
                                    std::move(localities_[i]), v[i].get()
                                );
#else
                            result.push_back(std::make_pair(
                                    std::move(localities_[i]), v[i].get()
                                ));
#endif
                        }
                        return result;
                    },
                    std::move(objs));
            }

            // handle special cases
            hpx::id_type id = get_next_target();

            hpx::future<std::vector<hpx::id_type> > f =
                stub_base<Component>::bulk_create_async(
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
        hpx::future<
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
        hpx::future<
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
                get_next_target(), priority, std::forward<Callback>(cb),
                std::forward<Ts>(vs)...);
        }

        template <typename Action, typename Callback, typename ...Ts>
        bool apply_cb(
            threads::thread_priority priority, Callback&& cb, Ts&&... vs) const
        {
            return hpx::detail::apply_cb_impl<Action>(
                get_next_target(), priority, std::forward<Callback>(cb),
                std::forward<Ts>(vs)...);
        }

        /// Returns the number of associated localities for this distribution
        /// policy
        ///
        /// \note This function is part of the creation policy implemented by
        ///       this class
        ///
        std::size_t get_num_localities() const
        {
            return (std::max)(std::size_t(1), localities_.size());
        }

        /// Returns the locality which is anticipated to be used for the next
        /// async operation
        hpx::id_type get_next_target() const
        {
            return localities_.empty() ? hpx::find_here() : localities_.front();
        }

    protected:
        /// \cond NOINTERNAL
        std::size_t
        get_num_items(std::size_t items, hpx::id_type const& loc) const
        {
            // make sure the given id is known to this distribution policy
            HPX_ASSERT(
                std::find(localities_.begin(), localities_.end(), loc) !=
                    localities_.end() ||
                (localities_.empty() && loc == hpx::find_here())
            );

            // this distribution policy places an equal number of items onto
            // each locality
            std::size_t locs = (std::max)(std::size_t(1), localities_.size());

            // the overall number of items to create is smaller than the number
            // of localities
            if (items < locs)
            {
                auto it = std::find(localities_.begin(), localities_.end(), loc);
                std::size_t num_loc = std::distance(localities_.begin(), it);
                return (items < num_loc) ? 1 : 0;
            }

            // the last locality might get less items
            if (localities_.size() > 1 && loc == localities_.back())
                return items - detail::round_to_multiple(items, locs, locs-1);

            // otherwise just distribute evenly
            return (items + locs - 1) / locs;
        }
        /// \endcond

    protected:
        /// \cond NOINTERNAL
        default_distribution_policy(std::vector<id_type> const& localities)
          : localities_(localities)
        {}

        default_distribution_policy(std::vector<id_type> && localities)
          : localities_(std::move(localities))
        {}

        default_distribution_policy(id_type const& locality)
        {
            localities_.push_back(locality);
        }

        friend class hpx::serialization::access;

        template <typename Archive>
        void serialize(Archive& ar, unsigned int const)
        {
            ar & localities_;
        }

        std::vector<id_type> localities_;   // localities to create things on
        /// \endcond
    };

    /// A predefined instance of the default \a distribution_policy. It will
    /// represent the local locality and will place all items to create here.
    static default_distribution_policy const default_layout;
}}

/// \cond NOINTERNAL
namespace hpx
{
    using hpx::components::default_distribution_policy;
    using hpx::components::default_layout;

    namespace traits
    {
        template <>
        struct is_distribution_policy<components::default_distribution_policy>
          : std::true_type
        {};
    }
}
/// \endcond

#endif
