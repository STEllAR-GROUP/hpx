//  Copyright (c) 2014-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file default_distribution_policy.hpp

#if !defined(HPX_COMPONENTS_DISTRIBUTION_POLICY_APR_07_2015_1246PM)
#define HPX_COMPONENTS_DISTRIBUTION_POLICY_APR_07_2015_1246PM

#include <hpx/config.hpp>
#include <hpx/traits/is_distribution_policy.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/naming/id_type.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/lcos/local/dataflow.hpp>
#include <hpx/util/move.hpp>

#include <algorithm>
#include <vector>

namespace hpx { namespace components
{
    ///////////////////////////////////////////////////////////////////////////
    /// \cond NOINTERNAL
    namespace detail
    {
        BOOST_FORCEINLINE std::size_t
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
#if defined(HPX_DEBUG)
            for (id_type const& loc: locs)
            {
                HPX_ASSERT(naming::is_locality(loc));
            }
#endif
            return default_distribution_policy(locs);
        }

        /// Create a new \a default_distribution policy representing the given
        /// locality
        ///
        /// \param loc     [in] The locality the new instance should
        ///                 represent
        default_distribution_policy operator()(id_type const& loc) const
        {
            HPX_ASSERT(naming::is_locality(loc));
            return default_distribution_policy(loc);
        }

        /// Return the list of localities represented by this
        /// \a distribution_policy
        std::vector<id_type> const& get_localities() const
        {
            return localities_;
        }

        /// Create one object on one of the localities associated by
        /// this policy instance
        ///
        /// \params vs  [in] The arguments which will be forwarded to the
        ///             constructor of the new object.
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

        /// Create multiple objects on the localities associated by
        /// this policy instance
        ///
        /// \param count [in] The number of objects to create
        /// \params vs   [in] The arguments which will be forwarded to the
        ///              constructors of the new objects.
        ///
        /// \returns A future holding the list of global addresses which
        ///          represent the newly created objects
        ///
        template <typename Component, typename ...Ts>
        hpx::future<std::vector<hpx::id_type> >
        bulk_create(std::size_t count, Ts&&... vs) const
        {
            using components::stub_base;

            // handle special cases
            if (localities_.size() == 0)
            {
                return stub_base<Component>::bulk_create_async(
                    hpx::find_here(), count, std::forward<Ts>(vs)...);
            }
            else if (localities_.size() == 1)
            {
                return stub_base<Component>::bulk_create_async(
                    localities_.front(), count, std::forward<Ts>(vs)...);
            }

            // schedule creation of all objects across given localities
            std::vector<hpx::future<std::vector<hpx::id_type> > > objs;
            objs.reserve(localities_.size());
            for (hpx::id_type const& loc: localities_)
            {
                objs.push_back(stub_base<Component>::bulk_create_async(
                    loc, get_num_items(count, loc), vs...));
            }

            // consolidate all results into single array
            return hpx::lcos::local::dataflow(
                [](std::vector<hpx::future<std::vector<hpx::id_type> > > && v)
                    -> std::vector<hpx::id_type>
                {
                    std::vector<hpx::id_type> result = v.front().get();
                    for (auto it = v.begin()+1; it != v.end(); ++it)
                    {
                        std::vector<id_type> r = it->get();
                        std::copy(r.begin(), r.end(),
                            std::back_inserter(result));
                    }
                    return result;
                },
                std::move(objs));
        }

    protected:
        // Return the number of items to place on the given locality.
        //
        // \param items    [in] The overall number of items to create based on
        //                 the given \a distribution_policy.
        // \param loc      [in] The locality for which the function will
        //                 return how many items to create.
        //
        // \note This function will calculate the number of items to be
        //       created on the given locality. A \a distribution_policy will
        //       evenly distribute the overall number of items over the given
        //       localities it represents.
        //
        // \returns The number of items to be created on the given locality
        //          \a loc.
        //
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

    protected:
        /// \cond NOINTERNAL
        default_distribution_policy(std::vector<id_type> const& localities)
          : localities_(localities)
        {}

        default_distribution_policy(id_type const& locality)
        {
            localities_.push_back(locality);
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

    namespace traits { namespace detail
    {
        template <>
        struct is_distribution_policy<components::default_distribution_policy>
          : std::true_type
        {};
    }}
}
/// \endcond

#endif
