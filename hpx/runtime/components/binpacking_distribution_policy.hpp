//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file binpacking_distribution_policy.hpp

#if !defined(HPX_COMPONENTS_BINPACKING_DISTRIBUTION_POLICY_APR_10_2015_0344PM)
#define HPX_COMPONENTS_BINPACKING_DISTRIBUTION_POLICY_APR_10_2015_0344PM

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
    /// This class specifies the parameters for a binpacking distribution policy
    /// to use for creating a given number of items on a given set of localities.
    /// The binpacking policy will distribute the new objects in a way such that
    /// each of the localities will equalize the number of overall objects of
    /// this type based on a given criteria (by default this criteria is the
    /// overall number of objects of this type).
    struct binpacking_distribution_policy
    {
    public:
        /// Default-construct a new instance of a \a binpacking_distribution_policy.
        /// This policy will represent one locality (the local locality).
        binpacking_distribution_policy()
          : counter_name_(default_counter_name)
        {}

        /// Create a new \a default_distribution policy representing the given
        /// set of localities.
        ///
        /// \param locs     [in] The list of localities the new instance should
        ///                 represent
        /// \param counter_name  [in] The name of the performance counter which
        ///                      should be used as the distribution criteria
        ///                      (by default the overall number of existing
        ///                      instances of the given component type will be
        ///                      used).
        ///
        binpacking_distribution_policy operator()(
            std::vector<id_type> const& locs,
            char const* counter_name = default_counter_name) const
        {
#if defined(HPX_DEBUG)
            for (id_type const& loc: locs)
            {
                HPX_ASSERT(naming::is_locality(loc));
            }
#endif
            return binpacking_distribution_policy(locs, counter_name);
        }

        /// Create a new \a default_distribution policy representing the given
        /// locality
        ///
        /// \param loc     [in] The locality the new instance should
        ///                 represent
        /// \param counter_name  [in] The name of the performance counter which
        ///                      should be used as the distribution criteria
        ///                      (by default the overall number of existing
        ///                      instances of the given component type will be
        ///                      used).
        ///
        binpacking_distribution_policy operator()(id_type const& loc,
            char const* counter_name = default_counter_name) const
        {
            HPX_ASSERT(naming::is_locality(loc));
            return binpacking_distribution_policy(loc, counter_name);
        }

        /// Create one object on one of the localities associated by
        /// this policy instance
        ///
        /// \param vs  [in] The arguments which will be forwarded to the
        ///            constructor of the new object.
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
        /// \param vs   [in] The arguments which will be forwarded to the
        ///             constructors of the new objects.
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
//     binpacking_factory::remote_result_type
//     binpacking_factory::create_components_counterbased(
//         components::component_type type, std::size_t count,
//         std::string const& countername) const
//     {
//         // make sure we get localities for derived component type, if any
//         components::component_type prefix_type = type;
//         if (type != components::get_base_type(type))
//             prefix_type = components::get_derived_type(type);
//
//         // get list of locality prefixes
//         std::vector<naming::id_type> localities =
//             hpx::find_all_localities(prefix_type);
//
//         if (localities.empty())
//         {
//             HPX_THROW_EXCEPTION(bad_component_type,
//                 "binpacking_factory::create_components_binpacked",
//                 "attempt to create component instance of unknown type: " +
//                 components::get_component_type_name(type));
//         }
//
//         if (count == std::size_t(-1))
//             count = localities.size();
//
//         // create performance counters on all localities
//         performance_counters::counter_path_elements p;
//         performance_counters::get_counter_type_path_elements(countername, p);
//
//         // FIXME: make loop asynchronous
//         typedef lcos::future<naming::id_type> future_type;
//
//         std::vector<future_type> lazy_counts;
//         BOOST_FOREACH(naming::id_type const& id, localities)
//         {
//             std::string name;
//             p.parentinstanceindex_ = naming::get_locality_id_from_id(id);
//             performance_counters::get_counter_name(p, name);
//             lazy_counts.push_back(performance_counters::get_counter_async(name));
//         }
//
//         // wait for counts to get back, collect statistics
//         long maxcount = 0;
//         long existing = 0;
//
//         // FIXME: make loop asynchronous
//         std::vector<long> counts;
//         counts.reserve(lazy_counts.size());
//         BOOST_FOREACH(future_type & f, lazy_counts)
//         {
//             performance_counters::counter_value value =
//                 performance_counters::stubs::performance_counter::get_value(f.get());
//             counts.push_back(value.get_value<long>());
//             maxcount = (std::max)(maxcount, counts.back());
//             existing += counts.back();
//         }
//
//         // distribute the number of components to create in a way, so that the
//         // overall number of component instances on all localities is
//         // approximately the same
//         HPX_ASSERT(std::size_t(maxcount) * counts.size() >= std::size_t(existing));
//         std::size_t missing = std::size_t(maxcount) * counts.size() -
//             std::size_t(existing);
//         if (missing == 0) missing = 1;
//
//         double hole_ratio = (std::min)(count, missing) / double(missing);
//         HPX_ASSERT(hole_ratio <= 1.);
//
//         std::size_t overflow_count =
//             (count > missing) ? (count - missing) / counts.size() : 0;
//         std::size_t excess = count - overflow_count * counts.size();
//
//         typedef std::vector<lazy_result> future_values_type;
//         typedef server::runtime_support::bulk_create_components_action
//             action_type;
//
//         std::size_t created_count = 0;
//         future_values_type v;
//
//         // start an asynchronous operation for each of the localities
//         for (std::size_t i = 0; i < counts.size(); ++i)
//         {
//             std::size_t numcreate =
//                 std::size_t((maxcount - counts[i]) * hole_ratio) + overflow_count;
//
//             if (excess != 0) {
//                 --excess;
//                 ++numcreate;
//             }
//
//             if (i == counts.size()-1) {
//                 // last bin gets all the rest
//                 if (created_count + numcreate < count)
//                     numcreate = count - created_count;
//             }
//
//             if (created_count + numcreate > count)
//                 numcreate = count - created_count;
//
//             if (numcreate == 0)
//                 break;
//
//             // create all components  for each locality at a time
//             v.push_back(future_values_type::value_type(localities[i].get_gid()));
//             lcos::packaged_action<action_type, std::vector<naming::gid_type> > p;
//             p.apply(launch::async, localities[i], type, numcreate);
//             v.back().gids_ = p.get_future();
//
//             created_count += numcreate;
//             if (created_count >= count)
//                 break;
//         }
//
//         // now wait for the results
//         remote_result_type results;
//
//         BOOST_FOREACH(lazy_result& lr, v)
//         {
//             results.push_back(remote_result_type::value_type(lr.locality_, type));
//             results.back().gids_ = std::move(lr.gids_.get());
//         }
//
//         return results;
//     }

    protected:
        /// \cond NOINTERNAL
        binpacking_distribution_policy(std::vector<id_type> const& localities,
                char const* counter_name)
          : localities_(localities),
            counter_name_(counter_name)
        {}

        binpacking_distribution_policy(id_type const& locality,
                char const* counter_name)
          : counter_name_(counter_name)
        {
            localities_.push_back(locality);
        }

        static char const* const default_counter_name;

        std::vector<id_type> localities_;   // localities to create things on
        std::string counter_name_;          // name of counter to use as criteria
        /// \endcond
    };

    /// \cond NOINTERNAL
    /// By default the binpacking policy uses the overall number of objects
    /// on the used localities.
    char const* const binpacking_distribution_policy::default_counter_name =
        "/runtime/count/component";
    /// \endcond

    /// A predefined instance of the binpacking \a distribution_policy. It will
    /// represent the local locality and will place all items to create here.
    static binpacking_distribution_policy const binpacked;
}}

/// \cond NOINTERNAL
namespace hpx
{
    using hpx::components::binpacking_distribution_policy;
    using hpx::components::binpacked;

    namespace traits { namespace detail
    {
        template <>
        struct is_distribution_policy<components::binpacking_distribution_policy>
          : std::true_type
        {};
    }}
}
/// \endcond

#endif
