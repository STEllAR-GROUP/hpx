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
#include <hpx/runtime/components/unique_component_name.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/naming/id_type.hpp>
#include <hpx/performance_counters/performance_counter.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/lcos/local/dataflow.hpp>
#include <hpx/util/move.hpp>
#include <hpx/util/unwrapped.hpp>

#include <algorithm>
#include <vector>

namespace hpx { namespace components
{
    static char const* const default_binpacking_counter_name =
        "/runtime/count/component@";

    /// This class specifies the parameters for a binpacking distribution policy
    /// to use for creating a given number of items on a given set of localities.
    /// The binpacking policy will distribute the new objects in a way such that
    /// each of the localities will equalize the number of overall objects of
    /// this type based on a given criteria (by default this criteria is the
    /// overall number of objects of this type).
    struct binpacking_distribution_policy
    {
    private:
        /// \cond NOINTERNAL
        static hpx::future<std::vector<hpx::id_type> >
        consolidate_result(
            std::vector<hpx::future<std::vector<hpx::id_type> > > && objs)
        {
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

        std::vector<std::size_t>
        static get_items_count(std::size_t count,
            std::vector<boost::uint64_t> && values)
        {
            std::size_t maxcount = 0;
            std::size_t existing = 0;

            for (boost::uint64_t value: values)
            {
                maxcount = (std::max)(maxcount, std::size_t(value));
                existing += value;
            }

            // distribute the number of components to create in a way, so that
            // the overall number of component instances on all localities is
            // approximately the same
            std::size_t num_localities = values.size();

            HPX_ASSERT(maxcount * num_localities >= std::size_t(existing));
            std::size_t missing = maxcount * num_localities - existing;
            if (missing == 0) missing = 1;

            double hole_ratio = (std::min)(count, missing) / double(missing);
            HPX_ASSERT(hole_ratio <= 1.);

            std::size_t overflow_count =
                (count > missing) ? (count - missing) / num_localities : 0;
            std::size_t excess = count - overflow_count * num_localities;

            // calculate the number of new instances to create on each of the
            // localities
            std::vector<std::size_t> to_create;
            to_create.reserve(num_localities);

            std::size_t created_count = 0;
            for (std::size_t i = 0; i != num_localities; ++i)
            {
                boost::uint64_t value = values[i];
                std::size_t numcreate = overflow_count +
                    std::size_t((maxcount - value) * hole_ratio);

                if (excess != 0) {
                    --excess;
                    ++numcreate;
                }

                if (i == num_localities-1) {
                    // last bin gets all the rest
                    if (created_count + numcreate < count)
                        numcreate = count - created_count;
                }

                if (created_count + numcreate > count)
                    numcreate = count - created_count;

                if (numcreate == 0)
                    break;

                // create all components  for each locality at a time
                to_create.push_back(numcreate);

                created_count += numcreate;
                if (created_count >= count)
                    break;
            }

            return to_create;
        }

        template <typename Component>
        struct create_helper
        {
            create_helper(binpacking_distribution_policy const& policy)
              : policy_(policy)
            {}

            template <typename ...Ts>
            hpx::future<std::vector<hpx::id_type> > operator()(
                hpx::future<std::vector<boost::uint64_t> > && values,
                std::size_t count, Ts... vs) const
            {
                std::vector<std::size_t> to_create =
                    policy_.get_items_count(count, values.get());

                std::vector<hpx::future<std::vector<hpx::id_type> > > objs;
                objs.reserve(policy_.localities_.size());

                for (std::size_t i = 0; i != to_create.size(); ++i)
                {
                    objs.push_back(stub_base<Component>::bulk_create_async(
                        policy_.localities_[i], to_create[i], vs...));
                }

                return policy_.consolidate_result(std::move(objs));
            }

            binpacking_distribution_policy const& policy_;
        };
        /// \endcond

    public:
        /// Default-construct a new instance of a \a binpacking_distribution_policy.
        /// This policy will represent one locality (the local locality).
        binpacking_distribution_policy()
          : counter_name_(default_binpacking_counter_name)
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
            char const* counter_name = default_binpacking_counter_name) const
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
            char const* counter_name = default_binpacking_counter_name) const
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

            if (!localities_.empty())
            {
                return stub_base<Component>::create_async(
                    localities_.front(), std::forward<Ts>(vs)...);
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
            hpx::future<std::vector<boost::uint64_t> > values =
                get_counter_values(
                    hpx::components::unique_component_name<
                        hpx::components::component_factory<
                            typename Component::wrapping_type
                        >
                    >::call());

            using hpx::util::placeholders::_1;
            return values.then(
                hpx::util::bind(create_helper<Component>(*this),
                    _1, count, std::forward<Ts>(vs)...));
        }

        /// Returns the name of the performance counter associated with this
        /// policy instance.
        std::string const& get_counter_name() const
        {
            return counter_name_;
        }

    protected:
        /// \cond NOINTERNAL
        static hpx::future<std::vector<boost::uint64_t> >
        get_counter_values_helper(
            std::vector<performance_counters::performance_counter> && counters)
        {
            using namespace hpx::performance_counters;

            std::vector<hpx::future<boost::uint64_t> > values;
            values.reserve(counters.size());

            for (performance_counter const& counter: counters)
                values.push_back(counter.get_value<boost::uint64_t>());

            return hpx::lcos::local::dataflow(
                hpx::util::unwrapped(
                    [](std::vector<boost::uint64_t> && values)
                    {
                        return values;
                    }),
                std::move(values));
        }

        hpx::future<std::vector<boost::uint64_t> > get_counter_values(
            std::string const& component_name) const
        {
            using namespace hpx::performance_counters;

            // create performance counters on all localities
            std::vector<performance_counter> counters;
            counters.reserve(localities_.size());

            if (counter_name_[sizeof(counter_name_)-2] == '@')
            {
                std::string counter_name(counter_name_ + component_name);
                for (hpx::id_type const& id: localities_)
                    counters.push_back(performance_counter(counter_name, id));
            }
            else
            {
                for (hpx::id_type const& id: localities_)
                    counters.push_back(performance_counter(counter_name_, id));
            }

            return hpx::lcos::local::dataflow(
                &binpacking_distribution_policy::get_counter_values_helper,
                std::move(counters));
        }
        /// \endcond

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

        std::vector<id_type> localities_;   // localities to create things on
        std::string counter_name_;          // name of counter to use as criteria
        /// \endcond
    };

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
