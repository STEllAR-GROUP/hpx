//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file binpacking_distribution_policy.hpp

#if !defined(HPX_COMPONENTS_BINPACKING_DISTRIBUTION_POLICY_APR_10_2015_0344PM)
#define HPX_COMPONENTS_BINPACKING_DISTRIBUTION_POLICY_APR_10_2015_0344PM

#include <hpx/config.hpp>
#include <hpx/dataflow.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/performance_counters/performance_counter.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>
#include <hpx/runtime/components/unique_component_name.hpp>
#include <hpx/runtime/naming/id_type.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/serialization/serialization_fwd.hpp>
#include <hpx/runtime/serialization/string.hpp>
#include <hpx/runtime/serialization/vector.hpp>
#include <hpx/traits/is_distribution_policy.hpp>
#include <hpx/util/unwrap.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace components
{
    static char const* const default_binpacking_counter_name =
        "/runtime{locality/total}/count/component@";

    namespace detail
    {
        /// \cond NOINTERNAL
        inline std::vector<std::size_t>
        get_items_count(std::size_t count, std::vector<std::uint64_t> const& values)
        {
            std::size_t maxcount = 0;
            std::size_t existing = 0;

            for (std::uint64_t value: values)
            {
                maxcount = (std::max)(maxcount, std::size_t(value));
                existing += std::size_t(value);
            }

            // distribute the number of components to create in a way, so that
            // the overall number of component instances on all localities is
            // approximately the same
            std::size_t num_localities = values.size();

            // calculate the number of new instances to create on each of the
            // localities
            std::vector<std::size_t> to_create(num_localities, 0);

            bool even_fill = true;
            while (even_fill)
            {
                even_fill = false;
                for (std::size_t i = 0; i != num_localities; ++i)
                {
                    if(values[i] + to_create[i] >= maxcount) continue;
                    even_fill = true;

                    ++to_create[i];
                    --count;
                    if (count == 0) break;
                }
            }
            std::size_t i = 0;
            while (count != 0)
            {
                ++to_create[i];
                i = (i + 1) % num_localities;
                --count;
            }

            return to_create;
        }

        inline hpx::future<std::vector<std::uint64_t> >
        retrieve_counter_values(
            std::vector<performance_counters::performance_counter> && counters)
        {
            using namespace hpx::performance_counters;

            std::vector<hpx::future<std::uint64_t> > values;
            values.reserve(counters.size());

            for (performance_counter const& counter: counters)
                values.push_back(counter.get_value<std::uint64_t>());

            return hpx::dataflow(hpx::launch::sync,
                hpx::util::unwrapping(
                    [](std::vector<std::uint64_t> && values)
                    {
                        return values;
                    }),
                std::move(values));
        }

        template <typename String>
        hpx::future<std::vector<std::uint64_t> > get_counter_values(
            String component_name, std::string const& counter_name,
            std::vector<hpx::id_type> const& localities)
        {
            using namespace hpx::performance_counters;

            // create performance counters on all localities
            std::vector<performance_counter> counters;
            counters.reserve(localities.size());

            if (counter_name[counter_name.size()-1] == '@')
            {
                std::string name(counter_name + component_name);

                for (hpx::id_type const& id: localities)
                    counters.push_back(performance_counter(name, id));
            }
            else
            {
                for (hpx::id_type const& id: localities)
                    counters.push_back(performance_counter(counter_name, id));
            }

            return hpx::dataflow(
                &retrieve_counter_values, std::move(counters));
        }

        inline hpx::id_type const& get_best_locality(
            hpx::future<std::vector<std::uint64_t> > && f,
            std::vector<hpx::id_type> const& localities)
        {
            std::vector<std::uint64_t> values = f.get();

            std::size_t best_locality = 0;
            std::uint64_t min_value =
                (std::numeric_limits<std::uint64_t>::max)();

            for (std::size_t i = 0; i != values.size(); ++i)
            {
                if (min_value > values[i])
                {
                    min_value = values[i];
                    best_locality = i;
                }
            }

            return localities[best_locality];
        }

        template <typename Component>
        struct create_helper
        {
            create_helper(std::vector<hpx::id_type> const& localities)
              : localities_(localities)
            {}

            template <typename ...Ts>
            hpx::future<hpx::id_type> operator()(
                hpx::future<std::vector<std::uint64_t> > && values,
                Ts&&... vs) const
            {
                hpx::id_type const& best_locality =
                    get_best_locality(std::move(values), localities_);

                return stub_base<Component>::create_async(
                    best_locality, std::forward<Ts>(vs)...);
            }

            std::vector<hpx::id_type> const& localities_;
        };

        template <typename Component>
        struct create_bulk_helper
        {
            typedef std::pair<hpx::id_type, std::vector<hpx::id_type> >
                bulk_locality_result;

            create_bulk_helper(std::vector<hpx::id_type> const& localities)
              : localities_(localities)
            {}

            template <typename ...Ts>
            hpx::future<std::vector<bulk_locality_result> >
            operator()(
                hpx::future<std::vector<std::uint64_t> > && values,
                std::size_t count, Ts&&... vs) const
            {
                std::vector<std::size_t> to_create =
                    detail::get_items_count(count, values.get());

                std::vector<hpx::future<std::vector<hpx::id_type> > > objs;
                objs.reserve(localities_.size());

                for (std::size_t i = 0; i != to_create.size(); ++i)
                {
                    objs.push_back(stub_base<Component>::bulk_create_async(
                        localities_[i], to_create[i], vs...));
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
                            result.emplace_back(
                                    std::move(localities_[i]), v[i].get()
                                );
                        }
                        return result;
                    },
                    std::move(objs));
            }

            std::vector<hpx::id_type> const& localities_;
        };
        /// \endcond
    }

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
          : counter_name_(default_binpacking_counter_name)
        {}

        /// Create a new \a default_distribution policy representing the given
        /// set of localities.
        ///
        /// \param locs     [in] The list of localities the new instance should
        ///                 represent
        /// \param perf_counter_name  [in] The name of the performance counter which
        ///                      should be used as the distribution criteria
        ///                      (by default the overall number of existing
        ///                      instances of the given component type will be
        ///                      used).
        ///
        binpacking_distribution_policy operator()(
            std::vector<id_type> const& locs,
            char const* perf_counter_name = default_binpacking_counter_name) const
        {
#if defined(HPX_DEBUG)
            for (id_type const& loc: locs)
            {
                HPX_ASSERT(naming::is_locality(loc));
            }
#endif
            return binpacking_distribution_policy(locs, perf_counter_name);
        }

        /// Create a new \a default_distribution policy representing the given
        /// set of localities.
        ///
        /// \param locs     [in] The list of localities the new instance should
        ///                 represent
        /// \param perf_counter_name  [in] The name of the performance counter which
        ///                      should be used as the distribution criteria
        ///                      (by default the overall number of existing
        ///                      instances of the given component type will be
        ///                      used).
        ///
        binpacking_distribution_policy operator()(
            std::vector<id_type> && locs,
            char const* perf_counter_name = default_binpacking_counter_name) const
        {
#if defined(HPX_DEBUG)
            for (id_type const& loc: locs)
            {
                HPX_ASSERT(naming::is_locality(loc));
            }
#endif
            return binpacking_distribution_policy(std::move(locs),
                perf_counter_name);
        }

        /// Create a new \a default_distribution policy representing the given
        /// locality
        ///
        /// \param loc     [in] The locality the new instance should
        ///                 represent
        /// \param perf_counter_name  [in] The name of the performance counter which
        ///                      should be used as the distribution criteria
        ///                      (by default the overall number of existing
        ///                      instances of the given component type will be
        ///                      used).
        ///
        binpacking_distribution_policy operator()(id_type const& loc,
            char const* perf_counter_name = default_binpacking_counter_name) const
        {
            HPX_ASSERT(naming::is_locality(loc));
            return binpacking_distribution_policy(loc, perf_counter_name);
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

            // handle special cases
            if (localities_.size() == 0)
            {
                return stub_base<Component>::create_async(
                    hpx::find_here(), std::forward<Ts>(vs)...);
            }
            else if (localities_.size() == 1)
            {
                return stub_base<Component>::create_async(
                    localities_.front(), std::forward<Ts>(vs)...);
            }

            // schedule creation of all objects across given localities
            hpx::future<std::vector<std::uint64_t> > values =
                detail::get_counter_values(
                    hpx::components::unique_component_name<
                        hpx::components::component_factory<
                            typename Component::wrapping_type
                        >
                    >::call(), counter_name_, localities_);

            using hpx::util::placeholders::_1;
            return values.then(hpx::util::bind(
                detail::create_helper<Component>(localities_),
                _1, std::forward<Ts>(vs)...));
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
                hpx::future<std::vector<std::uint64_t> > values =
                    detail::get_counter_values(
                        hpx::components::unique_component_name<
                            hpx::components::component_factory<
                                typename Component::wrapping_type
                            >
                        >::call(), counter_name_, localities_);

                using hpx::util::placeholders::_1;
                return values.then(
                    hpx::util::bind(
                        detail::create_bulk_helper<Component>(localities_),
                        _1, count, std::forward<Ts>(vs)...));
            }

            // handle special cases
            hpx::id_type id =
                localities_.empty() ? hpx::find_here() : localities_.front();

            hpx::future<std::vector<hpx::id_type> > f =
                stub_base<Component>::bulk_create_async(
                    id, count, std::forward<Ts>(vs)...);

            return f.then(hpx::launch::sync,
                [id](hpx::future<std::vector<hpx::id_type> > && f)
                    -> std::vector<bulk_locality_result>
                {
                    std::vector<bulk_locality_result> result;
                    result.emplace_back(id, f.get());
                    return result;
                });
        }

        /// Returns the name of the performance counter associated with this
        /// policy instance.
        std::string const& get_counter_name() const
        {
            return counter_name_;
        }

        /// Returns the number of associated localities for this distribution
        /// policy
        ///
        /// \note This function is part of the creation policy implemented by
        ///       this class
        ///
        std::size_t get_num_localities() const
        {
            return localities_.size();
        }

    protected:
        /// \cond NOINTERNAL
        binpacking_distribution_policy(std::vector<id_type> const& localities,
                char const* perf_counter_name)
          : localities_(localities),
            counter_name_(perf_counter_name)
        {}

        binpacking_distribution_policy(std::vector<id_type> && localities,
                char const* perf_counter_name)
          : localities_(std::move(localities)),
            counter_name_(perf_counter_name)
        {}

        binpacking_distribution_policy(id_type const& locality,
                char const* perf_counter_name)
          : counter_name_(perf_counter_name)
        {
            localities_.push_back(locality);
        }

        friend class hpx::serialization::access;

        template <typename Archive>
        void serialize(Archive& ar, unsigned int const)
        {
            ar & counter_name_ & localities_;
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

    namespace traits
    {
        template <>
        struct is_distribution_policy<components::binpacking_distribution_policy>
          : std::true_type
        {};
    }
}
/// \endcond

#endif
