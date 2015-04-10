//  Copyright (c) 2014-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file colocating_distribution_policy.hpp

#if !defined(HPX_COMPONENTS_colocating_distribution_policy_APR_10_2015_0227PM)
#define HPX_COMPONENTS_colocating_distribution_policy_APR_10_2015_0227PM

#include <hpx/config.hpp>
#include <hpx/traits/is_distribution_policy.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>
#include <hpx/runtime/components/client_base.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/naming/id_type.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/lcos/local/dataflow.hpp>
#include <hpx/util/move.hpp>

#include <algorithm>
#include <vector>
#include <type_traits>

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
        /// \param client  [in] The client side representation of the object with
        ///                which the new instances should be colocated on
        ///
        template <typename Client, typename Stub>
        colocating_distribution_policy operator()(
            client_base<Client, Stub> const& client) const
        {
            return colocating_distribution_policy(client.get_gid());
        }

        /// Create one object on the locality of the object this distribution
        /// policy instance is associated with
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

            if (!id_)
            {
                return stub_base<Component>::create_async(
                    hpx::find_here(), std::forward<Ts>(vs)...);
            }

            return stub_base<Component>::create_colocated_async(
                id_, std::forward<Ts>(vs)...);
        }

        /// Create multiple objects colocated with the object represented
        /// by this policy instance
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
            if (!id_)
            {
                return stub_base<Component>::bulk_create_async(
                    hpx::find_here(), count, std::forward<Ts>(vs)...);
            }

            return stub_base<Component>::bulk_create_colocated_async(
                id_, count, std::forward<Ts>(vs)...);
        }

    protected:
        /// \cond NOINTERNAL
        colocating_distribution_policy(id_type const& id)
          : id_(id)
        {}

        hpx::id_type id_;   // the global address of the object with which the
                            // new objects will be colocated
        /// \endcond
    };

    /// A predefined instance of the default \a distribution_policy. It will
    /// represent the local locality and will place all items to create here.
    static colocating_distribution_policy const colocated;
}}

/// \cond NOINTERNAL
namespace hpx
{
    using hpx::components::colocating_distribution_policy;
    using hpx::components::colocated;

    namespace traits { namespace detail
    {
        template <>
        struct is_distribution_policy<components::colocating_distribution_policy>
          : std::true_type
        {};
    }}
}
/// \endcond

#endif
