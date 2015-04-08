//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file new.hpp

#if !defined(HPX_RUNTIME_COMPONENTS_NEW_OCT_10_2012_1256PM)
#define HPX_RUNTIME_COMPONENTS_NEW_OCT_10_2012_1256PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/traits/is_distribution_policy.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>
#include <hpx/runtime/components/distribution_policy.hpp>
#include <hpx/util/move.hpp>
#include <hpx/traits/is_component.hpp>

#include <type_traits>
#include <vector>

#if defined(DOXYGEN)
namespace hpx
{
    /// \brief Create a new instance of the given Component type on the
    /// specified locality.
    ///
    /// This function creates a new instance of the given Component type
    /// on the specified locality and returns a future object for the
    /// global address which can be used to reference the new component
    /// instance.
    ///
    /// \param locality  [in] The global address of the locality where the
    ///                  new instance should be created on.
    /// \param vs        [in] Any number of arbitrary arguments (passed by
    ///                  value, by const reference or by rvalue reference)
    ///                  which will be forwarded to the constructor of
    ///                  the created component instance.
    ///
    /// \note    This function requires to specify an explicit template
    ///          argument which will define what type of component(s) to
    ///          create, for instance:
    ///          \code
    ///              hpx::future<hpx::id_type> f =
    ///                 hpx::new_<some_component>(hpx::find_here(), ...);
    ///              hpx::id_type id = f.get();
    ///          \endcode
    ///
    /// \returns The function returns different types depending on its use:\n
    ///          If the explicit template argument \a Component represents a
    ///          component type (traits::is_component<Component>::value
    ///          evaluates to true), the function will return an \a hpx::future
    ///          object instance which can be used to retrieve the global
    ///          address of the newly created component.\n
    ///          If the explicit template argument \a Component represents a
    ///          client side object (traits::is_client<Component>::value
    ///          evaluates to true), the function will return a new instance
    ///          of that type which can be used to refer to the newly created
    ///          component instance.
    ///
    template <typename Component, typename ...Ts>
    <unspecified>
    new_(id_type const& locality, Ts&&... vs);

    /// \brief Create a new instance of the given Component type on the
    /// specified locality.
    ///
    /// This function creates a new instance of the given Component type
    /// on the specified locality and returns a future object for the
    /// global address which can be used to reference the new component
    /// instance.
    ///
    /// \param policy    [in] The distribution policy used to decide where to
    ///                  place the newly created.
    /// \param vs        [in] Any number of arbitrary arguments (passed by
    ///                  value, by const reference or by rvalue reference)
    ///                  which will be forwarded to the constructor of
    ///                  the created component instance.
    ///
    /// \note    This function requires to specify an explicit template
    ///          argument which will define what type of component(s) to
    ///          create, for instance:
    ///          \code
    ///              hpx::future<hpx::id_type> f =
    ///                 hpx::new_<some_component>(hpx::default_layout, ...);
    ///              hpx::id_type id = f.get();
    ///          \endcode
    ///
    /// \returns The function returns different types depending on its use:\n
    ///          If the explicit template argument \a Component represents a
    ///          component type (traits::is_component<Component>::value
    ///          evaluates to true), the function will return an \a hpx::future
    ///          object instance which can be used to retrieve the global
    ///          address of the newly created component.\n
    ///          If the explicit template argument \a Component represents a
    ///          client side object (traits::is_client<Component>::value
    ///          evaluates to true), the function will return a new instance
    ///          of that type which can be used to refer to the newly created
    ///          component instance.
    ///
    template <typename Component, typename DistPolicy, typename ...Ts>
    <unspecified>
    new_(DistPolicy const& policy, Ts&&... vs);
}

#else

namespace hpx { namespace components
{
    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        // create a single instance of a component
        template <typename Component>
        struct new_impl
        {
            typedef hpx::future<hpx::id_type> type;

            template <typename ...Ts>
            static type call(hpx::id_type const& locality, Ts&&... vs)
            {
                using components::stub_base;
                return stub_base<Component>::create_async(
                    locality, std::forward<Ts>(vs)...);
            }

            template <typename DistPolicy, typename ...Ts>
            static type call(DistPolicy const& policy, Ts&&... vs)
            {
                using components::stub_base;

                for (hpx::id_type const& loc: policy.get_localities())
                {
                    if (policy.get_num_items(1, loc) != 0)
                    {
                        return stub_base<Component>::create_async(
                            loc, std::forward<Ts>(vs)...);
                    }
                }

                return stub_base<Component>::create_async(
                    hpx::find_here(), std::forward<Ts>(vs)...);
            }
        };

        // create several component instances
        template <typename Component>
        struct new_impl<Component[]>
        {
            typedef hpx::future<std::vector<hpx::id_type> > type;

            template <typename ...Ts>
            static type call(hpx::id_type const& locality, std::size_t count,
                Ts&&... vs)
            {
                using components::stub_base;
                return stub_base<Component>::bulk_create_async(
                    locality, count, std::forward<Ts>(vs)...);
            }

            template <typename DistPolicy, typename ...Ts>
            static type call(DistPolicy const& policy, Ts&&... vs)
            {
                using components::stub_base;

//                 for (hpx::id_type const& loc: policy.get_localities())
//                 {
//                     if (policy.get_num_items(1, loc) != 0)
//                     {
//                         return stub_base<Component>::create_async(
//                             loc, std::forward<Ts>(vs)...);
//                     }
//                 }

                return stub_base<Component>::bulk_create_async(
                    hpx::find_here(), count, std::forward<Ts>(vs)...);
            }
        };

        // create a given number of component instances
        template <typename Component, std::size_t N>
        struct new_impl<Component[N]>
        {
            typedef hpx::future<std::vector<hpx::id_type> > type;

            template <typename ...Ts>
            static type call(hpx::id_type const& locality, Ts&&... vs)
            {
                using components::stub_base;
                return stub_base<Component>::bulk_create_async(
                    locality, N, std::forward<Ts>(vs)...);
            }

            template <typename DistPolicy, typename ...Ts>
            static type call(DistPolicy const& policy, Ts&&... vs)
            {
                using components::stub_base;

//                 for (hpx::id_type const& loc: policy.get_localities())
//                 {
//                     if (policy.get_num_items(1, loc) != 0)
//                     {
//                         return stub_base<Component>::create_async(
//                             loc, std::forward<Ts>(vs)...);
//                     }
//                 }

                return stub_base<Component>::bulk_create_async(
                    hpx::find_here(), N, std::forward<Ts>(vs)...);
            }
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Component, typename ...Ts>
    inline typename std::enable_if<
        traits::is_component_or_component_array<Component>::value,
        typename detail::new_impl<Component>::type
    >::type
    new_(id_type const& locality, Ts&&... vs)
    {
        return detail::new_impl<Component>::call(
            locality, std::forward<Ts>(vs)...);
    }

    template <typename Component, typename DistPolicy, typename ...Ts>
    inline typename std::enable_if<
        traits::is_component_or_component_array<Component>::value &&
            traits::is_distribution_policy<DistPolicy>::value,
        typename detail::new_impl<Component>::type
    >::type
    new_(DistPolicy const& policy, Ts&&... vs)
    {
        return detail::new_impl<Component>::call(
            policy, std::forward<Ts>(vs)...);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Client, typename ...Ts>
    inline typename std::enable_if<
        traits::is_client_or_client_array<Client>::value, Client
    >::type
    new_(id_type const& locality, Ts&&... vs)
    {
        typedef typename Client::server_component_type component_type;
        return Client(detail::new_impl<component_type>::call(
            locality, std::forward<Ts>(vs)...));
    }

    template <typename Client, typename DistPolicy, typename ...Ts>
    inline typename std::enable_if<
        traits::is_client_or_client_array<Client>::value &&
            traits::is_distribution_policy<DistPolicy>::value,
        Client
    >::type
    new_(DistPolicy const& policy, Ts&&... vs)
    {
        typedef typename Client::server_component_type component_type;
        return Client(detail::new_impl<component_type>::call(
            policy, std::forward<Ts>(vs)...));
    }
}}

namespace hpx
{
    using hpx::components::new_;
}

#endif

#endif // HPX_NEW_OCT_10_2012_1256PM
