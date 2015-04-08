//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file new_colocated.hpp

#if !defined(HPX_RUNTIME_COMPONENTS_NEW_COLOCATED_APR_07_2015_0405PM)
#define HPX_RUNTIME_COMPONENTS_NEW_COLOCATED_APR_07_2015_0405PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>
#include <hpx/runtime/components/distribution_policy.hpp>
#include <hpx/util/move.hpp>
#include <hpx/traits/is_component.hpp>

#include <type_traits>

#if defined(DOXYGEN)
namespace hpx
{
    /// \brief Create a new instance of the given Component type on the
    /// co-located with the specified object.
    ///
    /// This function creates a new instance of the given Component type
    /// co-located with the given object and returns a future object for the
    /// global address which can be used to reference the new component instance.
    ///
    /// \param locality  [in] The global address of an object defining the
    ///                  locality where the new instance should be created on.
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
    ///                 hpx::new_colocated<some_component>(some_id, ...);
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
    new_colocated(id_type const& locality, Ts&&... vs);

    /// \brief Create a new instance of the given Component type on the
    /// co-located with the specified object.
    ///
    /// This function creates a new instance of the given Component type
    /// co-located with the given object and returns a future object for the
    /// global address which can be used to reference the new component instance.
    ///
    /// \param client    [in] The client side representation of an object
    ///                  defining the locality where the new instance should
    ///                  be created on.
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
    ///                 hpx::new_colocated<some_component>(some_client, ...);
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
    template <typename Component, typename Derived, typename Stub, typename ...Ts>
    <unspecified>
    new_colocated(client_base<Derived, Stub> const& client, Ts&&... vs);
}

#else

namespace hpx { namespace components
{
    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        template <typename Component>
        struct new_impl_colocated
        {
            typedef hpx::future<hpx::id_type> type;

            template <typename ...Ts>
            static type call(hpx::id_type const& locality, Ts&&... vs)
            {
                using components::stub_base;
                return stub_base<Component>::create_colocated_async(
                    locality, std::forward<Ts>(vs)...);
            }
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Component, typename ...Ts>
    inline typename std::enable_if<
        traits::is_component<Component>::value,
        typename detail::new_impl_colocated<Component>::type
    >::type
    new_colocated(id_type const& locality, Ts&&... vs)
    {
        return detail::new_impl_colocated<Component>::call(
            locality, std::forward<Ts>(vs)...);
    }

    template <typename Client, typename ...Ts>
    inline typename std::enable_if<
        traits::is_client<Client>::value, Client
    >::type
    new_colocated(id_type const& locality, Ts&&... vs)
    {
        typedef typename Client::server_component_type component_type;
        return Client(detail::new_impl_colocated<component_type>::call(
            locality, std::forward<Ts>(vs)...));
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Component, typename Derived, typename Stub, typename ...Ts>
    inline typename std::enable_if<
        traits::is_component<Component>::value,
        typename detail::new_impl_colocated<Component>::type
    >::type
    new_colocated(client_base<Derived, Stub> const& client, Ts&&... vs)
    {
        return detail::new_impl_colocated<Component>::call(
            client.get_gid(), std::forward<Ts>(vs)...);
    }

    template <typename Client, typename Derived, typename Stub, typename ...Ts>
    inline typename std::enable_if<
        traits::is_client<Client>::value, Client
    >::type
    new_colocated(client_base<Derived, Stub> const& client, Ts&&... vs)
    {
        typedef typename Client::server_component_type component_type;
        return Client(detail::new_impl_colocated<component_type>::call(
            client.get_gid(), std::forward<Ts>(vs)...));
    }
}}

namespace hpx
{
    using hpx::components::new_colocated;
}

#endif // DOXYGEN

#endif
