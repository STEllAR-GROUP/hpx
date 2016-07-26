//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file new.hpp

#if !defined(HPX_RUNTIME_COMPONENTS_NEW_OCT_10_2012_1256PM)
#define HPX_RUNTIME_COMPONENTS_NEW_OCT_10_2012_1256PM

#include <hpx/config.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/runtime/components/client_base.hpp>
#include <hpx/runtime/components/default_distribution_policy.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/traits/is_client.hpp>
#include <hpx/traits/is_component.hpp>
#include <hpx/traits/is_distribution_policy.hpp>

#include <algorithm>
#include <type_traits>
#include <vector>

#if defined(DOXYGEN)
namespace hpx
{
    /// \brief Create one or more new instances of the given Component type
    /// on the specified locality.
    ///
    /// This function creates one or more new instances of the given Component
    /// type on the specified locality and returns a future object for the
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
    ///          * If the explicit template argument \a Component represents a
    ///          component type (<code>traits::is_component<Component>::value</code>
    ///          evaluates to true), the function will return an \a hpx::future
    ///          object instance which can be used to retrieve the global
    ///          address of the newly created component.
    ///          * If the explicit template argument \a Component represents a
    ///          client side object (<code>traits::is_client<Component>::value</code>
    ///          evaluates to true), the function will return a new instance
    ///          of that type which can be used to refer to the newly created
    ///          component instance.
    ///
    template <typename Component, typename ...Ts>
    <unspecified>
    new_(id_type const& locality, Ts&&... vs);

    /// \brief Create multiple new instances of the given Component type on the
    /// specified locality.
    ///
    /// This function creates multiple new instances of the given Component type
    /// on the specified locality and returns a future object for the
    /// global address which can be used to reference the new component
    /// instance.
    ///
    /// \param locality  [in] The global address of the locality where the
    ///                  new instance should be created on.
    /// \param count     [in] The number of component instances to create
    /// \param vs        [in] Any number of arbitrary arguments (passed by
    ///                  value, by const reference or by rvalue reference)
    ///                  which will be forwarded to the constructor of
    ///                  the created component instance.
    ///
    /// \note    This function requires to specify an explicit template
    ///          argument which will define what type of component(s) to
    ///          create, for instance:
    ///          \code
    ///              hpx::future<std::vector<hpx::id_type> > f =
    ///                 hpx::new_<some_component[]>(hpx::find_here(), 10, ...);
    ///              hpx::id_type id = f.get();
    ///          \endcode
    ///
    /// \returns The function returns different types depending on its use:\n
    ///          * If the explicit template argument \a Component
    ///          represents an array of a component type (i.e. \a Component[],
    ///          where <code>traits::is_component<Component>::value</code>
    ///          evaluates to true), the function will return an \a hpx::future
    ///          object instance which holds a std::vector<hpx::id_type>, where
    ///          eahc of the items in this vector is a global address of one
    ///          of the newly created components.
    ///          * If the explicit template argument \a Component
    ///          represents an array of a client side object type (i.e. \a Component[],
    ///          where <code>traits::is_client<Component>::value</code>
    ///          evaluates to true), the function will return an \a hpx::future
    ///          object instance which holds a std::vector<hpx::id_type>, where
    ///          eahc of the items in this vector is a client side instance of
    ///          the given type, each representing one of the newly created
    ///          components.
    ///
    template <typename Component, typename ...Ts>
    <unspecified>
    new_(id_type const& locality, std::size_t count, Ts&&... vs);

    /// \brief Create one or more new instances of the given Component type
    /// based on the given distribution policy.
    ///
    /// This function creates one or more new instances of the given Component
    /// type on the localities defined by the given distribution policy and
    /// returns a future object for global address which can be used to reference
    /// the new component instance(s).
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
    ///          * If the explicit template argument \a Component represents a
    ///          component type (<code>traits::is_component<Component>::value</code>
    ///          evaluates to true), the function will return an \a hpx::future
    ///          object instance which can be used to retrieve the global
    ///          address of the newly created component.
    ///          * If the explicit template argument \a Component represents a
    ///          client side object (<code>traits::is_client<Component>::value</code>
    ///          evaluates to true), the function will return a new instance
    ///          of that type which can be used to refer to the newly created
    ///          component instance.
    ///
    template <typename Component, typename DistPolicy, typename ...Ts>
    <unspecified>
    new_(DistPolicy const& policy, Ts&&... vs);

    /// \brief Create multiple new instances of the given Component type on the
    /// localities as defined by the given distribution policy.
    ///
    /// This function creates multiple new instances of the given Component type
    /// on the localities defined by the given distribution policy and returns
    /// a future object for the global address which can be used to reference
    /// the new component instance.
    ///
    /// \param policy    [in] The distribution policy used to decide where to
    ///                  place the newly created.
    /// \param count     [in] The number of component instances to create
    /// \param vs        [in] Any number of arbitrary arguments (passed by
    ///                  value, by const reference or by rvalue reference)
    ///                  which will be forwarded to the constructor of
    ///                  the created component instance.
    ///
    /// \note    This function requires to specify an explicit template
    ///          argument which will define what type of component(s) to
    ///          create, for instance:
    ///          \code
    ///              hpx::future<std::vector<hpx::id_type> > f =
    ///                 hpx::new_<some_component[]>(hpx::default_layout, 10, ...);
    ///              hpx::id_type id = f.get();
    ///          \endcode
    ///
    /// \returns The function returns different types depending on its use:\n
    ///          * If the explicit template argument \a Component
    ///          represents an array of a component type (i.e. \a Component[],
    ///          where <code>traits::is_component<Component>::value</code>
    ///          evaluates to true), the function will return an \a hpx::future
    ///          object instance which holds a std::vector<hpx::id_type>, where
    ///          eahc of the items in this vector is a global address of one
    ///          of the newly created components.\n
    ///          * If the explicit template argument \a Component
    ///          represents an array of a client side object type (i.e. \a Component[],
    ///          where <code>traits::is_client<Component>::value</code>
    ///          evaluates to true), the function will return an \a hpx::future
    ///          object instance which holds a std::vector<hpx::id_type>, where
    ///          eahc of the items in this vector is a client side instance of
    ///          the given type, each representing one of the newly created
    ///          components.
    ///
    template <typename Component, typename DistPolicy, typename ...Ts>
    <unspecified>
    new_(DistPolicy const& policy, std::size_t count, Ts&&... vs);
}

#else

namespace hpx { namespace components
{
    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        // create a single instance of a component
        template <typename Component>
        struct new_component
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
                return policy.template create<Component>(std::forward<Ts>(vs)...);
            }
        };

        // create multiple component instances
        template <typename Component>
        struct new_component<Component[]>
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
            static type call(DistPolicy const& policy, std::size_t count,
                Ts&&... vs)
            {
                typedef typename DistPolicy::bulk_locality_result
                    bulk_locality_result;

                hpx::future<std::vector<bulk_locality_result> > f =
                    policy.template bulk_create<Component>(count,
                        std::forward<Ts>(vs)...);

                return f.then(launch::sync,
                    [count](hpx::future<std::vector<bulk_locality_result> >
                        && f) -> std::vector<hpx::id_type>
                    {
                        std::vector<hpx::id_type> result;
                        result.reserve(count);

                        for (bulk_locality_result& r: f.get())
                        {
                            std::move(r.second.begin(), r.second.end(),
                                std::back_inserter(result));
                        }
                        return result;
                    });
            }
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Component, typename ...Ts>
    inline typename std::enable_if<
        traits::is_component_or_component_array<Component>::value,
        detail::new_component<Component>
    >::type::type
    new_(id_type const& locality, Ts&&... vs)
    {
        return detail::new_component<Component>::call(
            locality, std::forward<Ts>(vs)...);
    }

    template <typename Component, typename DistPolicy, typename ...Ts>
    inline typename std::enable_if<
        traits::is_component_or_component_array<Component>::value &&
            traits::is_distribution_policy<DistPolicy>::value,
        detail::new_component<Component>
    >::type::type
    new_(DistPolicy const& policy, Ts&&... vs)
    {
        return detail::new_component<Component>::call(
            policy, std::forward<Ts>(vs)...);
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        // create a single instance of a component
        template <typename Client>
        struct new_client
        {
            typedef Client type;
            typedef typename Client::server_component_type component_type;

            template <typename ...Ts>
            static type call(hpx::id_type const& locality, Ts&&... vs)
            {
                using components::stub_base;
                return make_client<Client>(
                    stub_base<component_type>::create_async(
                        locality, std::forward<Ts>(vs)...));
            }

            template <typename DistPolicy, typename ...Ts>
            static type call(DistPolicy const& policy, Ts&&... vs)
            {
                return make_client<Client>(
                    policy.template create<component_type>(
                        std::forward<Ts>(vs)...));
            }
        };

        // create multiple client instances
        template <typename Client>
        struct new_client<Client[]>
        {
            typedef hpx::future<std::vector<Client> > type;
            typedef typename Client::server_component_type component_type;

            template <typename ...Ts>
            static type call(Ts&&... vs)
            {
                return new_component<component_type[]>::call(
                        std::forward<Ts>(vs)...
                    )
                    .then(
                        [](hpx::future<std::vector<hpx::id_type> > && v)
                            -> std::vector<Client>
                        {
                            return make_clients<Client>(v.get());
                        }
                    );
            }
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Client, typename ...Ts>
    inline typename std::enable_if<
        traits::is_client_or_client_array<Client>::value,
        detail::new_client<Client>
    >::type::type
    new_(id_type const& locality, Ts&&... vs)
    {
        return detail::new_client<Client>::call(
            locality, std::forward<Ts>(vs)...);
    }

    template <typename Client, typename DistPolicy, typename ...Ts>
    inline typename std::enable_if<
        traits::is_client_or_client_array<Client>::value &&
            traits::is_distribution_policy<DistPolicy>::value,
        detail::new_client<Client>
    >::type::type
    new_(DistPolicy const& policy, Ts&&... vs)
    {
        return detail::new_client<Client>::call(
            policy, std::forward<Ts>(vs)...);
    }
}}

namespace hpx
{
    using hpx::components::new_;
}

#endif

#endif // HPX_NEW_OCT_10_2012_1256PM
