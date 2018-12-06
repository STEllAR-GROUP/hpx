//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_RUNTIME_GET_LVA_JUN_22_2008_0451PM)
#define HPX_RUNTIME_GET_LVA_JUN_22_2008_0451PM

#include <hpx/config.hpp>
#include <hpx/runtime/components_fwd.hpp>
#include <hpx/runtime/naming/address.hpp>
#include <hpx/traits/is_component.hpp>

#include <type_traits>

///////////////////////////////////////////////////////////////////////////////
namespace hpx
{
    namespace detail
    {
        template <typename Component>
        struct is_simple_or_fixed_component
            : std::integral_constant<bool,
                  std::is_base_of<traits::detail::component_tag,
                      Component>::value ||
                      std::is_base_of<traits::detail::fixed_component_tag,
                          Component>::value>
        {};
    }

    /// The \a get_lva template is a helper structure allowing to convert a
    /// local virtual address as stored in a local address (returned from
    /// the function \a resolver_client#resolve) to the address of the
    /// component implementing the action.
    ///
    /// The default implementation uses the template argument \a Component
    /// to deduce the type wrapping the component implementing the action. This
    /// is used to get the needed address.
    ///
    /// The specialization for the \a runtime_support component is needed
    /// because this is not wrapped by a separate type as all the other
    /// components.
    ///
    /// \tparam Component  This is the type of the component implementing the
    ///                    action to execute.
    template <typename Component>
    struct get_lva
    {
        template <typename Address>
        static Component*
        call(Address lva, std::false_type)
        {
            typedef typename Component::wrapping_type wrapping_type;
            return reinterpret_cast<wrapping_type*>(lva)->get_checked();
        }

        template <typename Address>
        static Component*
        call(Address lva, std::true_type)
        {
            return reinterpret_cast<Component*>(lva);
        }

        static Component*
        call(naming::address::address_type lva)
        {
            return call(lva, detail::is_simple_or_fixed_component<Component>());
        }
    };

    template <typename Component>
    struct get_lva<Component const>
    {
        template <typename Address>
        static Component const*
        call(Address lva, std::false_type)
        {
            typedef typename std::add_const<
                typename Component::wrapping_type
            >::type wrapping_type;
            return reinterpret_cast<wrapping_type*>(lva)->get_checked();
        }

        template <typename Address>
        static Component const*
        call(Address lva, std::true_type)
        {
            return reinterpret_cast<Component const*>(lva);
        }

        static Component const*
        call(naming::address::address_type lva)
        {
            return call(lva, detail::is_simple_or_fixed_component<Component>());
        }
    };

    // specialization for components::server::runtime_support
    template <>
    struct get_lva<components::server::runtime_support>
    {
        // for server::runtime_support the provided lva is directly usable
        // as the required local address
        static components::server::runtime_support*
        call(naming::address::address_type lva)
        {
            return reinterpret_cast<components::server::runtime_support*>(lva);
        }
    };
    template <>
    struct get_lva<components::server::runtime_support const>
    {
        // for server::runtime_support the provided lva is directly usable
        // as the required local address
        static components::server::runtime_support const*
        call(naming::address::address_type lva)
        {
            return reinterpret_cast<components::server::runtime_support const*>(lva);
        }
    };

    // specialization for components::server::memory
    template <>
    struct get_lva<components::server::memory>
    {
        // for server::memory the provided lva is directly usable as the
        // required local address
        static components::server::memory*
        call(naming::address::address_type lva)
        {
            return reinterpret_cast<components::server::memory*>(lva);
        }
    };
    template <>
    struct get_lva<components::server::memory const>
    {
        // for server::memory the provided lva is directly usable as the
        // required local address
        static components::server::memory const*
        call(naming::address::address_type lva)
        {
            return reinterpret_cast<components::server::memory const*>(lva);
        }
    };
}

#endif

