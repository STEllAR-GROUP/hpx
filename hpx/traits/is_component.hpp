//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TRAITS_IS_COMPONENT_OCT_10_2012_0221PM)
#define HPX_TRAITS_IS_COMPONENT_OCT_10_2012_0221PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/traits.hpp>

#include <boost/mpl/bool.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_base_and_derived.hpp>

namespace hpx { namespace traits
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Component, typename Enable>
    struct is_component : boost::mpl::false_
    {};

    template <typename Component>
    struct is_component<Component const>
      : is_component<Component>
    {};

    ///////////////////////////////////////////////////////////////////////////
    // Simple components are components
    template <typename Component>
    struct is_component<Component,
            typename boost::enable_if<
                boost::is_base_and_derived<
                    components::detail::simple_component_tag, Component
                > >::type>
      : boost::mpl::true_
    {};

    // Fixed components are components
    template <typename Component>
    struct is_component<Component, 
            typename boost::enable_if<
                boost::is_base_and_derived<
                    components::detail::fixed_component_tag, Component
                > >::type>
      : boost::mpl::true_
    {};

    // Managed components are components
    template <typename Component>
    struct is_component<Component,
            typename boost::enable_if<
                boost::is_base_and_derived<
                    components::detail::managed_component_tag, Component
                > >::type>
      : boost::mpl::true_
    {};

    ///////////////////////////////////////////////////////////////////////////
    // And, we have a couple of hand rolled components
    template <>
    struct is_component<components::server::runtime_support>
      : boost::mpl::true_
    {};

    template <>
    struct is_component<components::server::memory>
      : boost::mpl::true_
    {};

    template <>
    struct is_component<components::server::memory_block>
      : boost::mpl::true_
    {};
}}

#endif

