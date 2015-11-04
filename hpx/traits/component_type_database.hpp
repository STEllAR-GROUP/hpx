//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TRAITS_COMPONENT_TYPE_DATABASE_OCT_27_2011_0420PM)
#define HPX_TRAITS_COMPONENT_TYPE_DATABASE_OCT_27_2011_0420PM

#include <hpx/traits.hpp>

namespace hpx { namespace components
{
    typedef boost::int32_t component_type;
}}

namespace hpx { namespace traits
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Component, typename Enable>
    struct component_type_database
    {
        static components::component_type value;

        HPX_ALWAYS_EXPORT static components::component_type get();
        HPX_ALWAYS_EXPORT static void set(components::component_type);
    };

    template <typename Component, typename Enable>
    components::component_type
    component_type_database<Component, Enable>::value =
        components::component_type(-1); //components::component_invalid;

    template <typename Component, typename Enable>
    struct component_type_database<Component const, Enable>
      : component_type_database<Component, Enable>
    {};
}}

#endif
