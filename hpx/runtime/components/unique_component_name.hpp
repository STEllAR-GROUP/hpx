//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_A7F46A4F_9AF9_4909_B0D8_5304FEFC5649)
#define HPX_A7F46A4F_9AF9_4909_B0D8_5304FEFC5649

#include <boost/preprocessor/stringize.hpp>

namespace hpx { namespace components
{
    struct base_name;
    struct derived_name;

    template <typename ComponentType, typename Type = derived_name>
    struct unique_component_name
    {
        static_assert(sizeof(ComponentType) == 0, "component name is not defined");
    };
}}

#define HPX_DEF_UNIQUE_COMPONENT_NAME(ComponentType, name)                    \
    namespace hpx { namespace components                                      \
    {                                                                         \
        template <>                                                           \
        struct unique_component_name<ComponentType >                          \
        {                                                                     \
            typedef char const* type;                                         \
                                                                              \
            static type call (void)                                           \
            {                                                                 \
                return BOOST_PP_STRINGIZE(name);                              \
            }                                                                 \
        };                                                                    \
    }}                                                                        \
    /***/

#define HPX_DEF_UNIQUE_DERIVED_COMPONENT_NAME(ComponentType, name, basename)  \
    namespace hpx { namespace components                                      \
    {                                                                         \
        template <>                                                           \
        struct unique_component_name<ComponentType >                          \
        {                                                                     \
            typedef char const* type;                                         \
                                                                              \
            static type call (void)                                           \
            {                                                                 \
                return BOOST_PP_STRINGIZE(name);                              \
            }                                                                 \
        };                                                                    \
                                                                              \
        template <>                                                           \
        struct unique_component_name<ComponentType, base_name >               \
        {                                                                     \
            typedef char const* type;                                         \
                                                                              \
            static type call (void)                                           \
            {                                                                 \
                return basename;                                              \
            }                                                                 \
        };                                                                    \
    }}                                                                        \
    /***/

#endif // HPX_A7F46A4F_9AF9_4909_B0D8_5304FEFC5649

