//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//  Copyright (c)      2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_PP_IS_ITERATING

#if !defined(HPX_RUNTIME_SUPPORT_IMPLEMENTATIONS_GCC44_JUN_02_2008_1145AM)
#define HPX_RUNTIME_SUPPORT_IMPLEMENTATIONS_GCC44_JUN_02_2008_1145AM

#if !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)
#  include <hpx/runtime/components/server/gcc44/preprocessed/runtime_support_implementations.hpp>
#else

#if defined(__WAVE__) && defined(HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(preserve: 1, line: 0, output: "preprocessed/runtime_support_implementations_" HPX_LIMIT_STR ".hpp")
#endif

#define BOOST_PP_ITERATION_PARAMS_1                                             \
    (3, (0, HPX_ACTION_ARGUMENT_LIMIT,                                          \
     "hpx/runtime/components/server/gcc44/runtime_support_implementations.hpp"))\
/**/

#include BOOST_PP_ITERATE()

#if defined(__WAVE__) && defined (HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(output: null)
#endif

#endif // !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)

#endif  // HPX_RUNTIME_SUPPORT_IMPLEMENTATIONS_GCC44_JUN_02_2008_1145AM

#else   // !BOOST_PP_IS_ITERATING

#define N BOOST_PP_ITERATION()

namespace hpx { namespace components { namespace server
{
#if N > 0
    template <typename Component, BOOST_PP_ENUM_PARAMS(N, typename A)>
    naming::gid_type runtime_support::BOOST_PP_CAT(create_component, N)(
        BOOST_PP_ENUM_BINARY_PARAMS(N, A, a))
    {
        components::component_type const type =
            components::get_component_type<
                typename Component::wrapped_type>();

        component_map_mutex_type::scoped_lock l(cm_mtx_);
        component_map_type::const_iterator it = components_.find(type);
        if (it == components_.end()) {
            hpx::util::osstream strm;
            strm << "attempt to create component instance of "
                << "invalid/unknown type: "
                << components::get_component_type_name(type)
                << " (component not found in map)";
            HPX_THROW_EXCEPTION(hpx::bad_component_type,
                "runtime_support::create_component",
                hpx::util::osstream_get_string(strm));
            return naming::invalid_gid;
        }

        if (!(*it).second.first) {
            hpx::util::osstream strm;
            strm << "attempt to create component instance of "
                << "invalid/unknown type: "
                << components::get_component_type_name(type)
                << " (map entry is NULL)";
            HPX_THROW_EXCEPTION(hpx::bad_component_type,
                "runtime_support::create_component",
                hpx::util::osstream_get_string(strm));
            return naming::invalid_gid;
        }

        naming::gid_type id;
        boost::shared_ptr<component_factory_base> factory((*it).second.first);
        {
            util::scoped_unlock<component_map_mutex_type::scoped_lock> ul(l);
            id = factory->create_with_args(
                BOOST_PP_CAT(component_constructor_functor, N)<
                    typename Component::wrapping_type,
                    BOOST_PP_ENUM_PARAMS(N, A)>(
                        HPX_ENUM_MOVE_IF_NO_REF_ARGS(N, A, a)));
        }
        LRT_(info) << "successfully created component " << id
            << " of type: " << components::get_component_type_name(type);

        return id;
    }
#endif

    ///////////////////////////////////////////////////////////////////////////
    // Actions used to create components with constructors of various arities.
    template <typename Component
      BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename A)>
    struct BOOST_PP_CAT(create_component_action, N)
      : BOOST_PP_CAT( ::hpx::actions::result_action, N)<
            runtime_support
          , naming::gid_type
          BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, A)
          , &runtime_support::BOOST_PP_CAT(create_component, N)<
                Component
              BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, A)>
          , BOOST_PP_CAT(create_component_action, N)<
                Component BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, A)> >
    {};

    template <typename Component
      BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename A)>
    struct BOOST_PP_CAT(create_component_direct_action, N)
      : BOOST_PP_CAT( ::hpx::actions::direct_result_action, N)<
            runtime_support
          , naming::gid_type
          BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, A)
          , &runtime_support::BOOST_PP_CAT(create_component, N)<
                Component BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, A)>
          , BOOST_PP_CAT(create_component_direct_action, N)<
                Component BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, A)> >
    {};
}}}

#undef N

#endif  // !BOOST_PP_IS_ITERATING
