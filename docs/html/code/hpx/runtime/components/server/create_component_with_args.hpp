//  Copyright (c)      2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_PP_IS_ITERATING

#ifndef HPX_RUNTIME_COMPONENTS_SERVER_CREATE_COMPONENT_WITH_ARGS_HPP
#define HPX_RUNTIME_COMPONENTS_SERVER_CREATE_COMPONENT_WITH_ARGS_HPP

#include <hpx/util/decay.hpp>
#include <hpx/util/move.hpp>
#include <hpx/runtime/components/server/create_component.hpp>

#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/remove_const.hpp>
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/enum_params.hpp>
#include <boost/preprocessor/iterate.hpp>
#include <boost/preprocessor/repeat.hpp>

namespace hpx { namespace components { namespace server
{

#define HPX_RUNTIME_SUPPORT_CTOR_M0(Z, N, D)                                  \
        BOOST_PP_CAT(a, N)(boost::forward<BOOST_PP_CAT(T, N)>                 \
            (BOOST_PP_CAT(t, N)))                                             \
    /**/
#define HPX_RUNTIME_SUPPORT_CTOR_M1(Z, N, D)                                  \
        typename util::decay<BOOST_PP_CAT(A, N)>::type                        \
            BOOST_PP_CAT(a, N);                                               \
    /**/
#define HPX_RUNTIME_SUPPORT_CTOR_M2(Z, N, D)                                  \
        BOOST_PP_CAT(a, N)(other. BOOST_PP_CAT(a, N))                         \
    /**/
#define HPX_RUNTIME_SUPPORT_CTOR_M3(Z, N, D)                                  \
        BOOST_PP_CAT(a, N)(boost::move(other. BOOST_PP_CAT(a, N)))            \
    /**/
#define HPX_RUNTIME_SUPPORT_CTOR_M4(Z, N, D)                                  \
        BOOST_PP_CAT(a, N) = other. BOOST_PP_CAT(a, N);                       \
    /**/
#define HPX_RUNTIME_SUPPORT_CTOR_M5(Z, N, D)                                  \
        BOOST_PP_CAT(a, N) = boost::move(other. BOOST_PP_CAT(a, N));          \
    /**/

#if !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)
#  include <hpx/runtime/components/server/preprocessed/create_component_with_args.hpp>
#else

#if defined(__WAVE__) && defined(HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(preserve: 1, line: 0, output: "preprocessed/create_component_with_args_" HPX_LIMIT_STR ".hpp")
#endif

#define BOOST_PP_ITERATION_PARAMS_1                                           \
    (                                                                         \
        3                                                                     \
      , (                                                                     \
            1                                                                 \
          , HPX_ACTION_ARGUMENT_LIMIT                                         \
          , "hpx/runtime/components/server/create_component_with_args.hpp"    \
        )                                                                     \
    )                                                                         \
    /**/

#include BOOST_PP_ITERATE()

#if defined(__WAVE__) && defined (HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(output: null)
#endif

#endif // !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)

}}}

#undef HPX_RUNTIME_SUPPORT_CTOR_M0
#undef HPX_RUNTIME_SUPPORT_CTOR_M1
#undef HPX_RUNTIME_SUPPORT_CTOR_M2
#undef HPX_RUNTIME_SUPPORT_CTOR_M3
#undef HPX_RUNTIME_SUPPORT_CTOR_M4
#undef HPX_RUNTIME_SUPPORT_CTOR_M5

#endif

#else

#define N BOOST_PP_ITERATION()

        template <typename Component, BOOST_PP_ENUM_PARAMS(N, typename A)>
        struct BOOST_PP_CAT(component_constructor_functor, N)
        {
            typedef void result_type;

            BOOST_PP_CAT(component_constructor_functor, N)(
                BOOST_PP_CAT(component_constructor_functor, N) const & other)
              : BOOST_PP_ENUM(N, HPX_RUNTIME_SUPPORT_CTOR_M2, _)
            {}

            BOOST_PP_CAT(component_constructor_functor, N)(
                BOOST_RV_REF(BOOST_PP_CAT(component_constructor_functor, N)) other)
              : BOOST_PP_ENUM(N, HPX_RUNTIME_SUPPORT_CTOR_M3, _)
            {}

            BOOST_PP_CAT(component_constructor_functor, N) & operator=(
                BOOST_COPY_ASSIGN_REF(BOOST_PP_CAT(component_constructor_functor, N)) other)
            {
                BOOST_PP_REPEAT(N, HPX_RUNTIME_SUPPORT_CTOR_M4, _)
                return *this;
            }

            BOOST_PP_CAT(component_constructor_functor, N) & operator=(
                BOOST_RV_REF(BOOST_PP_CAT(component_constructor_functor, N)) other)
            {
                BOOST_PP_REPEAT(N, HPX_RUNTIME_SUPPORT_CTOR_M5, _)
                return *this;
            }

            template <BOOST_PP_ENUM_PARAMS(N, typename T)>
#if N == 1
            explicit
#endif
            BOOST_PP_CAT(component_constructor_functor, N)(
                HPX_ENUM_FWD_ARGS(N, T, t)
#if N == 1
              , typename ::boost::disable_if<
                    typename boost::is_same<
                        BOOST_PP_CAT(component_constructor_functor, N)
                      , typename util::decay<T0>::type
                    >::type
                >::type * dummy = 0
#endif
            )
              : BOOST_PP_ENUM(N, HPX_RUNTIME_SUPPORT_CTOR_M0, _)
            {}

            result_type operator()(void* p)
            {
                new (p) typename Component::derived_type(HPX_ENUM_MOVE_ARGS(N, a));
            }
            BOOST_PP_REPEAT(N, HPX_RUNTIME_SUPPORT_CTOR_M1, _)

        private:
            BOOST_COPYABLE_AND_MOVABLE(BOOST_PP_CAT(component_constructor_functor, N))
        };

        template <typename Component, BOOST_PP_ENUM_PARAMS(N, typename A)>
        naming::gid_type create_with_args(HPX_ENUM_FWD_ARGS(N, A, a))
        {
            return server::create<Component>(
                BOOST_PP_CAT(component_constructor_functor, N)<
                    Component, BOOST_PP_ENUM_PARAMS(N, A)>(
                        HPX_ENUM_FORWARD_ARGS(N, A, a))
            );
        }
#undef N

#endif

