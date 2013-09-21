//  Copyright (c) 2011 Thomas Heller
//  Copyright (c) 2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !BOOST_PP_IS_ITERATING

#ifndef HPX_FUNCTION_DETAIL_VTABLE_HPP
#define HPX_FUNCTION_DETAIL_VTABLE_HPP

#include <hpx/config/forceinline.hpp>
#include <hpx/util/invoke.hpp>
#include <hpx/util/move.hpp>
#include <hpx/util/detail/add_rvalue_reference.hpp>

#include <boost/preprocessor/iteration/iterate.hpp>
#include <boost/preprocessor/punctuation/comma_if.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_trailing.hpp>
#include <boost/preprocessor/repetition/enum_trailing_params.hpp>

#include <typeinfo>

namespace hpx { namespace util { namespace detail {

    template <>
    struct vtable<true>
    {
        template <typename Functor>
        struct type_base
        {
            enum { empty = false };

            static std::type_info const& get_type()
            {
                return typeid(Functor);
            }

            static Functor & construct(void ** f)
            {
                new (f) Functor;
                return *reinterpret_cast<Functor *>(f);
            }

            static Functor & get(void **f)
            {
                return *reinterpret_cast<Functor *>(f);
            }

            static Functor const & get(void *const*f)
            {
                return *reinterpret_cast<Functor const *>(f);
            }

            static void static_delete(void ** f)
            {
                reinterpret_cast<Functor*>(f)->~Functor();
            }

            static void destruct(void ** f)
            {
                reinterpret_cast<Functor*>(f)->~Functor();
            }

            static void clone(void *const* src, void ** dest)
            {
                new (dest) Functor(*reinterpret_cast<Functor const*>(src));
            }

            static void copy(void *const* f, void ** dest)
            {
                reinterpret_cast<Functor*>(dest)->~Functor();
                *reinterpret_cast<Functor*>(dest) =
                    *reinterpret_cast<Functor const *>(f);
            }
        };

        template <typename Functor, typename Sig, typename IArchive, typename OArchive>
        struct type;

#       define BOOST_UTIL_DETAIL_VTABLE_ADD_RVALUE_REF(Z, N, D)                 \
        typename util::detail::add_rvalue_reference<BOOST_PP_CAT(D, N)>::type   \
        BOOST_PP_CAT(a, N)                                                      \
        /**/

#if !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)
#  include <hpx/util/detail/preprocessed/vtable.hpp>
#else

#if defined(__WAVE__) && defined(HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(preserve: 1, line: 0, output: "preprocessed/vtable_" HPX_LIMIT_STR ".hpp")
#endif

#define BOOST_PP_ITERATION_PARAMS_1                                             \
    (                                                                           \
        4                                                                       \
      , (                                                                       \
            0                                                                   \
          , HPX_FUNCTION_ARGUMENT_LIMIT                                         \
          , <hpx/util/detail/vtable.hpp>                                        \
          , 1                                                                   \
        )                                                                       \
    )                                                                           \
/**/
#include BOOST_PP_ITERATE()

#if defined(__WAVE__) && defined (HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(output: null)
#endif

#endif // !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)

    };

    template <>
    struct vtable<false>
    {
        template <typename Functor>
        struct type_base
        {
            enum { empty = false };

            static std::type_info const & get_type()
            {
                return typeid(Functor);
            }

            static Functor & construct(void ** f)
            {
                *f = new Functor;
                return **reinterpret_cast<Functor **>(f);
            }

            static Functor & get(void **f)
            {
                return **reinterpret_cast<Functor **>(f);
            }

            static Functor const & get(void *const*f)
            {
                return **reinterpret_cast<Functor *const *>(f);
            }

            static void static_delete(void ** f)
            {
                delete (*reinterpret_cast<Functor **>(f));
            }

            static void destruct(void ** f)
            {
                (*reinterpret_cast<Functor**>(f))->~Functor();
            }

            static void clone(void *const* src, void ** dest)
            {
                *dest = new Functor(**reinterpret_cast<Functor *const*>(src));
            }

            static void copy(void *const* f, void ** dest)
            {
                (*reinterpret_cast<Functor**>(dest))->~Functor();
                **reinterpret_cast<Functor**>(dest) =
                    **reinterpret_cast<Functor * const *>(f);
            }
        };

        template <typename Functor, typename Sig, typename IArchive, typename OArchive>
        struct type;

#if !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)
#  include <hpx/util/detail/preprocessed/vtable2.hpp>
#else

#if defined(__WAVE__) && defined(HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(preserve: 1, line: 0, output: "preprocessed/vtable2_" HPX_LIMIT_STR ".hpp")
#endif

#define BOOST_PP_ITERATION_PARAMS_1                                             \
    (                                                                           \
        4                                                                       \
      , (                                                                       \
            0                                                                   \
          , HPX_FUNCTION_ARGUMENT_LIMIT                                         \
          , <hpx/util/detail/vtable.hpp>                                        \
          , 2                                                                   \
        )                                                                       \
    )                                                                           \
/**/
#include BOOST_PP_ITERATE()

#if defined(__WAVE__) && defined (HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(output: null)
#endif

#endif // !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)

#       undef BOOST_UTIL_DETAIL_VTABLE_ADD_RVALUE_REF
    };
}}}

#endif

#else

#define N BOOST_PP_ITERATION()

#if BOOST_PP_ITERATION_FLAGS() == 1

        template <
            typename Functor
          , typename R
          BOOST_PP_ENUM_TRAILING_PARAMS(N, typename A)
          , typename IArchive
          , typename OArchive
        >
        struct type<
            Functor
          , R(BOOST_PP_ENUM_PARAMS(N, A))
          , IArchive
          , OArchive
        >
            : type_base<Functor>
        {
            static vtable_ptr_base<
                R(BOOST_PP_ENUM_PARAMS(N, A))
              , IArchive
              , OArchive
            > *get_ptr()
            {
                return
                    get_table<
                        Functor
                      , R(BOOST_PP_ENUM_PARAMS(N, A))
                    >::template get<IArchive, OArchive>();
            }

            BOOST_FORCEINLINE static R
            invoke(void ** f
                BOOST_PP_ENUM_TRAILING(N, BOOST_UTIL_DETAIL_VTABLE_ADD_RVALUE_REF, A))
            {
                return util::invoke_r<R>((*reinterpret_cast<Functor*>(f))
                    BOOST_PP_COMMA_IF(N) HPX_ENUM_FORWARD_ARGS(N, A, a));
            }
        };

#endif

#if BOOST_PP_ITERATION_FLAGS() == 2

        template <
            typename Functor
          , typename R
          BOOST_PP_ENUM_TRAILING_PARAMS(N, typename A)
          , typename IArchive
          , typename OArchive
        >
        struct type<
            Functor
          , R(BOOST_PP_ENUM_PARAMS(N, A))
          , IArchive
          , OArchive
        >
            : type_base<Functor>
        {
            static vtable_ptr_base<
                R(BOOST_PP_ENUM_PARAMS(N, A))
              , IArchive
              , OArchive
            > *get_ptr()
            {
                return
                    get_table<
                        Functor
                      , R(BOOST_PP_ENUM_PARAMS(N, A))
                    >::template get<IArchive, OArchive>();
            }

            BOOST_FORCEINLINE static R
            invoke(void ** f
                BOOST_PP_ENUM_TRAILING(N, BOOST_UTIL_DETAIL_VTABLE_ADD_RVALUE_REF, A))
            {
                return util::invoke_r<R>((**reinterpret_cast<Functor**>(f))
                    BOOST_PP_COMMA_IF(N) HPX_ENUM_FORWARD_ARGS(N, A, a));
            }
        };

#endif

#undef N

#endif

