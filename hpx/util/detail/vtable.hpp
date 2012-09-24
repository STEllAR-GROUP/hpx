//  Copyright (c) 2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !BOOST_PP_IS_ITERATING

#ifndef HPX_FUNCTION_DETAIL_VTABLE_HPP
#define HPX_FUNCTION_DETAIL_VTABLE_HPP

#include <boost/ref.hpp>
#include <boost/detail/sp_typeinfo.hpp>
#include <boost/preprocessor/iteration/iterate.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>

namespace hpx { namespace util { namespace detail {

    template <>
    struct vtable<true>
    {
        template <typename Functor, typename Sig, typename IArchive, typename OArchive>
        struct type;

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
        template <typename Functor, typename Sig, typename IArchive, typename OArchive>
        struct type;
#define BOOST_PP_ITERATION_PARAMS_1                                             \
    (                                                                           \
        4                                                                       \
      , (                                                                       \
            0                                                                   \
          , HPX_FUNCTION_ARGUMENT_LIMIT                                                  \
          , <hpx/util/detail/vtable.hpp>                                        \
          , 2                                                                   \
        )                                                                       \
    )                                                                           \
/**/
#include BOOST_PP_ITERATE()

    };
}}}

#endif

#else

#define N BOOST_PP_ITERATION()

#if BOOST_PP_ITERATION_FLAGS() == 1

        template <
            typename Functor
          , typename R
          BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename A)
          , typename IArchive
          , typename OArchive
        >
        struct type<
            Functor
          , R(BOOST_PP_ENUM_PARAMS(N, A))
          , IArchive
          , OArchive
        >
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

            static boost::detail::sp_typeinfo const & get_type()
            {
                return BOOST_SP_TYPEID(Functor);
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

            static R
            invoke(void ** f BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_BINARY_PARAMS(N, A, a))
            {
                return invoke_(f BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, a), typename boost::is_reference_wrapper<Functor>::type());
            }

            static R
            invoke_(void ** f BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_BINARY_PARAMS(N, A, a), boost::mpl::true_)
            {
                return (*reinterpret_cast<Functor*>(f)).get()(BOOST_PP_ENUM_PARAMS(N, a));
            }
            static R
            invoke_(void ** f BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_BINARY_PARAMS(N, A, a), boost::mpl::false_)
            {
                return (*reinterpret_cast<Functor*>(f))(BOOST_PP_ENUM_PARAMS(N, a));
            }
        };

        template <
            typename Functor
          BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename A)
          , typename IArchive
          , typename OArchive
        >
        struct type<
            Functor
          , void(BOOST_PP_ENUM_PARAMS(N, A))
          , IArchive
          , OArchive
        >
        {
            static vtable_ptr_base<
                void(BOOST_PP_ENUM_PARAMS(N, A))
              , IArchive
              , OArchive
            > *get_ptr()
            {
                return
                    get_table<
                        Functor
                      , void(BOOST_PP_ENUM_PARAMS(N, A))
                    >::template get<IArchive, OArchive>();
            }

            static boost::detail::sp_typeinfo const & get_type()
            {
                return BOOST_SP_TYPEID(Functor);
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

            static
            void
            invoke(void ** f BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_BINARY_PARAMS(N, A, a))
            {
                invoke_(f BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, a), typename boost::is_reference_wrapper<Functor>::type());
            }

            static
            void
            invoke_(void ** f BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_BINARY_PARAMS(N, A, a), boost::mpl::true_)
            {
                (*reinterpret_cast<Functor*>(f)).get()(BOOST_PP_ENUM_PARAMS(N, a));
            }
            static
            void
            invoke_(void ** f BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_BINARY_PARAMS(N, A, a), boost::mpl::false_)
            {
                (*reinterpret_cast<Functor*>(f))(BOOST_PP_ENUM_PARAMS(N, a));
            }
        };


#endif

#if BOOST_PP_ITERATION_FLAGS() == 2

        template <
            typename Functor
          , typename R
          BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename A)
          , typename IArchive
          , typename OArchive
        >
        struct type<
            Functor
          , R(BOOST_PP_ENUM_PARAMS(N, A))
          , IArchive
          , OArchive
        >
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

            static boost::detail::sp_typeinfo const & get_type()
            {
                return BOOST_SP_TYPEID(Functor);
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

            static R
            invoke(void ** f BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_BINARY_PARAMS(N, A, a))
            {
                return invoke_(f BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, a), typename boost::is_reference_wrapper<Functor>::type());
            }

            static R
            invoke_(void ** f BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_BINARY_PARAMS(N, A, a), boost::mpl::true_)
            {
                return (**reinterpret_cast<Functor**>(f)).get()(BOOST_PP_ENUM_PARAMS(N, a));
            }
            static R
            invoke_(void ** f BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_BINARY_PARAMS(N, A, a), boost::mpl::false_)
            {
                return (**reinterpret_cast<Functor**>(f))(BOOST_PP_ENUM_PARAMS(N, a));
            }
        };

        template <
            typename Functor
          BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename A)
          , typename IArchive
          , typename OArchive
        >
        struct type<
            Functor
          , void(BOOST_PP_ENUM_PARAMS(N, A))
          , IArchive
          , OArchive
        >
        {
            static vtable_ptr_base<
                void(BOOST_PP_ENUM_PARAMS(N, A))
              , IArchive
              , OArchive
            > *get_ptr()
            {
                return
                    get_table<
                        Functor
                      , void(BOOST_PP_ENUM_PARAMS(N, A))
                    >::template get<IArchive, OArchive>();
            }

            static boost::detail::sp_typeinfo const & get_type()
            {
                return BOOST_SP_TYPEID(Functor);
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

            static
            void
            invoke(void ** f BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_BINARY_PARAMS(N, A, a))
            {
                invoke_(f BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, a), typename boost::is_reference_wrapper<Functor>::type());
            }

            static
            void
            invoke_(void ** f BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_BINARY_PARAMS(N, A, a), boost::mpl::true_)
            {
                (**reinterpret_cast<Functor**>(f)).get()(BOOST_PP_ENUM_PARAMS(N, a));
            }
            static
            void
            invoke_(void ** f BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_BINARY_PARAMS(N, A, a), boost::mpl::false_)
            {
                (**reinterpret_cast<Functor**>(f))(BOOST_PP_ENUM_PARAMS(N, a));
            }
        };

#endif

#undef N

#endif

