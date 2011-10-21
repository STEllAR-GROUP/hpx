//  Copyright (c) 2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !BOOST_PP_IS_ITERATING

#ifndef HPX_FUNCTION_DETAIL_VTABLE_HPP
#define HPX_FUNCTION_DETAIL_VTABLE_HPP

#include <boost/detail/sp_typeinfo.hpp>
#include <boost/preprocessor/iteration/iterate.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>

namespace hpx { namespace util { namespace detail {

    template <>
    struct vtable<true>
    {
        template <typename Functor, typename Sig, typename IArchive, typename OArchive>
        struct type;
#define BOOST_PP_ITERATION_PARAMS_1                                             \
    (                                                                           \
        4                                                                       \
      , (                                                                       \
            0                                                                   \
          , HPX_FUNCTION_LIMIT                                                  \
          , <hpx/util/detail/vtable.hpp>                                        \
          , 1                                                                   \
        )                                                                       \
    )                                                                           \
/**/
#include BOOST_PP_ITERATE()

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
          , HPX_FUNCTION_LIMIT                                                  \
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

            static void move(void *const* f, void ** dest)
            {
                reinterpret_cast<Functor*>(dest)->~Functor();
                *reinterpret_cast<Functor*>(dest) =
                    *reinterpret_cast<Functor const *>(f);
            }

            static R invoke(
                void ** f
                BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_BINARY_PARAMS(N, A, a)
            )
            {
                return
                    (*reinterpret_cast<Functor *>(f))(
                        BOOST_PP_ENUM_PARAMS(N, a)
                    );
            }

            static void iregister_base(IArchive & ar)
            {
                ar.register_type(
                    static_cast<
                        vtable_ptr<
                            R(BOOST_PP_ENUM_PARAMS(N, A))
                          , IArchive
                          , OArchive
                          , type<
                                Functor
                              , R(BOOST_PP_ENUM_PARAMS(N, A))
                              , IArchive
                              , OArchive
                            >
                        > *
                    >(0)
                );
            }

            static void oregister_base(OArchive & ar)
            {
                ar.register_type(
                    static_cast<
                        vtable_ptr<
                            R(BOOST_PP_ENUM_PARAMS(N, A))
                          , IArchive
                          , OArchive
                          , type<
                                Functor
                              , R(BOOST_PP_ENUM_PARAMS(N, A))
                              , IArchive
                              , OArchive
                            >
                        > *
                    >(0)
                );
            }

            static void iserialize(void ** object, IArchive & ar, unsigned)
            {
                Functor f;
                ar & f;
                new (object) Functor(f);
            }

            static void oserialize(void * const * f, OArchive & ar, unsigned)
            {
                ar & *reinterpret_cast<Functor const *>(f);
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

            static void move(void *const* f, void ** dest)
            {
                (*reinterpret_cast<Functor**>(dest))->~Functor();
                **reinterpret_cast<Functor**>(dest) =
                    **reinterpret_cast<Functor * const *>(f);
            }

            static R invoke(
                void ** f
                BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_BINARY_PARAMS(N, A, a)
            )
            {
                return
                    (**reinterpret_cast<Functor **>(f))(
                        BOOST_PP_ENUM_PARAMS(N, a)
                    );
            }

            static void iregister_base(IArchive & ar)
            {
                ar.register_type(
                    static_cast<
                        vtable_ptr<
                            R(BOOST_PP_ENUM_PARAMS(N, A))
                          , IArchive
                          , OArchive
                          , type<
                                Functor
                              , R(BOOST_PP_ENUM_PARAMS(N, A))
                              , IArchive
                              , OArchive
                            >
                        > *
                    >(0)
                );
            }

            static void oregister_base(OArchive & ar)
            {
                ar.register_type(
                    static_cast<
                        vtable_ptr<
                            R(BOOST_PP_ENUM_PARAMS(N, A))
                          , IArchive
                          , OArchive
                          , type<
                                Functor
                              , R(BOOST_PP_ENUM_PARAMS(N, A))
                              , IArchive
                              , OArchive
                            >
                        > *
                    >(0)
                );
            }

            static void iserialize(void ** object, IArchive & ar, unsigned)
            {
                Functor f;
                ar & f;
                *object = new Functor(f);
            }

            static void oserialize(void *const* f, OArchive & ar, unsigned)
            {
                ar & **reinterpret_cast<Functor *const*>(f);
            }
        };

#endif

#endif
