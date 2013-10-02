//  Copyright (c) 2011 Thomas Heller
//  Copyright (c) 2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !BOOST_PP_IS_ITERATING

#ifndef HPX_FUNCTION_DETAIL_EMPTY_VTABLE_HPP
#define HPX_FUNCTION_DETAIL_EMPTY_VTABLE_HPP

#include <hpx/config/forceinline.hpp>
#include <hpx/util/add_rvalue_reference.hpp>
#include <boost/ref.hpp>

#include <boost/preprocessor/iteration/iterate.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_trailing_params.hpp>

#include <typeinfo>

namespace hpx { namespace util { namespace detail
{
    struct empty_vtable_base
    {
        enum { empty = true };

        static std::type_info const& get_type()
        {
            return typeid(void);
        }

        static void static_delete(void ** f) {}
        static void destruct(void ** f) {}
        static void clone(void *const* f, void ** dest) {}
        static void copy(void *const* f, void ** dest) {}

        // we can safely return an int here as those function will never
        // be called.
        static int& construct(void ** f)
        {
            hpx::throw_exception(bad_function_call,
                "empty function object should not be used",
                "empty_vtable_base::construct", __FILE__, __LINE__);
            static int t = 0;
            return t;
        }

        static int& get(void **f)
        {
            hpx::throw_exception(bad_function_call,
                "empty function object should not be used",
                "empty_vtable_base::get", __FILE__, __LINE__);
            static int t = 0;
            return t;
        }

        static int& get(void *const*f)
        {
            hpx::throw_exception(bad_function_call,
                "empty function object should not be used",
                "empty_vtable_base::get", __FILE__, __LINE__);
            static int t = 0;
            return t;
        }
    };

    template <typename Sig, typename IArchive, typename OArchive>
    struct empty_vtable;
}}}

#define BOOST_UTIL_DETAIL_EMPTY_VTABLE_ADD_RVALUE_REF(Z, N, D)                  \
    typename util::add_rvalue_reference<BOOST_PP_CAT(D, N)>::type               \
    BOOST_PP_CAT(a, N)                                                          \
    /**/

#if !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)
#  include <hpx/util/detail/preprocessed/empty_vtable.hpp>
#else

#if defined(__WAVE__) && defined(HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(preserve: 1, line: 0, output: "preprocessed/empty_vtable_" HPX_LIMIT_STR ".hpp")
#endif

#define BOOST_PP_ITERATION_PARAMS_1                                             \
    (                                                                           \
        3                                                                       \
      , (                                                                       \
            0                                                                   \
          , HPX_FUNCTION_ARGUMENT_LIMIT                                         \
          , <hpx/util/detail/empty_vtable.hpp>                                  \
        )                                                                       \
    )                                                                           \
/**/
#include BOOST_PP_ITERATE()

#if defined(__WAVE__) && defined (HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(output: null)
#endif

#endif // !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)

#undef BOOST_UTIL_DETAIL_EMPTY_VTABLE_ADD_RVALUE_REF

#endif

#else

#define N BOOST_PP_ITERATION()

namespace hpx { namespace util { namespace detail
{
    template <
        typename R
      BOOST_PP_ENUM_TRAILING_PARAMS(N, typename A)
      , typename IArchive
      , typename OArchive
    >
    struct empty_vtable<
        R(BOOST_PP_ENUM_PARAMS(N, A))
      , IArchive
      , OArchive
    >
        : empty_vtable_base
    {
        typedef R (*functor_type)(BOOST_PP_ENUM_PARAMS(N, A));

        static vtable_ptr_base<
            R(BOOST_PP_ENUM_PARAMS(N, A))
          , IArchive
          , OArchive
        > *get_ptr()
        {
            return
                get_empty_table<
                    R(BOOST_PP_ENUM_PARAMS(N, A))
                >::template get<IArchive, OArchive>();
        }

        BOOST_ATTRIBUTE_NORETURN static R
        invoke(void ** f
            BOOST_PP_ENUM_TRAILING(N, BOOST_UTIL_DETAIL_EMPTY_VTABLE_ADD_RVALUE_REF, A))
        {
            hpx::throw_exception(bad_function_call,
                "empty function object should not be used",
                "empty_vtable::operator()");
        }
    };
}}}

#undef N

#endif

