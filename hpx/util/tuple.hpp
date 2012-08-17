//  Copyright (c) 2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !BOOST_PP_IS_ITERATING

#ifndef HPX_UTIL_TUPLE_HPP
#define HPX_UTIL_TUPLE_HPP

#include <hpx/config.hpp>
#include <hpx/util/move.hpp>
#include <hpx/util/detail/remove_reference.hpp>
#include <hpx/util/detail/pp_strip_parens.hpp>
#include <boost/preprocessor/iteration/iterate.hpp>
#include <boost/preprocessor/repetition/enum.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/repeat.hpp>
#include <boost/fusion/include/adapt_struct.hpp>
#include <boost/fusion/include/at_c.hpp>

#define M0(Z, N, D)                                                             \
    typedef BOOST_PP_CAT(A, N) BOOST_PP_CAT(member_type, N);                    \
    BOOST_PP_CAT(A, N) BOOST_PP_CAT(a, N);                                      \
/**/
#define M1(Z, N, D)                                                             \
    (BOOST_PP_CAT(A, N))                                                        \
/**/
#define M2(Z, N, D)                                                             \
    (BOOST_PP_CAT(T, N))                                                        \
/**/
#define M3(Z, N, D)                                                             \
    (BOOST_PP_CAT(A, N), BOOST_PP_CAT(a, N))                                    \
/**/

namespace hpx { namespace util
{
    namespace detail
    {
#if defined(BOOST_NO_RVALUE_REFERENCES)
        template <typename T>
        struct env_value_type
        {
            typedef T type;
        }

        template <typename T>
        struct env_value_type<T const>
        {
            typedef T const type;
        }

        template <typename T>
        struct env_value_type<T &>
        {
            typedef typename hpx::util::detail::remove_reference<T>::type & type;
        };

        template <typename T>
        struct env_value_type<T const &>
        {
            typedef typename hpx::util::detail::remove_reference<T>::type const & type;
        };
#else
        template <typename T, bool IsRvalue = std::is_rvalue_reference<T>::value>
        struct env_value_type
        {
            typedef typename hpx::util::detail::remove_reference<T>::type type;
        };

        template <typename T>
        struct env_value_type<T, false>
        {
            typedef T type;
        };

        template <typename T>
        struct env_value_type<T const, false>
        {
            typedef T const type;
        };

        template <typename T>
        struct env_value_type<T const &, false>
        {
            typedef T const & type;
        };

        template <typename T>
        struct env_value_type<T &, false>
        {
            typedef T & type;
        };
#endif

        template <typename A>
        struct move_if_no_ref
        {
            static A call(A & a)
            {
                return boost::move(a);
            }
        };

        template <typename A>
        struct move_if_no_ref<A &>
        {
            static A & call(A & a)
            {
                return a;
            }
        };

        template <typename A>
        struct move_if_no_ref<A const &>
        {
            static A const & call(A const & a)
            {
                return a;
            }
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    struct is_tuple
      : boost::mpl::false_
    {};

    template <typename T>
    struct is_tuple<T&>
      : is_tuple<T>
    {};

    template <typename T>
    struct is_tuple<T const&>
      : is_tuple<T>
    {};

    ///////////////////////////////////////////////////////////////////////////
    template <typename Dummy = void>
    struct tuple0
    {
        typedef boost::mpl::int_<0> size_type;
        static const int size_value = 0;

        template <typename Archive>
        void serialize(Archive & ar, unsigned)
        {}
    };

    template <>
    struct is_tuple<tuple0<> >
      : boost::mpl::true_
    {};

    inline tuple0<> forward_as_tuple()
    {
        return tuple0<>();
    }
}}

namespace boost {
    namespace fusion {
        namespace traits
        {
            template <>
            struct tag_of<hpx::util::tuple0<> >
            {
                typedef struct_tag type;
            };

            template <>
            struct tag_of<hpx::util::tuple0<> const>
            {
                typedef struct_tag type;
            };
        }

        namespace extension
        {
            template<int I>
            struct access::struct_member<hpx::util::tuple0<>, I>
            {
                template<typename Seq> struct apply;
            };

            template<int I>
            struct struct_member_name<hpx::util::tuple0<>, I>
            {};

            template<>
            struct struct_size<hpx::util::tuple0<> > : mpl::int_<0> {};

            template<>
            struct struct_is_view<hpx::util::tuple0<> > : mpl::false_ {};
        }
    }
    namespace mpl
    {
        template<typename>
        struct sequence_tag;
        template<>
        struct sequence_tag<hpx::util::tuple0<> >
        {
            typedef fusion::fusion_sequence_tag type;
        };
        template<>
        struct sequence_tag<hpx::util::tuple0<> const >
        {
            typedef fusion::fusion_sequence_tag type;
        };
    }
}

#if !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)
#  include <hpx/util/preprocessed/tuple.hpp>
#else

#if defined(__WAVE__) && defined(HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(preserve: 1, line: 0, output: "preprocessed/tuple_" HPX_LIMIT_STR ".hpp")
#endif

#define BOOST_PP_ITERATION_PARAMS_1                                             \
    (                                                                           \
        3                                                                       \
      , (                                                                       \
            1                                                                   \
          , HPX_TUPLE_LIMIT                                                     \
          , <hpx/util/tuple.hpp>                                                \
        )                                                                       \
    )                                                                           \
/**/
#include BOOST_PP_ITERATE()

#if defined(__WAVE__) && defined (HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(output: null)
#endif

#endif // !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)

#undef M0
#undef M1
#undef M2
#undef M3

#endif

#else // !BOOST_PP_IS_ITERATING

#define N BOOST_PP_ITERATION()

#define HPX_UTIL_TUPLE_NAME BOOST_PP_CAT(tuple, N)

#define HPX_UTIL_TUPLE_INIT_MEMBER(Z, N, D)                                     \
    BOOST_PP_CAT(a, N)(HPX_FORWARD_ARGS(Z, N, D))                               \

#define HPX_UTIL_TUPLE_INIT_COPY_MEMBER(Z, N, D)                                \
    BOOST_PP_CAT(a, N)(BOOST_PP_CAT(other.a, N))                                \

#define HPX_UTIL_TUPLE_INIT_MOVE_MEMBER(Z, N, D)                                \
    BOOST_PP_CAT(a, N)(detail::move_if_no_ref<BOOST_PP_CAT(D, N)>::call(BOOST_PP_CAT(other.a, N)))        \

#define HPX_UTIL_TUPLE_ASSIGN_COPY_MEMBER(Z, N, D)                              \
    BOOST_PP_CAT(a, N) = BOOST_PP_CAT(other.a, N);                              \

#define HPX_UTIL_TUPLE_ASSIGN_MOVE_MEMBER(Z, N, D)                              \
    BOOST_PP_CAT(a, N) = detail::move_if_no_ref<BOOST_PP_CAT(D, N)>::call(BOOST_PP_CAT(other.a, N));      \

#define HPX_UTIL_TUPLE_SERIALIZE(Z, N, D)                                       \
    this->do_serialize(ar, BOOST_PP_CAT(a, N));                                 \

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util
{
    template <BOOST_PP_ENUM_PARAMS(N, typename A)>
    struct HPX_UTIL_TUPLE_NAME
    {
        BOOST_PP_REPEAT(N, M0, _)

        HPX_UTIL_TUPLE_NAME() {}

//#ifdef BOOST_NO_RVALUE_REFERENCES
        HPX_UTIL_TUPLE_NAME(HPX_UTIL_TUPLE_NAME const& other)
          : BOOST_PP_ENUM(N, HPX_UTIL_TUPLE_INIT_COPY_MEMBER, _)
        {}

        HPX_UTIL_TUPLE_NAME(BOOST_RV_REF(HPX_UTIL_TUPLE_NAME) other)
          : BOOST_PP_ENUM(N, HPX_UTIL_TUPLE_INIT_MOVE_MEMBER, A)
        {}

        HPX_UTIL_TUPLE_NAME & operator=(BOOST_COPY_ASSIGN_REF(HPX_UTIL_TUPLE_NAME) other)
        {
            BOOST_PP_REPEAT(N, HPX_UTIL_TUPLE_ASSIGN_COPY_MEMBER, _)
            return *this;
        }

        HPX_UTIL_TUPLE_NAME & operator=(BOOST_RV_REF(HPX_UTIL_TUPLE_NAME) other)
        {
            BOOST_PP_REPEAT(N, HPX_UTIL_TUPLE_ASSIGN_MOVE_MEMBER, A)
            return *this;
        }
/*
#else
        HPX_UTIL_TUPLE_NAME(HPX_UTIL_TUPLE_NAME const& other) = default;
        HPX_UTIL_TUPLE_NAME(HPX_UTIL_TUPLE_NAME && other) = default;
        HPX_UTIL_TUPLE_NAME & operator=(HPX_UTIL_TUPLE_NAME const & other) = default;
        HPX_UTIL_TUPLE_NAME & operator=(HPX_UTIL_TUPLE_NAME && other) = default;
#endif
*/

        template <BOOST_PP_ENUM_PARAMS(N, typename T)>
        HPX_UTIL_TUPLE_NAME & operator=(BOOST_RV_REF(HPX_UTIL_STRIP((
                HPX_UTIL_TUPLE_NAME<BOOST_PP_ENUM_PARAMS(N, T)>
            ))) other)
        {
            BOOST_PP_REPEAT(N, HPX_UTIL_TUPLE_ASSIGN_MOVE_MEMBER, T)
            return *this;
        }

#if N == 1
        template <typename Arg0>
        explicit tuple1(BOOST_FWD_REF(Arg0) arg0,
                typename boost::disable_if<is_tuple<Arg0> >::type* = 0)
          : a0(boost::forward<Arg0>(arg0))
        {}

        template <typename Arg0>
        explicit tuple1(BOOST_RV_REF(tuple1<Arg0>) arg0)
          : a0(detail::move_if_no_ref<Arg0>::call(arg0.a0))
        {}
#else
        template <BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        HPX_UTIL_TUPLE_NAME(HPX_ENUM_FWD_ARGS(N, Arg, arg))
          : BOOST_PP_ENUM(N, HPX_UTIL_TUPLE_INIT_MEMBER, (Arg, arg))
        {}

        template <BOOST_PP_ENUM_PARAMS(N, typename T)>
        HPX_UTIL_TUPLE_NAME(BOOST_RV_REF(HPX_UTIL_STRIP((
                    HPX_UTIL_TUPLE_NAME<BOOST_PP_ENUM_PARAMS(N, T)>
                ))) other)
          : BOOST_PP_ENUM(N, HPX_UTIL_TUPLE_INIT_MOVE_MEMBER, T)
        {}
#endif

        typedef boost::mpl::int_<N> size_type;
        static const int size_value = N;

    private:
        BOOST_COPYABLE_AND_MOVABLE(HPX_UTIL_TUPLE_NAME);
    };

    template <BOOST_PP_ENUM_PARAMS(N, typename T)>
    struct is_tuple<HPX_UTIL_TUPLE_NAME<BOOST_PP_ENUM_PARAMS(N, T)> >
      : boost::mpl::true_
    {};

#if !defined(BOOST_NO_RVALUE_REFERENCES)
#define HPX_UTIL_MAKE_ARGUMENT_PACK(Z, N, D)                                  \
    typename detail::env_value_type<BOOST_PP_CAT(D, N)>::type                 \
    /**/
#else
#define HPX_UTIL_MAKE_ARGUMENT_PACK(Z, N, D)                                  \
    BOOST_FWD_REF(BOOST_PP_CAT(D, N))                                         \
    /**/
#endif

    ///////////////////////////////////////////////////////////////////////////
    template <BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    inline HPX_UTIL_TUPLE_NAME<BOOST_PP_ENUM(N, HPX_UTIL_MAKE_ARGUMENT_PACK, Arg)>
    forward_as_tuple(HPX_ENUM_FWD_ARGS(N, Arg, arg))
    {
        return HPX_UTIL_TUPLE_NAME<
                BOOST_PP_ENUM(N, HPX_UTIL_MAKE_ARGUMENT_PACK, Arg)>(
            HPX_ENUM_FORWARD_ARGS(N , Arg, arg));
    }

#undef HPX_UTIL_ARGUMENT_PACK_ARGS
#undef HPX_UTIL_MAKE_ARGUMENT_PACK
}}

BOOST_FUSION_ADAPT_TPL_STRUCT(
    BOOST_PP_REPEAT(N, M1, _)
  , (BOOST_PP_CAT(hpx::util::tuple, N))BOOST_PP_REPEAT(N, M1, _)
  , BOOST_PP_REPEAT(N, M3, _)
)

#undef N
#undef HPX_UTIL_TUPLE_NAME
#undef HPX_UTIL_TUPLE_INIT_MEMBER
#undef HPX_UTIL_TUPLE_INIT_COPY_MEMBER
#undef HPX_UTIL_TUPLE_INIT_MOVE_MEMBER
#undef HPX_UTIL_TUPLE_ASSIGN_COPY_MEMBER
#undef HPX_UTIL_TUPLE_ASSIGN_MOVE_MEMBER
#undef HPX_UTIL_TUPLE_SERIALIZE

#endif
