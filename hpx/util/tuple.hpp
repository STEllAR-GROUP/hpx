//  Copyright (c) 2011-2013 Thomas Heller
//  Copyright (c) 2011-2013 Hartmut Kaiser
//  Copyright (c) 2013 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !BOOST_PP_IS_ITERATING

#ifndef HPX_UTIL_TUPLE_HPP
#define HPX_UTIL_TUPLE_HPP

#include <hpx/config.hpp>
#include <hpx/util/move.hpp>
#include <hpx/util/serialize_sequence.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/detail/pp_strip_parens.hpp>
#include <hpx/util/tuple_helper.hpp>
#include <hpx/util/add_rvalue_reference.hpp>

#include <boost/preprocessor/arithmetic/inc.hpp>
#include <boost/preprocessor/facilities/intercept.hpp>
#include <boost/preprocessor/iteration/iterate.hpp>
#include <boost/preprocessor/repetition/enum.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/repeat.hpp>
#include <boost/preprocessor/repetition/repeat_from_to.hpp>
#include <boost/fusion/include/adapt_struct.hpp>
#include <boost/fusion/include/at_c.hpp>
#include <boost/fusion/include/comparison.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/mpl/int.hpp>
#include <boost/mpl/eval_if.hpp>
#if !defined(HPX_GCC_VERSION) || HPX_GCC_VERSION >= 40500 || defined(BOOST_INTEL)
#include <boost/mpl/fold.hpp>
#include <boost/mpl/and.hpp>
#endif
#include <boost/type_traits/is_reference.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/utility/swap.hpp>

#include <boost/serialization/is_bitwise_serializable.hpp>

#define M0(Z, N, D)                                                           \
    typedef BOOST_PP_CAT(A, N) BOOST_PP_CAT(member_type, N);                  \
    BOOST_PP_CAT(A, N) BOOST_PP_CAT(a, N);                                    \
/**/
#define M1(Z, N, D)                                                           \
    (BOOST_PP_CAT(A, N))                                                      \
/**/
#define M2(Z, N, D)                                                           \
    (BOOST_PP_CAT(T, N))                                                      \
/**/
#define M3(Z, N, D)                                                           \
    (BOOST_PP_CAT(A, N), BOOST_PP_CAT(a, N))                                  \
/**/

namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
#if defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)
    template <BOOST_PP_ENUM_BINARY_PARAMS(HPX_TUPLE_LIMIT
      , typename A, = void BOOST_PP_INTERCEPT), typename Dummy = void>
    struct tuple;

    template <BOOST_PP_ENUM_PARAMS(HPX_TUPLE_LIMIT, typename A)>
    struct is_tuple<tuple<BOOST_PP_ENUM_PARAMS(HPX_TUPLE_LIMIT, A)> >
      : boost::mpl::true_
    {};

    template <BOOST_PP_ENUM_PARAMS(HPX_TUPLE_LIMIT, typename A)>
    void
    swap(
        tuple<BOOST_PP_ENUM_PARAMS(HPX_TUPLE_LIMIT, A)>& t0
      , tuple<BOOST_PP_ENUM_PARAMS(HPX_TUPLE_LIMIT, A)>& t1)
    {
        t0.swap(t1);
    }
#else
    template <BOOST_PP_ENUM_BINARY_PARAMS(HPX_PP_ROUND_UP_ADD3(HPX_TUPLE_LIMIT)
      , typename A, = void BOOST_PP_INTERCEPT), typename Dummy = void>
    struct tuple;

    template <BOOST_PP_ENUM_PARAMS(HPX_PP_ROUND_UP_ADD3(HPX_TUPLE_LIMIT), typename A)>
    struct is_tuple<tuple<BOOST_PP_ENUM_PARAMS(HPX_PP_ROUND_UP_ADD3(HPX_TUPLE_LIMIT), A)> >
      : boost::mpl::true_
    {};

    template <BOOST_PP_ENUM_PARAMS(HPX_PP_ROUND_UP_ADD3(HPX_TUPLE_LIMIT), typename A)>
    void
    swap(
        tuple<BOOST_PP_ENUM_PARAMS(HPX_PP_ROUND_UP_ADD3(HPX_TUPLE_LIMIT), A)>& t0
      , tuple<BOOST_PP_ENUM_PARAMS(HPX_PP_ROUND_UP_ADD3(HPX_TUPLE_LIMIT), A)>& t1)
    {
        t0.swap(t1);
    }
#endif

    template <>
    struct tuple<>
    {
        typedef boost::mpl::int_<0> size_type;
        static const int size_value = 0;

        void swap(tuple& other)
        {}

        template <typename Archive>
        void serialize(Archive & ar, unsigned)
        {}
    };

    ///////////////////////////////////////////////////////////////////////////
    BOOST_FORCEINLINE tuple<>
    make_tuple()
    {
        return tuple<>();
    }

    ///////////////////////////////////////////////////////////////////////////
    BOOST_FORCEINLINE tuple<> forward_as_tuple() BOOST_NOEXCEPT
    {
        return tuple<>();
    }

    ///////////////////////////////////////////////////////////////////////////
    BOOST_FORCEINLINE tuple<> tie() BOOST_NOEXCEPT
    {
        return tuple<>();
    }

    namespace detail
    {
        template <typename T>
        struct tuple_element_access
        {
            typedef const T& ctype;
            typedef T& type;
        };

        template <typename T>
        struct tuple_element_access<T&>
        {
            typedef T& ctype;
            typedef T& type;
        };

        template <int E, typename Tuple>
        struct tuple_element;
    }

    ///////////////////////////////////////////////////////////////////////////
    template <int N, typename Tuple, typename Enable = void>
    struct tuple_element;

    template <int N, typename Tuple>
    struct tuple_element<N, Tuple
      , typename boost::enable_if<is_tuple<Tuple> >::type>
    {
        typedef typename detail::tuple_element<N, Tuple>::type type;
    };

    template <int N, typename Tuple>
    struct tuple_element<N, Tuple const
      , typename boost::enable_if<is_tuple<Tuple> >::type>
      : boost::add_const<typename tuple_element<N, Tuple>::type>
    {};

    ///////////////////////////////////////////////////////////////////////////
    template <typename Tuple, typename Enable = void>
    struct tuple_size;

    template <>
    struct tuple_size<tuple<> >
    {
        static const std::size_t value = 0;
    };

    template <typename Tuple>
    struct tuple_size<Tuple const>
      : tuple_size<Tuple>
    {};

    namespace detail
    {
        template <std::size_t N, typename T0, typename T1, typename Enable = void>
        struct tuple_cat_element
        {
            typedef void type;
        };

        template <std::size_t N, typename T0, typename T1>
        struct tuple_cat_element<
            N, T0, T1
          , typename boost::enable_if_c<
                (N < util::decay<T0>::type::size_value)>::type>
        {
            typedef typename util::decay<T0>::type tuple0_type;
            typedef typename util::decay<T1>::type tuple1_type;

            typedef
                typename detail::tuple_element<N, tuple0_type>::type
                type;

            static BOOST_FORCEINLINE
            type
            call(tuple0_type const& t0, tuple1_type const& t1)
            {
                return t0.template get<N>();
            }
            static BOOST_FORCEINLINE
            typename util::add_rvalue_reference<type>::type
            call(BOOST_RV_REF(tuple0_type) t0, tuple1_type const& t1)
            {
                return boost::forward<type>(t0.template get<N>());
            }
        };

        template <std::size_t N, typename T0, typename T1>
        struct tuple_cat_element<
            N, T0, T1
          , typename boost::enable_if_c<
                (N >= util::decay<T0>::type::size_value &&
                N < util::decay<T1>::type::size_value + util::decay<T0>::type::size_value)>::type>
        {
            static const std::size_t offset =
                util::decay<T0>::type::size_value;
            
            typedef typename util::decay<T0>::type tuple0_type;
            typedef typename util::decay<T1>::type tuple1_type;

            typedef
                typename detail::tuple_element<N - offset, tuple1_type>::type
                type;

            static BOOST_FORCEINLINE
            type
            call(tuple0_type const& t0, tuple1_type const& t1)
            {
                return t1.template get<N - offset>();
            }
            static BOOST_FORCEINLINE
            typename util::add_rvalue_reference<type>::type
            call(tuple0_type const& t0, BOOST_RV_REF(tuple1_type) t1)
            {
                return boost::forward<type>(t1.template get<N - offset>());
            }
        };

        ///////////////////////////////////////////////////////////////////////
#       define HPX_TUPLE_TRAILING_PARAMS(Z, N, D)                             \
        , BOOST_PP_CAT(D, N)                                                  \
        /**/

#       if defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)
        template <BOOST_PP_ENUM_BINARY_PARAMS(HPX_TUPLE_LIMIT
          , typename T, = void BOOST_PP_INTERCEPT), typename Dummy = void>
        struct tuple_cat_result
          : tuple_cat_result<
                typename tuple_cat_result<T0, T1>::type
                BOOST_PP_REPEAT_FROM_TO(2, HPX_TUPLE_LIMIT
                  , HPX_TUPLE_TRAILING_PARAMS, T)
            >
        {};
#       else
        template <BOOST_PP_ENUM_BINARY_PARAMS(HPX_PP_ROUND_UP_ADD3(HPX_TUPLE_LIMIT)
          , typename T, = void BOOST_PP_INTERCEPT), typename Dummy = void>
        struct tuple_cat_result
          : tuple_cat_result<
                typename tuple_cat_result<T0, T1>::type
                BOOST_PP_REPEAT_FROM_TO(2, HPX_PP_ROUND_UP_ADD3(HPX_TUPLE_LIMIT)
                  , HPX_TUPLE_TRAILING_PARAMS, T)
            >
        {};
#       endif

#       undef HPX_TUPLE_TRAILING_PARAMS

        template <>
        struct tuple_cat_result<>
        {
            typedef tuple<> type;
        };

        template <typename T0>
        struct tuple_cat_result<T0>
        {
            typedef T0 type;
        };

        template <typename T0, typename T1>
        struct tuple_cat_result<T0, T1>
        {
#           define HPX_TUPLE_CAT_ELEM_TYPE(Z, N, D)                           \
            typename tuple_cat_element<N, T0, T1>::type                       \
            /**/

            typedef tuple<
                BOOST_PP_ENUM(HPX_TUPLE_LIMIT, HPX_TUPLE_CAT_ELEM_TYPE, _)> type;

#           undef HPX_TUPLE_CAT_ELEM_TYPE
        };
    }

    BOOST_FORCEINLINE tuple<>
    tuple_cat()
    {
        return tuple<>();
    }

    ///////////////////////////////////////////////////////////////////////////
    template <int N, typename Tuple>
    BOOST_FORCEINLINE BOOST_CONSTEXPR
    typename detail::tuple_element<N, Tuple>::rtype
    get(Tuple& t) BOOST_NOEXCEPT
    {
        return t.template get<N>();
    }

    template <int N, typename Tuple>
    BOOST_FORCEINLINE BOOST_CONSTEXPR
    typename detail::tuple_element<N, Tuple const>::crtype
    get(Tuple const& t) BOOST_NOEXCEPT
    {
        return t.template get<N>();
    }

    template <int N, typename Tuple>
    BOOST_FORCEINLINE BOOST_CONSTEXPR
    typename util::add_rvalue_reference<
        typename boost::lazy_disable_if<
            boost::is_reference<Tuple>
          , detail::tuple_element<N, Tuple>
        >::type
    >::type
    get(BOOST_RV_REF(Tuple) t) BOOST_NOEXCEPT
    {
        return
            boost::forward<typename detail::tuple_element<N, Tuple>::type>
                (t.template get<N>());
    }

    ///////////////////////////////////////////////////////////////////////////
    using boost::fusion::operator==;
    using boost::fusion::operator!=;
    using boost::fusion::operator<;
    using boost::fusion::operator>;
    using boost::fusion::operator<=;
    using boost::fusion::operator>=;
}}

namespace boost
{
    namespace serialization
    {
        template <>
        struct is_bitwise_serializable<hpx::util::tuple<> >
          : boost::mpl::true_
        {};
    }
    namespace fusion
    {
        namespace traits
        {
            template <>
            struct tag_of<hpx::util::tuple<> >
            {
                typedef struct_tag type;
            };

            template <>
            struct tag_of<hpx::util::tuple<> const>
            {
                typedef struct_tag type;
            };
        }

        namespace extension
        {
            template<int I>
            struct access::struct_member<hpx::util::tuple<>, I>
            {
                template<typename Seq> struct apply;
            };
            template<>
            struct access::struct_member<hpx::util::tuple<>, 0>
            {
                template<typename Seq> struct apply;
            };

            template<int I>
            struct struct_member_name<hpx::util::tuple<>, I>
            {};
            template<>
            struct struct_member_name<hpx::util::tuple<>, 0>
            {};

            template<>
            struct struct_size<hpx::util::tuple<> > : mpl::int_<0> {};

            template<>
            struct struct_is_view<hpx::util::tuple<> > : mpl::false_ {};
        }
    }
    namespace mpl
    {
        template<typename>
        struct sequence_tag;
        template<>
        struct sequence_tag<hpx::util::tuple<> >
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

///////////////////////////////////////////////////////////////////////////////
#define HPX_TUPLE_INDEX(Z, N, D)                                              \
    template <typename Tuple>                                                 \
    struct tuple_element<N, Tuple>                                            \
    {                                                                         \
        typedef typename Tuple::BOOST_PP_CAT(member_type, N) type;            \
        typedef typename detail::tuple_element_access<type>::type rtype;      \
        typedef typename detail::tuple_element_access<type>::ctype crtype;    \
                                                                              \
        static BOOST_CONSTEXPR rtype get(Tuple& t) BOOST_NOEXCEPT             \
            { return t.BOOST_PP_CAT(a, N); }                                  \
    };                                                                        \
/**/

#define HPX_TUPLE_INDEX_CONST(Z, N, D)                                        \
    template <typename Tuple>                                                 \
    struct tuple_element<N, Tuple const>                                      \
    {                                                                         \
        typedef typename boost::add_const<                                    \
            typename Tuple::BOOST_PP_CAT(member_type, N)>::type type;         \
        typedef typename detail::tuple_element_access<type>::type rtype;      \
        typedef typename detail::tuple_element_access<type>::ctype crtype;    \
                                                                              \
        static BOOST_CONSTEXPR crtype get(Tuple const& t) BOOST_NOEXCEPT      \
            { return t.BOOST_PP_CAT(a, N); }                                  \
    };                                                                        \
/**/

namespace hpx { namespace util { namespace detail
{
    BOOST_PP_REPEAT(HPX_TUPLE_LIMIT, HPX_TUPLE_INDEX, _)
    BOOST_PP_REPEAT(HPX_TUPLE_LIMIT, HPX_TUPLE_INDEX_CONST, _)
}}}

#undef HPX_TUPLE_INDEX_CONST
#undef HPX_TUPLE_INDEX

///////////////////////////////////////////////////////////////////////////////
#define BOOST_PP_ITERATION_PARAMS_1                                           \
    (                                                                         \
        3                                                                     \
      , (                                                                     \
            1                                                                 \
          , HPX_TUPLE_LIMIT                                                   \
          , <hpx/util/tuple.hpp>                                              \
        )                                                                     \
    )                                                                         \
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

#define HPX_UTIL_TUPLE_DEFAULT_CONSTRUCT_MEMBER(Z, N, D)                      \
    BOOST_PP_CAT(a, N)()                                                      \
/**/
#define HPX_UTIL_TUPLE_INIT_MEMBER(Z, N, D)                                   \
    BOOST_PP_CAT(a, N)(HPX_FORWARD_ARGS(Z, N, D))                             \
/**/
#define HPX_UTIL_TUPLE_INIT_COPY_MEMBER(Z, N, D)                              \
    BOOST_PP_CAT(a, N)(                                                       \
        detail::copy_construct<                                               \
            BOOST_PP_CAT(A, N)                                                \
          , typename boost::add_const<BOOST_PP_CAT(D, N)>::type               \
        >::call(BOOST_PP_CAT(other.a, N)))                                    \
/**/
#define HPX_UTIL_TUPLE_INIT_MOVE_MEMBER(Z, N, D)                              \
    BOOST_PP_CAT(a, N)(boost::forward<BOOST_PP_CAT(D, N)>(                    \
        BOOST_PP_CAT(other.a, N)))                                            \
/**/
#define HPX_UTIL_TUPLE_ASSIGN_COPY_MEMBER(Z, N, D)                            \
    BOOST_PP_CAT(a, N) = BOOST_PP_CAT(other.a, N);                            \
/**/
#define HPX_UTIL_TUPLE_ASSIGN_MOVE_MEMBER(Z, N, D)                            \
    BOOST_PP_CAT(a, N) = boost::forward<BOOST_PP_CAT(D, N)>(                  \
        BOOST_PP_CAT(other.a, N));                                            \
/**/
#define HPX_UTIL_TUPLE_SWAP_MEMBER(Z, N, D)                                   \
    boost::swap(BOOST_PP_CAT(a, N), BOOST_PP_CAT(other.a, N));                \
/**/

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    template <BOOST_PP_ENUM_PARAMS(N, typename A)>
    struct tuple<BOOST_PP_ENUM_PARAMS(N, A)>
    {
        BOOST_PP_REPEAT(N, M0, _)

        template <int E>
        typename detail::tuple_element<E, tuple>::rtype
        get() BOOST_NOEXCEPT
        {
            return detail::tuple_element<E, tuple>::get(*this);
        }

        template <int E>
        BOOST_CONSTEXPR
        typename detail::tuple_element<E, tuple const>::crtype
        get() const BOOST_NOEXCEPT
        {
            return detail::tuple_element<E, tuple const>::get(*this);
        }

        ///////////////////////////////////////////////////////////////////////
        tuple()
          : BOOST_PP_ENUM(N, HPX_UTIL_TUPLE_DEFAULT_CONSTRUCT_MEMBER, A)
        {}

#       if N == 1
        template <typename Arg0>
        tuple(BOOST_FWD_REF(Arg0) arg0
          , typename boost::disable_if<is_tuple<Arg0> >::type* = 0)
          : a0(boost::forward<Arg0>(arg0))
        {}

        template <typename Arg0>
        tuple(BOOST_FWD_REF(Arg0) arg0, detail::forwarding_tag)
          : a0(boost::forward<Arg0>(arg0))
        {}
#       else
        template <BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        tuple(HPX_ENUM_FWD_ARGS(N, Arg, arg))
          : BOOST_PP_ENUM(N, HPX_UTIL_TUPLE_INIT_MEMBER, (Arg, arg))
        {}
#       endif

        ///////////////////////////////////////////////////////////////////////
        tuple(tuple const& other)
          : BOOST_PP_ENUM(N, HPX_UTIL_TUPLE_INIT_COPY_MEMBER, A)
        {}

        tuple(BOOST_RV_REF(tuple) other)
          : BOOST_PP_ENUM(N, HPX_UTIL_TUPLE_INIT_MOVE_MEMBER, A)
        {}

        template <BOOST_PP_ENUM_PARAMS(N, typename T)>
        tuple(tuple<BOOST_PP_ENUM_PARAMS(N, T)> const& other)
          : BOOST_PP_ENUM(N, HPX_UTIL_TUPLE_INIT_COPY_MEMBER, T)
        {}

        template <BOOST_PP_ENUM_PARAMS(N, typename T)>
        tuple(BOOST_RV_REF(HPX_UTIL_STRIP((
                tuple<BOOST_PP_ENUM_PARAMS(N, T)>
            ))) other)
          : BOOST_PP_ENUM(N, HPX_UTIL_TUPLE_INIT_MOVE_MEMBER, T)
        {}

#       if N == 2
        template <typename U1, typename U2>
        tuple(std::pair<U1, U2> const& other)
          : a0(other.first)
          , a1(other.second)
        {}

        template <typename U1, typename U2>
        tuple(BOOST_RV_REF(HPX_UTIL_STRIP((std::pair<U1, U2>))) other)
          : a0(boost::move(other.first))
          , a1(boost::move(other.second))
        {}
#       endif

        ///////////////////////////////////////////////////////////////////////
        tuple& operator=(BOOST_COPY_ASSIGN_REF(tuple) other)
        {
            BOOST_PP_REPEAT(N, HPX_UTIL_TUPLE_ASSIGN_COPY_MEMBER, A);
            return *this;
        }

        tuple& operator=(BOOST_RV_REF(tuple) other)
        {
            BOOST_PP_REPEAT(N, HPX_UTIL_TUPLE_ASSIGN_MOVE_MEMBER, A);
            return *this;
        }

        template <BOOST_PP_ENUM_PARAMS(N, typename T)>
        tuple& operator=(BOOST_COPY_ASSIGN_REF(HPX_UTIL_STRIP((
                tuple<BOOST_PP_ENUM_PARAMS(N, T)>
            ))) other)
        {
            BOOST_PP_REPEAT(N, HPX_UTIL_TUPLE_ASSIGN_COPY_MEMBER, T);
            return *this;
        }

        template <BOOST_PP_ENUM_PARAMS(N, typename T)>
        tuple& operator=(BOOST_RV_REF(HPX_UTIL_STRIP((
                tuple<BOOST_PP_ENUM_PARAMS(N, T)>
            ))) other)
        {
            BOOST_PP_REPEAT(N, HPX_UTIL_TUPLE_ASSIGN_MOVE_MEMBER, T);
            return *this;
        }

#       if N == 2
        template <typename U1, typename U2>
        tuple& operator=(BOOST_COPY_ASSIGN_REF(HPX_UTIL_STRIP((
                std::pair<U1, U2>
            ))) other)
        {
            a0 = other.first;
            a1 = other.second;
            return *this;
        }

        template <typename U1, typename U2>
        tuple& operator=(BOOST_RV_REF(HPX_UTIL_STRIP((
                std::pair<U1, U2>
            ))) other)
        {
            a0 = boost::move(other.first);
            a1 = boost::move(other.second);
            return *this;
        }
#       endif

        void swap(tuple& other)
        {
            BOOST_PP_REPEAT(N, HPX_UTIL_TUPLE_SWAP_MEMBER, T);
        }

        typedef boost::mpl::int_<N> size_type;
        static const int size_value = N;

    private:
        BOOST_COPYABLE_AND_MOVABLE(tuple);
    };

    template <BOOST_PP_ENUM_PARAMS(N, typename A)>
    struct tuple_size<tuple<BOOST_PP_ENUM_PARAMS(N, A)> >
    {
        static const std::size_t value = N;
    };

#define HPX_UTIL_MAKE_TUPLE_ARG(Z, N, D)                                      \
    typename util::decay<BOOST_PP_CAT(D, N)>::type                            \
/**/
#define HPX_UTIL_TUPLE_TRAILING_FWD_ARG(Z, N, D)                              \
    , boost::forward<BOOST_PP_CAT(T, N)>(BOOST_PP_CAT(D, N))                  \
/**/
#define HPX_UTIL_MAKE_FWD_TUPLE_ARG(Z, N, D)                                  \
    typename util::add_rvalue_reference<BOOST_PP_CAT(D, N)>::type             \
/**/
#define HPX_UTIL_TUPLE_CAT_ELEM_CALL(Z, N, D)                                 \
    detail::tuple_cat_element<N, T0, T1>::call(t0, t1)                        \
/**/

#if N == 1
    ///////////////////////////////////////////////////////////////////////////
    template <BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    BOOST_FORCEINLINE
    tuple<HPX_UTIL_MAKE_TUPLE_ARG(_, 0, Arg)>
    make_tuple(BOOST_FWD_REF(Arg0) arg0)
    {
        typedef tuple<HPX_UTIL_MAKE_TUPLE_ARG(_, 0, Arg)> result_type;
        return result_type(boost::forward<Arg0>(arg0), detail::forwarding_tag());
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Arg0>
    BOOST_FORCEINLINE
    tuple<HPX_UTIL_MAKE_FWD_TUPLE_ARG(_, 0, Arg)>
    forward_as_tuple(BOOST_FWD_REF(Arg0) arg0) BOOST_NOEXCEPT
    {
        typedef tuple<HPX_UTIL_MAKE_FWD_TUPLE_ARG(_, 0, Arg)> result_type;
        return result_type(boost::forward<Arg0>(arg0), detail::forwarding_tag());
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Arg0>
    BOOST_FORCEINLINE
    tuple<Arg0&>
    tie(Arg0& arg0) BOOST_NOEXCEPT
    {
        typedef tuple<Arg0&> result_type;
        return result_type(arg0, detail::forwarding_tag());
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename T0>
    BOOST_FORCEINLINE T0
    tuple_cat(BOOST_FWD_REF(T0) t0)
    {
        return boost::forward<T0>(t0);
    }
#else
    ///////////////////////////////////////////////////////////////////////////
    template <BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    BOOST_FORCEINLINE
    tuple<BOOST_PP_ENUM(N, HPX_UTIL_MAKE_TUPLE_ARG, Arg)>
    make_tuple(HPX_ENUM_FWD_ARGS(N, Arg, arg))
    {
        return tuple<BOOST_PP_ENUM(N, HPX_UTIL_MAKE_TUPLE_ARG, Arg)>(
            HPX_ENUM_FORWARD_ARGS(N , Arg, arg));
    }

    ///////////////////////////////////////////////////////////////////////////
    template <BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    BOOST_FORCEINLINE
    tuple<BOOST_PP_ENUM(N, HPX_UTIL_MAKE_FWD_TUPLE_ARG, Arg)>
    forward_as_tuple(HPX_ENUM_FWD_ARGS(N, Arg, arg)) BOOST_NOEXCEPT
    {
        return tuple<
                BOOST_PP_ENUM(N, HPX_UTIL_MAKE_FWD_TUPLE_ARG, Arg)>(
            HPX_ENUM_FORWARD_ARGS(N , Arg, arg));
    }

    ///////////////////////////////////////////////////////////////////////////
    template <BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    BOOST_FORCEINLINE
    tuple<BOOST_PP_ENUM_BINARY_PARAMS(N, Arg, & BOOST_PP_INTERCEPT)>
    tie(BOOST_PP_ENUM_BINARY_PARAMS(N, Arg, & arg)) BOOST_NOEXCEPT
    {
        return tuple<
                BOOST_PP_ENUM_BINARY_PARAMS(N, Arg, & BOOST_PP_INTERCEPT)>(
            BOOST_PP_ENUM_PARAMS(N, arg));
    }

    ///////////////////////////////////////////////////////////////////////////
#   if N == 2
#   define HPX_UTIL_TUPLE_CAT_IMPL(Z, N, D)                                   \
    template <typename T0, typename T1>                                       \
    typename boost::lazy_enable_if_c<                                         \
        N == util::decay<T0>::type::size_value                                \
           + util::decay<T1>::type::size_value                                \
      , detail::tuple_cat_result<T0, T1>                                      \
    >::type                                                                   \
    tuple_cat(BOOST_FWD_REF(T0) t0, BOOST_FWD_REF(T1) t1)                     \
    {                                                                         \
        typedef typename detail::tuple_cat_result<T0, T1>::type result_type;  \
        return result_type(BOOST_PP_ENUM(N, HPX_UTIL_TUPLE_CAT_ELEM_CALL, _));\
    }                                                                         \
    /**/

    BOOST_PP_REPEAT(HPX_TUPLE_LIMIT, HPX_UTIL_TUPLE_CAT_IMPL, _)

#   undef HPX_UTIL_TUPLE_CAT_IMPL
#   else
    ///////////////////////////////////////////////////////////////////////////
    template <BOOST_PP_ENUM_PARAMS(N, typename T)>
    typename detail::tuple_cat_result<
        BOOST_PP_ENUM(N, HPX_UTIL_MAKE_TUPLE_ARG, T)
    >::type
    tuple_cat(HPX_ENUM_FWD_ARGS(N, T, t))
    {
        typedef
            typename detail::tuple_cat_result<T0, T1>::type
            head_type;

        head_type head =
            tuple_cat(boost::forward<T0>(t0), boost::forward<T1>(t1));
        return tuple_cat(boost::move(head)
                BOOST_PP_REPEAT_FROM_TO(2, N
                  , HPX_UTIL_TUPLE_TRAILING_FWD_ARG, t));
    }
#   endif
#endif

#undef HPX_UTIL_MAKE_TUPLE_ARG
#undef HPX_UTIL_TUPLE_TRAILING_FWD_ARG
#undef HPX_UTIL_MAKE_FWD_TUPLE_ARG
#undef HPX_UTIL_TUPLE_CAT_ELEM_CALL
}}

BOOST_FUSION_ADAPT_TPL_STRUCT(
    BOOST_PP_REPEAT(N, M1, _)
  , (hpx::util::tuple)BOOST_PP_REPEAT(N, M1, _)
  , BOOST_PP_REPEAT(N, M3, _)
)

namespace boost { namespace serialization
{
    ///////////////////////////////////////////////////////////////////////////
    template <BOOST_PP_ENUM_PARAMS(N, typename T)>
    struct is_bitwise_serializable<
            hpx::util::tuple<BOOST_PP_ENUM_PARAMS(N, T)> >
      : hpx::util::detail::sequence_is_bitwise_serializable<
            hpx::util::tuple<BOOST_PP_ENUM_PARAMS(N, T)> >
    {};

    ///////////////////////////////////////////////////////////////////////////
    template <typename Archive, BOOST_PP_ENUM_PARAMS(N, typename T)>
    BOOST_FORCEINLINE void serialize(Archive& ar,
        hpx::util::tuple<BOOST_PP_ENUM_PARAMS(N, T)>& t,
        unsigned int const version)
    {
        hpx::util::serialize_sequence(ar, t);
    }
}}

#undef N
#undef HPX_UTIL_TUPLE_DEFAULT_CONSTRUCT_MEMBER
#undef HPX_UTIL_TUPLE_INIT_MEMBER
#undef HPX_UTIL_TUPLE_INIT_COPY_MEMBER
#undef HPX_UTIL_TUPLE_INIT_MOVE_MEMBER
#undef HPX_UTIL_TUPLE_ASSIGN_COPY_MEMBER
#undef HPX_UTIL_TUPLE_ASSIGN_MOVE_MEMBER
#undef HPX_UTIL_TUPLE_SWAP_MEMBER

#endif
