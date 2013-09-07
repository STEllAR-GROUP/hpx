//  Copyright (c) 2011 Thomas Heller
//  Copyright (c) 2011-2013 Hartmut Kaiser
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

#include <boost/preprocessor/facilities/intercept.hpp>
#include <boost/preprocessor/iteration/iterate.hpp>
#include <boost/preprocessor/repetition/enum.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_params_with_a_default.hpp>
#include <boost/preprocessor/repetition/repeat.hpp>
#include <boost/fusion/include/adapt_struct.hpp>
#include <boost/fusion/include/at_c.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/mpl/int.hpp>
#if !defined(HPX_GCC_VERSION) || HPX_GCC_VERSION >= 40500 || defined(BOOST_INTEL)
#include <boost/mpl/fold.hpp>
#include <boost/mpl/and.hpp>
#endif
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
    namespace detail
    {
#if defined(BOOST_NO_CXX11_RVALUE_REFERENCES)
        template <typename T>
        struct env_value_type
        {
            typedef T type;
        };

        template <typename T>
        struct env_value_type<T const>
        {
            typedef T const type;
        };

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

        ///////////////////////////////////////////////////////////////////////
        struct ignore_type
        {
            template <typename T>
            ignore_type const& operator=(T const&) const
            {
                return *this;
            }
        };

// gcc 4.4.x is not able to cope with this, thus we disable the optimization
#if !defined(HPX_GCC_VERSION) || HPX_GCC_VERSION >= 40500 || defined(BOOST_INTEL)
        ///////////////////////////////////////////////////////////////////////
        struct compute_sequence_is_bitwise_serializable
        {
            template <typename State, typename T>
            struct apply
              : boost::mpl::and_<
                    boost::serialization::is_bitwise_serializable<T>, State>
            {};
        };

        template <typename Seq>
        struct sequence_is_bitwise_serializable
          : boost::mpl::fold<
                Seq, boost::mpl::true_, compute_sequence_is_bitwise_serializable>
        {};
#else
        ///////////////////////////////////////////////////////////////////////
        template <typename Seq>
        struct sequence_is_bitwise_serializable
          : boost::mpl::false_
        {};
#endif
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
    detail::ignore_type const ignore = detail::ignore_type{};

    ///////////////////////////////////////////////////////////////////////////
    template <typename Dummy = void>
    struct tuple0
    {
        typedef boost::mpl::int_<0> size_type;
        static const int size_value = 0;

        void swap(tuple0& other)
        {}

        template <typename Archive>
        void serialize(Archive & ar, unsigned)
        {}
    };

    template <>
    struct is_tuple<tuple0<> >
      : boost::mpl::true_
    {};

    inline tuple0<> tie() BOOST_NOEXCEPT
    {
        return tuple0<>();
    }

    inline tuple0<> forward_as_tuple() BOOST_NOEXCEPT
    {
        return tuple0<>();
    }

    ///////////////////////////////////////////////////////////////////////////
#if defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)
    template <BOOST_PP_ENUM_PARAMS_WITH_A_DEFAULT(
        HPX_TUPLE_LIMIT, typename A, void), typename Dummy = void>
    struct tuple;

    template <BOOST_PP_ENUM_PARAMS(HPX_TUPLE_LIMIT, typename A)>
    struct is_tuple<tuple<BOOST_PP_ENUM_PARAMS(HPX_TUPLE_LIMIT, A)> >
      : boost::mpl::true_
    {};
#else
    template <BOOST_PP_ENUM_PARAMS_WITH_A_DEFAULT(
        HPX_PP_ROUND_UP_ADD3(HPX_TUPLE_LIMIT), typename A, void), typename Dummy = void>
    struct tuple;

    template <BOOST_PP_ENUM_PARAMS(HPX_PP_ROUND_UP_ADD3(HPX_TUPLE_LIMIT), typename A)>
    struct is_tuple<tuple<BOOST_PP_ENUM_PARAMS(HPX_PP_ROUND_UP_ADD3(HPX_TUPLE_LIMIT), A)> >
      : boost::mpl::true_
    {};
#endif
    
    template <>
    struct tuple<> : tuple0<>
    {
        template <typename Archive>
        void serialize(Archive & ar, unsigned)
        {}
    };

    BOOST_FORCEINLINE tuple<>
    make_tuple()
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
      , typename boost::enable_if<
            is_tuple<typename boost::remove_cv<Tuple>::type>>::type>
    {
        typedef typename detail::tuple_element<N, Tuple>::type type;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Tuple, typename Enable = void>
    struct tuple_size;

    template <typename Tuple>
    struct tuple_size<Tuple
      , typename boost::enable_if<
            is_tuple<typename boost::remove_cv<Tuple>::type>>::type>
    {
        static const std::size_t value = Tuple::size_value;
    };

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
}}

namespace boost
{
    namespace serialization
    {
        template <>
        struct is_bitwise_serializable<hpx::util::tuple0<> >
          : boost::mpl::true_
        {};

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
            struct tag_of<hpx::util::tuple0<> >
            {
                typedef struct_tag type;
            };

            template <>
            struct tag_of<hpx::util::tuple0<> const>
            {
                typedef struct_tag type;
            };

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

            template<int I>
            struct access::struct_member<hpx::util::tuple<>, I>
            {
                template<typename Seq> struct apply;
            };

            template<int I>
            struct struct_member_name<hpx::util::tuple<>, I>
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
        struct sequence_tag<hpx::util::tuple0<> >
        {
            typedef fusion::fusion_sequence_tag type;
        };
        template<>
        struct sequence_tag<hpx::util::tuple0<> const>
        {
            typedef fusion::fusion_sequence_tag type;
        };

        template<typename>
        struct sequence_tag;
        template<>
        struct sequence_tag<hpx::util::tuple<> >
        {
            typedef fusion::fusion_sequence_tag type;
        };
        template<>
        struct sequence_tag<hpx::util::tuple<> const>
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

#define HPX_UTIL_TUPLE_NAME BOOST_PP_CAT(tuple, N)

#define HPX_UTIL_TUPLE_INIT_MEMBER(Z, N, D)                                   \
    BOOST_PP_CAT(a, N)(HPX_FORWARD_ARGS(Z, N, D))                             \
/**/
#define HPX_UTIL_TUPLE_INIT_COPY_MEMBER(Z, N, D)                              \
    BOOST_PP_CAT(a, N)(BOOST_PP_CAT(other.a, N))                              \
/**/
#define HPX_UTIL_TUPLE_INIT_MOVE_MEMBER(Z, N, D)                              \
    BOOST_PP_CAT(a, N)(detail::move_if_no_ref<BOOST_PP_CAT(D, N)>::call(      \
        BOOST_PP_CAT(other.a, N)))                                            \
/**/
#define HPX_UTIL_TUPLE_ASSIGN_COPY_MEMBER(Z, N, D)                            \
    BOOST_PP_CAT(a, N) = BOOST_PP_CAT(other.a, N);                            \
/**/
#define HPX_UTIL_TUPLE_ASSIGN_MOVE_MEMBER(Z, N, D)                            \
    BOOST_PP_CAT(a, N) = detail::move_if_no_ref<BOOST_PP_CAT(D, N)>::call(    \
        BOOST_PP_CAT(other.a, N));                                            \
/**/
#define HPX_UTIL_TUPLE_SWAP_MEMBER(Z, N, D)                                   \
    boost::swap(BOOST_PP_CAT(a, N), BOOST_PP_CAT(other.a, N));                \
/**/

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util
{
    template <BOOST_PP_ENUM_PARAMS(N, typename A)>
    struct HPX_UTIL_TUPLE_NAME
    {
        BOOST_PP_REPEAT(N, M0, _)

        template <int E>
        typename detail::tuple_element<E, HPX_UTIL_TUPLE_NAME>::rtype
        get() BOOST_NOEXCEPT
        {
            return detail::tuple_element<E, HPX_UTIL_TUPLE_NAME>::get(*this);
        }

        template <int E>
        BOOST_CONSTEXPR
        typename detail::tuple_element<E, HPX_UTIL_TUPLE_NAME const>::crtype
        get() const BOOST_NOEXCEPT
        {
            return detail::tuple_element<E, HPX_UTIL_TUPLE_NAME const>::get(*this);
        }

        HPX_UTIL_TUPLE_NAME() {}

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
        
        template <BOOST_PP_ENUM_PARAMS(N, typename T)>
        HPX_UTIL_TUPLE_NAME & operator=(BOOST_COPY_ASSIGN_REF(HPX_UTIL_STRIP((
                HPX_UTIL_TUPLE_NAME<BOOST_PP_ENUM_PARAMS(N, T)>
            ))) other)
        {
            BOOST_PP_REPEAT(N, HPX_UTIL_TUPLE_ASSIGN_COPY_MEMBER, T)
            return *this;
        }

        template <BOOST_PP_ENUM_PARAMS(N, typename T)>
        HPX_UTIL_TUPLE_NAME & operator=(BOOST_RV_REF(HPX_UTIL_STRIP((
                HPX_UTIL_TUPLE_NAME<BOOST_PP_ENUM_PARAMS(N, T)>
            ))) other)
        {
            BOOST_PP_REPEAT(N, HPX_UTIL_TUPLE_ASSIGN_MOVE_MEMBER, T)
            return *this;
        }

        void swap(HPX_UTIL_TUPLE_NAME& other)
        {
            BOOST_PP_REPEAT(N, HPX_UTIL_TUPLE_SWAP_MEMBER, T)
        }

#if N == 1
        template <typename Arg0>
        explicit tuple1(BOOST_FWD_REF(Arg0) arg0,
                typename boost::disable_if<is_tuple<Arg0> >::type* = 0)
          : a0(boost::forward<Arg0>(arg0))
        {}

        struct forwarding_tag {};

        template <typename Arg0>
        explicit tuple1(BOOST_FWD_REF(Arg0) arg0, forwarding_tag)
          : a0(boost::forward<Arg0>(arg0))
        {}

        template <typename Arg0>
        explicit tuple1(BOOST_RV_REF(tuple<Arg0>) arg0)
          : a0(detail::move_if_no_ref<Arg0>::call(arg0.a0))
        {}

        template <typename Arg0>
        explicit tuple1(BOOST_RV_REF(tuple1<Arg0>) arg0)
          : a0(detail::move_if_no_ref<Arg0>::call(arg0.a0))
        {}

        template <typename Arg0>
        explicit tuple1(tuple<Arg0> const& arg0)
          : a0(arg0.a0)
        {}

        template <typename Arg0>
        explicit tuple1(tuple1<Arg0> const& arg0)
          : a0(arg0.a0)
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

        template <BOOST_PP_ENUM_PARAMS(N, typename T)>
        HPX_UTIL_TUPLE_NAME(BOOST_RV_REF(HPX_UTIL_STRIP((
                    tuple<BOOST_PP_ENUM_PARAMS(N, T)>
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

#if !defined(BOOST_NO_CXX11_RVALUE_REFERENCES)
#define HPX_UTIL_MAKE_ARGUMENT_PACK(Z, N, D)                                  \
    typename detail::env_value_type<BOOST_PP_CAT(D, N)>::type                 \
/**/
#else
#define HPX_UTIL_MAKE_ARGUMENT_PACK(Z, N, D)                                  \
    BOOST_FWD_REF(BOOST_PP_CAT(D, N))                                         \
/**/
#endif

#if N == 1
    ///////////////////////////////////////////////////////////////////////////
    template <typename Arg0>
    BOOST_FORCEINLINE
    tuple1<Arg0> 
    tie(Arg0& arg0) BOOST_NOEXCEPT
    {
        typedef tuple1<Arg0> result_type;
        return result_type(arg0,
            typename result_type::forwarding_tag());
    }

    template <typename Arg0>
    BOOST_FORCEINLINE
    tuple1<HPX_UTIL_MAKE_ARGUMENT_PACK(_, 0, Arg)> 
    forward_as_tuple(BOOST_FWD_REF(Arg0) arg0) BOOST_NOEXCEPT
    {
        typedef tuple1<HPX_UTIL_MAKE_ARGUMENT_PACK(_, 0, Arg)> result_type;
        return result_type(boost::forward<Arg0>(arg0),
            typename result_type::forwarding_tag());
    }
#else
    ///////////////////////////////////////////////////////////////////////////
    template <BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    BOOST_FORCEINLINE
    HPX_UTIL_TUPLE_NAME<BOOST_PP_ENUM_BINARY_PARAMS(N, Arg, & BOOST_PP_INTERCEPT)>
    tie(BOOST_PP_ENUM_BINARY_PARAMS(N, Arg, & arg)) BOOST_NOEXCEPT
    {
        return HPX_UTIL_TUPLE_NAME<
                BOOST_PP_ENUM_BINARY_PARAMS(N, Arg, & BOOST_PP_INTERCEPT)>(
            BOOST_PP_ENUM_PARAMS(N, arg));
    }

    template <BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    BOOST_FORCEINLINE
    HPX_UTIL_TUPLE_NAME<BOOST_PP_ENUM(N, HPX_UTIL_MAKE_ARGUMENT_PACK, Arg)>
    forward_as_tuple(HPX_ENUM_FWD_ARGS(N, Arg, arg)) BOOST_NOEXCEPT
    {
        return HPX_UTIL_TUPLE_NAME<
                BOOST_PP_ENUM(N, HPX_UTIL_MAKE_ARGUMENT_PACK, Arg)>(
            HPX_ENUM_FORWARD_ARGS(N , Arg, arg));
    }
#endif

#undef HPX_UTIL_MAKE_ARGUMENT_PACK

    ///////////////////////////////////////////////////////////////////////////
    template <BOOST_PP_ENUM_PARAMS(N, typename A)>
    struct tuple<BOOST_PP_ENUM_PARAMS(N, A)>
      : HPX_UTIL_TUPLE_NAME<BOOST_PP_ENUM_PARAMS(N, A)>
    {
        typedef HPX_UTIL_TUPLE_NAME<BOOST_PP_ENUM_PARAMS(N, A)> base_tuple;

        tuple() {}

        tuple(tuple const& other)
          : base_tuple(other)
        {}
        tuple(base_tuple const& other)
          : base_tuple(other)
        {}

        tuple(BOOST_RV_REF(tuple) other)
          : base_tuple(other)
        {}
        tuple(BOOST_RV_REF(base_tuple) other)
          : base_tuple(other)
        {}

        tuple& operator=(BOOST_COPY_ASSIGN_REF(tuple) other)
        {
            this->base_tuple::operator=(other);
            return *this;
        }
        tuple& operator=(BOOST_COPY_ASSIGN_REF(base_tuple) other)
        {
            this->base_tuple::operator=(other);
            return *this;
        }

        tuple& operator=(BOOST_RV_REF(tuple) other)
        {
            this->base_tuple::operator=(other);
            return *this;
        }
        tuple& operator=(BOOST_RV_REF(base_tuple) other)
        {
            this->base_tuple::operator=(other);
            return *this;
        }

        template <BOOST_PP_ENUM_PARAMS(N, typename T)>
        tuple& operator=(BOOST_RV_REF(HPX_UTIL_STRIP((
                tuple<BOOST_PP_ENUM_PARAMS(N, T)>
            ))) other)
        {
            this->base_tuple::operator=(other);
            return *this;
        }
        template <BOOST_PP_ENUM_PARAMS(N, typename T)>
        tuple& operator=(BOOST_RV_REF(HPX_UTIL_STRIP((
                HPX_UTIL_TUPLE_NAME<BOOST_PP_ENUM_PARAMS(N, T)>
            ))) other)
        {
            this->base_tuple::operator=(other);
            return *this;
        }

#if N == 1
        template <typename Arg0>
        explicit tuple(BOOST_FWD_REF(Arg0) arg0,
                typename boost::disable_if<is_tuple<Arg0> >::type* = 0)
          : base_tuple(boost::forward<Arg0>(arg0))
        {}

        struct forwarding_tag {};

        template <typename Arg0>
        explicit tuple(BOOST_FWD_REF(Arg0) arg0, forwarding_tag t)
          : base_tuple(boost::forward<Arg0>(arg0), typename base_tuple::forwarding_tag())
        {}

        template <typename Arg0>
        explicit tuple(BOOST_RV_REF(tuple<Arg0>) arg0)
          : base_tuple(boost::move(arg0))
        {}

        template <typename Arg0>
        explicit tuple(BOOST_RV_REF(tuple1<Arg0>) arg0)
          : base_tuple(boost::move(arg0))
        {}

        template <typename Arg0>
        explicit tuple(tuple<Arg0> const& arg0)
          : base_tuple(arg0)
        {}

        template <typename Arg0>
        explicit tuple(tuple1<Arg0> const& arg0)
          : base_tuple(arg0)
        {}
#else
        template <BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        tuple(HPX_ENUM_FWD_ARGS(N, Arg, arg))
          : base_tuple(HPX_ENUM_FORWARD_ARGS(N, Arg, arg))
        {}

        template <BOOST_PP_ENUM_PARAMS(N, typename T)>
        tuple(BOOST_RV_REF(HPX_UTIL_STRIP((
                    tuple<BOOST_PP_ENUM_PARAMS(N, T)>
                ))) other)
          : base_tuple(other)
        {}
        template <BOOST_PP_ENUM_PARAMS(N, typename T)>
        tuple(BOOST_RV_REF(HPX_UTIL_STRIP((
                    HPX_UTIL_TUPLE_NAME<BOOST_PP_ENUM_PARAMS(N, T)>
                ))) other)
          : base_tuple(other)
        {}
#endif
    };

#define HPX_UTIL_MAKE_TUPLE_ARG(Z, N, D)                                      \
    typename util::decay<BOOST_PP_CAT(D, N)>::type                            \
/**/

#if N == 1
    ///////////////////////////////////////////////////////////////////////////
    template <BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    BOOST_FORCEINLINE
    tuple<HPX_UTIL_MAKE_TUPLE_ARG(_, 0, Arg)>
    make_tuple(BOOST_FWD_REF(Arg0) arg0)
    {
        typedef tuple<HPX_UTIL_MAKE_TUPLE_ARG(_, 0, Arg)> result_type;
        return result_type(boost::forward<Arg0>(arg0),
            typename result_type::forwarding_tag());
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
#endif

#undef HPX_UTIL_MAKE_TUPLE_ARG
}}

BOOST_FUSION_ADAPT_TPL_STRUCT(
    BOOST_PP_REPEAT(N, M1, _)
  , (BOOST_PP_CAT(hpx::util::tuple, N))BOOST_PP_REPEAT(N, M1, _)
  , BOOST_PP_REPEAT(N, M3, _)
)

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
            hpx::util::HPX_UTIL_TUPLE_NAME<BOOST_PP_ENUM_PARAMS(N, T)> >
       : hpx::util::detail::sequence_is_bitwise_serializable<
            hpx::util::HPX_UTIL_TUPLE_NAME<BOOST_PP_ENUM_PARAMS(N, T)> >
    {};

    template <BOOST_PP_ENUM_PARAMS(N, typename T)>
    struct is_bitwise_serializable<
            hpx::util::tuple<BOOST_PP_ENUM_PARAMS(N, T)> >
      : hpx::util::detail::sequence_is_bitwise_serializable<
            hpx::util::tuple<BOOST_PP_ENUM_PARAMS(N, T)> >
    {};

    ///////////////////////////////////////////////////////////////////////////
    template <typename Archive, BOOST_PP_ENUM_PARAMS(N, typename T)>
    BOOST_FORCEINLINE void serialize(Archive& ar,
        hpx::util::HPX_UTIL_TUPLE_NAME<BOOST_PP_ENUM_PARAMS(N, T)>& t,
        unsigned int const version)
    {
        hpx::util::serialize_sequence(ar, t);
    }

    template <typename Archive, BOOST_PP_ENUM_PARAMS(N, typename T)>
    BOOST_FORCEINLINE void serialize(Archive& ar,
        hpx::util::tuple<BOOST_PP_ENUM_PARAMS(N, T)>& t,
        unsigned int const version)
    {
        hpx::util::serialize_sequence(ar, t);
    }
}}

#undef N
#undef HPX_UTIL_TUPLE_NAME
#undef HPX_UTIL_TUPLE_INIT_MEMBER
#undef HPX_UTIL_TUPLE_INIT_COPY_MEMBER
#undef HPX_UTIL_TUPLE_INIT_MOVE_MEMBER
#undef HPX_UTIL_TUPLE_ASSIGN_COPY_MEMBER
#undef HPX_UTIL_TUPLE_ASSIGN_MOVE_MEMBER
#undef HPX_UTIL_TUPLE_SWAP_MEMBER

#endif
