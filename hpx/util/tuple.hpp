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
#include <hpx/util/decay.hpp>
#include <hpx/util/move.hpp>
#include <hpx/util/serialize_sequence.hpp>
#include <hpx/util/detail/pp_strip_parens.hpp>
#include <hpx/util/detail/qualify_as.hpp>

#include <boost/array.hpp>
#include <boost/fusion/adapted/struct/adapt_struct.hpp>
#include <utility>
#include <boost/mpl/bool.hpp>
#include <boost/mpl/eval_if.hpp>
#include <boost/mpl/identity.hpp>
#include <boost/mpl/size_t.hpp>
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/min.hpp>
#include <boost/preprocessor/arithmetic/add.hpp>
#include <boost/preprocessor/arithmetic/dec.hpp>
#include <boost/preprocessor/arithmetic/div.hpp>
#include <boost/preprocessor/arithmetic/inc.hpp>
#include <boost/preprocessor/arithmetic/mul.hpp>
#include <boost/preprocessor/facilities/intercept.hpp>
#include <boost/preprocessor/iteration/iterate.hpp>
#include <boost/preprocessor/repetition/enum.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>
#include <boost/preprocessor/repetition/repeat.hpp>
#include <boost/ref.hpp>
#include <boost/serialization/is_bitwise_serializable.hpp>
#include <boost/type_traits/add_const.hpp>
#include <boost/type_traits/add_cv.hpp>
#include <boost/type_traits/add_volatile.hpp>
#include <boost/type_traits/is_base_of.hpp>
#include <boost/type_traits/is_fundamental.hpp>
#include <boost/type_traits/is_pointer.hpp>
#include <boost/type_traits/is_reference.hpp>
#include <boost/type_traits/remove_cv.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/utility/swap.hpp>

#include <cstddef> // for size_t
#include <utility> // for pair

#if defined(BOOST_NO_SFINAE_EXPR) ||                                          \
    (defined(HPX_GCC_VERSION) && HPX_GCC_VERSION < 40800)
#   define HPX_UTIL_TUPLE_SFINAE_NOEXCEPT_IF(Predicate)
#else
#   define HPX_UTIL_TUPLE_SFINAE_NOEXCEPT_IF(Predicate)                       \
    BOOST_NOEXCEPT_IF(Predicate)                                              \
    /**/
#endif

#define N HPX_TUPLE_LIMIT

namespace hpx { namespace util
{
    template <
        BOOST_PP_ENUM_BINARY_PARAMS(N, typename T, = void BOOST_PP_INTERCEPT)
    >
    class tuple;

    template <class T>
    struct tuple_size; // undefined

    template <std::size_t I, typename T>
    struct tuple_element; // undefined

    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        template <typename T, typename Enable = void>
        struct tuple_member
        {
        public: // exposition-only
            T _value;

        public:
            // 20.4.2.1, tuple construction
            BOOST_CONSTEXPR tuple_member()
              : _value()
            {}

            template <typename U>
            BOOST_CONSTEXPR explicit tuple_member(U && value)
              : _value(std::forward<U>(value))
            {}

            BOOST_CONSTEXPR tuple_member(tuple_member const& other)
              : _value(other._value)
            {}
            BOOST_CONSTEXPR tuple_member(tuple_member && other)
              : _value(std::forward<T>(other._value))
            {}
        };

        template <typename T>
        struct tuple_member<T&>
        {
        public: // exposition-only
            T& _value;

        public:
            // 20.4.2.1, tuple construction
            BOOST_CONSTEXPR explicit tuple_member(T& value)
              : _value(value)
            {}

            BOOST_CONSTEXPR tuple_member(tuple_member const& other)
              : _value(other._value)
            {}
        };

#       if (defined(HPX_GCC_VERSION) && HPX_GCC_VERSION < 40600)              \
        || (defined(BOOST_MSVC) && BOOST_MSVC < 1800)
        template <typename T>
        struct tuple_member<T &&,
            typename boost::disable_if_c<
                boost::is_fundamental<T>::value || boost::is_pointer<T>::value
            >::type
        >
        {
        public: // exposition-only
            T && _value;

        public:
            // 20.4.2.1, tuple construction
            BOOST_CONSTEXPR explicit tuple_member(T && value)
              : _value(std::forward<T>(value))
            {}

            BOOST_CONSTEXPR tuple_member(tuple_member const& other)
              : _value(std::forward<T>(other._value))
            {}
        };

        template <typename T>
        struct tuple_member<
            T &&
          , typename boost::enable_if_c<
                boost::is_fundamental<T>::value || boost::is_pointer<T>::value
            >::type
        >
        {
        public: // exposition-only
            T _value;

        public:
            // 20.4.2.1, tuple construction
            BOOST_CONSTEXPR explicit tuple_member(T && value)
              : _value(std::forward<T>(value))
            {}

            BOOST_CONSTEXPR tuple_member(tuple_member const& other)
              : _value(other._value)
            {}
        };
#       else
        template <typename T>
        struct tuple_member<T &&>
        {
        public: // exposition-only
            T && _value;

        public:
            // 20.4.2.1, tuple construction
            BOOST_CONSTEXPR explicit tuple_member(T && value)
              : _value(std::forward<T>(value))
            {}

            BOOST_CONSTEXPR tuple_member(tuple_member const& other)
              : _value(std::forward<T>(other._value))
            {}
        };
#       endif

        ///////////////////////////////////////////////////////////////////////
        template <typename TTuple, typename UTuple, typename Enable = void>
        struct are_tuples_compatible
          : boost::mpl::false_
        {};

        template <typename UTuple>
        struct are_tuples_compatible<
            tuple<>, UTuple
          , typename boost::enable_if_c<
                tuple_size<typename boost::remove_reference<UTuple>::type>::value == 0
            >::type
        > : boost::mpl::true_
        {};

        ///////////////////////////////////////////////////////////////////////
        struct ignore_type
        {
            template <typename T>
            void operator=(T && t) const
            {}
        };
    }

    // 20.4.2, class template tuple
    template <>
    class tuple<>
    {
    public:
        // 20.4.2.1, tuple construction

        // constexpr tuple();
        // Value initializes each element.
        BOOST_CONSTEXPR tuple()
        {}

        // tuple(const tuple& u) = default;
        // Initializes each element of *this with the corresponding element
        // of u.
        BOOST_CONSTEXPR tuple(tuple const& /*other*/)
        {}

        // tuple(tuple&& u) = default;
        // For all i, initializes the ith element of *this with
        // std::forward<Ti>(get<i>(u)).
        BOOST_CONSTEXPR tuple(tuple && /*other*/)
        {}

        // 20.4.2.2, tuple assignment

        // tuple& operator=(const tuple& u);
        // Assigns each element of u to the corresponding element of *this.
        tuple& operator=(tuple const& /*other*/) BOOST_NOEXCEPT
        {
            return *this;
        }

        // tuple& operator=(tuple&& u) noexcept(see below );
        // For all i, assigns std::forward<Ti>(get<i>(u)) to get<i>(*this).
        tuple& operator=(tuple && /*other*/) BOOST_NOEXCEPT
        {
            return *this;
        }

        // 20.4.2.3, tuple swap

        // void swap(tuple& rhs) noexcept(see below);
        // Calls swap for each element in *this and its corresponding element
        // in rhs.
        void swap(tuple& /*other*/) BOOST_NOEXCEPT
        {}
    };

    // 20.4.2.5, tuple helper classes

    // template <class Tuple>
    // class tuple_size
    template <class T>
    struct tuple_size
    {};

    template <class T>
    struct tuple_size<const T>
      : tuple_size<T>
    {};

    template <class T>
    struct tuple_size<volatile T>
      : tuple_size<T>
    {};

    template <class T>
    struct tuple_size<const volatile T>
      : tuple_size<T>
    {};

    template <>
    struct tuple_size<tuple<> >
      : boost::mpl::size_t<0>
    {};

    template <typename T0, typename T1>
    struct tuple_size<std::pair<T0, T1> >
      : boost::mpl::size_t<2>
    {};

    template <typename Type, std::size_t Size>
    struct tuple_size<boost::array<Type, Size> >
      : boost::mpl::size_t<Size>
    {};

    // template <size_t I, class Tuple>
    // class tuple_element
    template <std::size_t I, typename T>
    struct tuple_element
    {};

    template <std::size_t I, typename T>
    struct tuple_element<I, const T>
      : boost::add_const<typename tuple_element<I, T>::type>
    {};

    template <std::size_t I, typename T>
    struct tuple_element<I, volatile T>
      : boost::add_volatile<typename tuple_element<I, T>::type>
    {};

    template <std::size_t I, typename T>
    struct tuple_element<I, const volatile T>
      : boost::add_cv<typename tuple_element<I, T>::type>
    {};

    template <typename T0, typename T1>
    struct tuple_element<0, std::pair<T0, T1> >
      : boost::mpl::identity<T0>
    {
        template <typename Tuple>
        static BOOST_CONSTEXPR BOOST_FORCEINLINE
        typename detail::qualify_as<T0, Tuple&>::type
        get(Tuple& tuple) BOOST_NOEXCEPT
        {
            return tuple.first;
        }
    };
    template <typename T0, typename T1>
    struct tuple_element<1, std::pair<T0, T1> >
      : boost::mpl::identity<T1>
    {
        template <typename Tuple>
        static BOOST_CONSTEXPR BOOST_FORCEINLINE
        typename detail::qualify_as<T1, Tuple&>::type
        get(Tuple& tuple) BOOST_NOEXCEPT
        {
            return tuple.second;
        }
    };

    template <std::size_t I, typename Type, std::size_t Size>
    struct tuple_element<I, boost::array<Type, Size> >
      : boost::mpl::identity<Type>
    {
        template <typename Tuple>
        static BOOST_CONSTEXPR BOOST_FORCEINLINE
        typename detail::qualify_as<Type, Tuple&>::type
        get(Tuple& tuple) BOOST_NOEXCEPT
        {
            return tuple[I];
        }
    };

    template <typename Tuple>
    struct tuple_decay
    {};
    
#   define HPX_TUPLE_DECAY_ELEM(Z, N, D)                                      \
    typename decay<BOOST_PP_CAT(T, N)>::type                                  \
    /**/
    template <BOOST_PP_ENUM_PARAMS(N, typename T)>
    struct tuple_decay<tuple<BOOST_PP_ENUM_PARAMS(N, T)> >
    {
        typedef tuple<BOOST_PP_ENUM(N, HPX_TUPLE_DECAY_ELEM, _)> type;
    };    
#undef HPX_TUPLE_DECAY_ELEM

    // 20.4.2.6, element access

    // template <size_t I, class... Types>
    // constexpr typename tuple_element<I, tuple<Types...> >::type&
    // get(tuple<Types...>& t) noexcept;
    template <std::size_t I, typename Tuple>
    BOOST_CONSTEXPR BOOST_FORCEINLINE
    typename detail::qualify_as<
        typename tuple_element<I, Tuple>::type
      , Tuple&
    >::type
    get(Tuple& t) BOOST_NOEXCEPT
    {
        return tuple_element<I, Tuple>::get(t);
    }

    // template <size_t I, class... Types>
    // constexpr typename tuple_element<I, tuple<Types...> >::type const&
    // get(const tuple<Types...>& t) noexcept;
    template <std::size_t I, typename Tuple>
    BOOST_CONSTEXPR BOOST_FORCEINLINE
    typename detail::qualify_as<
        typename tuple_element<I, Tuple>::type
      , Tuple const&
    >::type
    get(Tuple const& t) BOOST_NOEXCEPT
    {
        return tuple_element<I, Tuple>::get(t);
    }

    // template <size_t I, class... Types>
    // constexpr typename tuple_element<I, tuple<Types...> >::type&&
    // get(tuple<Types...>&& t) noexcept;
    template <std::size_t I, typename Tuple>
    BOOST_CONSTEXPR BOOST_FORCEINLINE
    typename detail::qualify_as<
        typename tuple_element<I, Tuple>::type
      , Tuple &&
    >::type
    get(Tuple && t) BOOST_NOEXCEPT
    {
        return std::forward<
            typename tuple_element<I, Tuple>::type>(util::get<I>(t));
    }

    // 20.4.2.4, tuple creation functions
    detail::ignore_type const ignore = {};

    // template<class... Types>
    // constexpr tuple<VTypes...> make_tuple(Types&&... t);
    namespace detail
    {
        template <typename T, typename U = typename decay<T>::type>
        struct make_tuple_element
          : boost::mpl::identity<U>
        {};

        template <typename T, typename X>
        struct make_tuple_element<T, boost::reference_wrapper<X> >
          : boost::mpl::identity<X&>
        {};
    }

    BOOST_CONSTEXPR BOOST_FORCEINLINE
    tuple<>
    make_tuple()
    {
        return tuple<>();
    }

    // template<class... Types>
    // tuple<Types&&...> forward_as_tuple(Types&&... t) noexcept;
    // Constructs a tuple of references to the arguments in t suitable for
    // forwarding as arguments to a function. Because the result may contain
    // references to temporary variables, a program shall ensure that the
    // return value of this function does not outlive any of its arguments.
    BOOST_FORCEINLINE
    tuple<>
    forward_as_tuple() BOOST_NOEXCEPT
    {
        return tuple<>();
    }

    // template<class... Types>
    // tuple<Types&...> tie(Types&... t) noexcept;
    BOOST_FORCEINLINE
    tuple<>
    tie() BOOST_NOEXCEPT
    {
        return tuple<>();
    }

    //template <class... Tuples>
    //constexpr tuple<Ctypes ...> tuple_cat(Tuples&&...);
    namespace detail
    {
        template <
            std::size_t I, typename T0, typename T1
          , typename Enable = void
        >
        struct tuple_cat_element
        {
            typedef void type;
        };

        template <std::size_t I, typename T0, typename T1>
        struct tuple_cat_element<
            I, T0, T1
          , typename boost::enable_if_c<
                (I < tuple_size<T0>::value)
            >::type
        >
        {
            typedef
                typename tuple_element<
                    I
                  , typename boost::remove_cv<T0>::type
                >::type
                type;

            template <typename TTuple, typename UTuple>
            static BOOST_FORCEINLINE
            typename detail::qualify_as<type, TTuple&&>::type
            call(TTuple && t, UTuple && u)
            {
                return util::get<I>(std::forward<TTuple>(t));
            }
        };

        template <std::size_t I, typename T0, typename T1>
        struct tuple_cat_element<
            I, T0, T1
          , typename boost::enable_if_c<
                (I >= tuple_size<T0>::value)
             && (I < tuple_size<T1>::value + tuple_size<T0>::value)
            >::type
        >
        {
            static const std::size_t offset =
                tuple_size<T0>::value;

            typedef
                typename tuple_element<
                    I - offset
                  , typename boost::remove_cv<T1>::type
                >::type
                type;

            template <typename TTuple, typename UTuple>
            static BOOST_FORCEINLINE
            typename detail::qualify_as<type, UTuple&&>::type
            call(TTuple && t, UTuple && u)
            {
                return util::get<I - offset>(std::forward<UTuple>(u));
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <
            BOOST_PP_ENUM_BINARY_PARAMS(N, typename T, = void BOOST_PP_INTERCEPT)
          , typename Enable = void
        >
        struct tuple_cat_result;

        template <>
        struct tuple_cat_result<>
        {
            typedef tuple<> type;
        };

        template <typename Tuple>
        struct tuple_cat_result<
            Tuple
          , typename boost::enable_if_c<tuple_size<Tuple>::value == 0>::type
        >
        {
            typedef tuple<> type;
        };

#       define HPX_UTIL_TUPLE_CAT_ELEMENT(Z, N, D)                            \
        typename tuple_cat_element<N, TTuple, UTuple>::type                   \
        /**/
        template <typename TTuple, typename UTuple>
        struct tuple_cat_result<TTuple, UTuple>
        {
            typedef tuple<BOOST_PP_ENUM(N, HPX_UTIL_TUPLE_CAT_ELEMENT, _)> type;
        };
#       undef HPX_UTIL_TUPLE_CAT_ELEMENT
    }

    BOOST_CONSTEXPR BOOST_FORCEINLINE
    tuple<>
    tuple_cat()
    {
        return tuple<>();
    }

    template <BOOST_PP_ENUM_PARAMS(N, typename T)>
    BOOST_CONSTEXPR BOOST_FORCEINLINE
    tuple<BOOST_PP_ENUM_PARAMS(N, T)>
    tuple_cat(tuple<BOOST_PP_ENUM_PARAMS(N, T)> const& t)
    {
        return t;
    }
    template <BOOST_PP_ENUM_PARAMS(N, typename T)>
    BOOST_CONSTEXPR BOOST_FORCEINLINE
    tuple<BOOST_PP_ENUM_PARAMS(N, T)>
    tuple_cat(tuple<BOOST_PP_ENUM_PARAMS(N, T)> && t)
    {
        return std::move(t);
    }

    template <typename Tuple>
    BOOST_CONSTEXPR BOOST_FORCEINLINE
    typename boost::enable_if_c<
        tuple_size<typename boost::remove_reference<Tuple>::type>::value == 0
      , tuple<>
    >::type
    tuple_cat(Tuple && t)
    {
        return tuple<>();
    }

    template <typename TTuple, typename UTuple>
    BOOST_CONSTEXPR BOOST_FORCEINLINE
    typename boost::enable_if_c<
        tuple_size<typename boost::remove_reference<TTuple>::type>::value
      + tuple_size<typename boost::remove_reference<UTuple>::type>::value == 0
      , tuple<>
    >::type
    tuple_cat(TTuple && /*t*/, UTuple && /*u*/)
    {
        return tuple<>();
    }

    // 20.4.2.7, relational operators

    // template<class... TTypes, class... UTypes>
    // constexpr bool operator==
    //     (const tuple<TTypes...>& t, const tuple<UTypes...>& u);
    // The elementary comparisons are performed in order from the zeroth index
    // upwards. No comparisons or element accesses are performed after the
    // first equality comparison that evaluates to false.
    namespace detail
    {
        template <std::size_t I, std::size_t Size>
        struct tuple_equal_to
        {
            template <typename TTuple, typename UTuple>
            static BOOST_CONSTEXPR BOOST_FORCEINLINE
            bool call(TTuple const& t, UTuple const&u)
            {
                return
                    util::get<I>(t) == util::get<I>(u)
                 && tuple_equal_to<I + 1, Size>::call(t, u);
            }
        };

        template <std::size_t Size>
        struct tuple_equal_to<Size, Size>
        {
            template <typename TTuple, typename UTuple>
            static BOOST_CONSTEXPR BOOST_FORCEINLINE
            bool call(TTuple const& t, UTuple const&u)
            {
                return true;
            }
        };
    }

    template <
        BOOST_PP_ENUM_PARAMS(N, typename T)
      , BOOST_PP_ENUM_PARAMS(N, typename U)
    >
    BOOST_CONSTEXPR BOOST_FORCEINLINE
    typename boost::enable_if_c<
        tuple_size<tuple<BOOST_PP_ENUM_PARAMS(N, T)> >::value
     == tuple_size<tuple<BOOST_PP_ENUM_PARAMS(N, U)> >::value
      , bool
    >::type
    operator==(
        tuple<BOOST_PP_ENUM_PARAMS(N, T)> const& t
      , tuple<BOOST_PP_ENUM_PARAMS(N, U)> const& u
    )
    {
        return
            detail::tuple_equal_to<
                0, tuple_size<tuple<BOOST_PP_ENUM_PARAMS(N, T)> >::value
            >::call(t, u);
    }

    // template<class... TTypes, class... UTypes>
    // constexpr bool operator!=
    //     (const tuple<TTypes...>& t, const tuple<UTypes...>& u);
    template <
        BOOST_PP_ENUM_PARAMS(N, typename T)
      , BOOST_PP_ENUM_PARAMS(N, typename U)
    >
    BOOST_CONSTEXPR BOOST_FORCEINLINE
    typename boost::enable_if_c<
        tuple_size<tuple<BOOST_PP_ENUM_PARAMS(N, T)> >::value
     == tuple_size<tuple<BOOST_PP_ENUM_PARAMS(N, U)> >::value
      , bool
    >::type
    operator!=(
        tuple<BOOST_PP_ENUM_PARAMS(N, T)> const& t
      , tuple<BOOST_PP_ENUM_PARAMS(N, U)> const& u
    )
    {
        return !(t == u);
    }

    // template<class... TTypes, class... UTypes>
    // constexpr bool operator<
    //     (const tuple<TTypes...>& t, const tuple<UTypes...>& u);
    // The result is defined as: (bool)(get<0>(t) < get<0>(u)) ||
    // (!(bool)(get<0>(u) < get<0>(t)) && ttail < utail), where rtail for some
    // tuple r is a tuple containing all but the first element of r. For any
    // two zero-length tuples e and f, e < f returns false.
    namespace detail
    {
        template <std::size_t I, std::size_t Size>
        struct tuple_less_than
        {
            template <typename TTuple, typename UTuple>
            static BOOST_CONSTEXPR BOOST_FORCEINLINE
            bool call(TTuple const& t, UTuple const&u)
            {
                return
                    util::get<I>(t) < util::get<I>(u)
                 || (
                        !(util::get<I>(u) < util::get<I>(t))
                     && tuple_less_than<I + 1, Size>::call(t, u)
                    );
            }
        };

        template <std::size_t Size>
        struct tuple_less_than<Size, Size>
        {
            template <typename TTuple, typename UTuple>
            static BOOST_CONSTEXPR BOOST_FORCEINLINE
            bool call(TTuple const& t, UTuple const&u)
            {
                return false;
            }
        };
    }

    template <
        BOOST_PP_ENUM_PARAMS(N, typename T)
      , BOOST_PP_ENUM_PARAMS(N, typename U)
    >
    BOOST_CONSTEXPR BOOST_FORCEINLINE
    typename boost::enable_if_c<
        tuple_size<tuple<BOOST_PP_ENUM_PARAMS(N, T)> >::value
     == tuple_size<tuple<BOOST_PP_ENUM_PARAMS(N, U)> >::value
      , bool
    >::type
    operator<(
        tuple<BOOST_PP_ENUM_PARAMS(N, T)> const& t
      , tuple<BOOST_PP_ENUM_PARAMS(N, U)> const& u
    )
    {
        return
            detail::tuple_less_than<
                0, tuple_size<tuple<BOOST_PP_ENUM_PARAMS(N, T)> >::value
            >::call(t, u);
    }

    // template<class... TTypes, class... UTypes>
    // constexpr bool operator>
    //     (const tuple<TTypes...>& t, const tuple<UTypes...>& u);
    template <
        BOOST_PP_ENUM_PARAMS(N, typename T)
      , BOOST_PP_ENUM_PARAMS(N, typename U)
    >
    BOOST_CONSTEXPR BOOST_FORCEINLINE
    typename boost::enable_if_c<
        tuple_size<tuple<BOOST_PP_ENUM_PARAMS(N, T)> >::value
     == tuple_size<tuple<BOOST_PP_ENUM_PARAMS(N, U)> >::value
      , bool
    >::type
    operator>(
        tuple<BOOST_PP_ENUM_PARAMS(N, T)> const& t
      , tuple<BOOST_PP_ENUM_PARAMS(N, U)> const& u
    )
    {
        return u < t;
    }

    // template<class... TTypes, class... UTypes>
    // constexpr bool operator<=
    //     (const tuple<TTypes...>& t, const tuple<UTypes...>& u);
    template <
        BOOST_PP_ENUM_PARAMS(N, typename T)
      , BOOST_PP_ENUM_PARAMS(N, typename U)
    >
    BOOST_CONSTEXPR BOOST_FORCEINLINE
    typename boost::enable_if_c<
        tuple_size<tuple<BOOST_PP_ENUM_PARAMS(N, T)> >::value
     == tuple_size<tuple<BOOST_PP_ENUM_PARAMS(N, U)> >::value
      , bool
    >::type
    operator<=(
        tuple<BOOST_PP_ENUM_PARAMS(N, T)> const& t
      , tuple<BOOST_PP_ENUM_PARAMS(N, U)> const& u
    )
    {
        return !(u < t);
    }

    // template<class... TTypes, class... UTypes>
    // constexpr bool operator>=
    //     (const tuple<TTypes...>& t, const tuple<UTypes...>& u);
    template <
        BOOST_PP_ENUM_PARAMS(N, typename T)
      , BOOST_PP_ENUM_PARAMS(N, typename U)
    >
    BOOST_CONSTEXPR BOOST_FORCEINLINE
    typename boost::enable_if_c<
        tuple_size<tuple<BOOST_PP_ENUM_PARAMS(N, T)> >::value
     == tuple_size<tuple<BOOST_PP_ENUM_PARAMS(N, U)> >::value
      , bool
    >::type
    operator>=(
        tuple<BOOST_PP_ENUM_PARAMS(N, T)> const& t
      , tuple<BOOST_PP_ENUM_PARAMS(N, U)> const& u
    )
    {
        return !(t < u);
    }

    // 20.4.2.9, specialized algorithms

    // template <class... Types>
    // void swap(tuple<Types...>& x, tuple<Types...>& y) noexcept(x.swap(y));
    // x.swap(y)
    template <
        BOOST_PP_ENUM_PARAMS(N, typename T)
    >
    BOOST_FORCEINLINE
    void swap(
        tuple<BOOST_PP_ENUM_PARAMS(N, T)>& x
      , tuple<BOOST_PP_ENUM_PARAMS(N, T)>& y
    )
        BOOST_NOEXCEPT_IF(BOOST_NOEXCEPT_EXPR((x.swap(y))))
    {
        x.swap(y);
    }
}}

#include <hpx/util/detail/fusion_adapt_tuple.hpp>

namespace boost { namespace serialization
{
    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        template <typename T>
        struct is_tuple_element_bitwise_serializable
          : boost::serialization::is_bitwise_serializable<T>
        {};

        template <>
        struct is_tuple_element_bitwise_serializable<void>
          : boost::mpl::true_
        {};
    }

#   define HPX_UTIL_TUPLE_IS_BITWISE_SERIALIZABLE(Z, N, D)                    \
     && detail::is_tuple_element_bitwise_serializable<BOOST_PP_CAT(T, N)>::value\
    /**/
    template <BOOST_PP_ENUM_PARAMS(N, typename T)>
    struct is_bitwise_serializable<
        ::hpx::util::tuple<BOOST_PP_ENUM_PARAMS(N, T)>
    > : boost::mpl::bool_<
            true BOOST_PP_REPEAT(N, HPX_UTIL_TUPLE_IS_BITWISE_SERIALIZABLE, _)
        >
    {};
#   undef HPX_UTIL_TUPLE_IS_BITWISE_SERIALIZABLE

    ///////////////////////////////////////////////////////////////////////////
    template <typename Archive, BOOST_PP_ENUM_PARAMS(N, typename T)>
    BOOST_FORCEINLINE
    void serialize(
        Archive& ar
      , ::hpx::util::tuple<BOOST_PP_ENUM_PARAMS(N, T)>& t
      , unsigned int const version
    )
    {
        ::hpx::util::serialize_sequence(ar, t);
    }
}}

#undef N

#   if !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)
#       include <hpx/util/preprocessed/tuple.hpp>
#   else
#       if defined(__WAVE__) && defined(HPX_CREATE_PREPROCESSED_FILES)
#           pragma wave option(preserve: 1, line: 0, output: "preprocessed/tuple_" HPX_LIMIT_STR ".hpp")
#       endif

        ///////////////////////////////////////////////////////////////////////
#       define BOOST_PP_ITERATION_PARAMS_1                                    \
        (                                                                     \
            3                                                                 \
          , (                                                                 \
                1                                                             \
              , HPX_TUPLE_LIMIT                                               \
              , <hpx/util/tuple.hpp>                                          \
            )                                                                 \
        )                                                                     \
        /**/
#       include BOOST_PP_ITERATE()

#       if defined(__WAVE__) && defined(HPX_CREATE_PREPROCESSED_FILES)
#           pragma wave option(output: null)
#       endif
#   endif // !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)

#endif

#else // !BOOST_PP_IS_ITERATING

#define N BOOST_PP_ITERATION()

namespace hpx { namespace util
{
    namespace detail
    {
        template <BOOST_PP_ENUM_PARAMS(N, typename T), typename UTuple>
        struct are_tuples_compatible<
            tuple<BOOST_PP_ENUM_PARAMS(N, T)>, UTuple
          , typename boost::enable_if_c<
                tuple_size<tuple<BOOST_PP_ENUM_PARAMS(N, T)> >::value == N
             && tuple_size<typename boost::remove_reference<UTuple>::type>::value == N
            >::type
        >
        {
            typedef char(&no_type)[1];
            typedef char(&yes_type)[2];

            static no_type call(...);
            static yes_type call(BOOST_PP_ENUM_PARAMS(N, T));

#           define HPX_UTIL_TUPLE_GET(Z, N, D)                                \
            util::get<N>(boost::declval<UTuple>())                            \
            /**/
            static bool const value =
                sizeof(
                    call(BOOST_PP_ENUM(N, HPX_UTIL_TUPLE_GET, _))
                ) == sizeof(yes_type);
#           undef HPX_UTIL_TUPLE_GET

            typedef boost::mpl::bool_<value> type;
        };
    }

    // 20.4.2, class template tuple
    template <BOOST_PP_ENUM_PARAMS(N, typename T)>
#   if N != HPX_TUPLE_LIMIT
    class tuple<BOOST_PP_ENUM_PARAMS(N, T)>
#   else
    class tuple
#   endif
    {
    public: // exposition-only
#       define HPX_UTIL_TUPLE_MEMBER(Z, N, D)                                 \
        detail::tuple_member<BOOST_PP_CAT(T, N)> BOOST_PP_CAT(_m, N);         \
        /**/
        BOOST_PP_REPEAT(N, HPX_UTIL_TUPLE_MEMBER, _);
#       undef HPX_UTIL_TUPLE_MEMBER

    public:
        // 20.4.2.1, tuple construction

        // constexpr tuple();
        // Value initializes each element.
#       define HPX_UTIL_TUPLE_DEFAULT_CONSTRUCT(Z, N, D)                      \
        BOOST_PP_CAT(_m, N)()                                                 \
        /**/
        BOOST_CONSTEXPR tuple()
          : BOOST_PP_ENUM(N, HPX_UTIL_TUPLE_DEFAULT_CONSTRUCT, _)
        {}
#       undef HPX_UTIL_TUPLE_DEFAULT_CONSTRUCT

        // explicit constexpr tuple(const Types&...);
        // Initializes each element with the value of the corresponding
        // parameter.
#       define HPX_UTIL_TUPLE_CONST_LVREF_PARAM(Z, N, D)                      \
        BOOST_PP_CAT(T, N) const& BOOST_PP_CAT(v, N)                          \
        /**/
#       define HPX_UTIL_TUPLE_COPY_CONSTRUCT(Z, N, D)                         \
        BOOST_PP_CAT(_m, N)(BOOST_PP_CAT(v, N))                               \
        /**/
        BOOST_CONSTEXPR explicit tuple(
            BOOST_PP_ENUM(N, HPX_UTIL_TUPLE_CONST_LVREF_PARAM, _)
        ) : BOOST_PP_ENUM(N, HPX_UTIL_TUPLE_COPY_CONSTRUCT, _)
        {}
#       undef HPX_UTIL_TUPLE_CONST_LVREF_PARAM
#       undef HPX_UTIL_TUPLE_COPY_CONSTRUCT

        // template <class... UTypes>
        // explicit constexpr tuple(UTypes&&... u);
        // Initializes the elements in the tuple with the corresponding value
        // in std::forward<UTypes>(u).
        // This constructor shall not participate in overload resolution
        // unless each type in UTypes is implicitly convertible to its
        // corresponding type in Types.
#       define HPX_UTIL_TUPLE_FWD_REF_PARAM(Z, N, D)                          \
        BOOST_PP_CAT(U, N &&) BOOST_PP_CAT(u, N)                  \
        /**/
#       define HPX_UTIL_TUPLE_FORWARD_CONSTRUCT(Z, N, D)                      \
        BOOST_PP_CAT(_m, N)                                                   \
            (std::forward<BOOST_PP_CAT(U, N)>(BOOST_PP_CAT(u, N)))          \
        /**/
        template <BOOST_PP_ENUM_PARAMS(N, typename U)>
        BOOST_CONSTEXPR explicit tuple(
            BOOST_PP_ENUM(N, HPX_UTIL_TUPLE_FWD_REF_PARAM, _)
          , typename boost::enable_if_c<
#       if N == 1
                boost::mpl::eval_if_c<
                    boost::is_base_of<
                        tuple, typename boost::remove_reference<U0>::type
                    >::value || detail::are_tuples_compatible<tuple, U0&&>::value
                  , boost::mpl::false_
                  , detail::are_tuples_compatible<
                        tuple
                      , tuple<BOOST_PP_ENUM_PARAMS(N, U)>&&
                    >
                >::type::value
#       else
                detail::are_tuples_compatible<
                    tuple
                  , tuple<BOOST_PP_ENUM_PARAMS(N, U)>&&
                >::value
#       endif
            >::type* = 0
        ) : BOOST_PP_ENUM(N, HPX_UTIL_TUPLE_FORWARD_CONSTRUCT, _)
        {}
#       undef HPX_UTIL_TUPLE_FWD_REF_PARAM
#       undef HPX_UTIL_TUPLE_FORWARD_CONSTRUCT

        // tuple(const tuple& u) = default;
        // Initializes each element of *this with the corresponding element
        // of u.
#       define HPX_UTIL_TUPLE_COPY_CONSTRUCT(Z, N, D)                         \
        BOOST_PP_CAT(_m, N)(BOOST_PP_CAT(other._m, N))                        \
        /**/
        BOOST_CONSTEXPR tuple(tuple const& other)
          : BOOST_PP_ENUM(N, HPX_UTIL_TUPLE_COPY_CONSTRUCT, _)
        {}
#       undef HPX_UTIL_TUPLE_COPY_CONSTRUCT

        // tuple(tuple&& u) = default;=
        // For all i, initializes the ith element of *this with
        // std::forward<Ti>(get<i>(u)).
#       define HPX_UTIL_TUPLE_MOVE_CONSTRUCT(Z, N, D)                         \
        BOOST_PP_CAT(_m, N)(std::move(BOOST_PP_CAT(other._m, N)))           \
        /**/
        BOOST_CONSTEXPR tuple(tuple && other)
          : BOOST_PP_ENUM(N, HPX_UTIL_TUPLE_MOVE_CONSTRUCT, _)
        {}
#       undef HPX_UTIL_TUPLE_MOVE_CONSTRUCT

        // template <class... UTypes> constexpr tuple(const tuple<UTypes...>& u);
        // template <class... UTypes> constexpr tuple(tuple<UTypes...>&& u);
        // For all i, initializes the ith element of *this with
        // get<i>(std::forward<U>(u).
        // This constructor shall not participate in overload resolution
        // unless each type in UTypes is implicitly convertible to its
        // corresponding type in Types
#       define HPX_UTIL_TUPLE_GET_CONSTRUCT(Z, N, D)                          \
        BOOST_PP_CAT(_m, N)(util::get<N>(std::forward<UTuple>(other)))      \
        /**/
        template <typename UTuple>
        BOOST_CONSTEXPR tuple(
            UTuple && other
          , typename boost::enable_if_c<
                detail::are_tuples_compatible<tuple, UTuple&&>::value
            >::type* = 0
        ) : BOOST_PP_ENUM(N, HPX_UTIL_TUPLE_GET_CONSTRUCT, _)
        {}
#       undef HPX_UTIL_TUPLE_GET_CONSTRUCT

        // 20.4.2.2, tuple assignment

        // tuple& operator=(const tuple& u);
        // Assigns each element of u to the corresponding element of *this.
#       define HPX_UTIL_TUPLE_COPY_ASSIGN_NOEXCEPT(Z, N, D)                   \
         && BOOST_NOEXCEPT_EXPR((                                             \
                BOOST_PP_CAT(_m, N)._value =                                  \
                    BOOST_PP_CAT(other._m, N)._value                          \
            ))                                                                \
        /**/
#       define HPX_UTIL_TUPLE_COPY_ASSIGN(Z, N, D)                            \
        BOOST_PP_CAT(_m, N)._value =                                          \
            BOOST_PP_CAT(other._m, N)._value;                                 \
        /**/
        tuple& operator=(tuple const& other)
            HPX_UTIL_TUPLE_SFINAE_NOEXCEPT_IF(
                true BOOST_PP_REPEAT(N, HPX_UTIL_TUPLE_COPY_ASSIGN_NOEXCEPT, _)
            )
        {
            BOOST_PP_REPEAT(N, HPX_UTIL_TUPLE_COPY_ASSIGN, _);
            return *this;
        }
#       undef HPX_UTIL_TUPLE_COPY_ASSIGN_NOEXCEPT
#       undef HPX_UTIL_TUPLE_COPY_ASSIGN

        // tuple& operator=(tuple&& u) noexcept(see below);
        // For all i, assigns std::forward<Ti>(get<i>(u)) to get<i>(*this).
#       define HPX_UTIL_TUPLE_MOVE_ASSIGN_NOEXCEPT(Z, N, D)                   \
         && BOOST_NOEXCEPT_EXPR((                                             \
                BOOST_PP_CAT(_m, N)._value =                                  \
                    std::forward<BOOST_PP_CAT(T, N)>                        \
                        (BOOST_PP_CAT(other._m, N)._value)                    \
            ))                                                                \
        /**/
#       define HPX_UTIL_TUPLE_MOVE_ASSIGN(Z, N, D)                            \
        BOOST_PP_CAT(_m, N)._value =                                          \
            std::forward<BOOST_PP_CAT(T, N)>                                \
                (BOOST_PP_CAT(other._m, N)._value);                           \
        /**/
        tuple& operator=(tuple && other)
            HPX_UTIL_TUPLE_SFINAE_NOEXCEPT_IF(
                true BOOST_PP_REPEAT(N, HPX_UTIL_TUPLE_MOVE_ASSIGN_NOEXCEPT, _)
            )
        {
            BOOST_PP_REPEAT(N, HPX_UTIL_TUPLE_MOVE_ASSIGN, _);
            return *this;
        }
#       undef HPX_UTIL_TUPLE_MOVE_ASSIGN_NOEXCEPT
#       undef HPX_UTIL_TUPLE_MOVE_ASSIGN

        // template <class... UTypes> tuple& operator=(const tuple<UTypes...>& u);
        // template <class... UTypes> tuple& operator=(tuple<UTypes...>&& u);
        // For all i, assigns get<i>(std::forward<U>(u)) to get<i>(*this).
#       define HPX_UTIL_TUPLE_GET_ASSIGN_NOEXCEPT(Z, N, D)                    \
         && BOOST_NOEXCEPT_EXPR((                                             \
                BOOST_PP_CAT(_m, N)._value =                                  \
                    util::get<N>(std::forward<UTuple>(other))               \
            ))                                                                \
        /**/
#       define HPX_UTIL_TUPLE_GET_ASSIGN(Z, N, D)                             \
        BOOST_PP_CAT(_m, N)._value =                                          \
            util::get<N>(std::forward<UTuple>(other));                      \
        /**/
        template <typename UTuple>
        typename boost::enable_if_c<
            tuple_size<typename boost::remove_reference<UTuple>::type>::value == N
          , tuple&
        >::type
        operator=(UTuple && other)
            HPX_UTIL_TUPLE_SFINAE_NOEXCEPT_IF(
                true BOOST_PP_REPEAT(N, HPX_UTIL_TUPLE_GET_ASSIGN_NOEXCEPT, _)
            )
        {
            BOOST_PP_REPEAT(N, HPX_UTIL_TUPLE_GET_ASSIGN, _);
            return *this;
        }
#       undef HPX_UTIL_TUPLE_GET_ASSIGN_NOEXCEPT
#       undef HPX_UTIL_TUPLE_GET_ASSIGN

        // 20.4.2.3, tuple swap

        // void swap(tuple& rhs) noexcept(see below );
        // Calls swap for each element in *this and its corresponding element
        // in rhs.
#       define HPX_UTIL_TUPLE_SWAP_NOEXCEPT(Z, N, D)                          \
         && BOOST_NOEXCEPT_EXPR((                                             \
                boost::swap(                                                  \
                    BOOST_PP_CAT(_m, N)._value                                \
                  , BOOST_PP_CAT(other._m, N)._value)                         \
                ))                                                            \
        /**/
#       define HPX_UTIL_TUPLE_SWAP(Z, N, D)                                   \
        boost::swap(                                                          \
            BOOST_PP_CAT(_m, N)._value                                        \
          , BOOST_PP_CAT(other._m, N)._value                                  \
        );                                                                    \
        /**/
        void swap(tuple& other)
            BOOST_NOEXCEPT_IF(
                true BOOST_PP_REPEAT(N, HPX_UTIL_TUPLE_SWAP_NOEXCEPT, _)
            )
        {
            BOOST_PP_REPEAT(N, HPX_UTIL_TUPLE_SWAP, _);
        }
#       undef HPX_UTIL_TUPLE_SWAP_NOEXCEPT
#       undef HPX_UTIL_TUPLE_SWAP
    };

    // 20.4.2.5, tuple helper classes

    // template <class Tuple>
    // class tuple_size
    template <BOOST_PP_ENUM_PARAMS(N, typename T)>
    struct tuple_size<tuple<BOOST_PP_ENUM_PARAMS(N, T)> >
      : boost::mpl::size_t<N>
    {};

    // template <size_t I, class Tuple>
    // class tuple_element
    template <BOOST_PP_ENUM_PARAMS(HPX_TUPLE_LIMIT, typename T)>
    struct tuple_element<
        BOOST_PP_DEC(N)
      , tuple<BOOST_PP_ENUM_PARAMS(HPX_TUPLE_LIMIT, T)>
    > : boost::mpl::identity<BOOST_PP_CAT(T, BOOST_PP_DEC(N))>
    {
        template <typename Tuple>
        static BOOST_CONSTEXPR BOOST_FORCEINLINE
        typename detail::qualify_as<
            BOOST_PP_CAT(T, BOOST_PP_DEC(N))
          , Tuple&
        >::type
        get(Tuple& tuple) BOOST_NOEXCEPT
        {
            return tuple.BOOST_PP_CAT(_m, BOOST_PP_DEC(N))._value;
        }
    };

    // 20.4.2.4, tuple creation functions

    // template<class... Types>
    // constexpr tuple<VTypes...> make_tuple(Types&&... t);
#   define HPX_UTIL_TUPLE_MAKE_ELEMENT(Z, N, D)                               \
    typename detail::make_tuple_element<BOOST_PP_CAT(T, N)>::type             \
    /**/
    template <BOOST_PP_ENUM_PARAMS(N, typename T)>
    BOOST_CONSTEXPR BOOST_FORCEINLINE
    tuple<BOOST_PP_ENUM(N, HPX_UTIL_TUPLE_MAKE_ELEMENT, _)>
    make_tuple(HPX_ENUM_FWD_ARGS(N, T, v))
    {
        return
            tuple<BOOST_PP_ENUM(N, HPX_UTIL_TUPLE_MAKE_ELEMENT, _)>(
                HPX_ENUM_FORWARD_ARGS(N, T, v)
            );
    }
#   undef HPX_UTIL_TUPLE_MAKE_ELEMENT

    // template<class... Types>
    // tuple<Types&&...> forward_as_tuple(Types&&... t) noexcept;
    // Constructs a tuple of references to the arguments in t suitable for
    // forwarding as arguments to a function. Because the result may contain
    // references to temporary variables, a program shall ensure that the
    // return value of this function does not outlive any of its arguments.
#   define HPX_UTIL_FORWARD_AS_TUPLE_ELEMENT(Z, N, D)                         \
    BOOST_PP_CAT(T, N) &&                                                     \
    /**/
    template <BOOST_PP_ENUM_PARAMS(N, typename T)>
    BOOST_FORCEINLINE
    tuple<BOOST_PP_ENUM(N, HPX_UTIL_FORWARD_AS_TUPLE_ELEMENT, _)>
    forward_as_tuple(HPX_ENUM_FWD_ARGS(N, T, v)) BOOST_NOEXCEPT
    {
        return
            tuple<BOOST_PP_ENUM(N, HPX_UTIL_FORWARD_AS_TUPLE_ELEMENT, _)>(
                HPX_ENUM_FORWARD_ARGS(N, T, v)
            );
    }
#   undef HPX_UTIL_FORWARD_AS_TUPLE_ELEMENT

    // template<class... Types>
    // tuple<Types&...> tie(Types&... t) noexcept;
#   define HPX_UTIL_TIE_ELEMENT(Z, N, D)                                      \
    BOOST_PP_CAT(T, N) &                                                      \
    /**/
    template <BOOST_PP_ENUM_PARAMS(N, typename T)>
    BOOST_FORCEINLINE
    tuple<BOOST_PP_ENUM(N, HPX_UTIL_TIE_ELEMENT, _)>
    tie(BOOST_PP_ENUM_BINARY_PARAMS(N, T, & v)) BOOST_NOEXCEPT
    {
        return
            tuple<BOOST_PP_ENUM(N, HPX_UTIL_TIE_ELEMENT, _)>(
                BOOST_PP_ENUM_PARAMS(N, v)
            );
    }
#   undef HPX_UTIL_TIE_ELEMENT

    //template <class... Tuples>
    //constexpr tuple<Ctypes ...> tuple_cat(Tuples&&...);
    namespace detail
    {
#       if N > 2
#       define HPX_UTIL_TUPLE_CAT_RESULT(Z, N, D)                             \
        BOOST_PP_COMMA_IF(N) typename tuple_cat_result<                       \
            BOOST_PP_CAT(T, BOOST_PP_MUL(N, 2))                               \
          , BOOST_PP_CAT(T, BOOST_PP_INC(BOOST_PP_MUL(N, 2)))                 \
        >::type                                                               \
        /**/
        template <BOOST_PP_ENUM_PARAMS(N, typename T)>
        struct tuple_cat_result<BOOST_PP_ENUM_PARAMS(N, T)>
          : tuple_cat_result<
                BOOST_PP_REPEAT(BOOST_PP_DIV(N, 2), HPX_UTIL_TUPLE_CAT_RESULT, _)
#           if N % 2 != 0
              , BOOST_PP_CAT(T, BOOST_PP_DEC(N))
#           endif
            >
        {};
#       undef HPX_UTIL_TUPLE_CAT_RESULT
#       endif

#       define HPX_UTIL_TUPLE_CAT_ELEMENT_TYPE(Z, N, D)                       \
        typename tuple_element<N, Tuple>::type                                \
        /**/
        template <typename Tuple>
        struct tuple_cat_result<
            Tuple
          , typename boost::enable_if_c<tuple_size<Tuple>::value == N>::type
        >
        {
            typedef
                tuple<BOOST_PP_ENUM(N, HPX_UTIL_TUPLE_CAT_ELEMENT_TYPE, _)>
                type;
        };
#       undef HPX_UTIL_TUPLE_CAT_ELEMENT_TYPE
    }

#   define HPX_UTIL_TUPLE_CAT_ELEMENT_CALL(Z, N, D)                           \
    util::get<N>(std::forward<Tuple>(t))                                      \
    /**/
    template <typename Tuple>
    BOOST_CONSTEXPR BOOST_FORCEINLINE
    typename boost::lazy_enable_if_c<
        tuple_size<typename boost::remove_reference<Tuple>::type>::value == N
      , detail::tuple_cat_result<
            typename boost::remove_reference<Tuple>::type
        >
    >::type
    tuple_cat(Tuple && t)
    {
        return
            typename detail::tuple_cat_result<
                typename boost::remove_reference<Tuple>::type
            >::type(
                BOOST_PP_ENUM(N, HPX_UTIL_TUPLE_CAT_ELEMENT_CALL, _)
            );
    }
#   undef HPX_UTIL_TUPLE_CAT_ELEMENT_CALL

#   define HPX_UTIL_TUPLE_CAT_ELEMENT_CALL(Z, N, D)                           \
    detail::tuple_cat_element<                                                \
        N                                                                     \
      , typename boost::remove_reference<TTuple>::type                        \
      , typename boost::remove_reference<UTuple>::type                        \
    >::call(std::forward<TTuple>(t), std::forward<UTuple>(u))                 \
    /**/
    template <typename TTuple, typename UTuple>
    BOOST_CONSTEXPR BOOST_FORCEINLINE
    typename boost::lazy_enable_if_c<
        tuple_size<typename boost::remove_reference<TTuple>::type>::value
      + tuple_size<typename boost::remove_reference<UTuple>::type>::value == N
      , detail::tuple_cat_result<
            typename boost::remove_reference<TTuple>::type
          , typename boost::remove_reference<UTuple>::type
        >
    >::type
    tuple_cat(TTuple && t, UTuple && u)
    {
        return
            typename detail::tuple_cat_result<
                typename boost::remove_reference<TTuple>::type
              , typename boost::remove_reference<UTuple>::type
            >::type(
                BOOST_PP_ENUM(N, HPX_UTIL_TUPLE_CAT_ELEMENT_CALL, _)
            );
    }
#   undef HPX_UTIL_TUPLE_CAT_ELEMENT_CALL

#   if N > 2
#   define HPX_UTIL_TUPLE_CAT_RESULT_ELEMENT(Z, N, D)                         \
    typename boost::remove_reference<BOOST_PP_CAT(T, N)>::type                \
    /**/
#   define HPX_UTIL_TUPLE_CAT(Z, N, D)                                        \
    BOOST_PP_COMMA_IF(N) util::tuple_cat(                                     \
        std::forward<BOOST_PP_CAT(T, BOOST_PP_MUL(N, 2))>                     \
            (BOOST_PP_CAT(t, BOOST_PP_MUL(N, 2)))                             \
      , std::forward<BOOST_PP_CAT(T, BOOST_PP_INC(BOOST_PP_MUL(N, 2)))>       \
            (BOOST_PP_CAT(t, BOOST_PP_INC(BOOST_PP_MUL(N, 2)))))              \
    /**/
    template <BOOST_PP_ENUM_PARAMS(N, typename T)>
    BOOST_CONSTEXPR BOOST_FORCEINLINE
    typename detail::tuple_cat_result<
        BOOST_PP_ENUM(N, HPX_UTIL_TUPLE_CAT_RESULT_ELEMENT, _)
    >::type
    tuple_cat(HPX_ENUM_FWD_ARGS(N, T, t))
    {
        return
            util::tuple_cat(
                BOOST_PP_REPEAT(BOOST_PP_DIV(N, 2), HPX_UTIL_TUPLE_CAT, _)
#           if N % 2 != 0
              , std::forward<BOOST_PP_CAT(T, BOOST_PP_DEC(N))>
                    (BOOST_PP_CAT(t, BOOST_PP_DEC(N)))
#           endif
            );
    }
#   undef HPX_UTIL_TUPLE_CAT_RESULT_ELEMENT
#   undef HPX_UTIL_TUPLE_CAT
#   endif
}}

#undef N

#endif
