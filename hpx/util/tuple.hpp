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
#include <hpx/util/detail/pack.hpp>
#include <hpx/util/detail/qualify_as.hpp>

#include <boost/array.hpp>
#include <boost/fusion/adapted/struct/adapt_struct.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/mpl/if.hpp>
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
#include <boost/preprocessor/repetition/repeat.hpp>
#include <boost/ref.hpp>
#include <boost/serialization/is_bitwise_serializable.hpp>
#include <boost/type_traits/add_const.hpp>
#include <boost/type_traits/add_cv.hpp>
#include <boost/type_traits/add_volatile.hpp>
#include <boost/type_traits/is_fundamental.hpp>
#include <boost/type_traits/is_pointer.hpp>
#include <boost/type_traits/is_reference.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/type_traits/remove_cv.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/utility/swap.hpp>

#include <cstddef> // for size_t
#include <utility>

#if defined(BOOST_NO_SFINAE_EXPR) ||                                          \
    (defined(HPX_GCC_VERSION) && HPX_GCC_VERSION < 40800)
#   define HPX_UTIL_TUPLE_SFINAE_NOEXCEPT_IF(Predicate)
#else
#   define HPX_UTIL_TUPLE_SFINAE_NOEXCEPT_IF(Predicate)                       \
    BOOST_NOEXCEPT_IF(Predicate)                                              \
    /**/
#endif

namespace hpx { namespace util
{
    template <typename ...Ts>
    class tuple;

    template <class T>
    struct tuple_size; // undefined

    template <std::size_t I, typename T>
    struct tuple_element; // undefined
    
    template <std::size_t I, typename Tuple>
    BOOST_CONSTEXPR BOOST_FORCEINLINE
    typename detail::qualify_as<
        typename tuple_element<I, Tuple>::type
      , Tuple&
    >::type
    get(Tuple& t) BOOST_NOEXCEPT;

    template <std::size_t I, typename Tuple>
    BOOST_CONSTEXPR BOOST_FORCEINLINE
    typename detail::qualify_as<
        typename tuple_element<I, Tuple>::type
      , Tuple const&
    >::type
    get(Tuple const& t) BOOST_NOEXCEPT;

    template <std::size_t I, typename Tuple>
    BOOST_CONSTEXPR BOOST_FORCEINLINE
    typename detail::qualify_as<
        typename tuple_element<I, Tuple>::type
      , Tuple &&
    >::type
    get(Tuple && t) BOOST_NOEXCEPT;

    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        template <typename T, typename Enable = void>
        struct tuple_member //-V690
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
        template <typename Indices, typename TTuple, typename UTuple>
        struct are_tuples_compatible_impl;

        template <std::size_t ...Is, typename ...Ts, typename UTuple>
        struct are_tuples_compatible_impl<
            detail::pack_c<std::size_t, Is...>, tuple<Ts...>, UTuple
        >
        {
            typedef char(&no_type)[1];
            typedef char(&yes_type)[2];

            static no_type call(...);
            static yes_type call(Ts...);

            static bool const value =
                sizeof(
                    call(util::get<Is>(boost::declval<UTuple>())...)
                ) == sizeof(yes_type);

            typedef boost::mpl::bool_<value> type;
            int m_;
        };

        template <typename TTuple, typename UTuple, typename Enable = void>
        struct are_tuples_compatible
          : boost::mpl::false_
        {};

        template <typename ...Ts, typename UTuple>
        struct are_tuples_compatible<
            tuple<Ts...>, UTuple
          , typename boost::enable_if_c<
                tuple_size<
                    typename boost::remove_reference<UTuple>::type
                >::value == tuple_size<tuple<Ts...> >::value
            >::type
        > : are_tuples_compatible_impl<
                typename detail::make_index_pack<sizeof...(Ts)>::type
              , tuple<Ts...>, UTuple
            >
        {};

        template <typename TTuple, typename UTuple>
        struct are_tuples_compatible_not_same
          : boost::mpl::if_c<
                boost::is_same<
                    typename decay<TTuple>::type, typename decay<UTuple>::type
                >::value
              , boost::mpl::false_
              , are_tuples_compatible<TTuple, UTuple>
            >::type
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

    template <typename ...Ts>
    struct tuple_size<tuple<Ts...> >
      : boost::mpl::size_t<sizeof...(Ts)>
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

    template <typename ...Ts>
    struct tuple_decay<tuple<Ts...> >
    {
        typedef tuple<typename decay<Ts>::type...> type;
    };

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
    template <typename ...Ts>
    BOOST_CONSTEXPR BOOST_FORCEINLINE
    tuple<typename decay_unwrap<Ts>::type...>
    make_tuple(Ts&&... vs)
    {
        return tuple<typename decay_unwrap<Ts>::type...>(
            std::forward<Ts>(vs)...);
    }

    // template<class... Types>
    // tuple<Types&&...> forward_as_tuple(Types&&... t) noexcept;
    // Constructs a tuple of references to the arguments in t suitable for
    // forwarding as arguments to a function. Because the result may contain
    // references to temporary variables, a program shall ensure that the
    // return value of this function does not outlive any of its arguments.
    template <typename ...Ts>
    BOOST_FORCEINLINE
    tuple<Ts&&...>
    forward_as_tuple(Ts&&... vs) BOOST_NOEXCEPT
    {
        return tuple<Ts&&...>(std::forward<Ts>(vs)...);
    }

    // template<class... Types>
    // tuple<Types&...> tie(Types&... t) noexcept;
    template <typename ...Ts>
    BOOST_FORCEINLINE
    tuple<Ts&...>
    tie(Ts&... vs) BOOST_NOEXCEPT
    {
        return tuple<Ts&...>(vs...);
    }

    //template <class... Tuples>
    //constexpr tuple<Ctypes ...> tuple_cat(Tuples&&...);
    namespace detail
    {
        template <std::size_t Size, typename Tuples>
        struct tuple_cat_size_impl;

        template <std::size_t Size>
        struct tuple_cat_size_impl<Size, detail::pack<> >
          : boost::mpl::size_t<Size>
        {};

        template <std::size_t Size, typename HeadTuple, typename ...TailTuples>
        struct tuple_cat_size_impl<
            Size, detail::pack<HeadTuple, TailTuples...>
        > : tuple_cat_size_impl<
                (Size + tuple_size<HeadTuple>::value), detail::pack<TailTuples...>
            >
        {};

        template <typename ...Tuples>
        struct tuple_cat_size
          : tuple_cat_size_impl<0, detail::pack<Tuples...> >
        {};

        ///////////////////////////////////////////////////////////////////////
        template <std::size_t I, typename Tuples, typename Enable = void>
        struct tuple_cat_element;

        template <std::size_t I, typename HeadTuple, typename ...TailTuples>
        struct tuple_cat_element<
            I, detail::pack<HeadTuple, TailTuples...>
          , typename boost::enable_if_c<
                (I < tuple_size<HeadTuple>::value)
            >::type
        > : tuple_element<I, HeadTuple>
        {
            typedef tuple_element<I, HeadTuple> base_type;

            template <typename HeadTuple_, typename ...TailTuples_>
            static BOOST_CONSTEXPR BOOST_FORCEINLINE
            typename detail::qualify_as<
                typename base_type::type
              , HeadTuple_&
            >::type
            get(HeadTuple_& head, TailTuples_& ...tail) BOOST_NOEXCEPT
            {
                return base_type::get(head);
            }
        };

        template <std::size_t I, typename HeadTuple, typename ...TailTuples>
        struct tuple_cat_element<
            I, detail::pack<HeadTuple, TailTuples...>
          , typename boost::enable_if_c<
                (I >= tuple_size<HeadTuple>::value)
            >::type
        > : tuple_cat_element<
                I - tuple_size<HeadTuple>::value
              , detail::pack<TailTuples...>
            >
        {
            typedef tuple_cat_element<
                I - tuple_size<HeadTuple>::value
              , detail::pack<TailTuples...>
            > base_type;

            template <typename HeadTuple_, typename ...TailTuples_>
            static BOOST_CONSTEXPR BOOST_FORCEINLINE
            typename detail::qualify_as<
                typename base_type::type
              , HeadTuple_&
            >::type
            get(HeadTuple_& head, TailTuples_& ...tail) BOOST_NOEXCEPT
            {
                return base_type::get(tail...);
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Indices, typename Tuples>
        struct tuple_cat_result_impl;

        template <std::size_t ...Is, typename ...Tuples>
        struct tuple_cat_result_impl<
            detail::pack_c<std::size_t, Is...>, detail::pack<Tuples...>
        >
        {
            typedef tuple<
                typename tuple_cat_element<Is, detail::pack<Tuples...> >::type...
            > type;

            template <typename ...Tuples_>
            static BOOST_CONSTEXPR BOOST_FORCEINLINE
            type make(Tuples_&&... tuples)
            {
                return type(tuple_cat_element<Is, detail::pack<Tuples...> >::get(
                    std::forward<Tuples_>(tuples)...)...);
            }
        };

        template <typename ...Tuples>
        struct tuple_cat_result
          : tuple_cat_result_impl<
                typename make_index_pack<tuple_cat_size<Tuples...>::value>::type
              , detail::pack<Tuples...>
            >
        {};
    }

    template <typename ...Tuples>
    BOOST_CONSTEXPR BOOST_FORCEINLINE
    typename detail::tuple_cat_result<Tuples...>::type
    tuple_cat(Tuples&&... tuples)
    {
        return detail::tuple_cat_result<Tuples...>::make(
            std::forward<Tuples>(tuples)...);
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

    template <typename ...Ts, typename ...Us>
    BOOST_CONSTEXPR BOOST_FORCEINLINE
    typename boost::enable_if_c<sizeof...(Ts) == sizeof...(Us), bool>::type
    operator==(tuple<Ts...> const& t, tuple<Us...> const& u)
    {
        return detail::tuple_equal_to<0, sizeof...(Ts)>::call(t, u);
    }

    // template<class... TTypes, class... UTypes>
    // constexpr bool operator!=
    //     (const tuple<TTypes...>& t, const tuple<UTypes...>& u);
    template <typename ...Ts, typename ...Us>
    BOOST_CONSTEXPR BOOST_FORCEINLINE
    typename boost::enable_if_c<sizeof...(Ts) == sizeof...(Us), bool>::type
    operator!=(tuple<Ts...> const& t, tuple<Us...> const& u)
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

    template <typename ...Ts, typename ...Us>
    BOOST_CONSTEXPR BOOST_FORCEINLINE
    typename boost::enable_if_c<sizeof...(Ts) == sizeof...(Us), bool>::type
    operator<(tuple<Ts...> const& t, tuple<Us...> const& u)
    {
        return detail::tuple_less_than<0, sizeof...(Ts)>::call(t, u);
    }

    // template<class... TTypes, class... UTypes>
    // constexpr bool operator>
    //     (const tuple<TTypes...>& t, const tuple<UTypes...>& u);
    template <typename ...Ts, typename ...Us>
    BOOST_CONSTEXPR BOOST_FORCEINLINE
    typename boost::enable_if_c<sizeof...(Ts) == sizeof...(Us), bool>::type
    operator>(tuple<Ts...> const& t, tuple<Us...> const& u)
    {
        return u < t;
    }

    // template<class... TTypes, class... UTypes>
    // constexpr bool operator<=
    //     (const tuple<TTypes...>& t, const tuple<UTypes...>& u);
    template <typename ...Ts, typename ...Us>
    BOOST_CONSTEXPR BOOST_FORCEINLINE
    typename boost::enable_if_c<sizeof...(Ts) == sizeof...(Us), bool>::type
    operator<=(tuple<Ts...> const& t, tuple<Us...> const& u)
    {
        return !(u < t);
    }

    // template<class... TTypes, class... UTypes>
    // constexpr bool operator>=
    //     (const tuple<TTypes...>& t, const tuple<UTypes...>& u);
    template <typename ...Ts, typename ...Us>
    BOOST_CONSTEXPR BOOST_FORCEINLINE
    typename boost::enable_if_c<sizeof...(Ts) == sizeof...(Us), bool>::type
    operator>=(tuple<Ts...> const& t, tuple<Us...> const& u)
    {
        return !(t < u);
    }

    // 20.4.2.9, specialized algorithms

    // template <class... Types>
    // void swap(tuple<Types...>& x, tuple<Types...>& y) noexcept(x.swap(y));
    // x.swap(y)
    template <typename ...Ts>
    BOOST_FORCEINLINE
    void swap(tuple<Ts...>& x, tuple<Ts...>& y)
        BOOST_NOEXCEPT_IF(BOOST_NOEXCEPT_EXPR((x.swap(y))))
    {
        x.swap(y);
    }
}}

#include <hpx/util/detail/fusion_adapt_tuple.hpp>

namespace boost { namespace serialization
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename ...Ts>
    struct is_bitwise_serializable<
        ::hpx::util::tuple<Ts...>
    > : ::hpx::util::detail::all_of< ::hpx::util::detail::pack<
            boost::serialization::is_bitwise_serializable<Ts>...
        > >
    {};

    ///////////////////////////////////////////////////////////////////////////
    template <typename Archive, typename ...Ts>
    BOOST_FORCEINLINE
    void serialize(
        Archive& ar
      , ::hpx::util::tuple<Ts...>& t
      , unsigned int const version
    )
    {
        ::hpx::util::serialize_sequence(ar, t);
    }
}}

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
    // 20.4.2, class template tuple
    template <BOOST_PP_ENUM_PARAMS(N, typename T)>
    class tuple<BOOST_PP_ENUM_PARAMS(N, T)>
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
                    boost::is_same<
                        tuple, typename boost::remove_reference<U0>::type
                    >::value || detail::are_tuples_compatible_not_same<
                        tuple, U0&&
                    >::value
                  , boost::mpl::false_
                  , detail::are_tuples_compatible<tuple, tuple<U0>&&>
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
                detail::are_tuples_compatible_not_same<tuple, UTuple&&>::value
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

    // template <size_t I, class Tuple>
    // class tuple_element
    template <BOOST_PP_ENUM_PARAMS(N, typename T), typename ...Tail>
    struct tuple_element<
        BOOST_PP_DEC(N)
      , tuple<BOOST_PP_ENUM_PARAMS(N, T), Tail...>
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
}}

#undef N

#endif
