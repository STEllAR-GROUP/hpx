//  Copyright (c) 2011-2013 Thomas Heller
//  Copyright (c) 2011-2013 Hartmut Kaiser
//  Copyright (c) 2013-2015 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_TUPLE_HPP
#define HPX_UTIL_TUPLE_HPP

#include <hpx/config.hpp>
#include <hpx/traits/is_bitwise_serializable.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/detail/pack.hpp>

#include <boost/array.hpp>
#include <boost/type_traits/integral_constant.hpp>

#include <algorithm>
#include <cstddef> // for size_t
#include <type_traits>
#include <utility>

#if defined(HPX_MSVC)
#pragma warning(push)
#pragma warning(disable: 4520) // multiple default constructors specified
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
    HPX_CONSTEXPR HPX_FORCEINLINE
    typename tuple_element<I, Tuple>::type&
    get(Tuple& t) HPX_NOEXCEPT;

    template <std::size_t I, typename Tuple>
    HPX_CONSTEXPR HPX_FORCEINLINE
    typename tuple_element<I, Tuple>::type const&
    get(Tuple const& t) HPX_NOEXCEPT;

    template <std::size_t I, typename Tuple>
    HPX_CONSTEXPR HPX_FORCEINLINE
    typename tuple_element<I, Tuple>::type&&
    get(Tuple&& t) HPX_NOEXCEPT;

    template <std::size_t I, typename Tuple>
    HPX_CONSTEXPR HPX_FORCEINLINE
    typename tuple_element<I, Tuple>::type const&&
    get(Tuple const&& t) HPX_NOEXCEPT;

    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        template <std::size_t I, typename T, typename Enable = void>
        struct tuple_member //-V690
        {
        public:
            HPX_CONSTEXPR tuple_member()
              : _value()
            {}

            template <typename U>
            explicit HPX_CONSTEXPR tuple_member(U&& value)
              : _value(std::forward<U>(value))
            {}

#if defined(HPX_HAVE_CXX11_DEFAULTED_FUNCTIONS)
            tuple_member(tuple_member const&) = default;
            tuple_member(tuple_member&&) = default;
#else
            HPX_CONSTEXPR tuple_member(tuple_member const& other)
              : _value(other.value())
            {}

            HPX_CONSTEXPR tuple_member(tuple_member&& other)
              : _value(std::forward<T>(other.value()))
            {}
#endif

            T& value() HPX_NOEXCEPT { return _value; }
            T const& value() const HPX_NOEXCEPT { return _value; }

        private:
            T _value;
        };

        template <std::size_t I, typename T>
        struct tuple_member<I, T,
            typename std::enable_if<
                std::is_empty<T>::value
#if defined(HPX_HAVE_CXX11_STD_IS_FINAL)
             && !std::is_final<T>::value
#endif
            >::type
        > : T
        {
        public:
            HPX_CONSTEXPR tuple_member()
              : T()
            {}

            template <typename U>
            explicit HPX_CONSTEXPR tuple_member(U&& value)
              : T(std::forward<U>(value))
            {}

#if defined(HPX_HAVE_CXX11_DEFAULTED_FUNCTIONS)
            tuple_member(tuple_member const&) = default;
            tuple_member(tuple_member&&) = default;
#else
            HPX_CONSTEXPR tuple_member(tuple_member const& other)
              : T(other.value())
            {}

            HPX_CONSTEXPR tuple_member(tuple_member&& other)
              : T(std::forward<T>(other.value()))
            {}
#endif

            T& value() HPX_NOEXCEPT { return *this; }
            T const& value() const HPX_NOEXCEPT { return *this; }
        };

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
                    call(util::get<Is>(std::declval<UTuple>())...)
                ) == sizeof(yes_type);

            typedef std::integral_constant<bool, value> type;
        };

        template <typename TTuple, typename UTuple, typename Enable = void>
        struct are_tuples_compatible
          : std::false_type
        {};

        template <typename ...Ts, typename UTuple>
        struct are_tuples_compatible<
            tuple<Ts...>, UTuple
          , typename std::enable_if<
                tuple_size<
                    typename std::remove_reference<UTuple>::type
                >::value == detail::pack<Ts...>::size
            >::type
        > : are_tuples_compatible_impl<
                typename detail::make_index_pack<sizeof...(Ts)>::type
              , tuple<Ts...>, UTuple
            >
        {};

        template <typename TTuple, typename UTuple>
        struct are_tuples_compatible_not_same
          : std::conditional<
                std::is_same<
                    typename std::decay<TTuple>::type, typename std::decay<UTuple>::type
                >::value
              , std::false_type
              , are_tuples_compatible<TTuple, UTuple>
            >::type
        {};

        ///////////////////////////////////////////////////////////////////////
        template <typename Is, typename ...Ts>
        struct tuple_impl;

        template <std::size_t ...Is, typename ...Ts>
        struct tuple_impl<detail::pack_c<std::size_t, Is...>, Ts...>
          : tuple_member<Is, Ts>...
        {
            // 20.4.2.1, tuple construction
            HPX_CONSTEXPR tuple_impl()
              : tuple_member<Is, Ts>()...
            {}

            template <typename ...Us, typename Enable =
                typename std::enable_if<
                    detail::pack<Us...>::size == detail::pack<Ts...>::size
                >::type>
            explicit HPX_CONSTEXPR tuple_impl(Us&&... vs)
              : tuple_member<Is, Ts>(std::forward<Us>(vs))...
            {}

#if defined(HPX_HAVE_CXX11_DEFAULTED_FUNCTIONS)
            tuple_impl(tuple_impl const&) = default;
            tuple_impl(tuple_impl&&) = default;
#else
            HPX_CONSTEXPR tuple_impl(tuple_impl const& other)
              : tuple_member<Is, Ts>(static_cast<tuple_member<Is, Ts> const&>(other))...
            {}

            HPX_CONSTEXPR tuple_impl(tuple_impl&& other)
              : tuple_member<Is, Ts>(static_cast<tuple_member<Is, Ts>&&>(other))...
            {}
#endif

            template <typename UTuple, typename Enable =
                typename std::enable_if<
                    are_tuples_compatible_not_same<tuple<Ts...>, UTuple&&>::value
                >::type>
            HPX_CONSTEXPR tuple_impl(UTuple&& other)
              : tuple_member<Is, Ts>(util::get<Is>(std::forward<UTuple>(other)))...
            {}

            tuple_impl& operator=(tuple_impl const& other)
            {
                int const _sequencer[]= {
                    ((this->get<Is>() = other.template get<Is>()), 0)...
                };
                (void)_sequencer;
                return *this;
            }

            tuple_impl& operator=(tuple_impl&& other)
            {
                int const _sequencer[]= {
                    ((this->get<Is>() = other.template get<Is>()), 0)...
                };
                (void)_sequencer;
                return *this;
            }

            template <typename UTuple>
            tuple_impl& operator=(UTuple&& other)
            {
                int const _sequencer[]= {
                    ((this->get<Is>() = util::get<Is>(other)), 0)...
                };
                (void)_sequencer;
                return *this;
            }

            void swap(tuple_impl& other)
            {
                using std::swap;
                int const _sequencer[] = {
                    ((swap(this->get<Is>(), other.template get<Is>())), 0)...
                };
                (void)_sequencer;
            }

            template <std::size_t I>
            typename detail::at_index<I, Ts...>::type&
            get() HPX_NOEXCEPT
            {
                return static_cast<tuple_member<
                        I, typename detail::at_index<I, Ts...>::type
                    >&>(*this).value();
            }

            template <std::size_t I>
            typename detail::at_index<I, Ts...>::type const&
            get() const HPX_NOEXCEPT
            {
                return static_cast<tuple_member<
                        I, typename detail::at_index<I, Ts...>::type
                    > const&>(*this).value();
            }

            template <typename Archive>
            void serialize(Archive& ar, unsigned int const version)
            {
                int const _sequencer[] = {
                    ((ar & this->get<Is>()), 0)...
                };
                (void)_sequencer;
            }
        };

        ///////////////////////////////////////////////////////////////////////
        struct ignore_type
        {
            template <typename T>
            void operator=(T&& t) const
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
        HPX_CONSTEXPR tuple()
        {}

        // tuple(const tuple& u) = default;
        // Initializes each element of *this with the corresponding element
        // of u.
        HPX_CONSTEXPR tuple(tuple const& /*other*/)
        {}

        // tuple(tuple&& u) = default;
        // For all i, initializes the ith element of *this with
        // std::forward<Ti>(get<i>(u)).
        HPX_CONSTEXPR tuple(tuple&& /*other*/)
        {}

        // 20.4.2.2, tuple assignment

        // tuple& operator=(const tuple& u);
        // Assigns each element of u to the corresponding element of *this.
        tuple& operator=(tuple const& /*other*/) HPX_NOEXCEPT
        {
            return *this;
        }

        // tuple& operator=(tuple&& u) noexcept(see below );
        // For all i, assigns std::forward<Ti>(get<i>(u)) to get<i>(*this).
        tuple& operator=(tuple&& /*other*/) HPX_NOEXCEPT
        {
            return *this;
        }

        // 20.4.2.3, tuple swap

        // void swap(tuple& rhs) noexcept(see below);
        // Calls swap for each element in *this and its corresponding element
        // in rhs.
        void swap(tuple& /*other*/) HPX_NOEXCEPT
        {}
    };

    template <typename ...Ts>
    class tuple
    {
    public: // exposition-only
        detail::tuple_impl<
            typename detail::make_index_pack<sizeof...(Ts)>::type
          , Ts...> _impl;

    public:
        // 20.4.2.1, tuple construction

        // constexpr tuple();
        // Value initializes each element.
        HPX_CONSTEXPR tuple()
          : _impl()
        {}

        // explicit constexpr tuple(const Types&...);
        // Initializes each element with the value of the corresponding
        // parameter.
        explicit HPX_CONSTEXPR tuple(Ts const&... vs)
          : _impl(vs...)
        {}

        // template <class... UTypes>
        // explicit constexpr tuple(UTypes&&... u);
        // Initializes the elements in the tuple with the corresponding value
        // in std::forward<UTypes>(u).
        // This constructor shall not participate in overload resolution
        // unless each type in UTypes is implicitly convertible to its
        // corresponding type in Types.
        template <typename U, typename ...Us, typename Enable =
            typename std::enable_if<
                detail::pack<U, Us...>::size == detail::pack<Ts...>::size
             && std::conditional<
                    detail::pack<Us...>::size == 0
                  , typename std::enable_if<
                        !std::is_same<tuple, typename std::decay<U>::type>::value
                     && !detail::are_tuples_compatible_not_same<tuple, U&&>::value
                      , detail::are_tuples_compatible<tuple, tuple<U>&&>
                    >::type
                  , detail::are_tuples_compatible<tuple, tuple<U, Us...>&&>
                >::type::value
            >::type>
        explicit HPX_CONSTEXPR tuple(U&& v, Us&&... vs)
          : _impl(std::forward<U>(v), std::forward<Us>(vs)...)
        {}

#if defined(HPX_HAVE_CXX11_DEFAULTED_FUNCTIONS)
        // tuple(const tuple& u) = default;
        // Initializes each element of *this with the corresponding element
        // of u.
        tuple(tuple const&) = default;

        // tuple(tuple&& u) = default;
        // For all i, initializes the ith element of *this with
        // std::forward<Ti>(get<i>(u)).
        tuple(tuple&&) = default;
#else
        // tuple(const tuple& u) = default;
        // Initializes each element of *this with the corresponding element
        // of u.
        HPX_CONSTEXPR tuple(tuple const& other)
          : _impl(other._impl)
        {}

        // tuple(tuple&& u) = default;
        // For all i, initializes the ith element of *this with
        // std::forward<Ti>(get<i>(u)).
        HPX_CONSTEXPR tuple(tuple&& other)
          : _impl(std::move(other._impl))
        {}
#endif

        // template <class... UTypes> constexpr tuple(const tuple<UTypes...>& u);
        // template <class... UTypes> constexpr tuple(tuple<UTypes...>&& u);
        // For all i, initializes the ith element of *this with
        // get<i>(std::forward<U>(u).
        // This constructor shall not participate in overload resolution
        // unless each type in UTypes is implicitly convertible to its
        // corresponding type in Types
        template <typename UTuple, typename Enable =
            typename std::enable_if<
                detail::are_tuples_compatible_not_same<tuple, UTuple&&>::value
            >::type>
        HPX_CONSTEXPR tuple(UTuple&& other)
          : _impl(std::forward<UTuple>(other))
        {}

        // 20.4.2.2, tuple assignment

        // tuple& operator=(const tuple& u);
        // Assigns each element of u to the corresponding element of *this.
        tuple& operator=(tuple const& other)
        {
            _impl = other._impl;
            return *this;
        }

        // tuple& operator=(tuple&& u) noexcept(see below);
        // For all i, assigns std::forward<Ti>(get<i>(u)) to get<i>(*this).
        tuple& operator=(tuple&& other)
        {
            _impl = std::move(other._impl);
            return *this;
        }

        // template <class... UTypes> tuple& operator=(const tuple<UTypes...>& u);
        // template <class... UTypes> tuple& operator=(tuple<UTypes...>&& u);
        // For all i, assigns get<i>(std::forward<U>(u)) to get<i>(*this).
        template <typename UTuple>
        typename std::enable_if<
            tuple_size<
                typename std::decay<UTuple>::type
            >::value == detail::pack<Ts...>::size
          , tuple&
        >::type operator=(UTuple&& other)
        {
            _impl = std::forward<UTuple>(other);
            return *this;
        }

        // 20.4.2.3, tuple swap

        // void swap(tuple& rhs) noexcept(see below );
        // Calls swap for each element in *this and its corresponding element
        // in rhs.
        void swap(tuple& other)
        {
            _impl.swap(other._impl);
        }
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
      : boost::integral_constant<std::size_t, sizeof...(Ts)>
    {};

    template <typename T0, typename T1>
    struct tuple_size<std::pair<T0, T1> >
      : boost::integral_constant<std::size_t, 2>
    {};

    template <typename Type, std::size_t Size>
    struct tuple_size<boost::array<Type, Size> >
      : boost::integral_constant<std::size_t, Size>
    {};

    // template <size_t I, class Tuple>
    // class tuple_element
    template <std::size_t I, typename T>
    struct tuple_element
    {};

    template <std::size_t I, typename T>
    struct tuple_element<I, const T>
      : std::add_const<typename tuple_element<I, T>::type>
    {};

    template <std::size_t I, typename T>
    struct tuple_element<I, volatile T>
      : std::add_volatile<typename tuple_element<I, T>::type>
    {};

    template <std::size_t I, typename T>
    struct tuple_element<I, const volatile T>
      : std::add_cv<typename tuple_element<I, T>::type>
    {};

    template <std::size_t I, typename ...Ts>
    struct tuple_element<I, tuple<Ts...> >
    {
        typedef typename detail::at_index<I, Ts...>::type type;

        static HPX_CONSTEXPR HPX_FORCEINLINE type&
        get(tuple<Ts...>& tuple) HPX_NOEXCEPT
        {
            return tuple._impl.template get<I>();
        }

        static HPX_CONSTEXPR HPX_FORCEINLINE type const&
        get(tuple<Ts...> const& tuple) HPX_NOEXCEPT
        {
            return tuple._impl.template get<I>();
        }
    };

    template <typename T0, typename T1>
    struct tuple_element<0, std::pair<T0, T1> >
    {
        typedef T0 type;

        static HPX_CONSTEXPR HPX_FORCEINLINE type&
        get(std::pair<T0, T1>& tuple) HPX_NOEXCEPT
        {
            return tuple.first;
        }

        static HPX_CONSTEXPR HPX_FORCEINLINE type const&
        get(std::pair<T0, T1> const& tuple) HPX_NOEXCEPT
        {
            return tuple.first;
        }
    };

    template <typename T0, typename T1>
    struct tuple_element<1, std::pair<T0, T1> >
    {
        typedef T1 type;

        static HPX_CONSTEXPR HPX_FORCEINLINE type&
        get(std::pair<T0, T1>& tuple) HPX_NOEXCEPT
        {
            return tuple.second;
        }

        static HPX_CONSTEXPR HPX_FORCEINLINE type const&
        get(std::pair<T0, T1> const& tuple) HPX_NOEXCEPT
        {
            return tuple.second;
        }
    };

    template <std::size_t I, typename Type, std::size_t Size>
    struct tuple_element<I, boost::array<Type, Size> >
    {
        typedef Type type;

        static HPX_CONSTEXPR HPX_FORCEINLINE type&
        get(boost::array<Type, Size>& tuple) HPX_NOEXCEPT
        {
            return tuple[I];
        }

        static HPX_CONSTEXPR HPX_FORCEINLINE type const&
        get(boost::array<Type, Size> const& tuple) HPX_NOEXCEPT
        {
            return tuple[I];
        }
    };

    // 20.4.2.6, element access

    // template <size_t I, class... Types>
    // constexpr typename tuple_element<I, tuple<Types...> >::type&
    // get(tuple<Types...>& t) noexcept;
    template <std::size_t I, typename Tuple>
    HPX_CONSTEXPR HPX_FORCEINLINE
    typename tuple_element<I, Tuple>::type&
    get(Tuple& t) HPX_NOEXCEPT
    {
        return tuple_element<I, Tuple>::get(t);
    }

    // template <size_t I, class... Types>
    // constexpr typename tuple_element<I, tuple<Types...> >::type const&
    // get(const tuple<Types...>& t) noexcept;
    template <std::size_t I, typename Tuple>
    HPX_CONSTEXPR HPX_FORCEINLINE
    typename tuple_element<I, Tuple>::type const&
    get(Tuple const& t) HPX_NOEXCEPT
    {
        return tuple_element<I, Tuple>::get(t);
    }

    // template <size_t I, class... Types>
    // constexpr typename tuple_element<I, tuple<Types...> >::type&&
    // get(tuple<Types...>&& t) noexcept;
    template <std::size_t I, typename Tuple>
    HPX_CONSTEXPR HPX_FORCEINLINE
    typename tuple_element<I, Tuple>::type&&
    get(Tuple&& t) HPX_NOEXCEPT
    {
        return std::forward<
            typename tuple_element<I, Tuple>::type>(util::get<I>(t));
    }

    // template <size_t I, class... Types>
    // constexpr typename tuple_element<I, tuple<Types...> >::type const&&
    // get(const tuple<Types...>&& t) noexcept;
    template <std::size_t I, typename Tuple>
    HPX_CONSTEXPR HPX_FORCEINLINE
    typename tuple_element<I, Tuple>::type const&&
    get(Tuple const&& t) HPX_NOEXCEPT
    {
        return std::forward<
            typename tuple_element<I, Tuple>::type const>(util::get<I>(t));
    }

    // 20.4.2.4, tuple creation functions
    detail::ignore_type const ignore = {};

    // template<class... Types>
    // constexpr tuple<VTypes...> make_tuple(Types&&... t);
    template <typename ...Ts>
    HPX_CONSTEXPR HPX_FORCEINLINE
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
    HPX_FORCEINLINE
    tuple<Ts&&...>
    forward_as_tuple(Ts&&... vs) HPX_NOEXCEPT
    {
        return tuple<Ts&&...>(std::forward<Ts>(vs)...);
    }

    // template<class... Types>
    // tuple<Types&...> tie(Types&... t) noexcept;
    template <typename ...Ts>
    HPX_FORCEINLINE
    tuple<Ts&...>
    tie(Ts&... vs) HPX_NOEXCEPT
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
        {
            static const std::size_t value = Size;
        };

        template <std::size_t Size, typename Head, typename ...Tail>
        struct tuple_cat_size_impl<
            Size, detail::pack<Head, Tail...>
        > : tuple_cat_size_impl<
                (Size + tuple_size<Head>::value), detail::pack<Tail...>
            >
        {};

        template <typename ...Tuples>
        struct tuple_cat_size
          : tuple_cat_size_impl<0, detail::pack<Tuples...> >
        {};

        ///////////////////////////////////////////////////////////////////////
        template <std::size_t I, typename Tuples, typename Enable = void>
        struct tuple_cat_element;

        template <std::size_t I, typename Head, typename ...Tail>
        struct tuple_cat_element<
            I, detail::pack<Head, Tail...>
          , typename std::enable_if<
                (I < tuple_size<Head>::value)
            >::type
        > : tuple_element<I, Head>
        {
            typedef tuple_element<I, Head> base_type;

            static HPX_CONSTEXPR HPX_FORCEINLINE
            typename base_type::type&
            get(Head& head, Tail& ...tail) HPX_NOEXCEPT
            {
                return base_type::get(head);
            }

            static HPX_CONSTEXPR HPX_FORCEINLINE
            typename base_type::type const&
            get(Head const& head, Tail& ...tail) HPX_NOEXCEPT
            {
                return base_type::get(head);
            }
        };

        template <std::size_t I, typename Head, typename ...Tail>
        struct tuple_cat_element<
            I, detail::pack<Head, Tail...>
          , typename std::enable_if<
                (I >= tuple_size<Head>::value)
            >::type
        > : tuple_cat_element<
                I - tuple_size<Head>::value
              , detail::pack<Tail...>
            >
        {
            typedef tuple_cat_element<
                I - tuple_size<Head>::value
              , detail::pack<Tail...>
            > base_type;

            static HPX_CONSTEXPR HPX_FORCEINLINE
            typename base_type::type&
            get(Head& head, Tail& ...tail) HPX_NOEXCEPT
            {
                return base_type::get(tail...);
            }

            static HPX_CONSTEXPR HPX_FORCEINLINE
            typename base_type::type const&
            get(Head const& head, Tail& ...tail) HPX_NOEXCEPT
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
            static HPX_CONSTEXPR HPX_FORCEINLINE
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
    HPX_CONSTEXPR HPX_FORCEINLINE
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
            static HPX_CONSTEXPR HPX_FORCEINLINE
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
            static HPX_CONSTEXPR HPX_FORCEINLINE
            bool call(TTuple const& t, UTuple const&u)
            {
                return true;
            }
        };
    }

    template <typename ...Ts, typename ...Us>
    HPX_CONSTEXPR HPX_FORCEINLINE
    typename std::enable_if<sizeof...(Ts) == sizeof...(Us), bool>::type
    operator==(tuple<Ts...> const& t, tuple<Us...> const& u)
    {
        return detail::tuple_equal_to<0, sizeof...(Ts)>::call(t, u);
    }

    // template<class... TTypes, class... UTypes>
    // constexpr bool operator!=
    //     (const tuple<TTypes...>& t, const tuple<UTypes...>& u);
    template <typename ...Ts, typename ...Us>
    HPX_CONSTEXPR HPX_FORCEINLINE
    typename std::enable_if<sizeof...(Ts) == sizeof...(Us), bool>::type
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
            static HPX_CONSTEXPR HPX_FORCEINLINE
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
            static HPX_CONSTEXPR HPX_FORCEINLINE
            bool call(TTuple const& t, UTuple const&u)
            {
                return false;
            }
        };
    }

    template <typename ...Ts, typename ...Us>
    HPX_CONSTEXPR HPX_FORCEINLINE
    typename std::enable_if<sizeof...(Ts) == sizeof...(Us), bool>::type
    operator<(tuple<Ts...> const& t, tuple<Us...> const& u)
    {
        return detail::tuple_less_than<0, sizeof...(Ts)>::call(t, u);
    }

    // template<class... TTypes, class... UTypes>
    // constexpr bool operator>
    //     (const tuple<TTypes...>& t, const tuple<UTypes...>& u);
    template <typename ...Ts, typename ...Us>
    HPX_CONSTEXPR HPX_FORCEINLINE
    typename std::enable_if<sizeof...(Ts) == sizeof...(Us), bool>::type
    operator>(tuple<Ts...> const& t, tuple<Us...> const& u)
    {
        return u < t;
    }

    // template<class... TTypes, class... UTypes>
    // constexpr bool operator<=
    //     (const tuple<TTypes...>& t, const tuple<UTypes...>& u);
    template <typename ...Ts, typename ...Us>
    HPX_CONSTEXPR HPX_FORCEINLINE
    typename std::enable_if<sizeof...(Ts) == sizeof...(Us), bool>::type
    operator<=(tuple<Ts...> const& t, tuple<Us...> const& u)
    {
        return !(u < t);
    }

    // template<class... TTypes, class... UTypes>
    // constexpr bool operator>=
    //     (const tuple<TTypes...>& t, const tuple<UTypes...>& u);
    template <typename ...Ts, typename ...Us>
    HPX_CONSTEXPR HPX_FORCEINLINE
    typename std::enable_if<sizeof...(Ts) == sizeof...(Us), bool>::type
    operator>=(tuple<Ts...> const& t, tuple<Us...> const& u)
    {
        return !(t < u);
    }

    // 20.4.2.9, specialized algorithms

    // template <class... Types>
    // void swap(tuple<Types...>& x, tuple<Types...>& y) noexcept(x.swap(y));
    // x.swap(y)
    template <typename ...Ts>
    HPX_FORCEINLINE
    void swap(tuple<Ts...>& x, tuple<Ts...>& y)
        HPX_NOEXCEPT_IF(HPX_NOEXCEPT_EXPR((x.swap(y))))
    {
        x.swap(y);
    }

#if defined(HPX_HAVE_TUPLE_RVALUE_SWAP)
    // BADBAD: This overload of swap is necessary to work around the problems
    //         caused by zip_iterator not being a real random access iterator.
    //         Dereferencing zip_iterator does not yield a true reference but
    //         only a temporary tuple holding true references.
    //
    // A real fix for this problem is proposed in PR0022R0
    // (http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2015/p0022r0.html)
    //
    template <typename ...Ts>
    HPX_FORCEINLINE
    void swap(tuple<Ts&...> && x, tuple<Ts&...> && y)
        HPX_NOEXCEPT_IF(HPX_NOEXCEPT_EXPR((x.swap(y))))
    {
        x.swap(y);
    }
#endif
}}

#include <hpx/util/detail/fusion_adapt_tuple.hpp>

namespace hpx { namespace traits
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename ...Ts>
    struct is_bitwise_serializable<
        ::hpx::util::tuple<Ts...>
    > : ::hpx::util::detail::all_of<
            hpx::traits::is_bitwise_serializable<Ts>...
        >
    {};
}}

namespace hpx { namespace serialization
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Archive, typename ...Ts>
    HPX_FORCEINLINE
    void serialize(
        Archive& ar
      , ::hpx::util::tuple<Ts...>& t
      , unsigned int const version = 0
    )
    {
        t._impl.serialize(ar, version);
    }

    template <typename Archive>
    HPX_FORCEINLINE
    void serialize(
        Archive& ar
      , ::hpx::util::tuple<>& t
      , unsigned int const version = 0
    )
    {}
}}

#if defined(HPX_MSVC)
#pragma warning(pop)
#endif

#endif
