//  Copyright (c) 2011-2013 Thomas Heller
//  Copyright (c) 2011-2013 Hartmut Kaiser
//  Copyright (c) 2013-2015 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_TUPLE_HPP
#define HPX_UTIL_TUPLE_HPP

#include <hpx/config.hpp>
#include <hpx/datastructures/config/defines.hpp>
#include <hpx/type_support/decay.hpp>
#include <hpx/type_support/pack.hpp>

#include <boost/array.hpp>

#include <algorithm>
#include <array>
#include <cstddef>    // for size_t
#include <tuple>
#include <type_traits>
#include <utility>

#if defined(HPX_MSVC_WARNING_PRAGMA)
#pragma warning(push)
#pragma warning(disable : 4520)    // multiple default constructors specified
#endif

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmismatched-tags"
#endif

namespace hpx { namespace util {

    template <typename... Ts>
    class tuple;

    template <class T>
    struct tuple_size;    // undefined

    template <std::size_t I, typename T>
    struct tuple_element;    // undefined

    // Hide implementations of get<> inside an internal namespace to be able to
    // import those into the namespace std below without pulling in all of
    // hpx::util.
    namespace adl_barrier {

        template <std::size_t I, typename Tuple>
        constexpr HPX_HOST_DEVICE HPX_FORCEINLINE
            typename tuple_element<I, Tuple>::type&
            get(Tuple& t) noexcept;

        template <std::size_t I, typename Tuple>
        constexpr HPX_HOST_DEVICE HPX_FORCEINLINE
            typename tuple_element<I, Tuple>::type const&
            get(Tuple const& t) noexcept;

        template <std::size_t I, typename Tuple>
        constexpr HPX_HOST_DEVICE HPX_FORCEINLINE
            typename tuple_element<I, Tuple>::type&&
            get(Tuple&& t) noexcept;

        template <std::size_t I, typename Tuple>
        constexpr HPX_HOST_DEVICE HPX_FORCEINLINE
            typename tuple_element<I, Tuple>::type const&&
            get(Tuple const&& t) noexcept;
    }    // namespace adl_barrier

    using hpx::util::adl_barrier::get;
}}    // namespace hpx::util

#if defined(HPX_DATASTRUCTURES_HAVE_ADAPT_STD_TUPLE)
// Adapt hpx::util::tuple to be usable with std::tuple
namespace std {

    template <typename... Ts>
    struct tuple_size<hpx::util::tuple<Ts...>>
      : public hpx::util::tuple_size<hpx::util::tuple<Ts...>>
    {
    };

    template <std::size_t I, typename... Ts>
    struct tuple_element<I, hpx::util::tuple<Ts...>>
      : public hpx::util::tuple_element<I, hpx::util::tuple<Ts...>>
    {
    };

    using hpx::util::adl_barrier::get;
}    // namespace std
#endif

namespace hpx { namespace util {

    namespace detail {

        ///////////////////////////////////////////////////////////////////////
        template <std::size_t I, typename T, typename Enable = void>
        struct tuple_member    //-V690
        {
        public:
            template <typename U = T,
                typename EnableDefault = typename std::enable_if<
                    std::is_constructible<U>::value>::type>
            constexpr HPX_HOST_DEVICE tuple_member()
              : _value()
            {
            }

            template <typename U,
                typename = typename std::enable_if<!std::is_same<tuple_member,
                    typename std::decay<U>::type>::value>::type>
            explicit constexpr HPX_HOST_DEVICE tuple_member(U&& value)
              : _value(std::forward<U>(value))
            {
            }

#if !defined(__NVCC__) && !defined(__CUDACC__)
            tuple_member(tuple_member const&) = default;
            tuple_member(tuple_member&&) = default;
#else
            constexpr HPX_HOST_DEVICE tuple_member(tuple_member const& other)
              : _value(other.value())
            {
            }

            constexpr HPX_HOST_DEVICE tuple_member(tuple_member&& other)
              : _value(std::forward<T>(other.value()))
            {
            }
#endif

            HPX_HOST_DEVICE T& value() noexcept
            {
                return _value;
            }
            HPX_HOST_DEVICE T const& value() const noexcept
            {
                return _value;
            }

        private:
            T _value;
        };

        template <std::size_t I, typename T>
        struct tuple_member<I, T,
            typename std::enable_if<std::is_empty<T>::value &&
                !std::is_final<T>::value>::type> : T
        {
        public:
            constexpr HPX_HOST_DEVICE tuple_member()
              : T()
            {
            }

            template <typename U,
                typename = typename std::enable_if<!std::is_same<tuple_member,
                    typename std::decay<U>::type>::value>::type>
            explicit constexpr HPX_HOST_DEVICE tuple_member(U&& value)
              : T(std::forward<U>(value))
            {
            }

#if !defined(__NVCC__) && !defined(__CUDACC__)
            tuple_member(tuple_member const&) = default;
            tuple_member(tuple_member&&) = default;
#else
            constexpr HPX_HOST_DEVICE tuple_member(tuple_member const& other)
              : T(other.value())
            {
            }

            constexpr HPX_HOST_DEVICE tuple_member(tuple_member&& other)
              : T(std::forward<T>(other.value()))
            {
            }
#endif

            HPX_HOST_DEVICE T& value() noexcept
            {
                return *this;
            }
            HPX_HOST_DEVICE T const& value() const noexcept
            {
                return *this;
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Indices, typename TTuple, typename UTuple,
            typename Enable = void>
        struct are_tuples_compatible_impl : std::false_type
        {
        };

        template <std::size_t... Is, typename... Ts, typename UTuple>
        struct are_tuples_compatible_impl<util::index_pack<Is...>, tuple<Ts...>,
            UTuple,
            typename std::enable_if<tuple_size<typename std::remove_reference<
                                        UTuple>::type>::value ==
                util::pack<Ts...>::size>::type>
          : util::all_of<std::is_convertible<
                decltype(util::get<Is>(std::declval<UTuple>())), Ts>...>
        {
        };

        template <typename TTuple, typename UTuple>
        struct are_tuples_compatible;

        template <typename... Ts, typename UTuple>
        struct are_tuples_compatible<tuple<Ts...>, UTuple>
          : are_tuples_compatible_impl<
                typename util::make_index_pack<sizeof...(Ts)>::type,
                tuple<Ts...>, UTuple>
        {
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Is, typename... Ts>
        struct tuple_impl;

        template <std::size_t... Is, typename... Ts>
        struct tuple_impl<util::index_pack<Is...>, Ts...>
          : tuple_member<Is, Ts>...
        {
            // 20.4.2.1, tuple construction
            template <typename Dependent = void,
                typename Enable = typename std::enable_if<
                    util::all_of<std::is_constructible<Ts>...>::value,
                    Dependent>::type>
            constexpr HPX_HOST_DEVICE tuple_impl()
              : tuple_member<Is, Ts>()...
            {
            }

            template <typename... Us,
                typename Enable = typename std::enable_if<
                    util::pack<Us...>::size == util::pack<Ts...>::size>::type>
            explicit constexpr HPX_HOST_DEVICE tuple_impl(
                std::piecewise_construct_t, Us&&... vs)
              : tuple_member<Is, Ts>(std::forward<Us>(vs))...
            {
            }

#if !defined(__NVCC__) && !defined(__CUDACC__)
            tuple_impl(tuple_impl const&) = default;
            tuple_impl(tuple_impl&&) = default;
#else
            constexpr HPX_HOST_DEVICE tuple_impl(tuple_impl const& other)
              : tuple_member<Is, Ts>(
                    static_cast<tuple_member<Is, Ts> const&>(other))...
            {
            }

            constexpr HPX_HOST_DEVICE tuple_impl(tuple_impl&& other)
              : tuple_member<Is, Ts>(
                    static_cast<tuple_member<Is, Ts>&&>(other))...
            {
            }
#endif

            template <typename UTuple,
                typename Enable =
                    typename std::enable_if<!std::is_same<tuple_impl,
                        typename std::decay<UTuple>::type>::value>::type,
                typename EnableCompatible = typename std::enable_if<
                    are_tuples_compatible<tuple<Ts...>, UTuple>::value>::type>
            constexpr HPX_HOST_DEVICE tuple_impl(UTuple&& other)
              : tuple_member<Is, Ts>(
                    util::get<Is>(std::forward<UTuple>(other)))...
            {
            }

            HPX_HOST_DEVICE tuple_impl& operator=(tuple_impl const& other)
            {
                int const _sequencer[] = {
                    ((this->get<Is>() = other.template get<Is>()), 0)...};
                (void) _sequencer;
                return *this;
            }

            HPX_HOST_DEVICE tuple_impl& operator=(tuple_impl&& other)
            {
                int const _sequencer[] = {((this->get<Is>() = std::forward<Ts>(
                                                other.template get<Is>())),
                    0)...};
                (void) _sequencer;
                return *this;
            }

            template <typename UTuple>
            HPX_HOST_DEVICE tuple_impl& operator=(UTuple&& other)
            {
                int const _sequencer[] = {((this->get<Is>() = util::get<Is>(
                                                std::forward<UTuple>(other))),
                    0)...};
                (void) _sequencer;
                return *this;
            }

            HPX_HOST_DEVICE void swap(tuple_impl& other)
            {
                using std::swap;
                int const _sequencer[] = {
                    ((swap(this->get<Is>(), other.template get<Is>())), 0)...};
                (void) _sequencer;
            }

            template <std::size_t I>
            HPX_HOST_DEVICE typename util::at_index<I, Ts...>::type&
            get() noexcept
            {
                return static_cast<
                    tuple_member<I, typename util::at_index<I, Ts...>::type>&>(
                    *this)
                    .value();
            }

            template <std::size_t I>
            HPX_HOST_DEVICE typename util::at_index<I, Ts...>::type const& get()
                const noexcept
            {
                return static_cast<tuple_member<I,
                    typename util::at_index<I, Ts...>::type> const&>(*this)
                    .value();
            }

#if defined(HPX_DATASTRUCTURES_HAVE_ADAPT_STD_TUPLE)
            HPX_HOST_DEVICE
            operator std::tuple<Ts...>() const&
            {
                return std::make_tuple(get<Is>()...);
            }

            HPX_HOST_DEVICE
            operator std::tuple<Ts...>() &&
            {
                return std::make_tuple(std::move(get<Is>())...);
            }

            HPX_HOST_DEVICE
            operator std::tuple<Ts...>() &
            {
                return std::make_tuple(get<Is>()...);
            }

            HPX_HOST_DEVICE
            operator std::tuple<Ts...>() const&&
            {
                return std::make_tuple(std::move(get<Is>())...);
            }
#endif
        };

        ///////////////////////////////////////////////////////////////////////
        struct ignore_type
        {
            template <typename T>
            void operator=(T&& t) const
            {
            }
        };
    }    // namespace detail

    // 20.4.2, class template tuple
    template <>
    class tuple<>
    {
    public:
        // 20.4.2.1, tuple construction

        // constexpr tuple();
        // Value initializes each element.
        constexpr HPX_HOST_DEVICE tuple() {}

        // tuple(const tuple& u) = default;
        // Initializes each element of *this with the corresponding element
        // of u.
        constexpr HPX_HOST_DEVICE tuple(tuple const& /*other*/) = default;

        // tuple(tuple&& u) = default;
        // For all i, initializes the ith element of *this with
        // std::forward<Ti>(get<i>(u)).
        constexpr HPX_HOST_DEVICE tuple(tuple&& /*other*/) = default;

        // 20.4.2.2, tuple assignment

        // tuple& operator=(const tuple& u);
        // Assigns each element of u to the corresponding element of *this.
        HPX_HOST_DEVICE tuple& operator=(tuple const& /*other*/) noexcept
        {
            return *this;
        }

        // tuple& operator=(tuple&& u) noexcept(see below );
        // For all i, assigns std::forward<Ti>(get<i>(u)) to get<i>(*this).
        HPX_HOST_DEVICE tuple& operator=(tuple&& /*other*/) noexcept
        {
            return *this;
        }

        // 20.4.2.3, tuple swap

        // void swap(tuple& rhs) noexcept(see below);
        // Calls swap for each element in *this and its corresponding element
        // in rhs.
        HPX_HOST_DEVICE void swap(tuple& /*other*/) noexcept {}

#if defined(HPX_DATASTRUCTURES_HAVE_ADAPT_STD_TUPLE)
        operator std::tuple<>() const
        {
            return std::tuple<>();
        }
#endif
    };

    template <typename... Ts>
    class tuple
    {
    public:    // exposition-only
        detail::tuple_impl<typename util::make_index_pack<sizeof...(Ts)>::type,
            Ts...>
            _impl;

    public:
        // 20.4.2.1, tuple construction

        // constexpr tuple();
        // Value initializes each element.
        template <typename Dependent = void,
            typename Enable = typename std::enable_if<
                util::all_of<std::is_constructible<Ts>...>::value,
                Dependent>::type>
        constexpr HPX_HOST_DEVICE tuple()
          : _impl()
        {
        }

        // explicit constexpr tuple(const Types&...);
        // Initializes each element with the value of the corresponding
        // parameter.
        explicit constexpr HPX_HOST_DEVICE tuple(Ts const&... vs)
          : _impl(std::piecewise_construct, vs...)
        {
        }

        // template <class... UTypes>
        // explicit constexpr tuple(UTypes&&... u);
        // Initializes the elements in the tuple with the corresponding value
        // in std::forward<UTypes>(u).
        // This constructor shall not participate in overload resolution
        // unless each type in UTypes is implicitly convertible to its
        // corresponding type in Types.
        template <typename U, typename... Us,
            typename Enable = typename std::enable_if<
                !std::is_same<tuple, typename std::decay<U>::type>::value ||
                util::pack<Us...>::size != 0>::type,
            typename EnableCompatible = typename std::enable_if<detail::
                    are_tuples_compatible<tuple, tuple<U, Us...>>::value>::type>
        explicit constexpr HPX_HOST_DEVICE tuple(U&& v, Us&&... vs)
          : _impl(std::piecewise_construct, std::forward<U>(v),
                std::forward<Us>(vs)...)
        {
        }

#if !defined(__NVCC__) && !defined(__CUDACC__)
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
        constexpr HPX_HOST_DEVICE tuple(tuple const& other)
          : _impl(other._impl)
        {
        }

        // tuple(tuple&& u) = default;
        // For all i, initializes the ith element of *this with
        // std::forward<Ti>(get<i>(u)).
        constexpr HPX_HOST_DEVICE tuple(tuple&& other)
          : _impl(std::move(other._impl))
        {
        }
#endif

        // template <class... UTypes> constexpr tuple(const tuple<UTypes...>& u);
        // template <class... UTypes> constexpr tuple(tuple<UTypes...>&& u);
        // For all i, initializes the ith element of *this with
        // get<i>(std::forward<U>(u).
        // This constructor shall not participate in overload resolution
        // unless each type in UTypes is implicitly convertible to its
        // corresponding type in Types
        template <typename UTuple,
            typename Enable = typename std::enable_if<!std::is_same<tuple,
                typename std::decay<UTuple>::type>::value>::type,
            typename EnableCompatible = typename std::enable_if<
                detail::are_tuples_compatible<tuple, UTuple>::value>::type>
        constexpr HPX_HOST_DEVICE tuple(UTuple&& other)
          : _impl(std::forward<UTuple>(other))
        {
        }

        // 20.4.2.2, tuple assignment

        // tuple& operator=(const tuple& u);
        // Assigns each element of u to the corresponding element of *this.
        HPX_HOST_DEVICE tuple& operator=(tuple const& other)
        {
            _impl = other._impl;
            return *this;
        }

        // tuple& operator=(tuple&& u) noexcept(see below);
        // For all i, assigns std::forward<Ti>(get<i>(u)) to get<i>(*this).
        HPX_HOST_DEVICE tuple& operator=(tuple&& other)
        {
            _impl = std::move(other._impl);
            return *this;
        }

        // template <class... UTypes> tuple& operator=(const tuple<UTypes...>& u);
        // template <class... UTypes> tuple& operator=(tuple<UTypes...>&& u);
        // For all i, assigns get<i>(std::forward<U>(u)) to get<i>(*this).
        template <typename UTuple>
        HPX_HOST_DEVICE typename std::enable_if<
            tuple_size<typename std::decay<UTuple>::type>::value ==
                util::pack<Ts...>::size,
            tuple&>::type
        operator=(UTuple&& other)
        {
            _impl = std::forward<UTuple>(other);
            return *this;
        }

        // 20.4.2.3, tuple swap

        // void swap(tuple& rhs) noexcept(see below );
        // Calls swap for each element in *this and its corresponding element
        // in rhs.
        HPX_HOST_DEVICE void swap(tuple& other)
        {
            _impl.swap(other._impl);
        }

#if defined(HPX_DATASTRUCTURES_HAVE_ADAPT_STD_TUPLE)
        operator std::tuple<Ts...>() const&
        {
            return _impl;
        }

        operator std::tuple<Ts...>() &
        {
            return _impl;
        }

        operator std::tuple<Ts...>() &&
        {
            return _impl;
        }

        operator std::tuple<Ts...>() const&&
        {
            return _impl;
        }
#endif
    };

    // 20.4.2.5, tuple helper classes

    // template <class Tuple>
    // class tuple_size
    template <class T>
    struct tuple_size
    {
    };

    template <class T>
    struct tuple_size<const T> : tuple_size<T>
    {
    };

    template <class T>
    struct tuple_size<volatile T> : tuple_size<T>
    {
    };

    template <class T>
    struct tuple_size<const volatile T> : tuple_size<T>
    {
    };

    template <typename... Ts>
    struct tuple_size<tuple<Ts...>>
      : std::integral_constant<std::size_t, sizeof...(Ts)>
    {
    };

    template <typename T0, typename T1>
    struct tuple_size<std::pair<T0, T1>>
      : std::integral_constant<std::size_t, 2>
    {
    };

    template <typename Type, std::size_t Size>
    struct tuple_size<boost::array<Type, Size>>
      : std::integral_constant<std::size_t, Size>
    {
    };

    template <typename Type, std::size_t Size>
    struct tuple_size<std::array<Type, Size>>
      : std::integral_constant<std::size_t, Size>
    {
    };

#if defined(HPX_DATASTRUCTURES_HAVE_ADAPT_STD_TUPLE)
    template <typename... Ts>
    struct tuple_size<std::tuple<Ts...>> : std::tuple_size<std::tuple<Ts...>>
    {
    };
#endif

    // template <size_t I, class Tuple>
    // class tuple_element
    template <std::size_t I, typename T>
    struct tuple_element
    {
    };

    template <std::size_t I, typename T>
    struct tuple_element<I, const T>
      : std::add_const<typename tuple_element<I, T>::type>
    {
    };

    template <std::size_t I, typename T>
    struct tuple_element<I, volatile T>
      : std::add_volatile<typename tuple_element<I, T>::type>
    {
    };

    template <std::size_t I, typename T>
    struct tuple_element<I, const volatile T>
      : std::add_cv<typename tuple_element<I, T>::type>
    {
    };

    template <std::size_t I, typename... Ts>
    struct tuple_element<I, tuple<Ts...>>
    {
        using type = typename util::at_index<I, Ts...>::type;

        static constexpr HPX_HOST_DEVICE HPX_FORCEINLINE type& get(
            tuple<Ts...>& tuple) noexcept
        {
            return tuple._impl.template get<I>();
        }

        static constexpr HPX_HOST_DEVICE HPX_FORCEINLINE type const& get(
            tuple<Ts...> const& tuple) noexcept
        {
            return tuple._impl.template get<I>();
        }
    };

    template <typename T0, typename T1>
    struct tuple_element<0, std::pair<T0, T1>>
    {
        using type = T0;

        static constexpr HPX_HOST_DEVICE HPX_FORCEINLINE type& get(
            std::pair<T0, T1>& tuple) noexcept
        {
            return tuple.first;
        }

        static constexpr HPX_HOST_DEVICE HPX_FORCEINLINE type const& get(
            std::pair<T0, T1> const& tuple) noexcept
        {
            return tuple.first;
        }
    };

    template <typename T0, typename T1>
    struct tuple_element<1, std::pair<T0, T1>>
    {
        using type = T1;

        static constexpr HPX_HOST_DEVICE HPX_FORCEINLINE type& get(
            std::pair<T0, T1>& tuple) noexcept
        {
            return tuple.second;
        }

        static constexpr HPX_HOST_DEVICE HPX_FORCEINLINE type const& get(
            std::pair<T0, T1> const& tuple) noexcept
        {
            return tuple.second;
        }
    };

    template <std::size_t I, typename Type, std::size_t Size>
    struct tuple_element<I, boost::array<Type, Size>>
    {
        using type = Type;

        static constexpr HPX_HOST_DEVICE HPX_FORCEINLINE type& get(
            boost::array<Type, Size>& tuple) noexcept
        {
            return tuple[I];
        }

        static constexpr HPX_HOST_DEVICE HPX_FORCEINLINE type const& get(
            boost::array<Type, Size> const& tuple) noexcept
        {
            return tuple[I];
        }
    };

    template <std::size_t I, typename Type, std::size_t Size>
    struct tuple_element<I, std::array<Type, Size>>
    {
        using type = Type;

        static constexpr HPX_HOST_DEVICE HPX_FORCEINLINE type& get(
            std::array<Type, Size>& tuple) noexcept
        {
            return tuple[I];
        }

        static constexpr HPX_HOST_DEVICE HPX_FORCEINLINE type const& get(
            std::array<Type, Size> const& tuple) noexcept
        {
            return tuple[I];
        }
    };

#if defined(HPX_DATASTRUCTURES_HAVE_ADAPT_STD_TUPLE)
    template <std::size_t I, typename... Ts>
    struct tuple_element<I, std::tuple<Ts...>>
    {
        using type = typename std::tuple_element<I, std::tuple<Ts...>>::type;

        static constexpr HPX_HOST_DEVICE HPX_FORCEINLINE type& get(
            std::tuple<Ts...>& tuple) noexcept
        {
            return std::get<I>(tuple);
        }

        static constexpr HPX_HOST_DEVICE HPX_FORCEINLINE type const& get(
            std::tuple<Ts...> const& tuple) noexcept
        {
            return std::get<I>(tuple);
        }

        static constexpr HPX_HOST_DEVICE HPX_FORCEINLINE type&& get(
            std::tuple<Ts...>&& tuple) noexcept
        {
            return std::get<I>(std::move(tuple));
        }

        static constexpr HPX_HOST_DEVICE HPX_FORCEINLINE type const&& get(
            std::tuple<Ts...> const&& tuple) noexcept
        {
            return std::get<I>(std::move(tuple));
        }
    };
#endif

    // 20.4.2.6, element access
    namespace adl_barrier {

        // template <size_t I, class... Types>
        // constexpr typename tuple_element<I, tuple<Types...> >::type&
        // get(tuple<Types...>& t) noexcept;
        template <std::size_t I, typename Tuple>
        constexpr HPX_HOST_DEVICE HPX_FORCEINLINE
            typename tuple_element<I, Tuple>::type&
            get(Tuple& t) noexcept
        {
            return tuple_element<I, Tuple>::get(t);
        }

        // template <size_t I, class... Types>
        // constexpr typename tuple_element<I, tuple<Types...> >::type const&
        // get(const tuple<Types...>& t) noexcept;
        template <std::size_t I, typename Tuple>
        constexpr HPX_HOST_DEVICE HPX_FORCEINLINE
            typename tuple_element<I, Tuple>::type const&
            get(Tuple const& t) noexcept
        {
            return tuple_element<I, Tuple>::get(t);
        }

        // template <size_t I, class... Types>
        // constexpr typename tuple_element<I, tuple<Types...> >::type&&
        // get(tuple<Types...>&& t) noexcept;
        template <std::size_t I, typename Tuple>
        constexpr HPX_HOST_DEVICE HPX_FORCEINLINE
            typename tuple_element<I, Tuple>::type&&
            get(Tuple&& t) noexcept
        {
            return std::forward<typename tuple_element<I, Tuple>::type>(
                util::get<I>(t));
        }

        // template <size_t I, class... Types>
        // constexpr typename tuple_element<I, tuple<Types...> >::type const&&
        // get(const tuple<Types...>&& t) noexcept;
        template <std::size_t I, typename Tuple>
        constexpr HPX_HOST_DEVICE HPX_FORCEINLINE
            typename tuple_element<I, Tuple>::type const&&
            get(Tuple const&& t) noexcept
        {
            return std::forward<typename tuple_element<I, Tuple>::type const>(
                util::get<I>(t));
        }
    }    // namespace adl_barrier

    // 20.4.2.4, tuple creation functions
    detail::ignore_type const ignore = {};

    // template<class... Types>
    // constexpr tuple<VTypes...> make_tuple(Types&&... t);
    template <typename... Ts>
    constexpr HPX_HOST_DEVICE HPX_FORCEINLINE
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
    template <typename... Ts>
    HPX_HOST_DEVICE HPX_FORCEINLINE tuple<Ts&&...> forward_as_tuple(
        Ts&&... vs) noexcept
    {
        return tuple<Ts&&...>(std::forward<Ts>(vs)...);
    }

    // template<class... Types>
    // tuple<Types&...> tie(Types&... t) noexcept;
    template <typename... Ts>
    HPX_HOST_DEVICE HPX_FORCEINLINE tuple<Ts&...> tie(Ts&... vs) noexcept
    {
        return tuple<Ts&...>(vs...);
    }

    //template <class... Tuples>
    //constexpr tuple<Ctypes ...> tuple_cat(Tuples&&...);
    namespace detail {

        /// Deduces to the overall size of all given tuples
        template <std::size_t Size, typename Tuples>
        struct tuple_cat_size_impl;

        template <std::size_t Size>
        struct tuple_cat_size_impl<Size, util::pack<>>
          : std::integral_constant<std::size_t, Size>
        {
        };

        template <std::size_t Size, typename Head, typename... Tail>
        struct tuple_cat_size_impl<Size, util::pack<Head, Tail...>>
          : tuple_cat_size_impl<(Size + tuple_size<Head>::value),
                util::pack<Tail...>>
        {
        };

        template <typename... Tuples>
        struct tuple_cat_size : tuple_cat_size_impl<0, util::pack<Tuples...>>
        {
        };

        ///////////////////////////////////////////////////////////////////////
        template <std::size_t I, typename Tuples, typename Enable = void>
        struct tuple_cat_element;

        template <std::size_t I, typename Head, typename... Tail>
        struct tuple_cat_element<I, util::pack<Head, Tail...>,
            typename std::enable_if<(I < tuple_size<Head>::value)>::type>
          : tuple_element<I, Head>
        {
            template <typename THead, typename... TTail>
            static constexpr HPX_HOST_DEVICE HPX_FORCEINLINE auto get(
                THead&& head, TTail&&... /*tail*/) noexcept
                -> decltype(hpx::util::get<I>(std::forward<THead>(head)))
            {
                return hpx::util::get<I>(std::forward<THead>(head));
            }
        };

        template <std::size_t I, typename Head, typename... Tail>
        struct tuple_cat_element<I, util::pack<Head, Tail...>,
            typename std::enable_if<(I >= tuple_size<Head>::value)>::type>
          : tuple_cat_element<I - tuple_size<Head>::value, util::pack<Tail...>>
        {
            using base_type = tuple_cat_element<I - tuple_size<Head>::value,
                util::pack<Tail...>>;

            template <typename THead, typename... TTail>
            static constexpr HPX_HOST_DEVICE HPX_FORCEINLINE auto get(
                THead&& /*head*/, TTail&&... tail) noexcept
                -> decltype(base_type::get(std::forward<TTail>(tail)...))
            {
                return base_type::get(std::forward<TTail>(tail)...);
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Indices, typename Tuples>
        struct tuple_cat_result_impl;

        template <std::size_t... Is, typename... Tuples>
        struct tuple_cat_result_impl<index_pack<Is...>, pack<Tuples...>>
        {
            using type =
                tuple<typename tuple_cat_element<Is, pack<Tuples...>>::type...>;
        };

        template <typename Indices, typename Tuples>
        using tuple_cat_result_of_t =
            typename tuple_cat_result_impl<Indices, Tuples>::type;

        template <std::size_t... Is, typename... Tuples, typename... Tuples_>
        constexpr HPX_HOST_DEVICE HPX_FORCEINLINE auto tuple_cat_impl(
            index_pack<Is...> is_pack, pack<Tuples...> tuple_pack,
            Tuples_&&... tuples)
            -> tuple_cat_result_of_t<decltype(is_pack), decltype(tuple_pack)>
        {
            return tuple_cat_result_of_t<decltype(is_pack),
                decltype(tuple_pack)>{
                tuple_cat_element<Is, pack<Tuples...>>::get(
                    std::forward<Tuples_>(tuples)...)...};
        }
    }    // namespace detail

    template <typename... Tuples>
    constexpr HPX_HOST_DEVICE HPX_FORCEINLINE auto tuple_cat(Tuples&&... tuples)
        -> decltype(detail::tuple_cat_impl(
            typename util::make_index_pack<detail::tuple_cat_size<
                typename std::decay<Tuples>::type...>::value>::type{},
            util::pack<typename std::decay<Tuples>::type...>{},
            std::forward<Tuples>(tuples)...))
    {
        return detail::tuple_cat_impl(
            typename util::make_index_pack<detail::tuple_cat_size<
                typename std::decay<Tuples>::type...>::value>::type{},
            util::pack<typename std::decay<Tuples>::type...>{},
            std::forward<Tuples>(tuples)...);
    }

    // 20.4.2.7, relational operators

    // template<class... TTypes, class... UTypes>
    // constexpr bool operator==
    //     (const tuple<TTypes...>& t, const tuple<UTypes...>& u);
    // The elementary comparisons are performed in order from the zeroth index
    // upwards. No comparisons or element accesses are performed after the
    // first equality comparison that evaluates to false.
    namespace detail {
        template <std::size_t I, std::size_t Size>
        struct tuple_equal_to
        {
            template <typename TTuple, typename UTuple>
            static constexpr HPX_HOST_DEVICE HPX_FORCEINLINE bool call(
                TTuple const& t, UTuple const& u)
            {
                return util::get<I>(t) == util::get<I>(u) &&
                    tuple_equal_to<I + 1, Size>::call(t, u);
            }
        };

        template <std::size_t Size>
        struct tuple_equal_to<Size, Size>
        {
            template <typename TTuple, typename UTuple>
            static constexpr HPX_HOST_DEVICE HPX_FORCEINLINE bool call(
                TTuple const& t, UTuple const& u)
            {
                return true;
            }
        };
    }    // namespace detail

    template <typename... Ts, typename... Us>
    constexpr HPX_HOST_DEVICE HPX_FORCEINLINE
        typename std::enable_if<sizeof...(Ts) == sizeof...(Us), bool>::type
        operator==(tuple<Ts...> const& t, tuple<Us...> const& u)
    {
        return detail::tuple_equal_to<0, sizeof...(Ts)>::call(t, u);
    }

    // template<class... TTypes, class... UTypes>
    // constexpr bool operator!=
    //     (const tuple<TTypes...>& t, const tuple<UTypes...>& u);
    template <typename... Ts, typename... Us>
    constexpr HPX_HOST_DEVICE HPX_FORCEINLINE
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
    namespace detail {
        template <std::size_t I, std::size_t Size>
        struct tuple_less_than
        {
            template <typename TTuple, typename UTuple>
            static constexpr HPX_HOST_DEVICE HPX_FORCEINLINE bool call(
                TTuple const& t, UTuple const& u)
            {
                return util::get<I>(t) < util::get<I>(u) ||
                    (!(util::get<I>(u) < util::get<I>(t)) &&
                        tuple_less_than<I + 1, Size>::call(t, u));
            }
        };

        template <std::size_t Size>
        struct tuple_less_than<Size, Size>
        {
            template <typename TTuple, typename UTuple>
            static constexpr HPX_HOST_DEVICE HPX_FORCEINLINE bool call(
                TTuple const& t, UTuple const& u)
            {
                return false;
            }
        };
    }    // namespace detail

    template <typename... Ts, typename... Us>
    constexpr HPX_HOST_DEVICE HPX_FORCEINLINE
        typename std::enable_if<sizeof...(Ts) == sizeof...(Us), bool>::type
        operator<(tuple<Ts...> const& t, tuple<Us...> const& u)
    {
        return detail::tuple_less_than<0, sizeof...(Ts)>::call(t, u);
    }

    // template<class... TTypes, class... UTypes>
    // constexpr bool operator>
    //     (const tuple<TTypes...>& t, const tuple<UTypes...>& u);
    template <typename... Ts, typename... Us>
    constexpr HPX_HOST_DEVICE HPX_FORCEINLINE
        typename std::enable_if<sizeof...(Ts) == sizeof...(Us), bool>::type
        operator>(tuple<Ts...> const& t, tuple<Us...> const& u)
    {
        return u < t;
    }

    // template<class... TTypes, class... UTypes>
    // constexpr bool operator<=
    //     (const tuple<TTypes...>& t, const tuple<UTypes...>& u);
    template <typename... Ts, typename... Us>
    constexpr HPX_HOST_DEVICE HPX_FORCEINLINE
        typename std::enable_if<sizeof...(Ts) == sizeof...(Us), bool>::type
        operator<=(tuple<Ts...> const& t, tuple<Us...> const& u)
    {
        return !(u < t);
    }

    // template<class... TTypes, class... UTypes>
    // constexpr bool operator>=
    //     (const tuple<TTypes...>& t, const tuple<UTypes...>& u);
    template <typename... Ts, typename... Us>
    constexpr HPX_HOST_DEVICE HPX_FORCEINLINE
        typename std::enable_if<sizeof...(Ts) == sizeof...(Us), bool>::type
        operator>=(tuple<Ts...> const& t, tuple<Us...> const& u)
    {
        return !(t < u);
    }

    // 20.4.2.9, specialized algorithms

    // template <class... Types>
    // void swap(tuple<Types...>& x, tuple<Types...>& y) noexcept(x.swap(y));
    // x.swap(y)
    template <typename... Ts>
    HPX_HOST_DEVICE HPX_FORCEINLINE void swap(
        tuple<Ts...>& x, tuple<Ts...>& y) noexcept(noexcept(x.swap(y)))
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
    template <typename... Ts>
    HPX_HOST_DEVICE HPX_FORCEINLINE void swap(
        tuple<Ts&...>&& x, tuple<Ts&...>&& y) noexcept(noexcept(x.swap(y)))
    {
        x.swap(y);
    }
#endif
}}    // namespace hpx::util

#if defined(HPX_MSVC_WARNING_PRAGMA)
#pragma warning(pop)
#endif

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#endif
