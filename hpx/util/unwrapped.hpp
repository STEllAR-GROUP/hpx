//  Copyright (c)      2013 Thomas Heller
//  Copyright (c)      2014 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_UNWRAPPED_HPP
#define HPX_UTIL_UNWRAPPED_HPP

#include <hpx/config.hpp>
#include <hpx/util_fwd.hpp>

#include <hpx/lcos/future.hpp>
#include <hpx/traits/future_traits.hpp>
#include <hpx/traits/is_future.hpp>
#include <hpx/traits/is_future_range.hpp>
#include <hpx/traits/is_future_tuple.hpp>
#include <hpx/util/detail/pack.hpp>
#include <hpx/util/invoke_fused.hpp>
#include <hpx/util/lazy_enable_if.hpp>
#include <hpx/util/result_of.hpp>
#include <hpx/util/tuple.hpp>

#if defined(HPX_HAVE_CXX11_STD_ARRAY)
#include <array>
#endif
#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace util
{
    /// \cond NOINTERNAL
    namespace detail
    {
        template <typename T, typename Enable = void>
        struct unwrap_impl;

        ///////////////////////////////////////////////////////////////////////
        // Implements first level future unwrapping for
        // - `hpx::lcos::(shared_)?future<T>`
        template <typename T>
        struct unwrap_impl<
            T,
            typename std::enable_if<traits::is_future<T>::value>::type
        >
        {
            typedef typename traits::future_traits<T>::type value_type;
            typedef std::is_void<value_type> is_void;

            typedef typename traits::future_traits<T>::result_type type;

            template <typename Future>
            static type call(Future& future, /*is_void=*/std::false_type)
            {
                return future.get();
            }

            template <typename Future>
            static type call(Future& future, /*is_void=*/std::true_type)
            {
                future.get();
            }

            template <typename Future>
            static type call(Future&& future)
            {
                return call(future, is_void());
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Range, typename New>
        struct rebind_range
        {
            typedef std::vector<New> type;
        };

#if defined(HPX_HAVE_CXX11_STD_ARRAY)
        template <typename T, std::size_t N, typename New>
        struct rebind_range<std::array<T, N>, New>
        {
            typedef std::array<New, N> type;
        };
#endif

        // Ranges (begin() and end()) of futures
        template <typename T>
        struct unwrap_impl<
            T,
            typename std::enable_if<
                traits::is_future_range<T>::value &&
                traits::detail::has_push_back<T>::value
            >::type
        >
        {
            typedef typename T::value_type future_type;
            typedef typename traits::future_traits<future_type>::type value_type;
            typedef std::is_void<value_type> is_void;

            typedef typename std::conditional<
                is_void::value, void, std::vector<value_type>
            >::type type;

            template <typename Range>
            static type call(Range& range, /*is_void=*/std::false_type)
            {
                type result;
                for (typename Range::value_type& f : range)
                {
                    result.push_back(unwrap_impl<future_type>::call(f));
                }

                return result;
            }

            // Edge case for void futures
            template <typename Range>
            static type call(Range& range, /*is_void=*/std::true_type)
            {
                for (typename Range::value_type& f : range)
                {
                    unwrap_impl<future_type>::call(f);
                }
            }

            // Tag dispatch trampoline
            template <typename Range>
            static type call(Range&& range)
            {
                return call(range, is_void());
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename T>
        struct unwrap_impl<
            T,
            typename std::enable_if<
                traits::is_future_range<T>::value &&
               !traits::detail::has_push_back<T>::value
            >::type
        >
        {
            typedef typename T::value_type future_type;
            typedef typename traits::future_traits<future_type>::type value_type;
            typedef std::is_void<value_type> is_void;

            typedef typename std::conditional<
                is_void::value, void,
                typename rebind_range<
                        typename std::decay<T>::type, value_type
                    >::type
            >::type type;

            template <typename Range>
            static type call(Range& range, /*is_void=*/std::false_type)
            {
                type result;

                std::size_t i = 0;
                for (typename Range::value_type& f : range)
                {
                    result[i++] = unwrap_impl<future_type>::call(f);
                }

                return result;
            }

            template <typename Range>
            static type call(Range& range, /*is_void=*/std::true_type)
            {
                for (typename Range::value_type& f : range)
                {
                    unwrap_impl<future_type>::call(f);
                }
            }

            template <typename Range>
            static type call(Range&& range)
            {
                return call(range, is_void());
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <
            typename Tuple,
            typename State = detail::pack<>, std::size_t I = 0,
            bool End =
                (I == util::tuple_size<typename std::decay<Tuple>::type>::value)
        > struct unwrap_tuple_fold;

        template <
            typename Tuple, typename State, std::size_t I,
            typename Future = decltype(util::get<I>(std::declval<Tuple&>())),
            bool IsVoid =
                unwrap_impl<typename std::decay<Future>::type>::is_void::value
        > struct unwrap_tuple_impl;

        template <typename Tuple, typename ...Vs, std::size_t I, typename Future>
        struct unwrap_tuple_impl<Tuple, detail::pack<Vs...>, I, Future, /*IsVoid=*/false>
        {
            typedef unwrap_impl<typename std::decay<Future>::type> unwrap_impl_t;
            typedef detail::pack<Vs..., typename unwrap_impl_t::type> next_state;

            typedef typename unwrap_tuple_fold<Tuple, next_state, I + 1>::type type;

            static type call(Tuple& tuple, Vs&&... vs)
            {
                return unwrap_tuple_fold<Tuple, next_state, I + 1>::call(
                    tuple, std::forward<Vs>(vs)...,
                    unwrap_impl_t::call(util::get<I>(tuple)));
            }
        };

        template <typename Tuple, typename ...Vs, std::size_t I, typename Future>
        struct unwrap_tuple_impl<Tuple, detail::pack<Vs...>, I, Future, /*IsVoid=*/true>
        {
            typedef unwrap_impl<typename std::decay<Future>::type> unwrap_impl_t;
            typedef detail::pack<Vs...> next_state;

            typedef typename unwrap_tuple_fold<Tuple, next_state, I + 1>::type type;

            static type call(Tuple& tuple, Vs&&... vs)
            {
                unwrap_impl_t::call(util::get<I>(tuple));
                return unwrap_tuple_fold<Tuple, next_state, I + 1>::call(
                    tuple, std::forward<Vs>(vs)...);
            }
        };

        template <typename Tuple, typename ...Vs, std::size_t I>
        struct unwrap_tuple_fold<Tuple, detail::pack<Vs...>, I, /*End=*/false>
          : unwrap_tuple_impl<Tuple, detail::pack<Vs...>, I>
        {};

        template <typename Tuple, typename ...Vs, std::size_t I>
        struct unwrap_tuple_fold<Tuple, detail::pack<Vs...>, I, /*End=*/true>
        {
            typedef util::tuple<Vs...> type;

            static type call(Tuple& /*tuple*/, Vs&&... vs)
            {
                return util::forward_as_tuple(std::forward<Vs>(vs)...);
            }
        };

        template <typename T>
        struct unwrap_impl<
            T,
            typename std::enable_if<traits::is_future_tuple<T>::value>::type
        >
        {
            typedef typename unwrap_tuple_fold<T>::type type;

            template <typename Tuple>
            static type call(Tuple&& tuple)
            {
                return unwrap_tuple_fold<Tuple&>::call(tuple);
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename F, typename T,
            typename TD = typename decay<T>::type, typename Enable = void>
        struct unwrapped_impl_result
        {};

        template <typename F, typename T, typename TD>
        struct unwrapped_impl_result<
            F, T, TD,
            typename std::enable_if<traits::is_future<TD>::value>::type
        > : util::detail::fused_result_of<
                F(typename unwrap_impl<util::tuple<TD> >::type)
            >
        {};

        template <typename F, typename T, typename TD>
        struct unwrapped_impl_result<
            F, T, TD,
            typename std::enable_if<traits::is_future_range<TD>::value>::type
        > : std::conditional<
                unwrap_impl<TD>::is_void::value
              , util::detail::fused_result_of<
                    F(util::tuple<>)
                >
              , util::detail::fused_result_of<
                    F(util::tuple<typename unwrap_impl<TD>::type>)
                >
            >::type
        {};

        template <typename F, typename T, typename TD>
        struct unwrapped_impl_result<
            F, T, TD,
            typename std::enable_if<traits::is_future_tuple<TD>::value>::type
        > : util::detail::fused_result_of<
                F(typename unwrap_impl<TD>::type)
            >
        {};

        template <typename F>
        struct unwrapped_impl
        {
            explicit unwrapped_impl(F const& f)
              : f_(f)
            {}

            explicit unwrapped_impl(F&& f)
              : f_(std::move(f))
            {}

            unwrapped_impl(unwrapped_impl && other)
              : f_(std::move(other.f_))
            {}

            unwrapped_impl(unwrapped_impl const & other)
              : f_(other.f_)
            {}

            unwrapped_impl &operator=(unwrapped_impl && other)
            {
                f_ = std::move(other.f_);
                return *this;
            }

            unwrapped_impl &operator=(unwrapped_impl const & other)
            {
                f_ = other.f_;
                return *this;
            }

            template <typename Delayed = unwrapped_impl>
            HPX_FORCEINLINE
            typename util::result_of<Delayed()>::type
            operator()()
            {
                return util::invoke_fused(f_,
                    util::make_tuple());
            }

            // future
            template <typename T0>
            HPX_FORCEINLINE
            typename util::lazy_enable_if<
                traits::is_future<typename decay<T0>::type>::value
              , unwrapped_impl_result<F, T0>
            >::type operator()(T0&& t0)
            {
                typedef
                    unwrap_impl<util::tuple<typename decay<T0>::type> >
                    unwrap_impl_t;

                return util::invoke_fused(f_,
                    unwrap_impl_t::call(util::forward_as_tuple(t0)));
            }

            // future-range
            template <typename T0>
            HPX_FORCEINLINE
            typename util::lazy_enable_if<
                traits::is_future_range<typename decay<T0>::type>::value
             && !unwrap_impl<typename decay<T0>::type>::is_void::value
              , unwrapped_impl_result<F, T0>
            >::type operator()(T0&& t0)
            {
                typedef
                    unwrap_impl<typename decay<T0>::type>
                    unwrap_impl_t;

                return util::invoke_fused(f_,
                    util::forward_as_tuple(unwrap_impl_t::call(t0)));
            }

            template <typename T0>
            HPX_FORCEINLINE
            typename util::lazy_enable_if<
                traits::is_future_range<typename decay<T0>::type>::value
             && unwrap_impl<typename decay<T0>::type>::is_void::value
              , unwrapped_impl_result<F, T0>
            >::type operator()(T0&& t0)
            {
                typedef
                    unwrap_impl<typename decay<T0>::type>
                    unwrap_impl_t;

                unwrap_impl_t::call(t0);
                return util::invoke_fused(f_,
                    util::forward_as_tuple());
            }

            // future-tuple
            template <typename T0>
            HPX_FORCEINLINE
            typename util::lazy_enable_if<
                traits::is_future_tuple<typename decay<T0>::type>::value
              , unwrapped_impl_result<F, T0>
            >::type operator()(T0&& t0)
            {
                typedef
                    unwrap_impl<typename decay<T0>::type>
                    unwrap_impl_t;

                return util::invoke_fused(f_,
                    unwrap_impl_t::call(t0));
            }

            template <typename ...Ts>
            HPX_FORCEINLINE
            typename unwrapped_impl_result<
                F,
                util::tuple<typename std::decay<Ts>::type...>
            >::type operator()(Ts&&... vs)
            {
                typedef
                    unwrap_impl<util::tuple<
                        typename std::decay<Ts>::type...> >
                    unwrap_impl_t;

                return util::invoke_fused(f_,
                    unwrap_impl_t::call(util::forward_as_tuple(
                        std::forward<Ts>(vs)...)));
            }

            F f_;
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Pack, typename Enable = void>
        struct unwrap_dispatch;

        // - hpx::lcos::(shared_)?future<T>
        // - Range (begin(), end()) of futures
        // - hpx::util::tuple<future<T>...>
        template <typename T>
        struct unwrap_dispatch<util::detail::pack<T>,
            typename std::enable_if<
                traits::is_future<T>::value ||
                    traits::is_future_range<T>::value ||
                    traits::is_future_tuple<T>::value
            >::type>
        {
            typedef typename unwrap_impl<T>::type type;

            template <typename Future>
            static typename unwrap_impl<typename decay<Future>::type>::type
            call(Future && f)
            {
                typedef unwrap_impl<typename decay<Future>::type> impl_type;
                return impl_type::call(std::forward<Future>(f));
            }
        };

        // Delayed function unwrapping: returns a callable object which unwraps
        template <typename T>
        struct unwrap_dispatch<util::detail::pack<T>,
            typename std::enable_if<
                !traits::is_future<T>::value &&
                    !traits::is_future_range<T>::value &&
                    !traits::is_future_tuple<T>::value
            >::type>
        {
            typedef unwrapped_impl<T> type;

            template <typename F>
            static unwrapped_impl<typename decay<F>::type>
            call(F && f)
            {
                // Return the callable object which performs the unwrap upon
                // invocation
                typedef unwrapped_impl<typename decay<F>::type> impl_type;
                return impl_type(std::forward<F>(f));
            }
        };

        // Unwraps a hpx::util::tuple which conatains futures
        template <typename ... Ts>
        struct unwrap_dispatch<util::detail::pack<Ts...>,
            typename std::enable_if<
                traits::is_future_tuple<util::tuple<Ts...> >::value &&
                    (sizeof...(Ts) > 1)
            >::type>
        {
            typedef typename unwrap_impl<util::tuple<Ts...> >::type type;

            template <typename ... Ts_>
            static typename unwrap_impl<
                util::tuple<typename decay<Ts_>::type...>
            >::type
            call(Ts_ &&... ts)
            {
                typedef unwrap_impl<
                        util::tuple<typename decay<Ts_>::type...>
                    > impl_type;

                return impl_type::call(
                        util::forward_as_tuple(std::forward<Ts_>(ts)...)
                    );
            }
        };

        template <typename ... Ts>
        struct unwrap_dispatch_result
        {
            typedef typename detail::unwrap_dispatch<
                    util::detail::pack<typename std::decay<Ts>::type...>
                >::type type;
        };
    } // end namespace detail
    /// \endcond

    /// A multi-usable helper function for retrieving the actual result of
    /// any hpx::lcos::future which is wrapped in an arbitrary way.
    ///
    /// unwrapped supports multiple applications, the underlying
    /// implementation is chosen based on the given arguments:
    ///
    /// - For a single callable object as argument,
    ///   the **deferred form** is used, which makes the function to return a
    ///   callable object that unwraps the input and passes it to the
    ///   given callable object upon invocation:
    ///   ```cpp
    ///   auto add = [](int left, int right) {
    ///       return left + right;
    ///   };
    ///   auto unwrapper = hpx:util:::unwrapped(add);
    ///   hpx::util::tuple<hpx::future<int>, hpx::future<int>> tuple = ...;
    ///   int result = unwrapper(tuple);
    ///   ```
    ///   The `unwrapper` object could be used to connect the `add` function
    ///   to the continuation handler of a hpx::future.
    ///
    /// - For any other input, the **immediate form** is used,
    ///   which unwraps the given pack of arguments,
    ///   so that any hpx::lcos::future object is replaced by
    ///   its future result type in the pack:
    ///       - `hpx::future<int>` -> `int`
    ///       - `hpx::future<std::vector<float>>` -> `std::vector<float>`
    ///       - `std::vector<future<float>>` -> `std::vector<float>`
    ///
    /// \param ts the argument pack that determines the used implementation
    ///
    /// \returns Depending on the chosen implementation the return type is
    ///          either a hpx::util::tuple containing unwrapped hpx::futures
    ///          when the *immediate form* is used.
    ///          If the *deferred form* is used, the function returns a
    ///          callable object, which unwrapps and forwards its arguments
    ///          when called, as desribed above.
    ///
    /// \throws std::exception like object in case the immediate application is
    ///         used and if any of the given hpx::lcos::future objects were
    ///         resolved through an exception.
    ///         See hpx::lcos::future::get() for details.
    ///
    template <typename ... Ts>
    typename detail::unwrap_dispatch_result<Ts...>::type
    unwrapped(Ts &&... ts)
    {
        // Select the underlying implementation
        typedef detail::unwrap_dispatch<
                util::detail::pack<typename std::decay<Ts>::type...>
            > impl_type;
        return impl_type::call(std::forward<Ts>(ts)...);
    }

    /// Provides an additional implementation of unwrapped which
    /// unwraps nested hpx::futures within a two-level depth.
    ///
    /// See hpx::util::unwrapped() for details.
    template <typename Future>
    typename util::lazy_enable_if<
        traits::is_future<typename decay<Future>::type>::value
     || traits::is_future_range<typename decay<Future>::type>::value
     || traits::is_future_tuple<typename decay<Future>::type>::value
      , detail::unwrap_impl<typename detail::unwrap_impl<
            typename decay<Future>::type
        > >
    >::type unwrapped2(Future&& f)
    {
        return unwrapped(unwrapped(std::forward<Future>(f)));
    }

    /// \copydoc unwrapped2
    template <typename F>
    typename std::enable_if<
        !traits::is_future<typename decay<F>::type>::value
     && !traits::is_future_range<typename decay<F>::type>::value
     && !traits::is_future_tuple<typename decay<F>::type>::value
      , detail::unwrapped_impl<detail::unwrapped_impl<
            typename std::decay<F>::type
        > >
    >::type unwrapped2(F && f)
    {
        typedef detail::unwrapped_impl<detail::unwrapped_impl<
            typename std::decay<F>::type
        > > result_type;

        detail::unwrapped_impl<typename std::decay<F>::type >
            res(std::forward<F>(f));

        return result_type(std::move(res));
    }
}}

#endif
