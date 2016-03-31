//  Copyright (c)      2013 Thomas Heller
//  Copyright (c)      2014 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_UNWRAPPED_HPP
#define HPX_UTIL_UNWRAPPED_HPP

#include <hpx/lcos/future.hpp>
#include <hpx/traits/is_callable.hpp>
#include <hpx/traits/is_future.hpp>
#include <hpx/traits/is_future_range.hpp>
#include <hpx/traits/is_future_tuple.hpp>
#include <hpx/util/invoke_fused.hpp>
#include <hpx/util/result_of.hpp>
#include <hpx/util/tuple.hpp>
#include <hpx/util/detail/pack.hpp>

#include <boost/mpl/eval_if.hpp>
#include <boost/utility/enable_if.hpp>

#include <type_traits>
#include <utility>

namespace hpx { namespace util
{
    namespace detail
    {
        template <typename T, typename Enable = void>
        struct unwrap_impl;

        ///////////////////////////////////////////////////////////////////////
        template <typename T>
        struct unwrap_impl<
            T,
            typename boost::enable_if<traits::is_future<T> >::type
        >
        {
            typedef typename traits::future_traits<T>::type value_type;
            typedef typename boost::is_void<value_type>::type is_void;

            typedef typename traits::future_traits<T>::result_type type;

            template <typename Future>
            static type call(Future& future, /*is_void=*/boost::mpl::false_)
            {
                return future.get();
            }

            template <typename Future>
            static type call(Future& future, /*is_void=*/boost::mpl::true_)
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
        template <typename T>
        struct unwrap_impl<
            T,
            typename boost::enable_if<traits::is_future_range<T> >::type
        >
        {
            typedef typename T::value_type future_type;
            typedef typename traits::future_traits<future_type>::type value_type;
            typedef typename boost::is_void<value_type>::type is_void;

            typedef typename boost::mpl::if_<
                is_void, void, std::vector<value_type>
            >::type type;

            template <typename Range>
            static type call(Range& range, /*is_void=*/boost::mpl::false_)
            {
                type result;
                for (typename Range::value_type& f : range)
                {
                    result.push_back(unwrap_impl<future_type>::call(f));
                }

                return result;
            }

            template <typename Range>
            static type call(Range& range, /*is_void=*/boost::mpl::true_)
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
            typename boost::enable_if<traits::is_future_tuple<T> >::type
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
            typename boost::enable_if<traits::is_future<TD> >::type
        > : util::detail::fused_result_of<
                F(typename unwrap_impl<util::tuple<TD> >::type)
            >
        {};

        template <typename F, typename T, typename TD>
        struct unwrapped_impl_result<
            F, T, TD,
            typename boost::enable_if<traits::is_future_range<TD> >::type
        > : boost::mpl::if_<
                typename unwrap_impl<TD>::is_void
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
            typename boost::enable_if<traits::is_future_tuple<TD> >::type
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

            template <typename Sig>
            struct result;

            template <typename This>
            struct result<This()>
              : boost::mpl::eval_if_c<
                    traits::is_callable<F()>::value
                  , util::result_of<F()>
                  , boost::mpl::identity<util::unused_type>
                >
            {};

            HPX_FORCEINLINE
            typename result<unwrapped_impl()>::type
            operator()()
            {
                return util::invoke_fused(f_,
                    util::make_tuple());
            }

            template <typename This, typename T, typename ...Ts>
            struct result<This(T, Ts...)>
              : boost::mpl::if_c<
                    (util::detail::pack<Ts...>::size == 0)
                  , unwrapped_impl_result<F, T>
                  , unwrapped_impl_result<F, util::tuple<
                        typename std::decay<T>::type
                      , typename std::decay<Ts>::type...> >
                >::type
            {};

            // future
            template <typename T0>
            HPX_FORCEINLINE
            typename boost::lazy_enable_if_c<
                traits::is_future<typename decay<T0>::type>::value
              , result<unwrapped_impl(T0)>
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
            typename boost::lazy_enable_if_c<
                traits::is_future_range<typename decay<T0>::type>::value
             && !unwrap_impl<typename decay<T0>::type>::is_void::value
              , result<unwrapped_impl(T0)>
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
            typename boost::lazy_enable_if_c<
                traits::is_future_range<typename decay<T0>::type>::value
             && unwrap_impl<typename decay<T0>::type>::is_void::value
              , result<unwrapped_impl(T0)>
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
            typename boost::lazy_enable_if_c<
                traits::is_future_tuple<typename decay<T0>::type>::value
              , result<unwrapped_impl(T0)>
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
            typename result<unwrapped_impl(Ts...)>::type
            operator()(Ts&&... vs)
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
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Future>
    typename boost::lazy_enable_if_c<
        traits::is_future<typename decay<Future>::type>::value
     || traits::is_future_range<typename decay<Future>::type>::value
     || traits::is_future_tuple<typename decay<Future>::type>::value
      , detail::unwrap_impl<typename decay<Future>::type>
    >::type unwrapped(Future&& f)
    {
        typedef
            detail::unwrap_impl<typename decay<Future>::type>
            unwrap_impl_t;

        return unwrap_impl_t::call(std::forward<Future>(f));
    }

    template <typename F>
    typename boost::disable_if_c<
        traits::is_future<typename decay<F>::type>::value
     || traits::is_future_range<typename decay<F>::type>::value
     || traits::is_future_tuple<typename decay<F>::type>::value
      , detail::unwrapped_impl<typename std::decay<F>::type >
    >::type unwrapped(F && f)
    {
        detail::unwrapped_impl<typename std::decay<F>::type >
            res(std::forward<F>(f));

        return res;
    }

    template <typename ...Ts>
    typename boost::lazy_enable_if_c<
        traits::is_future_tuple<util::tuple<
            typename std::decay<Ts>::type...
        > >::value
      , detail::unwrap_impl<util::tuple<
            typename std::decay<Ts>::type...
        > >
    >::type unwrapped(Ts&&... vs)
    {
        typedef detail::unwrap_impl<util::tuple<
            typename std::decay<Ts>::type...
        > > unwrap_impl_t;

        return unwrap_impl_t::call(util::forward_as_tuple(
            std::forward<Ts>(vs)...));
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Future>
    typename boost::lazy_enable_if_c<
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

    template <typename F>
    typename boost::disable_if_c<
        traits::is_future<typename decay<F>::type>::value
     || traits::is_future_range<typename decay<F>::type>::value
     || traits::is_future_tuple<typename decay<F>::type>::value
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
