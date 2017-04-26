//  Copyright (c) 2016-2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file lcos/split_future.hpp

#if !defined(HPX_LCOS_SPLIT_FUTURE_JUL_08_2016_0824AM)
#define HPX_LCOS_SPLIT_FUTURE_JUL_08_2016_0824AM

// sadly, MSVC12 is not able to cope with split_future
#if !defined(HPX_MSVC) || HPX_MSVC >= 1900

#if defined(DOXYGEN)
namespace hpx
{
    /// The function \a split_future is an operator allowing to split a given
    /// future of a sequence of values (any tuple, std::pair, or std::array)
    /// into an equivalent container of futures where each future represents
    /// one of the values from the original future. In some sense this function
    /// provides the inverse operation of \a when_all.
    ///
    /// \param f    [in] A future holding an arbitrary sequence of values stored
    ///             in a tuple-like container. This facility supports
    ///             \a hpx::util::tuple<>, \a std::pair<T1, T2>, and
    ///             \a std::array<T, N>
    ///
    /// \return     Returns an equivalent container (same container type as
    ///             passed as the argument) of futures, where each future refers
    ///             to the corresponding value in the input parameter. All of
    ///             the returned futures become ready once the input future has
    ///             become ready. If the input future is exceptional, all output
    ///             futures will be exceptional as well.
    ///
    /// \note       The following cases are special:
    /// \code
    ///     tuple<future<void> > split_future(future<tuple<> > && f);
    ///     array<future<void>, 1> split_future(future<array<T, 0> > && f);
    /// \endcode
    ///             here the returned futures are directly representing the
    ///             futures which were passed to the function.
    ///
    template <typename ... Ts>
    inline tuple<future<Ts>...>
    split_future(future<tuple<Ts...> > && f);
}

#else // DOXYGEN

#include <hpx/config.hpp>
#include <hpx/exception.hpp>
#include <hpx/lcos/detail/future_data.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/lcos/local/packaged_continuation.hpp>
#include <hpx/traits/acquire_future.hpp>
#include <hpx/traits/acquire_shared_state.hpp>
#include <hpx/traits/future_access.hpp>
#include <hpx/traits/future_traits.hpp>
#include <hpx/util/deferred_call.hpp>
#include <hpx/util/detail/pack.hpp>
#include <hpx/util/tuple.hpp>
#include <hpx/util/unused.hpp>

#include <boost/intrusive_ptr.hpp>

#if defined(HPX_HAVE_CXX11_STD_ARRAY)
#include <array>
#endif
#include <cstddef>
#include <type_traits>
#include <utility>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos
{
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        template <typename ContResult>
        class split_nth_continuation : public future_data<ContResult>
        {
            typedef future_data<ContResult> base_type;

        private:
            template <std::size_t I, typename T>
            void on_ready(
                typename traits::detail::shared_state_ptr_for<T>::type const&
                    state)
            {
                try {
                    typedef typename traits::future_traits<T>::type result_type;
                    result_type* result = state->get_result();
                    this->base_type::set_value(
                        std::move(hpx::util::get<I>(*result)));
                }
                catch (...) {
                    this->base_type::set_exception(boost::current_exception());
                }
            }

        public:
            template <std::size_t I, typename Future>
            void attach(Future& future)
            {
                typedef
                    typename traits::detail::shared_state_ptr_for<Future>::type
                    shared_state_ptr;

                // Bind an on_completed handler to this future which will wait
                // for the future and will transfer its result to the new
                // future.
                boost::intrusive_ptr<split_nth_continuation> this_(this);
                shared_state_ptr const& state =
                    hpx::traits::detail::get_shared_state(future);

                state->execute_deferred();
                state->set_on_completed(util::deferred_call(
                    &split_nth_continuation::on_ready<I, Future>,
                    std::move(this_), state));
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Result, typename Tuple, std::size_t I, typename Future>
        inline typename hpx::traits::detail::shared_state_ptr<
            typename hpx::util::tuple_element<I, Tuple>::type
        >::type
        extract_nth_continuation(Future& future)
        {
            typedef split_nth_continuation<Result> shared_state;

            typename hpx::traits::detail::shared_state_ptr<Result>::type
                p(new shared_state());

            static_cast<shared_state*>(p.get())->template attach<I>(future);
            return p;
        }

        ///////////////////////////////////////////////////////////////////////
        template <std::size_t I, typename Tuple>
        HPX_FORCEINLINE
        hpx::future<typename hpx::util::tuple_element<I, Tuple>::type>
        extract_nth_future(hpx::future<Tuple>& future)
        {
            typedef typename hpx::util::tuple_element<
                    I, Tuple
                >::type result_type;

            return hpx::traits::future_access<
                    hpx::future<result_type>
                >::create(
                    extract_nth_continuation<result_type, Tuple, I>(future));
        }

        template <std::size_t I, typename Tuple>
        HPX_FORCEINLINE
        hpx::future<typename hpx::util::tuple_element<I, Tuple>::type>
        extract_nth_future(hpx::shared_future<Tuple>& future)
        {
            typedef typename hpx::util::tuple_element<
                    I, Tuple
                >::type result_type;

            return hpx::traits::future_access<
                    hpx::future<result_type>
                >::create(
                    extract_nth_continuation<result_type, Tuple, I>(future));
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename ... Ts, std::size_t ... Is>
        HPX_FORCEINLINE
        hpx::util::tuple<hpx::future<Ts>...>
        split_future_helper(hpx::future<hpx::util::tuple<Ts...> > && f,
            hpx::util::detail::pack_c<std::size_t, Is...>)
        {
            return hpx::util::make_tuple(extract_nth_future<Is>(f)...);
        }

        template <typename ... Ts, std::size_t ... Is>
        HPX_FORCEINLINE
        hpx::util::tuple<hpx::future<Ts>...>
        split_future_helper(hpx::shared_future<hpx::util::tuple<Ts...> > && f,
            hpx::util::detail::pack_c<std::size_t, Is...>)
        {
            return hpx::util::make_tuple(extract_nth_future<Is>(f)...);
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename T1, typename T2>
        HPX_FORCEINLINE
        std::pair<hpx::future<T1>, hpx::future<T2> >
        split_future_helper(hpx::future<std::pair<T1, T2> > && f)
        {
            return std::make_pair(extract_nth_future<0>(f),
                extract_nth_future<1>(f));
        }

        template <typename T1, typename T2>
        HPX_FORCEINLINE
        std::pair<hpx::future<T1>, hpx::future<T2> >
        split_future_helper(hpx::shared_future<std::pair<T1, T2> > && f)
        {
            return std::make_pair(extract_nth_future<0>(f),
                extract_nth_future<1>(f));
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename ContResult>
        class split_continuation : public future_data<ContResult>
        {
            typedef future_data<ContResult> base_type;

        private:
            template <typename T>
            void on_ready(
                std::size_t i,
                typename traits::detail::shared_state_ptr_for<T>::type const&
                    state)
            {
                try {
                    typedef typename traits::future_traits<T>::type result_type;
                    result_type* result = state->get_result();
                    this->base_type::set_value(std::move((*result)[i]));
                }
                catch (...) {
                    this->base_type::set_exception(boost::current_exception());
                }
            }

        public:
            template <typename Future>
            void attach(std::size_t i, Future& future)
            {
                typedef
                    typename traits::detail::shared_state_ptr_for<Future>::type
                    shared_state_ptr;

                // Bind an on_completed handler to this future which will wait
                // for the future and will transfer its result to the new
                // future.
                boost::intrusive_ptr<split_continuation> this_(this);
                shared_state_ptr const& state =
                    hpx::traits::detail::get_shared_state(future);

                state->execute_deferred();
                state->set_on_completed(util::deferred_call(
                    &split_continuation::on_ready<Future>, std::move(this_),
                    i, state));
            }
        };

        template <typename T, typename Future>
        inline hpx::future<T>
        extract_future_array(std::size_t i, Future& future)
        {
            typedef split_continuation<T> shared_state;

            typename hpx::traits::detail::shared_state_ptr<T>::type
                p(new shared_state());

            static_cast<shared_state*>(p.get())->attach(i, future);
            return hpx::traits::future_access<hpx::future<T> >::create(p);
        }

#if defined(HPX_HAVE_CXX11_STD_ARRAY)
        template <std::size_t N, typename T, typename Future>
        inline std::array<hpx::future<T>, N>
        split_future_helper_array(Future && f)
        {
            std::array<hpx::future<T>, N> result;

            for (std::size_t i = 0; i != N; ++i)
                result[i] = extract_future_array<T>(i, f);

            return result;
        }
#endif
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename ... Ts>
    HPX_FORCEINLINE hpx::util::tuple<hpx::future<Ts>...>
    split_future(hpx::future<hpx::util::tuple<Ts...> > && f)
    {
        return detail::split_future_helper(
                std::move(f),
                typename hpx::util::detail::make_index_pack<sizeof...(Ts)>::type()
            );
    }

    HPX_FORCEINLINE hpx::util::tuple<hpx::future<void> >
    split_future(hpx::future<hpx::util::tuple<> > && f)
    {
        return hpx::util::make_tuple(hpx::future<void>(std::move(f)));
    }

    template <typename ... Ts>
    HPX_FORCEINLINE hpx::util::tuple<hpx::future<Ts>...>
    split_future(hpx::shared_future<hpx::util::tuple<Ts...> > && f)
    {
        return detail::split_future_helper(
                std::move(f),
                typename hpx::util::detail::make_index_pack<sizeof...(Ts)>::type()
            );
    }

    HPX_FORCEINLINE hpx::util::tuple<hpx::future<void> >
    split_future(hpx::shared_future<hpx::util::tuple<> > && f)
    {
        return hpx::util::make_tuple(hpx::make_future<void>(std::move(f)));
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename T1, typename T2>
    HPX_FORCEINLINE std::pair<hpx::future<T1>, hpx::future<T2> >
    split_future(hpx::future<std::pair<T1, T2> > && f)
    {
        return detail::split_future_helper(std::move(f));
    }

    template <typename T1, typename T2>
    HPX_FORCEINLINE std::pair<hpx::future<T1>, hpx::future<T2> >
    split_future(hpx::shared_future<std::pair<T1, T2> > && f)
    {
        return detail::split_future_helper(std::move(f));
    }

    ///////////////////////////////////////////////////////////////////////////
#if defined(HPX_HAVE_CXX11_STD_ARRAY)
    template <std::size_t N, typename T>
    HPX_FORCEINLINE std::array<hpx::future<T>, N>
    split_future(hpx::future<std::array<T, N> > && f)
    {
        return detail::split_future_helper_array<N, T>(std::move(f));
    }

    template <typename T>
    HPX_FORCEINLINE std::array<hpx::future<void>, 1>
    split_future(hpx::future<std::array<T, 0> > && f)
    {
        std::array<hpx::future<void>, 1> result;
        result[0] = hpx::future<void>(std::move(f));
        return result;
    }

    template <std::size_t N, typename T>
    HPX_FORCEINLINE std::array<hpx::future<T>, N>
    split_future(hpx::shared_future<std::array<T, N> > && f)
    {
        return detail::split_future_helper_array<N, T>(std::move(f));
    }

    template <typename T>
    HPX_FORCEINLINE std::array<hpx::future<void>, 1>
    split_future(hpx::shared_future<std::array<T, 0> > && f)
    {
        std::array<hpx::future<void>, 1> result;
        result[0] = hpx::make_future<void>(std::move(f));
        return result;
    }
#endif
}}

namespace hpx
{
    using lcos::split_future;
}

#endif
#endif
#endif
