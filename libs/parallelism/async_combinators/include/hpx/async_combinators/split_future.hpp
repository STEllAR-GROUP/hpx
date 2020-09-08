//  Copyright (c) 2016-2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file lcos/split_future.hpp

#pragma once

#if defined(DOXYGEN)
namespace hpx {
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
    template <typename... Ts>
    inline tuple<future<Ts>...> split_future(future<tuple<Ts...>>&& f);

    /// The function \a split_future is an operator allowing to split a given
    /// future of a sequence of values (any std::vector)
    /// into a std::vector of futures where each future represents
    /// one of the values from the original std::vector. In some sense this
    /// function provides the inverse operation of \a when_all.
    ///
    /// \param f    [in] A future holding an arbitrary sequence of values stored
    ///             in a std::vector.
    /// \param size [in] The number of elements the vector will hold once the
    ///             input future has become ready
    ///
    /// \return     Returns a std::vector of futures, where each future refers
    ///             to the corresponding value in the input parameter. All of
    ///             the returned futures become ready once the input future has
    ///             become ready. If the input future is exceptional, all output
    ///             futures will be exceptional as well.
    ///
    template <typename T>
    inline std::vector<future<T>> split_future(
        future<std::vector<T>>&& f, std::size_t size);
}    // namespace hpx

#else    // DOXYGEN

#include <hpx/config.hpp>
#include <hpx/datastructures/tuple.hpp>
#include <hpx/functional/deferred_call.hpp>
#include <hpx/futures/detail/future_data.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/futures/packaged_continuation.hpp>
#include <hpx/futures/traits/acquire_future.hpp>
#include <hpx/futures/traits/acquire_shared_state.hpp>
#include <hpx/futures/traits/future_access.hpp>
#include <hpx/futures/traits/future_traits.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/modules/memory.hpp>
#include <hpx/type_support/pack.hpp>
#include <hpx/type_support/unused.hpp>

#include <array>
#include <cstddef>
#include <exception>
#include <type_traits>
#include <utility>
#include <vector>

#if !defined(HPX_COMPUTE_DEVICE_CODE)

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos {
    namespace detail {
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
                std::exception_ptr p;

                try
                {
                    typedef typename traits::future_traits<T>::type result_type;
                    result_type* result = state->get_result();
                    this->base_type::set_value(
                        std::move(hpx::util::get<I>(*result)));
                    return;
                }
                catch (...)
                {
                    p = std::current_exception();
                }

                // The exception is set outside the catch block since
                // set_exception may yield. Ending the catch block on a
                // different worker thread than where it was started may lead
                // to segfaults.
                this->base_type::set_exception(std::move(p));
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
                hpx::intrusive_ptr<split_nth_continuation> this_(this);
                shared_state_ptr const& state =
                    hpx::traits::detail::get_shared_state(future);

                state->execute_deferred();
                state->set_on_completed(util::deferred_call(
                    &split_nth_continuation::on_ready<I, Future>,
                    std::move(this_), state));
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Result, typename Tuple, std::size_t I,
            typename Future>
        inline typename hpx::traits::detail::shared_state_ptr<
            typename hpx::util::tuple_element<I, Tuple>::type>::type
        extract_nth_continuation(Future& future)
        {
            typedef split_nth_continuation<Result> shared_state;

            typename hpx::traits::detail::shared_state_ptr<Result>::type p(
                new shared_state());

            static_cast<shared_state*>(p.get())->template attach<I>(future);
            return p;
        }

        ///////////////////////////////////////////////////////////////////////
        template <std::size_t I, typename Tuple>
        HPX_FORCEINLINE
            hpx::future<typename hpx::util::tuple_element<I, Tuple>::type>
            extract_nth_future(hpx::future<Tuple>& future)
        {
            typedef
                typename hpx::util::tuple_element<I, Tuple>::type result_type;

            return hpx::traits::future_access<hpx::future<result_type>>::create(
                extract_nth_continuation<result_type, Tuple, I>(future));
        }

        template <std::size_t I, typename Tuple>
        HPX_FORCEINLINE
            hpx::future<typename hpx::util::tuple_element<I, Tuple>::type>
            extract_nth_future(hpx::shared_future<Tuple>& future)
        {
            typedef
                typename hpx::util::tuple_element<I, Tuple>::type result_type;

            return hpx::traits::future_access<hpx::future<result_type>>::create(
                extract_nth_continuation<result_type, Tuple, I>(future));
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename... Ts, std::size_t... Is>
        HPX_FORCEINLINE hpx::util::tuple<hpx::future<Ts>...>
        split_future_helper(hpx::future<hpx::util::tuple<Ts...>>&& f,
            hpx::util::index_pack<Is...>)
        {
            return hpx::util::make_tuple(extract_nth_future<Is>(f)...);
        }

        template <typename... Ts, std::size_t... Is>
        HPX_FORCEINLINE hpx::util::tuple<hpx::future<Ts>...>
        split_future_helper(hpx::shared_future<hpx::util::tuple<Ts...>>&& f,
            hpx::util::index_pack<Is...>)
        {
            return hpx::util::make_tuple(extract_nth_future<Is>(f)...);
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename T1, typename T2>
        HPX_FORCEINLINE std::pair<hpx::future<T1>, hpx::future<T2>>
        split_future_helper(hpx::future<std::pair<T1, T2>>&& f)
        {
            return std::make_pair(
                extract_nth_future<0>(f), extract_nth_future<1>(f));
        }

        template <typename T1, typename T2>
        HPX_FORCEINLINE std::pair<hpx::future<T1>, hpx::future<T2>>
        split_future_helper(hpx::shared_future<std::pair<T1, T2>>&& f)
        {
            return std::make_pair(
                extract_nth_future<0>(f), extract_nth_future<1>(f));
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename ContResult>
        class split_continuation : public future_data<ContResult>
        {
            typedef future_data<ContResult> base_type;

        private:
            template <typename T>
            void on_ready(std::size_t i,
                typename traits::detail::shared_state_ptr_for<T>::type const&
                    state)
            {
                std::exception_ptr p;

                try
                {
                    typedef typename traits::future_traits<T>::type result_type;
                    result_type* result = state->get_result();
                    if (i >= result->size())
                    {
                        HPX_THROW_EXCEPTION(length_error,
                            "split_continuation::on_ready",
                            "index out of bounds");
                    }
                    this->base_type::set_value(std::move((*result)[i]));
                    return;
                }
                catch (...)
                {
                    p = std::current_exception();
                }

                // The exception is set outside the catch block since
                // set_exception may yield. Ending the catch block on a
                // different worker thread than where it was started may lead
                // to segfaults.
                this->base_type::set_exception(std::move(p));
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
                hpx::intrusive_ptr<split_continuation> this_(this);
                shared_state_ptr const& state =
                    hpx::traits::detail::get_shared_state(future);

                state->execute_deferred();
                state->set_on_completed(
                    util::deferred_call(&split_continuation::on_ready<Future>,
                        std::move(this_), i, state));
            }
        };

        template <typename T, typename Future>
        inline hpx::future<T> extract_future_array(
            std::size_t i, Future& future)
        {
            typedef split_continuation<T> shared_state;

            typename hpx::traits::detail::shared_state_ptr<T>::type p(
                new shared_state());

            static_cast<shared_state*>(p.get())->attach(i, future);
            return hpx::traits::future_access<hpx::future<T>>::create(p);
        }

        template <std::size_t N, typename T, typename Future>
        inline std::array<hpx::future<T>, N> split_future_helper_array(
            Future&& f)
        {
            std::array<hpx::future<T>, N> result;

            for (std::size_t i = 0; i != N; ++i)
                result[i] = extract_future_array<T>(i, f);

            return result;
        }

        template <typename T, typename Future>
        inline std::vector<hpx::future<T>> split_future_helper_vector(
            Future&& f, std::size_t size)
        {
            std::vector<hpx::future<T>> result;
            result.reserve(size);

            for (std::size_t i = 0; i != size; ++i)
                result.push_back(extract_future_array<T>(i, f));

            return result;
        }
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    template <typename... Ts>
    HPX_FORCEINLINE hpx::util::tuple<hpx::future<Ts>...> split_future(
        hpx::future<hpx::util::tuple<Ts...>>&& f)
    {
        return detail::split_future_helper(std::move(f),
            typename hpx::util::make_index_pack<sizeof...(Ts)>::type());
    }

    HPX_FORCEINLINE hpx::util::tuple<hpx::future<void>> split_future(
        hpx::future<hpx::util::tuple<>>&& f)
    {
        return hpx::util::make_tuple(hpx::future<void>(std::move(f)));
    }

    template <typename... Ts>
    HPX_FORCEINLINE hpx::util::tuple<hpx::future<Ts>...> split_future(
        hpx::shared_future<hpx::util::tuple<Ts...>>&& f)
    {
        return detail::split_future_helper(std::move(f),
            typename hpx::util::make_index_pack<sizeof...(Ts)>::type());
    }

    HPX_FORCEINLINE hpx::util::tuple<hpx::future<void>> split_future(
        hpx::shared_future<hpx::util::tuple<>>&& f)
    {
        return hpx::util::make_tuple(hpx::make_future<void>(std::move(f)));
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename T1, typename T2>
    HPX_FORCEINLINE std::pair<hpx::future<T1>, hpx::future<T2>> split_future(
        hpx::future<std::pair<T1, T2>>&& f)
    {
        return detail::split_future_helper(std::move(f));
    }

    template <typename T1, typename T2>
    HPX_FORCEINLINE std::pair<hpx::future<T1>, hpx::future<T2>> split_future(
        hpx::shared_future<std::pair<T1, T2>>&& f)
    {
        return detail::split_future_helper(std::move(f));
    }

    ///////////////////////////////////////////////////////////////////////////
    template <std::size_t N, typename T>
    HPX_FORCEINLINE std::array<hpx::future<T>, N> split_future(
        hpx::future<std::array<T, N>>&& f)
    {
        return detail::split_future_helper_array<N, T>(std::move(f));
    }

    template <typename T>
    HPX_FORCEINLINE std::array<hpx::future<void>, 1> split_future(
        hpx::future<std::array<T, 0>>&& f)
    {
        std::array<hpx::future<void>, 1> result;
        result[0] = hpx::future<void>(std::move(f));
        return result;
    }

    template <std::size_t N, typename T>
    HPX_FORCEINLINE std::array<hpx::future<T>, N> split_future(
        hpx::shared_future<std::array<T, N>>&& f)
    {
        return detail::split_future_helper_array<N, T>(std::move(f));
    }

    template <typename T>
    HPX_FORCEINLINE std::array<hpx::future<void>, 1> split_future(
        hpx::shared_future<std::array<T, 0>>&& f)
    {
        std::array<hpx::future<void>, 1> result;
        result[0] = hpx::make_future<void>(std::move(f));
        return result;
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    HPX_FORCEINLINE std::vector<hpx::future<T>> split_future(
        hpx::future<std::vector<T>>&& f, std::size_t size)
    {
        return detail::split_future_helper_vector<T>(std::move(f), size);
    }

    template <typename T>
    HPX_FORCEINLINE std::vector<hpx::future<T>> split_future(
        hpx::shared_future<std::vector<T>>&& f, std::size_t size)
    {
        return detail::split_future_helper_vector<T>(std::move(f), size);
    }
}}    // namespace hpx::lcos

namespace hpx {
    using lcos::split_future;
}
#endif

#endif
