//  Copyright (c) 2007-2017 Hartmut Kaiser
//  Copyright (c) 2013 Agustin Berge
//  Copyright (c) 2017 Denis Blank
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file lcos/when_all.hpp

#if !defined(HPX_LCOS_WHEN_ALL_APR_19_2012_1140AM)
#define HPX_LCOS_WHEN_ALL_APR_19_2012_1140AM

#if defined(DOXYGEN)
namespace hpx
{
    /// The function \a when_all is an operator allowing to join on the result
    /// of all given futures. It AND-composes all future objects given and
    /// returns a new future object representing the same list of futures
    /// after they finished executing.
    ///
    /// \param first    [in] The iterator pointing to the first element of a
    ///                 sequence of \a future or \a shared_future objects for
    ///                 which \a when_all should wait.
    /// \param last     [in] The iterator pointing to the last element of a
    ///                 sequence of \a future or \a shared_future objects for
    ///                 which \a when_all should wait.
    ///
    /// \return   Returns a future holding the same list of futures as has
    ///           been passed to \a when_all.
    ///           - future<Container<future<R>>>: If the input cardinality is
    ///             unknown at compile time and the futures are all of the
    ///             same type. The order of the futures in the output container
    ///             will be the same as given by the input iterator.
    ///
    /// \note Calling this version of \a when_all where first == last, returns
    ///       a future with an empty container that is immediately ready.
    ///       Each future and shared_future is waited upon and then copied into
    ///       the collection of the output (returned) future, maintaining the
    ///       order of the futures in the input collection.
    ///       The future returned by \a when_all will not throw an exception,
    ///       but the futures held in the output collection may.
    template <typename InputIter, typename Container =
        vector<future<typename std::iterator_traits<InputIter>::value_type>>>
    future<Container>
    when_all(InputIter first, InputIter last);

    /// The function \a when_all is an operator allowing to join on the result
    /// of all given futures. It AND-composes all future objects given and
    /// returns a new future object representing the same list of futures
    /// after they finished executing.
    ///
    /// \param values   [in] A range holding an arbitrary amount of \a future
    ///                 or \a shared_future objects for which \a when_all
    ///                 should wait.
    ///
    /// \return   Returns a future holding the same list of futures as has
    ///           been passed to when_all.
    ///           - future<Container<future<R>>>: If the input cardinality is
    ///             unknown at compile time and the futures are all of the
    ///             same type.
    ///
    /// \note Calling this version of \a when_all where the input container is
    ///       empty, returns a future with an empty container that is immediately
    ///       ready.
    ///       Each future and shared_future is waited upon and then copied into
    ///       the collection of the output (returned) future, maintaining the
    ///       order of the futures in the input collection.
    ///       The future returned by \a when_all will not throw an exception,
    ///       but the futures held in the output collection may.
    template <typename Range>
    future<Range>
    when_all(Range&& values);

    /// The function \a when_all is an operator allowing to join on the result
    /// of all given futures. It AND-composes all future objects given and
    /// returns a new future object representing the same list of futures
    /// after they finished executing.
    ///
    /// \param futures  [in] An arbitrary number of \a future or \a shared_future
    ///                 objects, possibly holding different types for which
    ///                 \a when_all should wait.
    ///
    /// \return   Returns a future holding the same list of futures as has
    ///           been passed to \a when_all.
    ///           - future<tuple<future<T0>, future<T1>, future<T2>...>>: If
    ///             inputs are fixed in number and are of heterogeneous types.
    ///             The inputs can be any arbitrary number of future objects.
    ///           - future<tuple<>> if \a when_all is called with zero arguments.
    ///             The returned future will be initially ready.
    ///
    /// \note Each future and shared_future is waited upon and then copied into
    ///       the collection of the output (returned) future, maintaining the
    ///       order of the futures in the input collection.
    ///       The future returned by \a when_all will not throw an exception,
    ///       but the futures held in the output collection may.
    template <typename ...T>
    future<tuple<future<T>...>>
    when_all(T &&... futures);

    /// The function \a when_all_n is an operator allowing to join on the result
    /// of all given futures. It AND-composes all future objects given and
    /// returns a new future object representing the same list of futures
    /// after they finished executing.
    ///
    /// \param begin    [in] The iterator pointing to the first element of a
    ///                 sequence of \a future or \a shared_future objects for
    ///                 which \a wait_all_n should wait.
    /// \param count    [in] The number of elements in the sequence starting at
    ///                 \a first.
    ///
    /// \return   Returns a future holding the same list of futures as has
    ///           been passed to \a when_all_n.
    ///           - future<Container<future<R>>>: If the input cardinality is
    ///             unknown at compile time and the futures are all of the
    ///             same type. The order of the futures in the output vector
    ///             will be the same as given by the input iterator.
    ///
    /// \throws This function will throw errors which are encountered while
    ///         setting up the requested operation only. Errors encountered
    ///         while executing the operations delivering the results to be
    ///         stored in the futures are reported through the futures
    ///         themselves.
    ///
    /// \note     As long as \a ec is not pre-initialized to \a hpx::throws this
    ///           function doesn't throw but returns the result code using the
    ///           parameter \a ec. Otherwise it throws an instance of
    ///           hpx::exception.
    ///
    /// \note     None of the futures in the input sequence are invalidated.
    template <typename InputIter, typename Container =
        vector<future<typename std::iterator_traits<InputIter>::value_type>>>
    future<Container>
    when_all_n(InputIter begin, std::size_t count);
}

#else // DOXYGEN

#include <hpx/config.hpp>
#include <hpx/lcos/detail/future_data.hpp>
#include <hpx/lcos/detail/future_transforms.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/traits/acquire_future.hpp>
#include <hpx/traits/future_access.hpp>
#include <hpx/traits/is_future.hpp>
#include <hpx/traits/is_future_range.hpp>
#include <hpx/util/pack_traversal_async.hpp>
#include <hpx/util/tuple.hpp>

#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos
{
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        template <typename T, typename Enable = void>
        struct when_all_result
        {
            typedef T type;

            static type call(T&& t)
            {
                return std::move(t);
            }
        };

        template <typename T>
        struct when_all_result<util::tuple<T>,
            typename std::enable_if<
                traits::is_future_range<T>::value
            >::type>
        {
            typedef T type;

            static type call(util::tuple<T>&& t)
            {
                return std::move(util::get<0>(t));
            }
        };

        template <typename Tuple>
        class async_when_all_frame
          : public future_data<typename when_all_result<Tuple>::type>
        {
        public:
            typedef typename when_all_result<Tuple>::type result_type;
            typedef hpx::lcos::future<result_type> type;
            typedef hpx::lcos::detail::future_data<result_type> base_type;

            explicit async_when_all_frame(
                typename base_type::init_no_addref no_addref)
              : future_data<typename when_all_result<Tuple>::type>(no_addref)
            {
            }

            template <typename T>
            auto operator()(util::async_traverse_visit_tag, T&& current)
                -> decltype(async_visit_future(std::forward<T>(current)))
            {
                return async_visit_future(std::forward<T>(current));
            }

            template <typename T, typename N>
            auto operator()(
                util::async_traverse_detach_tag, T&& current, N&& next)
                -> decltype(async_detach_future(
                    std::forward<T>(current), std::forward<N>(next)))
            {
                return async_detach_future(
                    std::forward<T>(current), std::forward<N>(next));
            }

            template <typename T>
            void operator()(util::async_traverse_complete_tag, T&& pack)
            {
                this->set_value(
                    when_all_result<Tuple>::call(std::forward<T>(pack)));
            }
        };

        template <typename... T>
        typename detail::async_when_all_frame<
            util::tuple<
                typename traits::acquire_future<T>::type...
            >
        >::type
        when_all_impl(T&&... args)
        {
            typedef util::tuple<typename traits::acquire_future<T>::type...>
                result_type;
            typedef detail::async_when_all_frame<result_type> frame_type;

            traits::acquire_future_disp func;

            typename frame_type::base_type::init_no_addref no_addref;

            auto frame = util::traverse_pack_async(
                util::async_traverse_in_place_tag<frame_type>{}, no_addref,
                func(std::forward<T>(args))...);

            using traits::future_access;
            return future_access<typename frame_type::type>::create(
                std::move(frame));
        }
    }

    template <typename First, typename Second>
    auto when_all(First&& first, Second&& second)
        -> decltype(detail::when_all_impl(
            std::forward<First>(first), std::forward<Second>(second)))
    {
        return detail::when_all_impl(
            std::forward<First>(first), std::forward<Second>(second));
    }
    template <typename Iterator,
        typename Container = std::vector<
            typename detail::future_iterator_traits<Iterator>::type>>
    future<Container> when_all(Iterator begin, Iterator end)
    {
        return detail::when_all_impl(
            detail::acquire_future_iterators<Iterator, Container>(begin, end));
    }

    inline lcos::future<util::tuple<> > //-V524
    when_all()
    {
        typedef util::tuple<> result_type;
        return lcos::make_ready_future(result_type());
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Iterator,
        typename Container = std::vector<
            typename lcos::detail::future_iterator_traits<Iterator>::type>>
    lcos::future<Container> when_all_n(Iterator begin, std::size_t count)
    {
        return detail::when_all_impl(
            detail::acquire_future_n<Iterator, Container>(begin, count));
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename... Args,
        typename std::enable_if<(sizeof...(Args) == 1U) ||
            (sizeof...(Args) > 2U)>::type* = nullptr>
    auto when_all(Args&&... args)
        -> decltype(detail::when_all_impl(std::forward<Args>(args)...))
    {
        return detail::when_all_impl(std::forward<Args>(args)...);
    }
}}

namespace hpx
{
    using lcos::when_all;
    using lcos::when_all_n;
}

#endif // DOXYGEN
#endif
