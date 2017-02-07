//  Copyright (c) 2007-2017 Hartmut Kaiser
//  Copyright (c) 2013 Agustin Berge
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
#include <hpx/lcos/future.hpp>
#include <hpx/lcos/when_some.hpp>
#include <hpx/traits/acquire_future.hpp>
#include <hpx/traits/acquire_shared_state.hpp>
#include <hpx/traits/future_access.hpp>
#include <hpx/traits/is_future.hpp>
#include <hpx/traits/is_future_range.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/deferred_call.hpp>
#include <hpx/util/tuple.hpp>
#include <hpx/util/unwrap_ref.hpp>

#include <boost/intrusive_ptr.hpp>
#include <boost/range/functions.hpp>

#include <algorithm>
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

        ///////////////////////////////////////////////////////////////////////
        template <typename Tuple>
        struct when_all_frame //-V690
          : hpx::lcos::detail::future_data<typename when_all_result<Tuple>::type>
        {
            typedef typename when_all_result<Tuple>::type result_type;
            typedef hpx::lcos::future<result_type> type;
            typedef hpx::lcos::detail::future_data<result_type> base_type;

        private:
            // workaround gcc regression wrongly instantiating constructors
            when_all_frame();
            when_all_frame(when_all_frame const&);

            template <std::size_t I>
            struct is_end
              : std::integral_constant<
                    bool,
                    util::tuple_size<Tuple>::value == I
                >
            {};

        public:
            typedef typename base_type::init_no_addref init_no_addref;

            template <typename Tuple_>
            when_all_frame(Tuple_&& t)
              : t_(std::forward<Tuple_>(t))
            {}

            template <typename Tuple_>
            when_all_frame(Tuple_&& t, init_no_addref no_addref)
              : base_type(no_addref), t_(std::forward<Tuple_>(t))
            {}

        protected:
            // End of the tuple is reached
            template <std::size_t I>
            HPX_FORCEINLINE
            void do_await(std::true_type)
            {
                this->set_value(when_all_result<Tuple>::call(std::move(t_)));
            }

            // Current element is a range of futures
            template <std::size_t I, typename Iter>
            void await_range(Iter next, Iter end)
            {
                typedef typename std::iterator_traits<Iter>::value_type
                    future_type;
                typedef typename traits::future_traits<future_type>::type
                    future_result_type;

                void (when_all_frame::*f)(Iter, Iter) =
                    &when_all_frame::await_range<I>;

                for (/**/; next != end; ++next)
                {
                    typename traits::detail::shared_state_ptr<
                            future_result_type
                        >::type next_future_data =
                            traits::detail::get_shared_state(*next);

                    if (!next_future_data->is_ready())
                    {
                        next_future_data->execute_deferred();

                        // execute_deferred might have made the future ready
                        if (!next_future_data->is_ready())
                        {
                            // Attach a continuation to this future which will
                            // re-evaluate it and continue to the next element
                            // in the sequence (if any).
                            boost::intrusive_ptr<when_all_frame> this_(this);
                            next_future_data->set_on_completed(util::deferred_call(
                                f, std::move(this_),
                                std::move(next), std::move(end)));
                            return;
                        }
                    }
                }

                do_await<I + 1>(is_end<I + 1>());
            }

            template <std::size_t I>
            HPX_FORCEINLINE
            void await_next(std::false_type, std::true_type)
            {
                await_range<I>(
                    boost::begin(util::unwrap_ref(util::get<I>(t_))),
                    boost::end(util::unwrap_ref(util::get<I>(t_))));
            }

            // Current element is a simple future
            template <std::size_t I>
            HPX_FORCEINLINE
            void await_next(std::true_type, std::false_type)
            {
                typedef typename util::decay_unwrap<
                    typename util::tuple_element<I, Tuple>::type
                >::type future_type;

                future_type& f_ = util::get<I>(t_);

                typedef typename traits::future_traits<future_type>::type
                    future_result_type;

                typename traits::detail::shared_state_ptr<
                        future_result_type
                    >::type next_future_data =
                        traits::detail::get_shared_state(f_);

                if (!next_future_data->is_ready())
                {
                    next_future_data->execute_deferred();

                    // execute_deferred might have made the future ready
                    if (!next_future_data->is_ready())
                    {
                        // Attach a continuation to this future which will
                        // re-evaluate it and continue to the next argument
                        // (if any).
                        void (when_all_frame::*f)(std::true_type, std::false_type) =
                            &when_all_frame::await_next<I>;

                        boost::intrusive_ptr<when_all_frame> this_(this);
                        next_future_data->set_on_completed(util::deferred_call(
                            f, std::move(this_), std::true_type(), std::false_type()));
                        return;
                    }
                }

                do_await<I + 1>(is_end<I + 1>());
            }

            template <std::size_t I>
            HPX_FORCEINLINE
            void do_await(std::false_type)
            {
                typedef typename util::decay_unwrap<
                    typename util::tuple_element<I, Tuple>::type
                >::type future_type;

                typedef traits::is_future<future_type> is_future;
                typedef traits::is_future_range<future_type> is_range;

                await_next<I>(is_future(), is_range());
            }

        public:
            HPX_FORCEINLINE void do_await()
            {
                do_await<0>(is_end<0>());
            }

        private:
            Tuple t_;
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Range>
    typename std::enable_if<traits::is_future_range<Range>::value,
        lcos::future<typename std::decay<Range>::type> >::type //-V659
    when_all(Range&& values)
    {
        typedef detail::when_all_frame<util::tuple<Range> > frame_type;
        typedef typename frame_type::init_no_addref init_no_addref;

        boost::intrusive_ptr<frame_type> p(new frame_type(
            util::forward_as_tuple(std::move(values)), init_no_addref()),
            false);
        p->do_await();

        using traits::future_access;
        return future_access<typename frame_type::type>::create(std::move(p));
    }

    template <typename Range>
    typename std::enable_if<traits::is_future_range<Range>::value,
        lcos::future<typename std::decay<Range>::type> >::type
    when_all(Range& values)
    {
        Range values_ = traits::acquire_future<Range>()(values);
        return lcos::when_all(std::move(values_));
    }

    template <typename Iterator, typename Container =
        std::vector<typename lcos::detail::future_iterator_traits<Iterator>::type> >
    lcos::future<Container>
    when_all(Iterator begin, Iterator end)
    {
        Container values;

        typename std::iterator_traits<Iterator>::
            difference_type difference = std::distance(begin, end);
        if (difference > 0)
            traits::detail::reserve_if_vector(
                values, static_cast<std::size_t>(difference));

        std::transform(begin, end, std::back_inserter(values),
            traits::acquire_future_disp());

        return lcos::when_all(std::move(values));
    }

    inline lcos::future<util::tuple<> > //-V524
    when_all()
    {
        typedef util::tuple<> result_type;
        return lcos::make_ready_future(result_type());
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Iterator, typename Container =
        std::vector<typename lcos::detail::future_iterator_traits<Iterator>::type> >
    lcos::future<Container>
    when_all_n(Iterator begin, std::size_t count)
    {
        Container values;
        traits::detail::reserve_if_vector(values, count);

        traits::acquire_future_disp func;
        for (std::size_t i = 0; i != count; ++i)
            values.push_back(func(*begin++));

        return lcos::when_all(std::move(values));
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename... Ts>
    typename detail::when_all_frame<
        util::tuple<typename traits::acquire_future<Ts>::type...>
    >::type
    when_all(Ts&&... ts)
    {
        typedef util::tuple<
                typename traits::acquire_future<Ts>::type...
            > result_type;
        typedef detail::when_all_frame<result_type> frame_type;
        typedef typename frame_type::init_no_addref init_no_addref;

        traits::acquire_future_disp func;
        result_type values(func(std::forward<Ts>(ts))...);

        boost::intrusive_ptr<frame_type> p(
            new frame_type(std::move(values), init_no_addref()), false);
        p->do_await();

        using traits::future_access;
        return future_access<typename frame_type::type>::create(std::move(p));
    }
}}

namespace hpx
{
    using lcos::when_all;
    using lcos::when_all_n;
}

#endif // DOXYGEN
#endif
