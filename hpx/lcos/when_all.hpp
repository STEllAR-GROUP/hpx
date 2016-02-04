//  Copyright (c) 2007-2015 Hartmut Kaiser
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

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/lcos/when_some.hpp>
#include <hpx/util/always_void.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/move.hpp>
#include <hpx/util/tuple.hpp>
#include <hpx/traits/acquire_future.hpp>
#include <hpx/traits/acquire_shared_state.hpp>

#include <boost/mpl/bool.hpp>
#include <boost/fusion/include/begin.hpp>
#include <boost/fusion/include/end.hpp>
#include <boost/fusion/include/deref.hpp>
#include <boost/fusion/include/next.hpp>
#include <boost/range/functions.hpp>
#include <boost/ref.hpp>
#include <boost/type_traits/is_same.hpp>

#include <algorithm>
#include <iterator>
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

        private:
            // workaround gcc regression wrongly instantiating constructors
            when_all_frame();
            when_all_frame(when_all_frame const&);

            typedef typename boost::fusion::result_of::end<Tuple>::type
                end_type;

        public:
            template <typename Tuple_>
            when_all_frame(Tuple_&& t)
              : t_(std::forward<Tuple_>(t))
            {}

        protected:
            // End of the tuple is reached
            template <typename TupleIter>
            HPX_FORCEINLINE
            void do_await(TupleIter&&, boost::mpl::true_)
            {
                this->set_value(when_all_result<Tuple>::call(std::move(t_)));
            }

            // Current element is a range of futures
            template <typename TupleIter, typename Iter>
            void await_range(TupleIter iter, Iter next, Iter end)
            {
                typedef typename std::iterator_traits<Iter>::value_type
                    future_type;
                typedef typename traits::future_traits<future_type>::type
                    future_result_type;

                void (when_all_frame::*f)(TupleIter, Iter, Iter) =
                    &when_all_frame::await_range;

                for (/**/; next != end; ++next)
                {
                    boost::intrusive_ptr<
                        lcos::detail::future_data<future_result_type>
                    > next_future_data =
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
                            next_future_data->set_on_completed(util::bind(
                                f, std::move(this_), std::move(iter),
                                std::move(next), std::move(end)));
                            return;
                        }
                    }
                }

                typedef typename boost::fusion::result_of::next<TupleIter>::type
                    next_type;
                typedef boost::is_same<next_type, end_type> pred;

                do_await(boost::fusion::next(iter), pred());
            }

            template <typename TupleIter>
            HPX_FORCEINLINE
            void await_next(TupleIter iter, boost::mpl::false_, boost::mpl::true_)
            {
                await_range(iter,
                    boost::begin(boost::unwrap_ref(boost::fusion::deref(iter))),
                    boost::end(boost::unwrap_ref(boost::fusion::deref(iter))));
            }

            // Current element is a simple future
            template <typename TupleIter>
            HPX_FORCEINLINE
            void await_next(TupleIter iter, boost::mpl::true_, boost::mpl::false_)
            {
                typedef typename util::decay_unwrap<
                    typename boost::fusion::result_of::deref<TupleIter>::type
                >::type future_type;

                using boost::mpl::false_;
                using boost::mpl::true_;

                future_type& f_ = boost::fusion::deref(iter);

                typedef typename traits::future_traits<future_type>::type
                    future_result_type;

                    boost::intrusive_ptr<
                        lcos::detail::future_data<future_result_type>
                    > next_future_data =
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
                        void (when_all_frame::*f)(TupleIter, true_, false_) =
                            &when_all_frame::await_next;

                        boost::intrusive_ptr<when_all_frame> this_(this);
                        next_future_data->set_on_completed(
                            hpx::util::bind(
                                f, std::move(this_), std::move(iter),
                                true_(), false_()));
                        return;
                    }
                }

                typedef typename boost::fusion::result_of::next<TupleIter>::type
                    next_type;
                typedef boost::is_same<next_type, end_type> pred;

                do_await(boost::fusion::next(iter), pred());
            }

            template <typename TupleIter>
            HPX_FORCEINLINE
            void do_await(TupleIter&& iter, boost::mpl::false_)
            {
                typedef typename util::decay_unwrap<
                    typename boost::fusion::result_of::deref<TupleIter>::type
                >::type future_type;

                typedef typename traits::is_future<future_type>::type is_future;
                typedef typename traits::is_future_range<future_type>::type is_range;

                await_next(std::move(iter), is_future(), is_range());
            }

        public:
            HPX_FORCEINLINE void do_await()
            {
                typedef typename boost::fusion::result_of::begin<Tuple>::type
                    begin_type;
                typedef boost::is_same<begin_type, end_type> pred;

                do_await(boost::fusion::begin(t_), pred());
            }

        private:
            Tuple t_;
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Range>
    typename boost::enable_if<traits::is_future_range<Range>,
        lcos::future<typename util::decay<Range>::type> >::type //-V659
    when_all(Range&& values)
    {
        typedef detail::when_all_frame<
                hpx::util::tuple<Range>
            > frame_type;

        boost::intrusive_ptr<frame_type> p(new frame_type(
            hpx::util::forward_as_tuple(std::move(values))));
        p->do_await();

        using traits::future_access;
        return future_access<typename frame_type::type>::create(std::move(p));
    }

    template <typename Range>
    typename boost::enable_if<traits::is_future_range<Range>,
        lcos::future<typename util::decay<Range>::type> >::type
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

    inline lcos::future<hpx::util::tuple<> > //-V524
    when_all()
    {
        typedef hpx::util::tuple<> result_type;
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
        hpx::util::tuple<typename traits::acquire_future<Ts>::type...>
    >::type
    when_all(Ts&&... ts)
    {
        typedef hpx::util::tuple<
                typename traits::acquire_future<Ts>::type...
            > result_type;
        typedef detail::when_all_frame<result_type> frame_type;

        traits::acquire_future_disp func;
        result_type values(func(std::forward<Ts>(ts))...);

        boost::intrusive_ptr<frame_type> p(new frame_type(std::move(values)));
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
