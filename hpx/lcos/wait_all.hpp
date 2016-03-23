//  Copyright (c) 2007-2015 Hartmut Kaiser
//  Copyright (c) 2013 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file lcos/wait_all.hpp

#if !defined(HPX_LCOS_WAIT_ALL_APR_19_2012_1140AM)
#define HPX_LCOS_WAIT_ALL_APR_19_2012_1140AM

#if defined(DOXYGEN)
namespace hpx
{
    /// The function \a wait_all is a operator allowing to join on the result
    /// of all given futures. It AND-composes all future objects given and
    /// returns after they finished executing.
    ///
    /// \param first    The iterator pointing to the first element of a
    ///                 sequence of \a future or \a shared_future objects for
    ///                 which \a wait_all should wait.
    /// \param last     The iterator pointing to the last element of a
    ///                 sequence of \a future or \a shared_future objects for
    ///                 which \a wait_all should wait.
    ///
    /// \note The function \a wait_all returns after all futures have become
    ///       ready. All input futures are still valid after \a wait_all
    ///       returns.
    ///
    template <typename InputIter>
    void wait_all(InputIter first, InputIter last);

    /// The function \a wait_all is a operator allowing to join on the result
    /// of all given futures. It AND-composes all future objects given and
    /// returns after they finished executing.
    ///
    /// \param futures  A vector holding an arbitrary amount of \a future or
    ///                 \a shared_future objects for which \a wait_all should
    ///                 wait.
    ///
    /// \note The function \a wait_all returns after all futures have become
    ///       ready. All input futures are still valid after \a wait_all
    ///       returns.
    ///
    template <typename R>
    void wait_all(std::vector<future<R>>&& futures);

    /// The function \a wait_all is a operator allowing to join on the result
    /// of all given futures. It AND-composes all future objects given and
    /// returns after they finished executing.
    ///
    /// \param futures  An arbitrary number of \a future or \a shared_future
    ///                 objects, possibly holding different types for which
    ///                 \a wait_all should wait.
    ///
    /// \note The function \a wait_all returns after all futures have become
    ///       ready. All input futures are still valid after \a wait_all
    ///       returns.
    ///
    template <typename ...T>
    void wait_all(T &&... futures);

    /// The function \a wait_all_n is a operator allowing to join on the result
    /// of all given futures. It AND-composes all future objects given and
    /// returns after they finished executing.
    ///
    /// \param begin    The iterator pointing to the first element of a
    ///                 sequence of \a future or \a shared_future objects for
    ///                 which \a wait_all_n should wait.
    /// \param count    The number of elements in the sequence starting at
    ///                 \a first.
    ///
    /// \return         The function \a wait_all_n will return an iterator
    ///                 referring to the first element in the input sequence
    ///                 after the last processed element.
    ///
    /// \note The function \a wait_all_n returns after all futures have become
    ///       ready. All input futures are still valid after \a wait_all_n
    ///       returns.
    ///
    template <typename InputIter>
    InputIter wait_all_n(InputIter begin, std::size_t count);
}

#else // DOXYGEN

#include <hpx/config.hpp>
#include <hpx/traits/acquire_shared_state.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/lcos/wait_some.hpp>
#include <hpx/util/always_void.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/move.hpp>
#include <hpx/util/tuple.hpp>

#include <boost/mpl/bool.hpp>
#include <boost/fusion/include/begin.hpp>
#include <boost/fusion/include/end.hpp>
#include <boost/fusion/include/deref.hpp>
#include <boost/fusion/include/next.hpp>
#include <boost/range/functions.hpp>
#include <boost/ref.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/utility/enable_if.hpp>

#include <algorithm>
#include <iterator>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos
{
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        template <typename Future, typename Enable = void>
        struct is_future_or_shared_state
          : traits::is_future<Future>
        {};

        template <typename R>
        struct is_future_or_shared_state<
                boost::intrusive_ptr<future_data<R> > >
          : boost::mpl::true_
        {};

        ///////////////////////////////////////////////////////////////////////
        template <typename Range, typename Enable = void>
        struct is_future_or_shared_state_range
            : boost::mpl::false_
        {};

        template <typename T>
        struct is_future_or_shared_state_range<std::vector<T> >
            : is_future_or_shared_state<T>
        {};

        ///////////////////////////////////////////////////////////////////////
        template <typename Future, typename Enable = void>
        struct future_or_shared_state_result;

        template <typename Future>
        struct future_or_shared_state_result<Future,
                typename boost::enable_if<traits::is_future<Future> >::type>
          : traits::future_traits<Future>
        {};

        template <typename R>
        struct future_or_shared_state_result<
            boost::intrusive_ptr<future_data<R> > >
        {
            typedef R type;
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Tuple>
        struct wait_all_frame //-V690
          : hpx::lcos::detail::future_data<void>
        {
        private:
            // workaround gcc regression wrongly instantiating constructors
            wait_all_frame();
            wait_all_frame(wait_all_frame const&);

            typedef typename boost::fusion::result_of::end<Tuple const>::type
                end_type;

        public:
            wait_all_frame(Tuple const& t)
              : t_(t)
            {}

        protected:
            // End of the tuple is reached
            template <typename TupleIter>
            HPX_FORCEINLINE
            void do_await(TupleIter&&, boost::mpl::true_)
            {
                this->set_value(util::unused);     // simply make ourself ready
            }

            // Current element is a range (vector) of futures
            template <typename TupleIter, typename Iter>
            void await_range(TupleIter iter, Iter next, Iter end)
            {
                typedef typename std::iterator_traits<Iter>::value_type
                    future_type;
                typedef typename detail::future_or_shared_state_result<
                        future_type
                    >::type future_result_type;

                void (wait_all_frame::*f)(TupleIter, Iter, Iter) =
                    &wait_all_frame::await_range;

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
                            next_future_data->set_on_completed(
                                util::bind(
                                    f, this, std::move(iter),
                                    std::move(next), std::move(end)));
                            return;
                        }
                    }
                }

                // All elements of the sequence are ready now, proceed to the
                // next argument.
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

                typedef typename detail::future_or_shared_state_result<
                        future_type
                    >::type future_result_type;

                using boost::mpl::false_;
                using boost::mpl::true_;

                boost::intrusive_ptr<
                    lcos::detail::future_data<future_result_type>
                > next_future_data = traits::detail::get_shared_state(
                    boost::fusion::deref(iter));

                if (!next_future_data->is_ready())
                {
                    next_future_data->execute_deferred();

                    // execute_deferred might have made the future ready
                    if (!next_future_data->is_ready())
                    {
                        // Attach a continuation to this future which will
                        // re-evaluate it and continue to the next argument
                        // (if any).
                        void (wait_all_frame::*f)(TupleIter, true_, false_) =
                            &wait_all_frame::await_next;

                        next_future_data->set_on_completed(hpx::util::bind(
                            f, this, std::move(iter), true_(), false_()));
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

                typedef typename detail::is_future_or_shared_state<future_type>::type
                    is_future;
                typedef typename detail::is_future_or_shared_state_range<future_type>
                      ::type
                    is_range;

                await_next(std::forward<TupleIter>(iter), is_future(), is_range());
            }

        public:
            void wait_all()
            {
                typedef typename boost::fusion::result_of::begin<Tuple const>::type
                    begin_type;
                typedef boost::is_same<begin_type, end_type> pred;

                do_await(boost::fusion::begin(t_), pred());

                // If there are still futures which are not ready, suspend and
                // wait.
                if (!this->is_ready())
                    this->wait();
            }

        private:
            Tuple const& t_;
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Future>
    void wait_all(std::vector<Future> const& values)
    {
        typedef hpx::util::tuple<std::vector<Future> const&> result_type;
        typedef detail::wait_all_frame<result_type> frame_type;

        result_type data(values);
        frame_type frame(data);
        frame.wait_all();
    }

    template <typename Future>
    HPX_FORCEINLINE void wait_all(std::vector<Future>& values)
    {
        lcos::wait_all(const_cast<std::vector<Future> const&>(values));
    }

    template <typename Future>
    HPX_FORCEINLINE void wait_all(std::vector<Future>&& values)
    {
        lcos::wait_all(const_cast<std::vector<Future> const&>(values));
    }

    template <typename Iterator>
    typename util::always_void<
        typename lcos::detail::future_iterator_traits<Iterator>::type
    >::type
    wait_all(Iterator begin, Iterator end)
    {
        typedef typename lcos::detail::future_iterator_traits<Iterator>::type
            future_type;
        typedef typename traits::detail::shared_state_ptr_for<future_type>::type
            shared_state_ptr;
        typedef std::vector<shared_state_ptr> result_type;

        result_type values;
        std::transform(begin, end, std::back_inserter(values),
            detail::wait_get_shared_state<future_type>());

        lcos::wait_all(values);
    }

    template <typename Iterator>
    Iterator
    wait_all_n(Iterator begin, std::size_t count)
    {
        typedef typename lcos::detail::future_iterator_traits<Iterator>::type
            future_type;
        typedef typename traits::detail::shared_state_ptr_for<future_type>::type
            shared_state_ptr;
        typedef std::vector<shared_state_ptr> result_type;

        result_type values;
        values.reserve(count);

        detail::wait_get_shared_state<future_type> func;
        for (std::size_t i = 0; i != count; ++i)
            values.push_back(func(*begin++));

        lcos::wait_all(std::move(values));

        return begin;
    }

    inline void wait_all()
    {
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename... Ts>
    void wait_all(Ts&&... ts)
    {
        typedef hpx::util::tuple<
                typename traits::detail::shared_state_ptr_for<Ts>::type...
            > result_type;
        typedef detail::wait_all_frame<result_type> frame_type;

        result_type values =
            result_type(traits::detail::get_shared_state(ts)...);

        frame_type frame(values);
        frame.wait_all();
    }
}}

namespace hpx
{
    using lcos::wait_all;
    using lcos::wait_all_n;
}

#endif // DOXYGEN
#endif
