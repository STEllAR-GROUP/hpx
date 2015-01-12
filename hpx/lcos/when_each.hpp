//  Copyright (c) 2007-2015 Hartmut Kaiser
//  Copyright (c) 2013 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file lcos/when_each.hpp

#if !defined(HPX_LCOS_WHEN_EACH_JUN_16_2014_0206PM)
#define HPX_LCOS_WHEN_EACH_JUN_16_2014_0206PM

#if defined(DOXYGEN)
namespace hpx
{
    /// The function \a when_each is a operator allowing to join on the results
    /// of all given futures. It AND-composes all future objects given and
    /// returns a new future object representing the event of all those futures
    /// having finished executing. It also calls the supplied callback
    /// for each of the futures which becomes ready.
    ///
    /// \param f        The function which will be called for each of the
    ///                 input futures once the future has become ready.
    /// \param futures  A vector holding an arbitrary amount of \a future or
    ///                 \a shared_future objects for which \a wait_each should
    ///                 wait.
    ///
    /// \note This function consumes the futures as they are passed on to the
    ///       supplied function.
    ///
    /// \return   Returns a future representing the event of all input futures
    ///           being ready.
    ///
    template <typename F, typename Future>
    future<void> when_each(F&& f, std::vector<Future>&& lazy_values);

    /// The function \a when_each is a operator allowing to join on the results
    /// of all given futures. It AND-composes all future objects given and
    /// returns a new future object representing the event of all those futures
    /// having finished executing. It also calls the supplied callback
    /// for each of the futures which becomes ready.
    ///
    /// \param f        The function which will be called for each of the
    ///                 input futures once the future has become ready.
    /// \param first    The iterator pointing to the first element of a
    ///                 sequence of \a future or \a shared_future objects for
    ///                 which \a wait_each should wait.
    /// \param last     The iterator pointing to the last element of a
    ///                 sequence of \a future or \a shared_future objects for
    ///                 which \a wait_each should wait.
    ///
    /// \note This function consumes the futures as they are passed on to the
    ///       supplied function.
    ///
    /// \return   Returns a future representing the event of all input futures
    ///           being ready.
    ///
    template <typename F, typename Iterator>
    future<Iterator> when_each(F&& f, Iterator begin, Iterator end);

    /// The function \a when_each is a operator allowing to join on the results
    /// of all given futures. It AND-composes all future objects given and
    /// returns a new future object representing the event of all those futures
    /// having finished executing. It also calls the supplied callback
    /// for each of the futures which becomes ready.
    ///
    /// \param f        The function which will be called for each of the
    ///                 input futures once the future has become ready.
    /// \param futures  An arbitrary number of \a future or \a shared_future
    ///                 objects, possibly holding different types for which
    ///                 \a wait_each should wait.
    ///
    /// \note This function consumes the futures as they are passed on to the
    ///       supplied function.
    ///
    /// \return   Returns a future representing the event of all input futures
    ///           being ready.
    ///
    template <typename F, typename... Ts>
    future<void> when_each(F&& f, Ts&&... ts);

    /// The function \a when_each is a operator allowing to join on the results
    /// of all given futures. It AND-composes all future objects given and
    /// returns a new future object representing the event of all those futures
    /// having finished executing. It also calls the supplied callback
    /// for each of the futures which becomes ready.
    ///
    /// \param f        The function which will be called for each of the
    ///                 input futures once the future has become ready.
    /// \param begin    The iterator pointing to the first element of a
    ///                 sequence of \a future or \a shared_future objects for
    ///                 which \a wait_each_n should wait.
    /// \param count    The number of elements in the sequence starting at
    ///                 \a first.
    ///
    /// \note This function consumes the futures as they are passed on to the
    ///       supplied function.
    ///
    /// \return   Returns a future holding the iterator pointing to the first
    ///           element after the last one.
    ///
    template <typename F, typename Iterator>
    future<Iterator> when_each_n(F&& f, Iterator begin, std::size_t count);
}

#else // DOXYGEN

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/when_some.hpp>
#include <hpx/util/functional/boolean_ops.hpp>

#include <boost/mpl/bool.hpp>
#include <boost/mpl/not.hpp>
#include <boost/mpl/and.hpp>
#include <boost/utility/enable_if.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos
{
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        template <typename Sequence, typename F>
        struct when_each;

        template <typename Sequence, typename F>
        struct set_when_each_on_completed_callback_impl
        {
            explicit set_when_each_on_completed_callback_impl(
                    boost::shared_ptr<when_each<Sequence, F> > when)
              : when_(when)
            {}

            template <typename Future>
            static void on_future_ready(Future& future,
                boost::shared_ptr<when_each<Sequence, F> > when,
                threads::thread_id_type const& id)
            {
                when->f_(std::move(future));

                if (when->count_.fetch_add(1) + 1 == when->needed_count_)
                {
                    // reactivate waiting thread only if it's not us
                    if (id != threads::get_self_id())
                        threads::set_thread_state(id, threads::pending);
                    else
                        when->goal_reached_on_calling_thread_ = true;
                }
            }

            template <typename Future>
            void operator()(Future& future) const
            {
                std::size_t counter = when_->count_.load(boost::memory_order_seq_cst);
                if (counter < when_->needed_count_ && !future.is_ready()) {
                    // handle future only if not enough futures are ready yet
                    // also, do not touch any futures which are already ready

                    typedef
                        typename lcos::detail::shared_state_ptr_for<Future>::type
                        shared_state_ptr;

                    shared_state_ptr const& shared_state =
                        lcos::detail::get_shared_state(future);

                    shared_state->set_on_completed(util::bind(
                        &set_when_each_on_completed_callback_impl::on_future_ready<Future>,
                        boost::ref(future), when_, threads::get_self_id()));
                }
                else {
                    if (when_->count_.fetch_add(1) + 1 == when_->needed_count_)
                    {
                        when_->goal_reached_on_calling_thread_ = true;
                    }
                    when_->f_(std::move(future));    // invoke callback right away
                }
            }

            template <typename Sequence_>
            void apply(Sequence_& sequence, typename boost::enable_if<
                boost::fusion::traits::is_sequence<Sequence_> >::type* = 0) const
            {
                boost::fusion::for_each(sequence, *this);
            }

            template <typename Sequence_>
            void apply(Sequence_& sequence, typename boost::disable_if<
                boost::fusion::traits::is_sequence<Sequence_> >::type* = 0) const
            {
                std::for_each(sequence.begin(), sequence.end(), *this);
            }

            boost::shared_ptr<when_each<Sequence, F> > when_;
        };

        template <typename Sequence, typename F>
        void set_on_completed_callback(
            boost::shared_ptr<when_each<Sequence, F> > when)
        {
            set_when_each_on_completed_callback_impl<Sequence, F>
                set_on_completed_callback_helper(when);
            set_on_completed_callback_helper.apply(when->lazy_values_);
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename Sequence, typename F>
        struct when_each
          : boost::enable_shared_from_this<when_each<Sequence, F> >
        {
        private:
            HPX_MOVABLE_BUT_NOT_COPYABLE(when_each)

        public:
            typedef Sequence argument_type;
            typedef void result_type;

            template <typename F_>
            when_each(argument_type && lazy_values, F_ && f, std::size_t n)
              : lazy_values_(std::move(lazy_values)),
                count_(0),
                f_(std::forward<F>(f)),
                needed_count_(n),
                goal_reached_on_calling_thread_(false)
            {}

            result_type operator()()
            {
                count_.store(0);
                goal_reached_on_calling_thread_ = false;

                // set callback functions to executed when future is ready
                set_on_completed_callback(this->shared_from_this());

                // If all of the requested futures are already set then our
                // callback above has already been called, otherwise we suspend
                // ourselves.
                if (!goal_reached_on_calling_thread_)
                {
                    // wait for all of the futures to return to become ready
                    this_thread::suspend(threads::suspended,
                        "hpx::lcos::detail::when_each::operator()");
                }

                // all futures should be ready
                HPX_ASSERT(count_.load(boost::memory_order_seq_cst) == needed_count_);
            }

            argument_type lazy_values_;
            boost::atomic<std::size_t> count_;
            F f_;
            std::size_t const needed_count_;
            bool goal_reached_on_calling_thread_;
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename F, typename Future>
    lcos::future<void>
    when_each(F&& func, std::vector<Future>& lazy_values)
    {
        BOOST_STATIC_ASSERT_MSG(
            traits::is_future<Future>::value, "invalid use of when_each");

        typedef void result_type;
        typedef std::vector<Future> argument_type;
        typedef typename util::decay<F>::type func_type;
        typedef detail::when_each<argument_type, func_type> when_each_type;

        std::vector<Future> lazy_values_;
        lazy_values_.reserve(lazy_values.size());

        std::transform(lazy_values.begin(), lazy_values.end(),
            std::back_inserter(lazy_values_),
            traits::acquire_future_disp());

        boost::shared_ptr<when_each_type> f =
            boost::make_shared<when_each_type>(std::move(lazy_values_),
                std::forward<F>(func), lazy_values_.size());

        lcos::local::futures_factory<result_type()> p(
            util::bind(&when_each_type::operator(), f));

        p.apply();
        return p.get_future();
    }

    template <typename F, typename Future>
    lcos::future<void> //-V659
    when_each(F&& f, std::vector<Future>&& lazy_values)
    {
        return lcos::when_each(std::forward<F>(f), lazy_values);
    }

    namespace detail
    {
        template <typename Iterator>
        Iterator return_iterator(hpx::future<void>&& fut, Iterator end)
        {
            fut.get();      // rethrow exceptions, if any
            return end;
        }
    }

    template <typename F, typename Iterator>
    lcos::future<Iterator>
    when_each(F&& f, Iterator begin, Iterator end)
    {
        typedef
            typename lcos::detail::future_iterator_traits<Iterator>::type
            future_type;

        std::vector<future_type> lazy_values_;
        std::transform(begin, end, std::back_inserter(lazy_values_),
            traits::acquire_future_disp());

        return lcos::when_each(std::forward<F>(f), lazy_values_).then(
            util::bind(&detail::return_iterator<Iterator>,
                util::placeholders::_1, end));
    }

    template <typename F, typename Iterator>
    lcos::future<Iterator>
    when_each_n(F&& f, Iterator begin, std::size_t count)
    {
        typedef
            typename lcos::detail::future_iterator_traits<Iterator>::type
            future_type;

        std::vector<future_type> lazy_values_;
        lazy_values_.reserve(count);

        traits::acquire_future_disp func;
        for (std::size_t i = 0; i != count; ++i)
            lazy_values_.push_back(func(*begin++));

        return lcos::when_each(std::forward<F>(f), lazy_values_).then(
            util::bind(&detail::return_iterator<Iterator>,
                util::placeholders::_1, begin));
    }

    template <typename F>
    inline lcos::future<void>
    when_each(F&& f)
    {
        return lcos::make_ready_future();
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename F, typename... Ts>
    typename boost::disable_if<
        boost::mpl::or_<
            traits::is_future<typename util::decay<F>::type>,
            util::functional::any_of<boost::mpl::not_<traits::is_future<Ts> >...>
        >,
        lcos::future<void>
    >::type
    when_each(F&& f, Ts&&... ts)
    {
        typedef hpx::util::tuple<
                typename traits::acquire_future<Ts>::type...
            > argument_type;

        typedef void result_type;
        typedef typename util::decay<F>::type func_type;
        typedef detail::when_each<argument_type, func_type> when_each_type;

        traits::acquire_future_disp func;
        argument_type lazy_values(func(std::forward<Ts>(ts))...);

        boost::shared_ptr<when_each_type> fptr =
            boost::make_shared<when_each_type>(std::move(lazy_values),
                std::forward<F>(f), sizeof...(Ts));

        lcos::local::futures_factory<result_type()> p(
            util::bind(&when_each_type::operator(), fptr));

        p.apply();
        return p.get_future();
    }
}}

namespace hpx
{
    using lcos::when_each;
    using lcos::when_each_n;
}

#endif // DOXYGEN
#endif

