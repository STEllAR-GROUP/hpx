//  Copyright (c) 2007-2014 Hartmut Kaiser
//  Copyright (c) 2013 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !BOOST_PP_IS_ITERATING

#if !defined(HPX_LCOS_WAIT_SOME_APR_19_2012_0203PM)
#define HPX_LCOS_WAIT_SOME_APR_19_2012_0203PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/lcos/local/packaged_task.hpp>
#include <hpx/lcos/local/packaged_continuation.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/always_void.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/move.hpp>
#include <hpx/util/tuple.hpp>
#include <hpx/util/detail/pp_strip_parens.hpp>

#include <boost/atomic.hpp>
#include <boost/enable_shared_from_this.hpp>
#include <boost/fusion/include/for_each.hpp>
#include <boost/fusion/include/is_sequence.hpp>
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/enum.hpp>
#include <boost/preprocessor/iterate.hpp>
#include <boost/preprocessor/repeat.hpp>
#include <boost/shared_ptr.hpp>
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
        template <typename Sequence>
        struct wait_some;

        template <typename Sequence, typename Callback>
        struct set_wait_on_completed_callback_impl
        {
            explicit set_wait_on_completed_callback_impl(
                    wait_some<Sequence>& wait, Callback const& callback)
              : wait_(wait),
                callback_(callback)
            {}

            template <typename SharedState>
            void operator()(SharedState& shared_state) const
            {
                std::size_t counter = wait_.count_.load(boost::memory_order_seq_cst);
                if (counter < wait_.needed_count_ && !shared_state->is_ready()) {
                    // handle future only if not enough futures are ready yet
                    // also, do not touch any futures which are already ready

                    shared_state->set_on_completed(Callback(callback_));
                }
                else {
                    ++wait_.count_;
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

            wait_some<Sequence>& wait_;
            Callback const& callback_;
        };

        template <typename Sequence, typename Callback>
        void set_on_completed_callback(wait_some<Sequence>& wait,
            Callback const& callback)
        {
            set_wait_on_completed_callback_impl<Sequence, Callback>
                set_on_completed_callback_helper(wait, callback);
            set_on_completed_callback_helper.apply(wait.lazy_values_);
        }

        template <typename Sequence>
        struct wait_some : boost::enable_shared_from_this<wait_some<Sequence> > //-V690
        {
        private:
            void on_future_ready(threads::thread_id_type const& id)
            {
                if (count_.fetch_add(1) + 1 == needed_count_)
                {
                    // reactivate waiting thread only if it's not us
                    if (id != threads::get_self_id())
                        threads::set_thread_state(id, threads::pending);
                }
            }

        private:
            // workaround gcc regression wrongly instantiating constructors
            wait_some();
            wait_some(wait_some const&);

        public:
            typedef Sequence argument_type;
            typedef void result_type;

            wait_some(argument_type && lazy_values, std::size_t n)
              : lazy_values_(std::move(lazy_values))
              , count_(0)
              , needed_count_(n)
            {}

            result_type operator()()
            {
                // set callback functions to executed wait future is ready
                set_on_completed_callback(*this,
                    util::bind(
                        &wait_some::on_future_ready, this->shared_from_this(),
                        threads::get_self_id()));

                // if all of the requested futures are already set, our
                // callback above has already been called often enough, otherwise
                // we suspend ourselves
                if (count_.load(boost::memory_order_seq_cst) < needed_count_)
                {
                    // wait for any of the futures to return to become ready
                    this_thread::suspend(threads::suspended,
                        "hpx::detail::when_some::operator()");
                }

                // at least N futures should be ready
                HPX_ASSERT(count_.load(boost::memory_order_seq_cst) >= needed_count_);
            }

            argument_type lazy_values_;
            boost::atomic<std::size_t> count_;
            std::size_t const needed_count_;
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Future>
        struct wait_get_shared_state
        {
            typedef
                typename lcos::detail::shared_state_ptr_for<Future>::type const&
                result_type;

            BOOST_FORCEINLINE result_type
            operator()(Future const& f) const
            {
                return lcos::detail::get_shared_state(f);
            }
        };
    }

    /// The function \a wait_some is a operator allowing to join on the result
    /// of all given futures. It AND-composes all future objects given and
    /// returns the same list of futures after they finished executing.
    ///
    /// \a wait_some returns after n futures have been triggered.
    ///
    /// \note There are three variations of wait_some. The first takes a pair
    ///       of InputIterators. The second takes an std::vector of future<R>.
    ///       The third takes any arbitrary number of future<R>, where R need
    ///       not be the same type.
    ///
    /// \return   The same list of futures as has been passed to wait_some.
    ///           - future<vector<future<R>>>: If the input cardinality is
    ///             unknown at compile time and the futures are all of the
    ///             same type.
    ///           - future<tuple<future<R0>, future<R1>, future<R2>...>>: If
    ///             inputs are fixed in number and are of heterogeneous types.
    ///             The inputs can be any arbitrary number of future objects.

    template <typename Future>
    void wait_some(std::size_t n,
        std::vector<Future> const& lazy_values,
        error_code& ec = throws)
    {
        BOOST_STATIC_ASSERT_MSG(
            traits::is_future<Future>::value, "invalid use of wait_some");

        typedef
            typename lcos::detail::shared_state_ptr_for<Future>::type
            shared_state_ptr;
        typedef std::vector<shared_state_ptr> result_type;

        if (n == 0)
        {
            return;
        }

        if (n > lazy_values.size())
        {
            HPX_THROWS_IF(ec, hpx::bad_parameter,
                "hpx::lcos::wait_some",
                "number of results to wait for is out of bounds");
            return;
        }

        result_type lazy_values_;
        std::transform(lazy_values.begin(), lazy_values.end(),
            std::back_inserter(lazy_values_),
            detail::wait_get_shared_state<Future>());

        boost::shared_ptr<detail::wait_some<result_type> > f =
            boost::make_shared<detail::wait_some<result_type> >(
                std::move(lazy_values_), n);

        return (*f.get())();
    }

    template <typename Future>
    void wait_some(std::size_t n,
        std::vector<Future>& lazy_values,
        error_code& ec = throws)
    {
        return lcos::wait_some(
            n, const_cast<std::vector<Future> const&>(lazy_values), ec);
    }

    template <typename Future>
    void wait_some(std::size_t n,
        std::vector<Future> && lazy_values,
        error_code& ec = throws)
    {
        return lcos::wait_some(
            n, const_cast<std::vector<Future> const&>(lazy_values), ec);
    }

    template <typename Iterator>
    typename util::always_void<
        typename lcos::detail::future_iterator_traits<Iterator>::type
    >::type wait_some(std::size_t n, Iterator begin, Iterator end,
        error_code& ec = throws)
    {
        typedef
            typename lcos::detail::future_iterator_traits<Iterator>::type
            future_type;
        typedef
            typename lcos::detail::shared_state_ptr_for<future_type>::type
            shared_state_ptr;
        typedef std::vector<shared_state_ptr> result_type;

        result_type lazy_values_;
        std::transform(begin, end, std::back_inserter(lazy_values_),
            detail::wait_get_shared_state<future_type>());

        boost::shared_ptr<detail::wait_some<result_type> > f =
            boost::make_shared<detail::wait_some<result_type> >(
                std::move(lazy_values_), n);

        return (*f.get())();
    }

    template <typename Iterator>
    Iterator wait_some_n(std::size_t n, Iterator begin,
        std::size_t count, error_code& ec = throws)
    {
        typedef
            typename lcos::detail::future_iterator_traits<Iterator>::type
            future_type;
        typedef
            typename lcos::detail::shared_state_ptr_for<future_type>::type
            shared_state_ptr;
        typedef std::vector<shared_state_ptr> result_type;

        result_type lazy_values_;
        lazy_values_.resize(count);
        detail::wait_get_shared_state<future_type> func;
        for (std::size_t i = 0; i != count; ++i)
            lazy_values_.push_back(func(*begin++));

        boost::shared_ptr<detail::wait_some<result_type> > f =
            boost::make_shared<detail::wait_some<result_type> >(
                std::move(lazy_values_), n);

        (*f.get())();

        return begin;
    }

    inline void wait_some(std::size_t n, error_code& ec = throws)
    {
        if (n == 0)
        {
            return;
        }

        //if (n > 0)
        //{
            HPX_THROWS_IF(ec, hpx::bad_parameter,
                "hpx::lcos::wait_some",
                "number of results to wait for is out of bounds");
            return;
        //}
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    void wait_some(std::size_t n, hpx::future<T> && f, error_code& ec = throws)
    {
        if (n != 1)
        {
            HPX_THROWS_IF(ec, hpx::bad_parameter,
                "hpx::lcos::wait_some",
                "number of results to wait for is out of bounds");
            return;
        }

        f.wait();
    }

    template <typename T>
    void wait_some(std::size_t n, hpx::shared_future<T> && f, error_code& ec = throws)
    {
        if (n != 1)
        {
            HPX_THROWS_IF(ec, hpx::bad_parameter,
                "hpx::lcos::wait_some",
                "number of results to wait for is out of bounds");
            return;
        }

        f.wait();
    }
}}

#if !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)
#  include <hpx/lcos/preprocessed/wait_some.hpp>
#else

#if defined(__WAVE__) && defined(HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(preserve: 1, line: 0, output: "preprocessed/wait_some_" HPX_LIMIT_STR ".hpp")
#endif

#define BOOST_PP_ITERATION_PARAMS_1                                           \
    (3, (1, HPX_WAIT_ARGUMENT_LIMIT, <hpx/lcos/wait_some.hpp>))               \
/**/
#include BOOST_PP_ITERATE()

#if defined(__WAVE__) && defined (HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(output: null)
#endif

#endif // !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)

namespace hpx
{
    using lcos::wait_some;
    using lcos::wait_some_n;
}

#endif

///////////////////////////////////////////////////////////////////////////////
#else // BOOST_PP_IS_ITERATING

#define N BOOST_PP_ITERATION()

#define HPX_WAIT_SOME_SHARED_STATE_FOR_FUTURE(Z, N, D)                        \
    typename lcos::detail::shared_state_ptr_for<BOOST_PP_CAT(T, N)>::type     \
    /**/
#define HPX_WAIT_SOME_GET_SHARED_STATE(Z, N, D)                               \
    lcos::detail::get_shared_state(BOOST_PP_CAT(f, N))                        \
    /**/

namespace hpx { namespace lcos
{
    ///////////////////////////////////////////////////////////////////////////
    template <BOOST_PP_ENUM_PARAMS(N, typename T)>
    void wait_some(std::size_t n, HPX_ENUM_FWD_ARGS(N, T, f),
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<
            BOOST_PP_ENUM(N, HPX_WAIT_SOME_SHARED_STATE_FOR_FUTURE, _)>
            result_type;

        result_type lazy_values_(
            BOOST_PP_ENUM(N, HPX_WAIT_SOME_GET_SHARED_STATE, _));

        if (n == 0)
        {
            return;
        }

        if (n > N)
        {
            HPX_THROWS_IF(ec, hpx::bad_parameter,
                "hpx::lcos::wait_some",
                "number of results to wait for is out of bounds");
            return;
        }

        boost::shared_ptr<detail::wait_some<result_type> > f =
            boost::make_shared<detail::wait_some<result_type> >(
                std::move(lazy_values_), n);

        return (*f.get())();
    }
}}

#undef HPX_WAIT_SOME_SHARED_STATE_FOR_FUTURE
#undef HPX_WAIT_SOME_GET_SHARED_STATE
#undef N

#endif

