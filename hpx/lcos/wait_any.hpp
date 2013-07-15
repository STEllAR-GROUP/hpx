//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !BOOST_PP_IS_ITERATING

#if !defined(HPX_LCOS_WHEN_ANY_APR_17_2012_1143AM)
#define HPX_LCOS_WHEN_ANY_APR_17_2012_1143AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/move.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/lcos/local/packaged_task.hpp>
#include <hpx/lcos/local/packaged_continuation.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/tuple.hpp>
#include <hpx/util/detail/pp_strip_parens.hpp>

#include <vector>

#include <boost/assert.hpp>
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/repeat.hpp>
#include <boost/preprocessor/enum.hpp>
#include <boost/preprocessor/iterate.hpp>

#include <boost/fusion/include/for_each.hpp>
#include <boost/fusion/include/accumulate.hpp>
#include <boost/fusion/include/size.hpp>
#include <boost/fusion/include/tuple.hpp>

#include <boost/atomic.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/enable_shared_from_this.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx
{
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        template <typename T>
        struct when_any : boost::enable_shared_from_this<when_any<T> >
        {
        private:
            BOOST_MOVABLE_BUT_NOT_COPYABLE(when_any)

            enum { index_error = -1 };

            void on_future_ready(std::size_t idx, threads::thread_id_type id)
            {
                std::size_t index_not_initialized =
                    static_cast<std::size_t>(index_error);
                if (index_.compare_exchange_strong(index_not_initialized, idx))
                {
                    // reactivate waiting thread only if it's not us
                    if (id != threads::get_self_id())
                        threads::set_thread_state(id, threads::pending);
                }
            }

        public:
            typedef HPX_STD_TUPLE<int, lcos::future<T> > result_type;
            typedef std::vector<lcos::future<T> > argument_type;

            when_any(argument_type const& lazy_values)
              : lazy_values_(lazy_values),
                index_(static_cast<std::size_t>(index_error))
            {}

            when_any(BOOST_RV_REF(argument_type) lazy_values)
              : lazy_values_(boost::move(lazy_values)),
                index_(static_cast<std::size_t>(index_error))
            {}

            when_any(BOOST_RV_REF(when_any) rhs)
              : lazy_values_(boost::move(rhs.lazy_values_)),
                index_(rhs.index_.load())
            {
                rhs.index_.store(static_cast<std::size_t>(index_error));
            }

            when_any& operator= (BOOST_RV_REF(when_any) rhs)
            {
                if (this != &rhs) {
                    lazy_values_ = boost::move(rhs.lazy_values_);
                    index_ = rhs.index_;
                    rhs.index_.store(static_cast<std::size_t>(index_error));
                }
                return *this;
            }

            result_type operator()()
            {
                using lcos::detail::get_future_data;

                index_.store(static_cast<std::size_t>(index_error));

                std::size_t size = lazy_values_.size();

                // set callback functions to execute when future is ready
                threads::thread_id_type id = threads::get_self_id();
                for (std::size_t i = 0; i != size; ++i)
                {
                    lcos::detail::future_data_base<T>* current =
                        get_future_data(lazy_values_[i]);

                    current->set_on_completed(util::bind(
                        &when_any::on_future_ready, this->shared_from_this(), i, id));
                }

                // If one of the futures is already set, our callback above has
                // already been called, otherwise we suspend ourselves
                if (index_.load() == static_cast<std::size_t>(index_error))
                {
                    // wait for any of the futures to return to become ready
                    this_thread::suspend(threads::suspended);
                }

                // that should not happen
                BOOST_ASSERT(index_.load() != static_cast<std::size_t>(index_error));

                return result_type(static_cast<int>(index_), lazy_values_[index_]);
            }

            std::vector<lcos::future<T> > lazy_values_;
            boost::atomic<std::size_t> index_;
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Tuple, typename T>
        struct when_any_tuple
          : boost::enable_shared_from_this<when_any_tuple<Tuple, T> >
        {
        private:
            BOOST_MOVABLE_BUT_NOT_COPYABLE(when_any_tuple)

            struct init_when
            {
                init_when(boost::shared_ptr<when_any_tuple>& outer,
                        threads::thread_id_type id)
                  : outer_(outer), id_(id)
                {}

                typedef std::size_t result_type;

                result_type operator()(std::size_t i, lcos::future<T> f) const
                {
                    lcos::detail::future_data_base<T>* current =
                        lcos::detail::get_future_data(f);

                    completed_callback_type cb = boost::move(
                        current->set_on_completed(completed_callback_type()));

                    current->set_on_completed(
                        util::bind(&when_any_tuple::on_future_ready,
                            outer_, i, id_));

                    return ++i;
                }

                boost::shared_ptr<when_any_tuple> outer_;
                threads::thread_id_type id_;
            };
            friend struct init_when;

            enum { index_error = -1 };

            void on_future_ready(std::size_t idx, threads::thread_id_type id)
            {
                std::size_t index_not_initialized =
                    static_cast<std::size_t>(index_error);
                if (index_.compare_exchange_strong(index_not_initialized, idx))
                {
                    // reactivate waiting thread only if it's not us
                    if (id != threads::get_self_id())
                        threads::set_thread_state(id, threads::pending);
                }
            }

            // extract element idx from given tuple
            template <typename First, typename Last>
            static lcos::future<T> const&
            get_element(First const& first, Last const& last, std::size_t idx,
                boost::mpl::false_)
            {
                if (idx == 0)
                    return boost::fusion::deref(first);

                return get_element(boost::fusion::next(first), last, --idx,
                    boost::fusion::result_of::equal_to<
                        typename boost::fusion::result_of::next<First>::type,
                        Last
                    >());
            }

            template <typename First, typename Last>
            static lcos::future<T> const&
            get_element(First const&, Last const&, std::size_t,
                boost::mpl::true_)
            {
                static lcos::future<T> f;
                return f;   // shouldn't ever be called
            }

            static lcos::future<T> const&
            get_element(Tuple const& t, std::size_t idx)
            {
                BOOST_ASSERT(idx <
                    static_cast<std::size_t>(boost::fusion::result_of::size<Tuple>::value));
                return get_element(
                    boost::fusion::begin(t), boost::fusion::end(t), idx,
                    boost::fusion::result_of::equal_to<
                        typename boost::fusion::result_of::begin<Tuple>::type,
                        typename boost::fusion::result_of::end<Tuple>::type>());
            }

        public:
            typedef HPX_STD_TUPLE<int, lcos::future<T> > result_type;
            typedef Tuple argument_type;

            when_any_tuple(argument_type const& lazy_values)
              : lazy_values_(lazy_values),
                index_(static_cast<std::size_t>(index_error))
            {}

            when_any_tuple(BOOST_RV_REF(argument_type) lazy_values)
              : lazy_values_(boost::move(lazy_values)),
                index_(static_cast<std::size_t>(index_error))
            {}

            when_any_tuple(BOOST_RV_REF(when_any_tuple) rhs)
              : lazy_values_(boost::move(rhs.lazy_values_)),
                index_(rhs.index_.load())
            {
                rhs.index_.store(static_cast<std::size_t>(index_error));
            }

            when_any_tuple& operator= (BOOST_RV_REF(when_any_tuple) rhs)
            {
                if (this != &rhs) {
                    lazy_values_ = boost::move(rhs.lazy_values_);
                    index_ = rhs.index_;
                    rhs.index_.store(static_cast<std::size_t>(index_error));
                }
                return *this;
            }

        public:
            result_type operator()()
            {
                index_.store(static_cast<std::size_t>(index_error));

                // set callback functions to execute when future is ready
                boost::fusion::accumulate(lazy_values_, std::size_t(0),
                    init_when(this->shared_from_this(), threads::get_self_id()));

                // If one of the futures is already set then our callback above
                // has already been called, otherwise we suspend ourselves.
                if (index_.load() == static_cast<std::size_t>(index_error))
                {
                    // wait for any of the futures to return to become ready
                    this_thread::suspend(threads::suspended);
                }

                // that should not happen
                BOOST_ASSERT(index_.load() != static_cast<std::size_t>(index_error));

                return result_type(static_cast<int>(index_),
                    get_element(lazy_values_, index_));
            }

            Tuple lazy_values_;
            boost::atomic<std::size_t> index_;
        };
    }

    /// The function \a when_any is a non-deterministic choice operator. It
    /// OR-composes all future objects stored in the given vector and returns
    /// a new future object representing the first future from that list which
    /// finishes execution.
    ///
    /// \return   The returned future holds a pair of values, the first value
    ///           is the index of the future which returned first and the second
    ///           value represents the actual future which returned first.

    template <typename Iterator>
    lcos::future<HPX_STD_TUPLE<int, lcos::future<
        typename lcos::future_iterator_traits<Iterator>::traits_type::value_type
    > > >
    when_any(Iterator begin, Iterator end)
    {
        typedef typename lcos::future_iterator_traits<
            Iterator>::traits_type::value_type value_type;
        typedef HPX_STD_TUPLE<int, lcos::future<value_type> > return_type;

        std::vector<lcos::future<value_type> > lazy_values;
        std::copy(begin, end, std::back_inserter(lazy_values));

        if (lazy_values.empty()) {
            HPX_THROW_EXCEPTION(hpx::bad_parameter,
                "hpx::lcos::when_any",
                "empty container passed to when_any");
            return lcos::make_ready_future(return_type());
        }

        boost::shared_ptr<detail::when_any<value_type> > f =
            boost::make_shared<detail::when_any<value_type> >(
                boost::move(lazy_values));

        lcos::local::futures_factory<return_type()> p(
            util::bind(&detail::when_any<value_type>::operator(), f));

        p.apply();
        return p.get_future();
    }

    template <typename T>
    lcos::future<HPX_STD_TUPLE<int, lcos::future<T> > >
    when_any(BOOST_RV_REF(HPX_UTIL_STRIP((
        std::vector<lcos::future<T> >))) lazy_values)
    {
        typedef HPX_STD_TUPLE<int, lcos::future<T> > return_type;

        if (lazy_values.empty()) {
            HPX_THROW_EXCEPTION(hpx::bad_parameter,
                "hpx::lcos::when_any",
                "empty container passed to when_any");
            return lcos::make_ready_future(return_type());
        }

        boost::shared_ptr<detail::when_any<T> > f =
            boost::make_shared<detail::when_any<T> >(
                boost::move(lazy_values));

        lcos::local::futures_factory<return_type()> p =
            lcos::local::futures_factory<return_type()>(
                util::bind(&detail::when_any<T>::operator(), f));

        p.apply();
        return p.get_future();
    }

    template <typename T>
    lcos::future<HPX_STD_TUPLE<int, lcos::future<T> > > //-V659
    when_any(std::vector<lcos::future<T> > const& lazy_values)
    {
        typedef HPX_STD_TUPLE<int, lcos::future<T> > return_type;

        if (lazy_values.empty()) {
            HPX_THROW_EXCEPTION(hpx::bad_parameter,
                "hpx::lcos::when_any",
                "empty container passed to when_any");
            return lcos::make_ready_future(return_type());
        }

        boost::shared_ptr<detail::when_any<T> > f =
            boost::make_shared<detail::when_any<T> >(lazy_values);

        lcos::local::futures_factory<return_type()> p =
            lcos::local::futures_factory<return_type()>(
                util::bind(&detail::when_any<T>::operator(), f));

        p.apply();
        return p.get_future();
    }

    /// The function \a wait_any is a non-deterministic choice operator. It
    /// OR-composes all future objects stored in the given vector and returns
    /// a new future object representing the first future from that list which
    /// finishes execution.
    ///
    /// \return   The returned tuple holds a pair of values, the first value
    ///           is the index of the future which returned first and the second
    ///           value represents the actual future which returned first.

    template <typename Iterator>
    HPX_STD_TUPLE<int, lcos::future<
        typename lcos::future_iterator_traits<Iterator>::traits_type::value_type
    > >
    wait_any(Iterator begin, Iterator end, error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<int, lcos::future<
            typename lcos::future_iterator_traits<Iterator>::traits_type::value_type
        > > result_type;

        lcos::future<result_type> f = when_any(begin, end);
        if (!f.valid()) {
            HPX_THROWS_IF(ec, uninitialized_value, "lcos::wait_any",
                "lcos::when_any didn't return a valid future");
            return result_type();
        }

        return f.get(ec);
    }

    template <typename T>
    HPX_STD_TUPLE<int, lcos::future<T> >
    wait_any(BOOST_RV_REF(HPX_UTIL_STRIP((
        std::vector<lcos::future<T> >))) lazy_values,
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<int, lcos::future<T> > result_type;

        lcos::future<result_type> f = when_any(lazy_values);
        if (!f.valid()) {
            HPX_THROWS_IF(ec, uninitialized_value, "lcos::wait_any",
                "lcos::when_any didn't return a valid future");
            return result_type();
        }

        return f.get(ec);
    }

    template <typename T>
    HPX_STD_TUPLE<int, lcos::future<T> >
    wait_any(std::vector<lcos::future<T> > const& lazy_values,
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<int, lcos::future<T> > result_type;

        lcos::future<result_type> f = when_any(lazy_values);
        if (!f.valid()) {
            HPX_THROWS_IF(ec, uninitialized_value, "lcos::wait_any",
                "lcos::when_any didn't return a valid future");
            return result_type();
        }

        return f.get(ec);
    }
}

#if !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)
#  include <hpx/lcos/preprocessed/wait_any.hpp>
#else

#if defined(__WAVE__) && defined(HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(preserve: 1, line: 0, output: "preprocessed/wait_any_" HPX_LIMIT_STR ".hpp")
#endif

#define BOOST_PP_ITERATION_PARAMS_1                                           \
    (3, (1, HPX_WAIT_ARGUMENT_LIMIT, <hpx/lcos/wait_any.hpp>))                \
/**/
#include BOOST_PP_ITERATE()

#if defined(__WAVE__) && defined (HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(output: null)
#endif

#endif // !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)

#endif

///////////////////////////////////////////////////////////////////////////////
#else // BOOST_PP_IS_ITERATING

#define N BOOST_PP_ITERATION()

#define HPX_WHEN_ANY_FUTURE_ARG(z, n, _)                                      \
        lcos::future<T> BOOST_PP_CAT(f, n)                                    \
    /**/
#define HPX_WHEN_ANY_FUTURE_TYPE(z, n, _)                                     \
        lcos::future<T>                                                       \
    /**/
#define HPX_WHEN_ANY_FUTURE_VAR(z, n, _) BOOST_PP_CAT(f, n)                   \
    /**/

namespace hpx
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    lcos::future<HPX_STD_TUPLE<int, lcos::future<T> > >
    when_any (BOOST_PP_ENUM(N, HPX_WHEN_ANY_FUTURE_ARG, _))
    {
        typedef HPX_STD_TUPLE<int, lcos::future<T> > return_type;
        typedef boost::fusion::tuple<
            BOOST_PP_ENUM(N, HPX_WHEN_ANY_FUTURE_TYPE, _)
        > argument_type;

        boost::shared_ptr<detail::when_any_tuple<argument_type, T> > f =
            boost::make_shared<detail::when_any_tuple<argument_type, T> >(
                argument_type(BOOST_PP_ENUM(N, HPX_WHEN_ANY_FUTURE_VAR, _)));

        lcos::local::futures_factory<return_type()> p =
            lcos::local::futures_factory<return_type()>(
                util::bind(&detail::when_any_tuple<argument_type, T>::operator(), f));

        p.apply();
        return p.get_future();
    }

    template <typename T>
    HPX_STD_TUPLE<int, lcos::future<T> >
    wait_any (BOOST_PP_ENUM(N, HPX_WHEN_ANY_FUTURE_ARG, _),
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<int, lcos::future<T> > result_type;

        lcos::future<result_type> f = when_any(
            BOOST_PP_ENUM(N, HPX_WHEN_ANY_FUTURE_VAR, _));
        if (!f.valid()) {
            HPX_THROWS_IF(ec, uninitialized_value, "lcos::wait_any",
                "lcos::when_any didn't return a valid future");
            return result_type();
        }

        return f.get(ec);
    }
}

#undef HPX_WHEN_ANY_FUTURE_ARG
#undef HPX_WHEN_ANY_FUTURE_TYPE
#undef HPX_WHEN_ANY_FUTURE_VAR
#undef N

#endif

