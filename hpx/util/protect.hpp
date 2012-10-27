//  Copyright (c) 2002 Peter Dimov and Multi Media Ltd.
//  Copyright (c) 2009 Steven Watanabe
//  Copyright (c) 2011-2012 Hartmut Kaiser
//
// Distributed under the Boost Software License, Version 1.0. (See
// accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_PROTECT_SEP_23_2011_1230PM)
#define HPX_UTIL_PROTECT_SEP_23_2011_1230PM

#include <boost/config.hpp>
#include <hpx/util/move.hpp>
#include <hpx/util/detail/remove_reference.hpp>

namespace hpx { namespace util { namespace detail
{
    template <typename F>
    class protected_bind
    {
    public:
        typedef typename F::result_type result_type;

        // copy constructor
        protected_bind(protected_bind const& other)
          : f_(other.f_)
        {
        }

        // move constructor
        protected_bind(BOOST_RV_REF(protected_bind) other)
          : f_(boost::move(other.f_))
        {
        }

        explicit protected_bind(F const & f)
          : f_(f)
        {}

        explicit protected_bind(BOOST_FWD_REF(F) f)
          : f_(boost::move(f))
        {}

        protected_bind& operator=(BOOST_COPY_ASSIGN_REF(protected_bind) rhs)
        {
            if (this != &rhs)
                f_ = rhs.f_;
            return *this;
        }

        protected_bind& operator=(BOOST_RV_REF(protected_bind) rhs)
        {
            if (this != &rhs)
               f_ = boost::move(rhs.f_);
            return *this;
        }

        result_type operator()()
        {
            return f_();
        }

        result_type operator()() const
        {
            return f_();
        }

        template <typename A1>
        result_type operator()(BOOST_FWD_REF(A1) a1)
        {
          return f_(boost::forward<A1>(a1));
        }

        template <typename A1>
        result_type operator()(BOOST_FWD_REF(A1) a1) const
        {
            return f_(boost::forward<A1>(a1));
        }

        template <typename A1, typename A2>
        result_type operator()(BOOST_FWD_REF(A1) a1, BOOST_FWD_REF(A2) a2)
        {
            return f_(boost::forward<A1>(a1), boost::forward<A2>(a2));
        }

        template <typename A1, typename A2>
        result_type operator()(BOOST_FWD_REF(A1) a1, BOOST_FWD_REF(A2) a2) const
        {
            return f_(boost::forward<A1>(a1), boost::forward<A2>(a2));
        }

        template <typename A1, typename A2, typename A3>
        result_type operator()(BOOST_FWD_REF(A1) a1, BOOST_FWD_REF(A2) a2,
            BOOST_FWD_REF(A3) a3)
        {
            return f_(boost::forward<A1>(a1), boost::forward<A2>(a2),
                boost::forward<A3>(a3));
        }

        template <typename A1, typename A2, typename A3>
        result_type operator()(BOOST_FWD_REF(A1) a1, BOOST_FWD_REF(A2) a2,
            BOOST_FWD_REF(A3) a3) const
        {
            return f_(boost::forward<A1>(a1), boost::forward<A2>(a2),
                boost::forward<A3>(a3));
        }

        template <typename A1, typename A2, typename A3, typename A4>
        result_type operator()(BOOST_FWD_REF(A1) a1, BOOST_FWD_REF(A2) a2,
            BOOST_FWD_REF(A3) a3, BOOST_FWD_REF(A4) a4)
        {
            return f_(boost::forward<A1>(a1), boost::forward<A2>(a2),
                boost::forward<A3>(a3), boost::forward<A4>(a4));
        }

        template <typename A1, typename A2, typename A3, typename A4>
        result_type operator()(BOOST_FWD_REF(A1) a1, BOOST_FWD_REF(A2) a2,
            BOOST_FWD_REF(A3) a3, BOOST_FWD_REF(A4) a4) const
        {
            return f_(boost::forward<A1>(a1), boost::forward<A2>(a2),
                boost::forward<A3>(a3), boost::forward<A4>(a4));
        }

        template <typename A1, typename A2, typename A3, typename A4, typename A5>
        result_type operator()(BOOST_FWD_REF(A1) a1, BOOST_FWD_REF(A2) a2,
            BOOST_FWD_REF(A3) a3, BOOST_FWD_REF(A4) a4, BOOST_FWD_REF(A5) a5)
        {
            return f_(boost::forward<A1>(a1), boost::forward<A2>(a2),
                boost::forward<A3>(a3), boost::forward<A4>(a4),
                boost::forward<A5>(a5));
        }

        template <typename A1, typename A2, typename A3, typename A4, typename A5>
        result_type operator()(BOOST_FWD_REF(A1) a1, BOOST_FWD_REF(A2) a2,
            BOOST_FWD_REF(A3) a3, BOOST_FWD_REF(A4) a4, BOOST_FWD_REF(A5) a5) const
        {
            return f_(boost::forward<A1>(a1), boost::forward<A2>(a2),
                boost::forward<A3>(a3), boost::forward<A4>(a4),
                boost::forward<A5>(a5));
        }

    private:
        F f_;
        BOOST_COPYABLE_AND_MOVABLE(protected_bind)
    };
}}} // namespace hpx::util::detail

namespace hpx { namespace util
{
    template <typename F>
    detail::protected_bind<typename detail::remove_reference<F>::type> protect(BOOST_FWD_REF(F) f)
    {
        return detail::protected_bind<typename detail::remove_reference<F>::type>(boost::forward<F>(f));
    }
}} // namespace hpx::util

#endif
