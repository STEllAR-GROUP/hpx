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

        template <typename>
        struct result;

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

        explicit protected_bind(BOOST_RV_REF(F) f)
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

        template <typename This>
        struct result<This()>
        {
            typedef typename boost::result_of<F()>::type type;
        };

        BOOST_FORCEINLINE 
        typename boost::result_of<F()>::type operator()()
        {
            return f_();
        }

        template <typename This>
        struct result<This const ()>
        {
            typedef typename boost::result_of<F const ()>::type type;
        };

        BOOST_FORCEINLINE
        typename boost::result_of<F const ()>::type operator()() const
        {
            return f_();
        }

#define HPX_UTIL_PROTECT_OPERATOR(Z, N, D)                                      \
        template <typename This, BOOST_PP_ENUM_PARAMS(N, typename A)>           \
        struct result<This(BOOST_PP_ENUM_PARAMS(N, A))>                         \
        {                                                                       \
            typedef                                                             \
                typename boost::result_of<F(BOOST_PP_ENUM_PARAMS(N, A))>::type  \
                type;                                                           \
        };                                                                      \
                                                                                \
        template <BOOST_PP_ENUM_PARAMS(N, typename A)>                          \
        BOOST_FORCEINLINE                                                       \
        typename boost::result_of<F(BOOST_PP_ENUM_PARAMS(N, A))>::type          \
        operator()(HPX_ENUM_FWD_ARGS(N, A, a))                                  \
        {                                                                       \
          return f_(HPX_ENUM_FORWARD_ARGS(N, A, a));                            \
        }                                                                       \
                                                                                \
        template <typename This, BOOST_PP_ENUM_PARAMS(N, typename A)>           \
        struct result<This const (BOOST_PP_ENUM_PARAMS(N, A))>                  \
        {                                                                       \
            typedef                                                             \
                typename boost::result_of<                                      \
                    F const (BOOST_PP_ENUM_PARAMS(N, A))                        \
                >::type                                                         \
                type;                                                           \
        };                                                                      \
                                                                                \
        template <BOOST_PP_ENUM_PARAMS(N, typename A)>                          \
        BOOST_FORCEINLINE                                                       \
        typename boost::result_of<F const (BOOST_PP_ENUM_PARAMS(N, A))>::type   \
        operator()(HPX_ENUM_FWD_ARGS(N, A, a)) const                            \
        {                                                                       \
          return f_(HPX_ENUM_FORWARD_ARGS(N, A, a));                            \
        }                                                                       \
    /**/
        
        BOOST_PP_REPEAT_FROM_TO(1, HPX_FUNCTION_ARGUMENT_LIMIT,
                HPX_UTIL_PROTECT_OPERATOR, _)

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

namespace boost {
    template <typename F>
    struct result_of<hpx::util::detail::protected_bind<F>()>
    {
        typedef typename result_of<F()>::type type;
    };
}

#endif
