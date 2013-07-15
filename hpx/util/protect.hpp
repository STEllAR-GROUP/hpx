//  Copyright (c) 2002 Peter Dimov and Multi Media Ltd.
//  Copyright (c) 2009 Steven Watanabe
//  Copyright (c) 2011-2013 Hartmut Kaiser
//
// Distributed under the Boost Software License, Version 1.0. (See
// accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_PROTECT_SEP_23_2011_1230PM)
#define HPX_UTIL_PROTECT_SEP_23_2011_1230PM

#include <boost/config.hpp>
#include <hpx/util/move.hpp>
#include <hpx/util/always_void.hpp>
#include <hpx/util/detail/remove_reference.hpp>

#include <boost/mpl/identity.hpp>
#include <boost/utility/result_of.hpp>

namespace hpx { namespace util { namespace detail
{
    // Handle the nullary operator()() separately
    template <typename F, typename Enable = void>
    class nullary_protected_bind
    {
    public:
        nullary_protected_bind(nullary_protected_bind const& other)
          : f_(other.f_)
        {
        }

        // move constructor
        nullary_protected_bind(BOOST_RV_REF(nullary_protected_bind) other)
          : f_(boost::move(other.f_))
        {
        }

        explicit nullary_protected_bind(F const & f)
          : f_(f)
        {}

        explicit nullary_protected_bind(BOOST_RV_REF(F) f)
          : f_(boost::move(f))
        {}

        nullary_protected_bind& operator=(BOOST_COPY_ASSIGN_REF(nullary_protected_bind) rhs)
        {
            if (this != &rhs)
                f_ = rhs.f_;
            return *this;
        }

        nullary_protected_bind& operator=(BOOST_RV_REF(nullary_protected_bind) rhs)
        {
            if (this != &rhs)
               f_ = boost::move(rhs.f_);
            return *this;
        }

    protected:
        F f_;

    private:
        BOOST_COPYABLE_AND_MOVABLE(nullary_protected_bind)
    };

    template <typename F>
    class nullary_protected_bind<
        F, typename util::always_void<typename F::result_type>::type>
    {
    public:
        nullary_protected_bind(nullary_protected_bind const& other)
          : f_(other.f_)
        {
        }

        // move constructor
        nullary_protected_bind(BOOST_RV_REF(nullary_protected_bind) other)
          : f_(boost::move(other.f_))
        {
        }

        explicit nullary_protected_bind(F const & f)
          : f_(f)
        {}

        explicit nullary_protected_bind(BOOST_RV_REF(F) f)
          : f_(boost::move(f))
        {}

        nullary_protected_bind& operator=(BOOST_COPY_ASSIGN_REF(nullary_protected_bind) rhs)
        {
            if (this != &rhs)
                f_ = rhs.f_;
            return *this;
        }

        nullary_protected_bind& operator=(BOOST_RV_REF(nullary_protected_bind) rhs)
        {
            if (this != &rhs)
               f_ = boost::move(rhs.f_);
            return *this;
        }

        //
        typedef typename F::result_type result_type;

        template <typename>
        struct result;

        template <typename This>
        struct result<This()>
        {
            typedef typename F::result_type type;
        };

        template <typename This>
        struct result<This const ()>
        {
            typedef typename F::result_type type;
        };

        BOOST_FORCEINLINE
        typename F::result_type operator()()
        {
            return f_();
        }

        BOOST_FORCEINLINE
        typename F::result_type operator()() const
        {
            return f_();
        }

    protected:
        F f_;

    private:
        BOOST_COPYABLE_AND_MOVABLE(nullary_protected_bind)
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename F>
    class protected_bind : public nullary_protected_bind<F>
    {
    public:
        // copy constructor
        protected_bind(protected_bind const& other)
          : nullary_protected_bind<F>(other)
        {
        }

        // move constructor
        protected_bind(BOOST_RV_REF(protected_bind) other)
          : nullary_protected_bind<F>(boost::move(other))
        {
        }

        explicit protected_bind(F const & f)
          : nullary_protected_bind<F>(f)
        {}

        explicit protected_bind(BOOST_RV_REF(F) f)
          : nullary_protected_bind<F>(boost::move(f))
        {}

        template <typename>
        struct result;

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
            return f_(HPX_ENUM_FORWARD_ARGS(N, A, a));                          \
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
            return this->f_(HPX_ENUM_FORWARD_ARGS(N, A, a));                    \
        }                                                                       \
    /**/

        BOOST_PP_REPEAT_FROM_TO(1, HPX_FUNCTION_ARGUMENT_LIMIT,
                HPX_UTIL_PROTECT_OPERATOR, _)

    private:
        BOOST_COPYABLE_AND_MOVABLE(protected_bind)
    };
}}} // namespace hpx::util::detail

namespace hpx { namespace util
{
    template <typename F>
    detail::protected_bind<typename detail::remove_reference<F>::type>
    protect(BOOST_FWD_REF(F) f)
    {
        return detail::protected_bind<
            typename detail::remove_reference<F>::type
        >(boost::forward<F>(f));
    }

    // handle nullary functions separately as those don't have any typename
    // F::result_type defined
    template <typename R>
    typename boost::mpl::identity<R>::type
    protect(R (*f)())
    {
        return f;
    }
}} // namespace hpx::util

namespace boost
{
    template <typename F>
    struct result_of<hpx::util::detail::protected_bind<F>()>
    {
        typedef typename result_of<F()>::type type;
    };
}

#endif
