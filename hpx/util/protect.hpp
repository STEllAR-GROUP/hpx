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
#include <hpx/traits/is_callable.hpp>
#include <hpx/util/move.hpp>
#include <hpx/util/detail/remove_reference.hpp>

#include <boost/mpl/identity.hpp>
#include <boost/type_traits/is_class.hpp>
#include <boost/utility/enable_if.hpp>
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
        
        template <typename>
        struct result
        {};

        BOOST_FORCEINLINE
        void operator()() const
        {
            f_();
        }

    protected:
        F f_;

    private:
        BOOST_COPYABLE_AND_MOVABLE(nullary_protected_bind)
    };

    template <typename F>
    class nullary_protected_bind<
        F, typename boost::enable_if<traits::is_callable<F()>>::type>
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

        template <typename>
        struct result;

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

        BOOST_FORCEINLINE
        typename boost::result_of<F const()>::type operator()() const
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

        protected_bind& operator=(BOOST_COPY_ASSIGN_REF(protected_bind) rhs)
        {
            nullary_protected_bind<F>::operator=(rhs);
            return *this;
        }

        protected_bind& operator=(BOOST_RV_REF(protected_bind) rhs)
        {
            nullary_protected_bind<F>::operator=(boost::move(rhs));
            return *this;
        }

        template <typename S>
        struct result
          : nullary_protected_bind<F>::template result<S>
        {};

        using nullary_protected_bind<F>::operator();

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
            return this->f_(HPX_ENUM_FORWARD_ARGS(N, A, a));                    \
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
    typename boost::enable_if<
        boost::is_class<F>
      , detail::protected_bind<typename detail::remove_reference<F>::type>
    >::type
    protect(BOOST_FWD_REF(F) f)
    {
        return detail::protected_bind<
            typename detail::remove_reference<F>::type
        >(boost::forward<F>(f));
    }
    
    // handle everything that is not a class separately as those
    // don't have any typename T::result_type defined
    template <typename T>
    typename boost::disable_if<
        boost::is_class<T>
      , BOOST_FWD_REF(T)
    >::type
    protect(BOOST_FWD_REF(T) v)
    {
        return boost::forward<T>(v);
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
