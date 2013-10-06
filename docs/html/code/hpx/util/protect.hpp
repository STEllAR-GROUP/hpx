//  Copyright (c) 2002 Peter Dimov and Multi Media Ltd.
//  Copyright (c) 2009 Steven Watanabe
//  Copyright (c) 2011-2013 Hartmut Kaiser
//  Copyright (c) 2013 Agustin Berge
//
// Distributed under the Boost Software License, Version 1.0. (See
// accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_PROTECT_SEP_23_2011_1230PM)
#define HPX_UTIL_PROTECT_SEP_23_2011_1230PM

#include <boost/config.hpp>
#include <hpx/traits/is_bind_expression.hpp>
#include <hpx/util/move.hpp>
#include <hpx/util/decay.hpp>

#include <boost/utility/enable_if.hpp>
#include <boost/utility/result_of.hpp>

namespace hpx { namespace util { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename F>
    class protected_bind : public F
    {
    public:
        // copy constructor
        protected_bind(protected_bind const& other)
          : F(other)
        {}

        // move constructor
        protected_bind(BOOST_RV_REF(protected_bind) other)
          : F(boost::move(other))
        {}

        explicit protected_bind(F const & f)
          : F(f)
        {}

        explicit protected_bind(BOOST_RV_REF(F) f)
          : F(boost::move(f))
        {}

        protected_bind& operator=(BOOST_COPY_ASSIGN_REF(protected_bind) rhs)
        {
            F::operator=(rhs);
            return *this;
        }

        protected_bind& operator=(BOOST_RV_REF(protected_bind) rhs)
        {
            F::operator=(boost::move(rhs));
            return *this;
        }
    };
}}} // namespace hpx::util::detail

namespace hpx { namespace util
{
    template <typename T>
    typename boost::enable_if<
        traits::is_bind_expression<typename util::decay<T>::type>
      , detail::protected_bind<typename util::decay<T>::type>
    >::type
    protect(BOOST_FWD_REF(T) f)
    {
        return detail::protected_bind<
            typename util::decay<T>::type
        >(boost::forward<T>(f));
    }
    
    // leave everything that is not a bind expression as is
#   ifndef BOOST_NO_CXX_RVALUE_REFERENCES
    template <typename T>
    typename boost::disable_if<
        traits::is_bind_expression<typename util::decay<T>::type>
      , T
    >::type
    protect(T&& v)
    {
        return boost::forward<T>(v);
    }
#   else
    template <typename T>
    typename boost::disable_if<
        traits::is_bind_expression<typename util::decay<T>::type>
      , T&
    >::type
    protect(T& v)
    {
        return v;
    }
    template <typename T>
    typename boost::disable_if<
        traits::is_bind_expression<typename util::decay<T>::type>
      , T
    >::type
    protect(BOOST_FWD_REF(T) v)
    {
        return boost::forward<T>(v);
    }
#   endif

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
