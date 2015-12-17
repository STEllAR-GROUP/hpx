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

#include <hpx/traits/is_bind_expression.hpp>
#include <hpx/util/decay.hpp>

#include <utility>

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
        protected_bind(protected_bind && other)
          : F(std::move(other))
        {}

        explicit protected_bind(F const & f)
          : F(f)
        {}

        explicit protected_bind(F && f)
          : F(std::move(f))
        {}

        protected_bind& operator=(protected_bind const & rhs)
        {
            F::operator=(rhs);
            return *this;
        }

        protected_bind& operator=(protected_bind && rhs)
        {
            F::operator=(std::move(rhs));
            return *this;
        }
    };
}}} // namespace hpx::util::detail

namespace hpx { namespace util
{
    template <typename T>
    typename std::enable_if<
        traits::is_bind_expression<typename util::decay<T>::type>::value
      , detail::protected_bind<typename util::decay<T>::type>
    >::type
    protect(T && f)
    {
        return detail::protected_bind<
            typename util::decay<T>::type
        >(std::forward<T>(f));
    }

    // leave everything that is not a bind expression as is
    template <typename T>
    typename std::enable_if<
        !traits::is_bind_expression<typename util::decay<T>::type>::value
      , T
    >::type
    protect(T && v) //-V659
    {
        return std::forward<T>(v);
    }
}} // namespace hpx::util

#endif
