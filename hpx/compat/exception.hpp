//  Copyright (c) 2017 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_COMPAT_EXCEPTION_HPP
#define HPX_COMPAT_EXCEPTION_HPP

// hpxinspect:nodeprecatedinclude:boost/exception.hpp
// hpxinspect:nodeprecatedname:boost::exception_ptr
// hpxinspect:nodeprecatedname:boost::make_exception_ptr
// hpxinspect:nodeprecatedname:boost::current_exception
// hpxinspect:nodeprecatedname:boost::rethrow_exception

#include <hpx/config.hpp>

#if defined(HPX_HAVE_EXCEPTION_PTR_COMPATIBILITY)
///////////////////////////////////////////////////////////////////////////////
#include <boost/exception_ptr.hpp>

namespace hpx { namespace compat
{
    using exception_ptr = boost::exception_ptr;

    template <typename E>
    inline exception_ptr make_exception_ptr(E e) noexcept
    {
        return boost::copy_exception(e);
    }

    inline exception_ptr current_exception() noexcept
    {
        return boost::current_exception();
    }

    HPX_ATTRIBUTE_NORETURN inline void exception_ptr rethrow_exception(exception_ptr e)
    {
        return boost::rethrow_exception(e);
    }
}}
#else
///////////////////////////////////////////////////////////////////////////////
#include <exception>

namespace hpx { namespace compat
{
    using exception_ptr = std::exception_ptr;
    using std::make_exception_ptr;
    using std::current_exception;
    using std::rethrow_exception;
}}
#endif

#endif /*HPX_COMPAT_EXCEPTION_HPP*/
