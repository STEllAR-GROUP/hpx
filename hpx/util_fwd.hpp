//  Copyright (c) 2007-2014 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file util_fwd.hpp

#ifndef HPX_UTIL_FWD_HPP
#define HPX_UTIL_FWD_HPP

#include <hpx/config.hpp>

namespace hpx { namespace util
{
    /// \cond NOINTERNAL
    class backtrace;

    template <typename Sig, bool Serializable = true>
    class function;

#ifdef HPX_HAVE_CXX11_ALIAS_TEMPLATES
    template <typename Sig>
    using function_nonser = function<Sig, false>;
#else
    template <typename T>
    class function_nonser;
#endif

    class HPX_EXPORT io_service_pool;

    class HPX_EXPORT runtime_configuration;
    class HPX_EXPORT section;

    template <typename Sig, bool Serializable = true>
    class unique_function;

#ifdef HPX_HAVE_CXX11_ALIAS_TEMPLATES
    template <typename Sig>
    using unique_function_nonser = unique_function<Sig, false>;
#else
    template <typename T>
    class unique_function_nonser;
#endif
    /// \endcond
}}

#endif /*HPX_UTIL_FWD_HPP*/
