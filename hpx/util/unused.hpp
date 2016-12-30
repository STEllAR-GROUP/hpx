/*=============================================================================
    Copyright (c) 2001-2011 Joel de Guzman
    Copyright (c) 2007-2013 Hartmut Kaiser

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
==============================================================================*/
#if !defined(HPX_UNUSED_FEB_01_2009_1217PM)
#define HPX_UNUSED_FEB_01_2009_1217PM

#include <hpx/config.hpp>
#if defined(HPX_MSVC)
# pragma warning(push)
# pragma warning(disable: 4522) // multiple assignment operators specified warning
#endif

namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    // We do not import fusion::unused_type anymore to avoid boost::fusion
    // being turned into an associate namespace, as this interferes with ADL
    // in unexpected ways. We rather copy the full unused_type implementation.
    ///////////////////////////////////////////////////////////////////////////
    struct unused_type
    {
        HPX_CONSTEXPR HPX_HOST_DEVICE unused_type() HPX_NOEXCEPT
        {
        }

        template <typename T>
        HPX_CONSTEXPR HPX_HOST_DEVICE unused_type(T const&) HPX_NOEXCEPT
        {
        }

        template <typename T>
        HPX_CONSTEXPR HPX_HOST_DEVICE unused_type const&
        operator=(T const&) const HPX_NOEXCEPT
        {
            return *this;
        }

        template <typename T>
        HPX_HOST_DEVICE unused_type&
        operator=(T const&) HPX_NOEXCEPT
        {
            return *this;
        }

        HPX_CONSTEXPR HPX_HOST_DEVICE unused_type const&
        operator=(unused_type const&) const HPX_NOEXCEPT
        {
            return *this;
        }

        HPX_HOST_DEVICE unused_type&
        operator=(unused_type const&) HPX_NOEXCEPT
        {
            return *this;
        }
    };

    HPX_CONSTEXPR_OR_CONST unused_type unused = unused_type();
}}

//////////////////////////////////////////////////////////////////////////////
// use this to silence compiler warnings related to unused function arguments.
#if defined(__CUDA_ARCH__)
#define HPX_UNUSED(x)  (void)x
#else
#define HPX_UNUSED(x)  ::hpx::util::unused = (x)
#endif

#if defined(HPX_MSVC)
# pragma warning(pop)
#endif

#endif
