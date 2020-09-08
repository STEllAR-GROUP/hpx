//  Copyright (c) 2007-2014 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file exception_fwd.hpp

#pragma once

#include <hpx/config.hpp>
#include <hpx/errors/error.hpp>

namespace hpx {
    /// \cond NOINTERNAL
    // forward declaration
    class error_code;

    class HPX_CORE_EXPORT exception;

    struct HPX_CORE_EXPORT thread_interrupted;
    /// \endcond

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Encode error category for new error_code.
    enum throwmode
    {
        plain = 0,
        rethrow = 1,
        lightweight =
            0x80,    // do not generate an exception for this error_code
        /// \cond NODETAIL
        lightweight_rethrow = lightweight | rethrow
        /// \endcond
    };

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Predefined error_code object used as "throw on error" tag.
    ///
    /// The predefined hpx::error_code object \a hpx::throws is supplied for use as
    /// a "throw on error" tag.
    ///
    /// Functions that specify an argument in the form 'error_code& ec=throws'
    /// (with appropriate namespace qualifiers), have the following error
    /// handling semantics:
    ///
    /// If &ec != &throws and an error occurred: ec.value() returns the
    /// implementation specific error number for the particular error that
    /// occurred and ec.category() returns the error_category for ec.value().
    ///
    /// If &ec != &throws and an error did not occur, ec.clear().
    ///
    /// If an error occurs and &ec == &throws, the function throws an exception
    /// of type \a hpx::exception or of a type derived from it. The exception's
    /// \a get_errorcode() member function returns a reference to an
    /// \a hpx::error_code object with the behavior as specified above.
    ///
#if defined(HPX_COMPUTE_DEVICE_CODE)
    // We can't actually refer to this in device code. This is only to satisfy
    // the compiler.
    extern HPX_DEVICE error_code throws;
#else
    HPX_CORE_EXPORT extern error_code throws;
#endif
}    // namespace hpx

#include <hpx/errors/throw_exception.hpp>
