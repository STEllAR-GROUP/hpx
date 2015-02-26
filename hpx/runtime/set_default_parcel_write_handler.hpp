//  Copyright (c) 2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_RUNTIME_SET_DEFAULT_PARCEL_WRITE_HANDLER_FEB_25_2015_0806PM)
#define HPX_RUNTIME_SET_DEFAULT_PARCEL_WRITE_HANDLER_FEB_25_2015_0806PM

#include <hpx/config.hpp>
#include <hpx/util/function.hpp>

#include <boost/system/error_code.hpp>

namespace hpx
{
    typedef util::function_nonser<
            void(boost::system::error_code const&, parcelset::parcel const&)
        > parcel_write_handler_type;

    /// Set the default parcel write handler which is invoked once a parcel has
    /// been sent if no explicit write handler was specified.
    ///
    /// \param f    The new parcel write handler to use from this point on
    ///
    /// \returns The function returns the parcel write handler which was
    ///          installed before this function was called.
    ///
    HPX_API_EXPORT parcel_write_handler_type set_default_parcel_write_handler(
        parcel_write_handler_type const& f);
}

#endif
