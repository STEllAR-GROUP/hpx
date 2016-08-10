//  Copyright (c) 2007-2014 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_RUNTIME_PARCELSET_FWD_HPP
#define HPX_RUNTIME_PARCELSET_FWD_HPP

#include <hpx/config.hpp>
#include <hpx/exception_fwd.hpp>
#include <hpx/util/function.hpp>

#include <boost/system/error_code.hpp>

namespace hpx {
    ///////////////////////////////////////////////////////////////////////////
    /// \namespace parcelset
    namespace parcelset
    {
        class HPX_API_EXPORT locality;

        class HPX_API_EXPORT parcel;
        class HPX_API_EXPORT parcelport;
        class HPX_API_EXPORT parcelhandler;

        namespace policies
        {
            struct message_handler;
        }

        HPX_API_EXPORT policies::message_handler* get_message_handler(
            parcelhandler* ph, char const* name, char const* type,
            std::size_t num, std::size_t interval, locality const& l,
            error_code& ec = throws);

        HPX_API_EXPORT bool do_background_work(std::size_t num_thread = 0);

        typedef util::function_nonser<
            void(boost::system::error_code const&, parcel const&)
        > write_handler_type;
    }
}

#endif /*HPX_RUNTIME_PARCELSET_FWD_HPP*/
