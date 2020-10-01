//  Copyright (c) 2007-2018 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_NETWORKING)
#include <hpx/modules/errors.hpp>
#include <hpx/functional/function.hpp>

#include <cstddef>
#include <system_error>

namespace hpx {
    ///////////////////////////////////////////////////////////////////////////
    /// \namespace parcelset
    namespace parcelset
    {
        class HPX_EXPORT locality;

        class HPX_EXPORT parcel;
        class HPX_EXPORT parcelport;
        class HPX_EXPORT parcelhandler;

        namespace policies
        {
            struct message_handler;
        }

        namespace detail
        {
            struct create_parcel;
        }

        //////////////////////////////////////////////////////////////////////
        /// \brief Return a pointer of type message handler between parcel
        ///        deliveries.
        ///
        /// This function returns a pointer of type message_handler during parecel
        /// exchange.
        ///
        /// \param ph this represents parcelhandle type.
        ///
        /// \param name this represents the name of the function.
        ///
        /// \param type this represents message type.
        ///
        /// \param num this represents the number of messages.
        ///
        /// \param interval this represents the intervals messages should be sent.
        ///
        /// \param locality this represents the locality type.
        ///
        /// \param ec[int, out] this represents the error code during exit.

        HPX_EXPORT policies::message_handler* get_message_handler(
            parcelhandler* ph, char const* name, char const* type,
            std::size_t num, std::size_t interval, locality const& l,
            error_code& ec = throws);

        ////////////////////////////////////////////////////////////////////////
        /// \brief Return boolean value when thread processing is completed.
        ///
        /// This returns true/false based on the background work.
        ///
        /// \param num_thread this represents the number of threads.

        /// Type of background work to perform
        enum parcelport_background_mode
        {
            /// perform buffer flush operations
            parcelport_background_mode_flush_buffers = 0x01,
            /// perform send operations (includes buffer flush)
            parcelport_background_mode_send = 0x03,
            /// perform receive operations
            parcelport_background_mode_receive = 0x04,
            /// perform all operations
            parcelport_background_mode_all = 0x07
        };

        HPX_EXPORT bool do_background_work(std::size_t num_thread = 0,
            parcelport_background_mode mode = parcelport_background_mode_all);

        typedef util::function_nonser<
            void(std::error_code const&, parcel const&)
        > write_handler_type;

        ///////////////////////////////////////////////////////////////////////
        /// Hand a parcel to the underlying parcel layer for delivery.
        HPX_EXPORT void put_parcel(parcel&& p, write_handler_type&& f);

        /// Hand a parcel to the underlying parcel layer for delivery.
        /// Wait for the operation to finish before returning to the user.
        HPX_EXPORT void sync_put_parcel(parcelset::parcel&& p);
    }
}

#endif
