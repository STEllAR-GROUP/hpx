//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#if defined(HPX_WINDOWS)

#if !defined(WIN32)
#  define WIN32
#endif
#include <winsock2.h>
#include <windows.h>
#include <boost/asio/detail/winsock_init.hpp>

namespace hpx { namespace detail
{
    // Make sure the Winsocket library is explicitly initialized before main
    // is executed.
    struct winsocket_init_helper
    {
        static const boost::asio::detail::winsock_init<> init;
    };

    boost::asio::detail::winsock_init<> const winsocket_init_helper::init =
        boost::asio::detail::winsock_init<>(false);

    // This function makes sure this file is actually linked to the executable.
    void init_winsocket()
    {
        static winsocket_init_helper init;
    }
}}

#endif

