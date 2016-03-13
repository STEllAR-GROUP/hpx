// Copyright (c) 2006, 2007 Julio M. Merino Vidal
// Copyright (c) 2008 Ilya Sokolov, Boris Schaeling
// Copyright (c) 2009 Boris Schaeling
// Copyright (c) 2010 Felipe Tanus, Boris Schaeling
// Copyright (c) 2011, 2012 Jeff Flinn, Boris Schaeling
// Copyright (c) 2016 Hartmut Kaiser
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PROCESS_MITIGATE_HPP
#define HPX_PROCESS_MITIGATE_HPP

#if defined(HPX_WINDOWS)
#include <boost/asio/windows/stream_handle.hpp>
#else
#include <boost/asio/posix/stream_handle.hpp>
#endif

namespace hpx { namespace components { namespace process  { namespace util {

#if defined(HPX_WINDOWS)
typedef boost::asio::windows::stream_handle pipe_end;
#else
typedef boost::asio::posix::stream_descriptor pipe_end;
#endif

inline const char *zero_device()
{
#if defined(HPX_WINDOWS)
    return "NUL";
#else
    return "/dev/zero";
#endif
}

inline const char *null_device()
{
#if defined(HPX_WINDOWS)
    return "NUL";
#else
    return "/dev/null";
#endif
}

}}}}

#endif
