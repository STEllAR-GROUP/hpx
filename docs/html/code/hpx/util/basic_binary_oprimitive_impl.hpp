// Copyright (c) 2007-2012 Hartmut Kaiser
// (C) Copyright 2002 Robert Ramey - http://www.rrsd.com .
//
// Use, modification and distribution is subject to the Boost Software
// License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#ifndef BASIC_BINARY_IPRIMITIVE_IMPL_HPP
#define BASIC_BINARY_IPRIMITIVE_IMPL_HPP

// MS compatible compilers support #pragma once
#if defined(_MSC_VER) && (_MSC_VER >= 1020)
# pragma once
#endif

#include <hpx/util/basic_binary_oprimitive.hpp>
#include <boost/archive/archive_exception.hpp>

namespace hpx { namespace util
{
    template <typename Archive>
    void basic_binary_oprimitive<Archive>::init(unsigned flags)
    {
        if (flags & boost::archive::no_header)
            return;

        // Record native sizes of fundamental types. This is to permit 
        // detection of attempts to pass native binary archives accross 
        // incompatible machines. This is not foolproof but its better 
        // than nothing.
        This()->save(static_cast<unsigned char>(sizeof(int)));
        This()->save(static_cast<unsigned char>(sizeof(long)));
        This()->save(static_cast<unsigned char>(sizeof(float)));
        This()->save(static_cast<unsigned char>(sizeof(double)));

        // for checking endianness
        This()->save(int(1));
    }
}}

#endif

