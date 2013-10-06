////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/components/iostreams/manipulators.hpp>

namespace hpx { namespace iostreams
{
// #if !defined(BOOST_WINDOWS)
    async_flush_type async_flush = { };
    async_endl_type async_endl = { };
    flush_type flush = flush_type();
    endl_type endl = endl_type();
    sync_flush_type sync_flush = { };
    sync_endl_type sync_endl = { };
    local_flush_type local_flush = { };
    local_endl_type local_endl = { };
// #endif
}}

