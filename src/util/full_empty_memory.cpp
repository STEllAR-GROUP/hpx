//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/static.hpp>
#include <hpx/util/full_empty_store.hpp>
#include <hpx/util/full_empty_memory.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    full_empty_base::store_type& full_empty_base::get_store()
    {
        // ensure thread-safe initialization
        util::static_<store_type, full_empty_tag> store;
        return store.get();
    }

}}}
