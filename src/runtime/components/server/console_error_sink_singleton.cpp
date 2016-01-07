//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/util/reinitializable_static.hpp>
#include <hpx/runtime/components/server/console_error_sink_singleton.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace server
{
    struct error_dispatcher_tag {};

    console_error_dispatcher& get_error_dispatcher()
    {
        util::reinitializable_static<console_error_dispatcher,
            error_dispatcher_tag> disp;
        return disp.get();
    }
}}}

