//  Copyright (c) 2011 Bryce Lelbach
//  Copyright (c) 2011-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_FF038007_9487_465F_B750_9452CF6D6693)
#define HPX_FF038007_9487_465F_B750_9452CF6D6693

#include <hpx/include/async.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>
#include <hpx/components/iostreams/server/output_stream.hpp>

namespace hpx { namespace iostreams { namespace stubs
{
    struct output_stream : components::stub_base<server::output_stream>
    {
        static void
        write_sync(naming::id_type const& gid, detail::buffer const& in)
        {
            typedef server::output_stream::write_sync_action action_type;
            hpx::async<action_type>(gid, in).get();
        }

        static void
        write_async(naming::id_type const& gid, detail::buffer const& in)
        {
            typedef server::output_stream::write_async_action action_type;
            hpx::async<action_type>(gid, in).get();
        }
    };
}}}

#endif // HPX_FF038007_9487_465F_B750_9452CF6D6693

