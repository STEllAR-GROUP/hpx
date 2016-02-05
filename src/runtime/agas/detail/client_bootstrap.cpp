//  Copyright (c) 2016 Hartmut Kaiser
//  Copyright (c) 2016 Parsa Amini
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/runtime/agas/detail/client_bootstrap.hpp>

namespace hpx { namespace agas { namespace detail
{
    void client_bootstrap::set_local_locality(naming::gid_type const& g)
    {
        data_.primary_ns_server_.set_local_locality(g);
    }
}}}

