//  Copyright (c) 2016 Hartmut Kaiser
//  Copyright (c) 2016 Parsa Amini
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_AGAS_CLIENT_BASE_FEB_05_2016_1144AM)
#define HPX_AGAS_CLIENT_BASE_FEB_05_2016_1144AM

namespace hpx { namespace agas { namespace detail
{
    struct bootstrap_data_type;
    struct hosted_data_type;

    struct client_implementation_base
    {
        virtual ~client_implementation_base() {}

        virtual void set_local_locality(naming::gid_type const& g) = 0;
    };
}}}

#endif

