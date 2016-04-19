//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_THROTTLE_AUG_09_2011_0659PM)
#define HPX_THROTTLE_AUG_09_2011_0659PM

#include <hpx/hpx.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/include/client.hpp>

#include "stubs/throttle.hpp"

namespace throttle
{
    ///////////////////////////////////////////////////////////////////////////
    class throttle
      : public hpx::components::client_base<throttle, stubs::throttle>
    {
    private:
        typedef hpx::components::client_base<throttle, stubs::throttle> base_type;

    public:
        // create a new partition instance and initialize it synchronously
        throttle()
          : base_type(stubs::throttle::create(hpx::find_here()))
        {}

        throttle(hpx::future<hpx::naming::id_type> && gid)
          : base_type(std::move(gid))
        {}

        ~throttle()
        {
        }

        void suspend(std::size_t thread_num) const
        {
            return stubs::throttle::suspend(this->get_id(), thread_num);
        }

        void resume(std::size_t thread_num) const
        {
            return stubs::throttle::resume(this->get_id(), thread_num);
        }
    };
}

#endif
