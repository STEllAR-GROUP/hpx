//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_BARRIER_MAR_10_2010_0307PM)
#define HPX_LCOS_BARRIER_MAR_10_2010_0307PM

#include <hpx/config.hpp>
#include <hpx/include/client.hpp>
#include <hpx/lcos/stubs/barrier.hpp>

#include <boost/exception_ptr.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos
{
    class barrier
      : public components::client_base<barrier, lcos::stubs::barrier>
    {
        typedef components::client_base<barrier, lcos::stubs::barrier> base_type;

    public:
        barrier()
        {}

        /// Create a client side representation for the existing
        /// \a server#barrier instance with the given global id \a gid.
        barrier(naming::id_type gid)
          : base_type(gid)
        {}
        barrier(lcos::shared_future<naming::id_type> gid)
          : base_type(gid)
        {}

        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        lcos::future<void> wait_async()
        {
            return this->base_type::wait_async(get_id());
        }

        void wait()
        {
            this->lcos::stubs::barrier::wait(get_id());
        }

        lcos::future<void> set_exception_async(boost::exception_ptr const& e)
        {
            return this->base_type::set_exception_async(get_id(), e);
        }

        void set_exception(boost::exception_ptr const& e)
        {
            this->base_type::set_exception(get_id(), e);
        }
    };
}}

#endif

