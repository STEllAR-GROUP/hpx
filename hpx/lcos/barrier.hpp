//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_BARRIER_MAR_10_2010_0307PM)
#define HPX_LCOS_BARRIER_MAR_10_2010_0307PM

#include <hpx/exception.hpp>
#include <hpx/runtime/components/client_base.hpp>
#include <hpx/lcos/stubs/barrier.hpp>

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

        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        lcos::future<void> wait_async()
        {
            BOOST_ASSERT(gid_);
            return this->base_type::wait_async(gid_);
        }

        void wait()
        {
            BOOST_ASSERT(gid_);
            this->base_type::wait(gid_);
        }

        lcos::future<void> set_error_async(boost::exception_ptr const& e)
        {
            BOOST_ASSERT(gid_);
            return this->base_type::set_error_async(gid_, e);
        }

        void set_error(boost::exception_ptr const& e)
        {
            BOOST_ASSERT(gid_);
            this->base_type::set_error(gid_, e);
        }
    };
}}

#endif

