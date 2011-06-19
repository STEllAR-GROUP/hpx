//  Copyright (c)      2011 Bryce Lelbach
//  Copyright (c) 2007-2011 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_1FB4A979_B6B6_4845_BD95_3CEC605011A2)
#define HPX_1FB4A979_B6B6_4845_BD95_3CEC605011A2

#include <hpx/runtime/components/client_base.hpp>
#include <hpx/lcos/stubs/object_semaphore.hpp>

namespace hpx { namespace lcos 
{

template <typename ValueType, typename RemoteType>
struct object_semaphore 
  : components::client_base<
        object_semaphore<ValueType, RemoteType>, 
        lcos::stubs::object_semaphore<ValueType, RemoteType>
    >
{
    typedef components::client_base<
        object_semaphore,
        lcos::stubs::object_semaphore<ValueType, RemoteType>
    > base_type;

    object_semaphore() {}

    object_semaphore(naming::id_type gid) : base_type(gid) {}

    ///////////////////////////////////////////////////////////////////////////
    lcos::future_value<void> signal_async(
        ValueType const& val
      , boost::uint64_t count = 1
    ) {
        BOOST_ASSERT(this->gid_);
        return this->base_type::signal_async(this->gid_, val, count);
    }

    void signal_sync(
        ValueType const& val
      , boost::uint64_t count = 1
    ) {
        BOOST_ASSERT(this->gid_);
        return this->base_type::signal_sync(this->gid_, val, count);
    }

    void signal(
        ValueType const& val
      , boost::uint64_t count = 1
    ) {
        signal_sync(val, count);
    }

    ///////////////////////////////////////////////////////////////////////////
    lcos::future_value<ValueType> wait_async()
    {
        BOOST_ASSERT(this->gid_);
        return this->base_type::wait_async(this->gid_);
    }

    ValueType wait_sync()
    {
        BOOST_ASSERT(this->gid_);
        return this->base_type::wait_sync(this->gid_);
    }

    ValueType wait()
    { return wait_sync(); }

    ///////////////////////////////////////////////////////////////////////////
    void abort_pending()
    {
        BOOST_ASSERT(this->gid_);

        try
        {
            HPX_THROW_EXCEPTION(no_success, "object_semaphore::abort_pending", 
                "interrupt all pending requests");
        }

        catch (...)
        {
            this->base_type::abort_pending
                (this->gid_, boost::current_exception());
        }
    }
};

}}

#endif // HPX_1FB4A979_B6B6_4845_BD95_3CEC605011A2

