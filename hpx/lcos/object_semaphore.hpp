//  Copyright (c)      2011 Bryce Lelbach
//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_1FB4A979_B6B6_4845_BD95_3CEC605011A2)
#define HPX_1FB4A979_B6B6_4845_BD95_3CEC605011A2

#include <hpx/exception.hpp>
#include <hpx/include/client.hpp>
#include <hpx/lcos/stubs/object_semaphore.hpp>

namespace hpx { namespace lcos
{

template <typename ValueType>
struct object_semaphore
  : components::client_base<
        object_semaphore<ValueType>,
        lcos::stubs::object_semaphore<ValueType>
    >
{
    typedef components::client_base<
        object_semaphore,
        lcos::stubs::object_semaphore<ValueType>
    > base_type;

    object_semaphore() {}

    object_semaphore(naming::id_type gid) : base_type(gid) {}

    ///////////////////////////////////////////////////////////////////////////
    lcos::future<void> signal_async(
        ValueType const& val
      , boost::uint64_t count = 1)
    {
        HPX_ASSERT(this->get_gid());
        return this->base_type::signal_async(this->get_gid(), val, count);
    }

    void signal_sync(
        ValueType const& val
      , boost::uint64_t count = 1)
    {
        HPX_ASSERT(this->get_gid());
        return this->base_type::signal_sync(this->get_gid(), val, count);
    }

    void signal(
        ValueType const& val
      , boost::uint64_t count = 1)
    {
        signal_sync(val, count);
    }

    ///////////////////////////////////////////////////////////////////////////
    lcos::future<ValueType> get_async()
    {
        HPX_ASSERT(this->get_gid());
        return this->base_type::get_async(this->get_gid());
    }

    ValueType get_sync()
    {
        HPX_ASSERT(this->get_gid());
        return this->base_type::get_sync(this->get_gid());
    }

    ValueType get()
    { return get_sync(); }

    ///////////////////////////////////////////////////////////////////////////
    void abort_pending_async(error ec = no_success)
    {
        HPX_ASSERT(this->get_gid());
        this->base_type::abort_pending_sync(this->get_gid(), ec);
    }

    void abort_pending_sync(error ec = no_success)
    {
        HPX_ASSERT(this->get_gid());
        this->base_type::abort_pending_sync(this->get_gid(), ec);
    }

    void abort_pending(error ec = no_success)
    { abort_pending_sync(ec); }

    ///////////////////////////////////////////////////////////////////////////
    void wait_async()
    {
        HPX_ASSERT(this->get_gid());
        this->base_type::wait_sync(this->get_gid());
    }

    void wait_sync()
    {
        HPX_ASSERT(this->get_gid());
        this->base_type::wait_sync(this->get_gid());
    }

    void wait()
    { wait_sync(); }
};

}}

#endif // HPX_1FB4A979_B6B6_4845_BD95_3CEC605011A2

