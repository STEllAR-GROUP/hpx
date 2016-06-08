//  Copyright (c)      2011 Bryce Lelbach
//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_18186AAE_AF96_4ADC_88AF_215F13F18004)
#define HPX_18186AAE_AF96_4ADC_88AF_215F13F18004

#include <hpx/async.hpp>
#include <hpx/lcos/base_lco.hpp>
#include <hpx/lcos/server/object_semaphore.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>

namespace hpx { namespace lcos { namespace stubs
{

template <typename ValueType>
struct object_semaphore : components::stub_base<
    lcos::server::object_semaphore<ValueType> >
{
    typedef lcos::server::object_semaphore<ValueType> server_type;

    ///////////////////////////////////////////////////////////////////////////
    static lcos::future<void> signal_async(
        naming::id_type const& gid
      , ValueType const& val
      , boost::uint64_t count)
    {
        typedef typename server_type::signal_action action_type;
        return hpx::async<action_type>(gid, val, count);
    }

    static void signal_sync(
        naming::id_type const& gid
      , ValueType const& val
      , boost::uint64_t count)
    {
        signal_async(gid, val, count).get();
    }

    static void signal(
        naming::id_type const& gid
      , ValueType const& val
      , boost::uint64_t count)
    {
        signal_async(gid, val, count).get();
    }

    ///////////////////////////////////////////////////////////////////////////
    static lcos::future<ValueType>
    get_async(naming::id_type const& gid)
    {
        typedef typename server_type::get_action action_type;
        lcos::promise<ValueType> lco;
        hpx::apply<action_type>(gid, lco.get_id());
        return lco.get_future();
    }

    static ValueType get_sync(naming::id_type const& gid)
    {
        return get_async(gid).get();
    }

    static ValueType get(naming::id_type const& gid)
    {
        return get_async(gid).get();
    }

    ///////////////////////////////////////////////////////////////////////////
    static lcos::future<void> abort_pending_async(
        naming::id_type const& gid
      , error ec)
    {
        typedef typename server_type::abort_pending_action action_type;
        return hpx::async<action_type>(gid, ec);
    }

    static void abort_pending_sync(
        naming::id_type const& gid
      , error ec)
    {
        abort_pending_async(gid, ec).get();
    }

    static void abort_pending(
        naming::id_type const& gid
      , error ec)
    {
        abort_pending_async(gid, ec).get();
    }

    ///////////////////////////////////////////////////////////////////////////
    static lcos::future<void> wait_async(
        naming::id_type const& gid)
    {
        typedef typename server_type::wait_action action_type;
        return hpx::async<action_type>(gid);
    }

    static void wait_sync(naming::id_type const& gid)
    {
        wait_async(gid).get();
    }

    static void wait(naming::id_type const& gid)
    {
        wait_sync(gid);
    }
};

}}}

#endif // HPX_18186AAE_AF96_4ADC_88AF_215F13F18004

