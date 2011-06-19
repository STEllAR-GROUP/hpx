//  Copyright (c)      2011 Bryce Lelbach
//  Copyright (c) 2007-2011 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_18186AAE_AF96_4ADC_88AF_215F13F18004)
#define HPX_18186AAE_AF96_4ADC_88AF_215F13F18004

#include <hpx/runtime/components/stubs/stub_base.hpp>
#include <hpx/lcos/base_lco.hpp>
#include <hpx/lcos/eager_future.hpp>
#include <hpx/lcos/server/object_semaphore.hpp>

namespace hpx { namespace lcos { namespace stubs 
{

template <typename ValueType, typename RemoteType>
struct object_semaphore : components::stubs::stub_base<
  lcos::server::object_semaphore<ValueType, RemoteType>
> {
    typedef lcos::server::object_semaphore<ValueType, RemoteType> server_type;

    typedef typename server_type::value_type value_type;
    typedef typename server_type::remote_type remote_type;

    ///////////////////////////////////////////////////////////////////////////
    static lcos::future_value<void> signal_async(
        naming::id_type const& gid
      , ValueType const& val
      , boost::uint64_t count
    ) {
        typedef typename server_type::set_result_action action_type; 
        return lcos::eager_future<action_type>(gid, remote_type(val, count));
    }

    static void signal_sync(
        naming::id_type const& gid
      , ValueType const& val
      , boost::uint64_t count
    ) {
        signal_async(gid, val, count).get();
    }

    static void signal(
        naming::id_type const& gid
      , ValueType const& val
      , boost::uint64_t count
    ) {
        signal_async(gid, val, count).get();
    }

    ///////////////////////////////////////////////////////////////////////////
    static lcos::local_dataflow_variable<ValueType>
    wait_async(naming::id_type const& gid)
    {
        typedef typename server_type::add_lco_action action_type; 
        lcos::local_dataflow_variable<ValueType> data;
        hpx::applier::apply<action_type>(gid, data.get_gid());
        return data;
    }

    static ValueType wait_sync(naming::id_type const& gid)
    {
        return wait_async(gid).get();
    }

    static ValueType wait(naming::id_type const& gid)
    {
        return wait_async(gid).get();
    }

    ///////////////////////////////////////////////////////////////////////////
    static void 
    abort_pending(
        naming::id_type const& gid
      , boost::exception_ptr const& e
    ) {
        typedef lcos::base_lco::set_error_action action_type;
        hpx::applier::apply<action_type>(gid, e);
    }
};

}}}

#endif // HPX_18186AAE_AF96_4ADC_88AF_215F13F18004

