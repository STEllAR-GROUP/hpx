////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_0C9D09E0_725D_4FA6_A879_8226DE97C6B9)
#define HPX_0C9D09E0_725D_4FA6_A879_8226DE97C6B9

#include <hpx/config.hpp>
#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/runtime.hpp>
#include <hpx/runtime/naming/address.hpp>
#include <hpx/runtime/parcelset/parcelhandler.hpp>
#include <hpx/runtime/parcelset/parcelport.hpp>
#include <hpx/runtime/parcelset/put_parcel.hpp>
#include <hpx/runtime/parcelset/detail/parcel_await.hpp>
#include <hpx/util/connection_cache.hpp>
#include <hpx/util/io_service_pool.hpp>
#include <hpx/util_fwd.hpp>
#include <boost/lockfree/queue.hpp>

#include <boost/thread/condition_variable.hpp>
#include <boost/thread/mutex.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx { namespace agas
{

struct HPX_EXPORT big_boot_barrier
{
  private:
    HPX_NON_COPYABLE(big_boot_barrier);

  private:
    parcelset::parcelport* pp;
    parcelset::endpoints_type const& endpoints;

    service_mode const service_type;
    parcelset::locality const bootstrap_agas;

    boost::condition_variable cond;
    boost::mutex mtx;
    std::size_t connected;

    boost::lockfree::queue<util::unique_function_nonser<void()>* > thunks;

    void spin();

    void notify();

  public:
    struct scoped_lock
    {
      private:
        big_boot_barrier& bbb;

      public:
        scoped_lock(
            big_boot_barrier& bbb_
            )
          : bbb(bbb_)
        {
            bbb.mtx.lock();
        }

        ~scoped_lock()
        {
            bbb.notify();
        }
    };

    big_boot_barrier(
        parcelset::parcelport* pp_
      , parcelset::endpoints_type const& endpoints_
      , util::runtime_configuration const& ini_
        );

    ~big_boot_barrier()
    {
        util::unique_function_nonser<void()>* f;
        while (thunks.pop(f))
            delete f;
    }

    parcelset::locality here() { return bootstrap_agas; }
    parcelset::endpoints_type const &get_endpoints() { return endpoints; }

    template <typename Action, typename... Args>
    void apply(
        std::uint32_t source_locality_id
      , std::uint32_t target_locality_id
      , parcelset::locality const & dest
      , Action act
      , Args &&... args
    ) { // {{{
        HPX_ASSERT(pp);
        naming::address addr(naming::get_gid_from_locality_id(target_locality_id));
        parcelset::parcel p(
            parcelset::detail::create_parcel::call(std::false_type(), std::false_type(),
                naming::get_gid_from_locality_id(target_locality_id),
                std::move(addr), act, std::forward<Args>(args)...));
#if defined(HPX_HAVE_PARCEL_PROFILING)
        if (!p.parcel_id())
        {
            p.parcel_id() = parcelset::parcel::generate_unique_id(source_locality_id);
        }
#endif
        auto f = [this, dest](parcelset::parcel p)
            {
                pp->send_early_parcel(dest, std::move(p));
            };
        typedef
            parcelset::detail::parcel_await<decltype(f)>
            parcel_await;
        parcel_await(std::move(p), 0, std::move(f)).apply();
    } // }}}

    template <typename Action, typename... Args>
    void apply_late(
        std::uint32_t source_locality_id
      , std::uint32_t target_locality_id
      , parcelset::locality const & dest
      , Action act
      , Args &&... args
    ) { // {{{
        naming::address addr(naming::get_gid_from_locality_id(target_locality_id));

        parcelset::put_parcel(
            naming::id_type(
                naming::get_gid_from_locality_id(target_locality_id),
                naming::id_type::unmanaged),
            std::move(addr), act, std::forward<Args>(args)...);
    } // }}}

    void wait_bootstrap();
    void wait_hosted(std::string const& locality_name,
        naming::address::address_type primary_ns_ptr,
        naming::address::address_type symbol_ns_ptr);

    // no-op on non-bootstrap localities
    void trigger();

    void add_thunk(util::unique_function_nonser<void()>* f)
    {
        std::size_t k = 0;
        while(!thunks.push(f))
        {
            // Wait until succesfully pushed ...
            hpx::lcos::local::spinlock::yield(k);
            ++k;
        }
    }
};

HPX_EXPORT void create_big_boot_barrier(
    parcelset::parcelport* pp_
  , parcelset::endpoints_type const& endpoints_
  , util::runtime_configuration const& ini_
    );

HPX_EXPORT void destroy_big_boot_barrier();

HPX_EXPORT big_boot_barrier& get_big_boot_barrier();

}}

#include <hpx/config/warnings_suffix.hpp>

#endif // HPX_0C9D09E0_725D_4FA6_A879_8226DE97C6B9

