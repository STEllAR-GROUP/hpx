////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_0C9D09E0_725D_4FA6_A879_8226DE97C6B9)
#define HPX_0C9D09E0_725D_4FA6_A879_8226DE97C6B9

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/io_service_pool.hpp>
#include <hpx/util/connection_cache.hpp>
#include <hpx/util/unique_function.hpp>
#include <boost/lockfree/queue.hpp>
#include <hpx/runtime.hpp>
#include <hpx/runtime/naming/address.hpp>
#include <hpx/runtime/parcelset/parcelhandler.hpp>
#include <hpx/runtime/parcelset/parcelport.hpp>

#include <hpx/config/warnings_prefix.hpp>

#include <boost/cstdint.hpp>
#include <boost/noncopyable.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/condition_variable.hpp>

namespace hpx { namespace agas
{

struct HPX_EXPORT big_boot_barrier : boost::noncopyable
{
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
        boost::uint32_t source_locality_id
      , boost::uint32_t target_locality_id
      , parcelset::locality const & dest
      , Action act
      , Args &&... args
    ) { // {{{
        HPX_ASSERT(pp);
        naming::address addr(naming::get_gid_from_locality_id(target_locality_id));
        parcelset::parcel p(naming::get_id_from_locality_id(target_locality_id),
                addr, act, std::forward<Args>(args)...);
        if (!p.parcel_id())
            p.parcel_id() = parcelset::parcel::generate_unique_id(source_locality_id);
        pp->send_early_parcel(dest, std::move(p));
    } // }}}

    template <typename Action, typename... Args>
    void apply_late(
        boost::uint32_t source_locality_id
      , boost::uint32_t target_locality_id
      , parcelset::locality const & dest
      , Action act
      , Args &&... args
    ) { // {{{
        naming::address addr(naming::get_gid_from_locality_id(target_locality_id));
        parcelset::parcel p(naming::get_id_from_locality_id(target_locality_id),
                addr, act, std::forward<Args>(args)...);
        if (!p.parcel_id())
            p.parcel_id() = parcelset::parcel::generate_unique_id(source_locality_id);
        get_runtime().get_parcel_handler().put_parcel(std::move(p));
    } // }}}

    void wait_bootstrap();
    void wait_hosted(std::string const& locality_name,
        void* primary_ns_ptr, void* symbol_ns_ptr);

    // no-op on non-bootstrap localities
    void trigger();

    void add_thunk(util::unique_function_nonser<void()>* f)
    {
        thunks.push(f);
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

