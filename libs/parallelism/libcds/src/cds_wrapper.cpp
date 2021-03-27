//  Copyright (c) 2020 Weile Wei
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/modules/libcds.hpp>

#include <cds/gc/details/hp_common.h>
#include <cds/gc/hp.h>
#include <cds/init.h>
#include <cds/urcu/general_buffered.h>

#include <cstddef>

namespace hpx { namespace cds {

    namespace detail {

        get_num_concurrent_hazard_pointer_threads_type&
        get_get_num_concurrent_hazard_pointer_threads()
        {
            static get_num_concurrent_hazard_pointer_threads_type f;
            return f;
        }

        void set_get_num_concurrent_hazard_pointer_threads(
            get_num_concurrent_hazard_pointer_threads_type f)
        {
            get_get_num_concurrent_hazard_pointer_threads() = f;
        }

        std::size_t get_num_concurrent_hazard_pointer_threads()
        {
            if (get_get_num_concurrent_hazard_pointer_threads())
            {
                return get_get_num_concurrent_hazard_pointer_threads()();
            }
            else
            {
                return 128;
            }
        }
    }    // namespace detail

    // hazard_pointer_count, max_thread_count, max_retired_pointer_count
    // are only used in hazard pointer
    //
    // hazard_pointer_count is corresponding var nHazardPtrCount
    // in libcds that defines Hazard pointer count per thread;
    //
    // max_concurrent_attach_thread_ is corresponding var nMaxThreadCount
    // in libcds that defines Max count of simultaneous working thread
    // in your application
    //
    // max_retired_pointer_count is corresponding var
    // nMaxRetiredPtrCount in libcds that defines capacity
    // of the array of retired objects for the thread
    libcds_wrapper::libcds_wrapper(smr_t smr_type,
        std::size_t hazard_pointer_count, std::size_t max_thread_count,
        std::size_t max_retired_pointer_count)
      : smr_(smr_type)
    {
        // Initialize libcds
        ::cds::Initialize();

        switch (smr_type)
        {
        default:
            HPX_FALLTHROUGH;
        case smr_t::hazard_pointer_hpxthread:
        {
            ::cds::gc::hp::custom_smr<::cds::gc::hp::details::HPXDataHolder>::
                construct(hazard_pointer_count, max_thread_count,
                    max_retired_pointer_count);
        }
        break;

        case smr_t::hazard_pointer_stdthread:
        {
            ::cds::gc::hp::custom_smr<::cds::gc::hp::details::
                    DefaultDataHolder>::construct(hazard_pointer_count,
                max_thread_count, max_retired_pointer_count);
        }
        break;

        case smr_t::rcu:
        {
            using rcu_gpb = ::cds::urcu::gc<::cds::urcu::general_buffered<>>;
            // Initialize general_buffered RCU
            ::cds::urcu::general_buffered<>::Construct(256);
            ::cds::threading::Manager::attachThread();
        }
        break;
        }
    }

    libcds_wrapper::~libcds_wrapper()
    {
        // Terminate libcds
        ::cds::Terminate();
        switch (smr_)
        {
        default:
            HPX_FALLTHROUGH;
        case smr_t::hazard_pointer_hpxthread:
            ::cds::gc::hp::custom_smr<
                ::cds::gc::hp::details::HPXDataHolder>::destruct(true);
            break;

        case smr_t::hazard_pointer_stdthread:
            ::cds::gc::hp::custom_smr<
                ::cds::gc::hp::details::DefaultDataHolder>::destruct(true);
            break;

        case smr_t::rcu:
            ::cds::threading::Manager::detachThread();
            ::cds::urcu::general_buffered<>::Destruct(true);
            break;
        }
    }
}}    // namespace hpx::cds
