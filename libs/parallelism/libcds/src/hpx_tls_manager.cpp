//  Copyright (c) 2020 Weile Wei
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/assert.hpp>
#include <hpx/modules/libcds.hpp>
#include <hpx/modules/threading.hpp>
#include <hpx/modules/threading_base.hpp>

#include <cds/gc/details/hp_common.h>
#include <cds/gc/hp.h>
#include <cds/init.h>

#include <atomic>
#include <cstddef>

namespace cds { namespace gc { namespace hp { namespace details {

    thread_data* HPXDataHolder::getTLS()
    {
        auto thread_id = hpx::threads::get_self_id();
        std::size_t hpx_tls_data =
            hpx::threads::get_libcds_hazard_pointer_data(thread_id);
        return reinterpret_cast<thread_data*>(hpx_tls_data);
    }

    void HPXDataHolder::setTLS(thread_data* new_tls)
    {
        auto thread_id = hpx::threads::get_self_id();
        std::size_t hp_tls_data = reinterpret_cast<std::size_t>(new_tls);
        hpx::threads::set_libcds_hazard_pointer_data(thread_id, hp_tls_data);
    }

    generic_smr<HPXDataHolder>* hpx_data_holder_instance_ = nullptr;

    generic_smr<HPXDataHolder>* HPXDataHolder::getInstance()
    {
        return hpx_data_holder_instance_;
    }

    void HPXDataHolder::setInstance(generic_smr<HPXDataHolder>* new_instance)
    {
        hpx_data_holder_instance_ = new_instance;
    }
}}}}    // namespace cds::gc::hp::details

namespace hpx { namespace cds {

    ///////////////////////////////////////////////////////////////////////////
    // the boolean use_libcds option is provided to make comparison
    // of certain tests with/without libcds easier
    // @TODO : we should remove it one day
    hpxthread_manager_wrapper::hpxthread_manager_wrapper(bool use_libcds)
      : uselibcds_(use_libcds)
    {
        if (uselibcds_)
        {
            if (++thread_counter_ >
                detail::get_num_concurrent_hazard_pointer_threads())
            {
                HPX_THROW_EXCEPTION(invalid_status,
                    "hpx::cds::hpxthread_manager_wrapper ",
                    "attaching more threads than number of maximum allowed "
                    "detached threads, consider update "
                    "--hpx:ini=hpx.cds.num_concurrent_hazard_pointer_"
                    "threads to a larger number");
            }

            if (::cds::gc::hp::custom_smr<
                    ::cds::gc::hp::details::HPXDataHolder>::isUsed())
            {
                ::cds::gc::hp::custom_smr<
                    ::cds::gc::hp::details::HPXDataHolder>::attach_thread();
            }
            else
            {
                HPX_THROW_EXCEPTION(invalid_status,
                    "hpx::cds::hpxthread_manager_wrapper ",
                    "failed to attach_thread to HPXDataHolder, please check"
                    "if hazard pointer is constructed.");
            }
        }
    }

    hpxthread_manager_wrapper::~hpxthread_manager_wrapper()
    {
        if (uselibcds_)
        {
            HPX_ASSERT(thread_counter_-- == 0);

            ::cds::gc::hp::custom_smr<
                ::cds::gc::hp::details::HPXDataHolder>::detach_thread();
        }
    }

    std::atomic<std::size_t> hpxthread_manager_wrapper::thread_counter_{0};
}}    // namespace hpx::cds
