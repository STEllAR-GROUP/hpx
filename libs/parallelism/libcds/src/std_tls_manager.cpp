//  Copyright (c) 2020 Weile Wei
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

// hpxinspect:noassert_macro

#include <hpx/modules/libcds.hpp>
#include <hpx/modules/threading.hpp>
#include <hpx/modules/threading_base.hpp>

#include <cds/gc/details/hp_common.h>
#include <cds/gc/hp.h>
#include <cds/init.h>

#include <cassert>
#include <stdexcept>

namespace hpx { namespace cds {

    ///////////////////////////////////////////////////////////////////////////
    stdthread_manager_wrapper::stdthread_manager_wrapper()
    {
        if (++hpxthread_manager_wrapper::thread_counter_ >
            detail::get_num_concurrent_hazard_pointer_threads())
        {
            throw std::runtime_error(
                "attaching more threads than number of maximum allowed "
                "detached threads, consider update "
                "--hpx:ini=hpx.cds.num_concurrent_hazard_pointer_"
                "threads to a larger number");
        }

        if (::cds::gc::hp::custom_smr<
                ::cds::gc::hp::details::DefaultDataHolder>::isUsed())
        {
            ::cds::gc::hp::custom_smr<
                ::cds::gc::hp::details::DefaultDataHolder>::attach_thread();
        }
        else
        {
            throw std::runtime_error(
                "failed to attach_thread to DefaultDataHolder, please check"
                "if hazard pointer is constructed.");
        }
    }

    stdthread_manager_wrapper::~stdthread_manager_wrapper()
    {
        assert(hpxthread_manager_wrapper::thread_counter_-- == 0);

        ::cds::gc::hp::custom_smr<
            ::cds::gc::hp::details::DefaultDataHolder>::detach_thread();
    }
}}    // namespace hpx::cds
