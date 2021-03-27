//  Copyright (c) 2020 Weile Wei
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/modules/functional.hpp>
//
#include <cds/gc/hp.h>

#include <atomic>
#include <cstddef>

#include <hpx/config/warnings_prefix.hpp>

/// \cond NODETAIL
namespace cds { namespace gc { namespace hp { namespace details {

    struct HPX_PARALLELISM_EXPORT HPXDataHolder
    {
        static thread_data* getTLS();
        static void setTLS(thread_data*);
        static generic_smr<HPXDataHolder>* getInstance();
        static void setInstance(generic_smr<HPXDataHolder>* new_instance);
    };
}}}}    // namespace cds::gc::hp::details

namespace hpx { namespace cds {

    ///////////////////////////////////////////////////////////////////////////
    // this wrapper will initialize an HPX thread/task for use with libCDS
    // algorithms
    struct HPX_PARALLELISM_EXPORT hpxthread_manager_wrapper
    {
        // allow the std thread wrapper to use the same counter because
        // the libCDS backend does not distinguish between them yet.
        friend struct stdthread_manager_wrapper;

        // the boolean uselibcs option is provided to make comparison
        // of certain tests with/without libcds easier
        // @TODO : we should remove it one day
        explicit hpxthread_manager_wrapper(bool uselibcds = true);

        ~hpxthread_manager_wrapper();

        // max_concurrent_attach_thread is corresponding variable to
        // the variable nMaxThreadCount in Hazard Pointer class. It defines
        // max count of simultaneous working thread in the application,
        // default 100 and it is public to user for use

    private:
        static std::atomic<std::size_t> thread_counter_;
        bool uselibcds_;
    };
}}    // namespace hpx::cds

#include <hpx/config/warnings_suffix.hpp>
