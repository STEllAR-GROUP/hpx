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
#include <cds/gc/details/hp_common.h>
#include <cds/gc/hp.h>
#include <cds/init.h>
#include <cds/threading/details/cxx11_manager.h>
#include <cds/urcu/general_buffered.h>

#include <atomic>
#include <cstddef>
#include <iostream>
#include <string>

/// \cond NODETAIL
namespace cds { namespace gc { namespace hp { namespace details {

    class HPXDataHolder
    {
    public:
        CDS_EXPORT_API static CDS_EXPORT_API thread_data* getTLS();
        CDS_EXPORT_API static CDS_EXPORT_API void setTLS(thread_data*);
        CDS_EXPORT_API static CDS_EXPORT_API generic_smr<HPXDataHolder>*
        getInstance();
        CDS_EXPORT_API static CDS_EXPORT_API void setInstance(
            generic_smr<HPXDataHolder>* new_instance);
    };

}}}}    // namespace cds::gc::hp::details

namespace hpx { namespace cds {

    namespace detail {
        /// \cond NOINTERNAL
        using get_num_concurrent_hazard_pointer_threads_type =
            hpx::util::function_nonser<std::size_t()>;
        HPX_EXPORT void set_get_num_concurrent_hazard_pointer_threads(
            get_num_concurrent_hazard_pointer_threads_type f);
        HPX_EXPORT std::size_t get_num_concurrent_hazard_pointer_threads();
        /// \endcond
    }    // namespace detail

    enum class smr_t
    {
        hazard_pointer_hpxthread,
        hazard_pointer_stdthread,
        rcu
    };

    // this wrapper will initialize libCDS
    struct libcds_wrapper
    {
        friend struct hpxthread_manager_wrapper;
        friend struct stdthread_manager_wrapper;

        libcds_wrapper(smr_t smr_type = smr_t::hazard_pointer_hpxthread,
            std::size_t hazard_pointer_count = 1,
            std::size_t max_thread_count =
                detail::get_num_concurrent_hazard_pointer_threads(),
            std::size_t max_retired_pointer_count = 16)
          : smr_(smr_type)
        {
            // Initialize libcds
            ::cds::Initialize();

            // hazard_pointer_count, max_thread_count, max_retired_pointer_count
            // are only used in hazard pointer

            // hazard_pointer_count is corresponding var nHazardPtrCount
            // in libcds that defines Hazard pointer count per thread;

            // max_concurrent_attach_thread_ is corresponding var nMaxThreadCount
            // in libcds that defines Max count of simultaneous working thread
            // in your application

            // max_retired_pointer_count is corresponding var
            // nMaxRetiredPtrCount in libcds that defines Capacity
            // of the array of retired objects for the thread

            switch (smr_type)
            {
            default:
            case smr_t::hazard_pointer_hpxthread:
                ::cds::gc::hp::custom_smr<::cds::gc::hp::details::
                        HPXDataHolder>::construct(hazard_pointer_count,
                    max_thread_count, max_retired_pointer_count);
                break;
            case smr_t::hazard_pointer_stdthread:
                ::cds::gc::hp::custom_smr<::cds::gc::hp::details::
                        DefaultDataHolder>::construct(hazard_pointer_count,
                    max_thread_count, max_retired_pointer_count);
                break;
            case smr_t::rcu:
                typedef ::cds::urcu::gc<::cds::urcu::general_buffered<>>
                    rcu_gpb;
                // Initialize general_buffered RCU
                ::cds::urcu::general_buffered<>::Construct(256);
                ::cds::threading::Manager::attachThread();
                break;
            }
        }

        ~libcds_wrapper()
        {
            // Terminate libcds
            ::cds::Terminate();
            switch (smr_)
            {
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

    private:
        smr_t smr_;
    };

    // this wrapper will initialize an HPX thread/task for use with libCDS
    // algorithms
    struct hpxthread_manager_wrapper
    {
        // allow the std thread wrapper to use the same counter because
        // the libCDS backend does not distinguish between them yet.
        friend struct stdthread_manager_wrapper;

        // the boolean uselibcs option is provided to make comparison
        // of certain tests with/without libcds easier
        // @TODO : we should remove it one day
        explicit hpxthread_manager_wrapper(bool uselibcds = true)
          : uselibcds_(uselibcds)
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

        ~hpxthread_manager_wrapper()
        {
            if (uselibcds_)
            {
                if (thread_counter_-- == 0)
                {
                    HPX_THROW_EXCEPTION(invalid_status,
                        "hpx::cds::hpxthread_manager_wrapper",
                        "detaching more threads than number of attached "
                        "threads");
                }
                ::cds::gc::hp::custom_smr<
                    ::cds::gc::hp::details::HPXDataHolder>::detach_thread();
            }
        }

        // max_concurrent_attach_thread is corresponding variable to
        // the variable nMaxThreadCount in Hazard Pointer class. It defines
        // max count of simultaneous working thread in the application, default 100
        // and it is public to user for use

    private:
        static std::atomic<std::size_t> thread_counter_;
        bool uselibcds_;
    };

    // this wrapper will initialize a std::thread for use with libCDS
    // algorithms
    struct stdthread_manager_wrapper
    {
        explicit stdthread_manager_wrapper()
        {
            if (++hpxthread_manager_wrapper::thread_counter_ >
                detail::get_num_concurrent_hazard_pointer_threads())
            {
                HPX_THROW_EXCEPTION(invalid_status,
                    "hpx::cds::stdthread_manager_wrapper ",
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
                HPX_THROW_EXCEPTION(invalid_status,
                    "hpx::cds::stdthread_manager_wrapper ",
                    "failed to attach_thread to DefaultDataHolder, please check"
                    "if hazard pointer is constructed.");
            }
        }

        ~stdthread_manager_wrapper()
        {
            if (hpxthread_manager_wrapper::thread_counter_-- == 0)
            {
                HPX_THROW_EXCEPTION(invalid_status,
                    "hpx::cds::stdthread_manager_wrapper",
                    "detaching more threads than number of attached "
                    "threads");
            }
            ::cds::gc::hp::custom_smr<
                ::cds::gc::hp::details::DefaultDataHolder>::detach_thread();
        }
    };

}}    // namespace hpx::cds
