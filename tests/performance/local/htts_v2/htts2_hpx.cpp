//  Copyright (c) 2011-2014 Bryce Adelstein-Lelbach
//  Copyright (c) 2007-2014 Hartmut Kaiser
//  Copyright (c) 2013-2014 Thomas Heller
//  Copyright (c) 2013-2014 Patricia Grubel
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/lcos/local/barrier.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/format.hpp>

#include "htts2.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

template <typename BaseClock = std::chrono::steady_clock>
struct hpx_driver : htts2::driver
{
    hpx_driver(int argc, char** argv)
      : htts2::driver(argc, argv, true)
//      , count_(0)
    {}

    void run()
    {
        std::vector<std::string> const cfg = {
            "hpx.os_threads=" + std::to_string(osthreads_),
            "hpx.run_hpx_main!=0",
            "hpx.commandline.allow_unknown!=1"
        };

        hpx::util::function_nonser<int(boost::program_options::variables_map& vm)> f;
        boost::program_options::options_description desc;

        using hpx::util::placeholders::_1;
        hpx::init(hpx::util::bind(&hpx_driver::run_impl, std::ref(*this), _1),
            desc, argc_, argv_, cfg);
    }

  private:
    int run_impl(boost::program_options::variables_map&)
    {
        // Cold run
        //kernel();

        // Hot run
        results_type results = kernel();
        print_results(results);

        return hpx::finalize();
    }

    hpx::threads::thread_result_type payload_thread_function(
        hpx::threads::thread_state_ex_enum ex = hpx::threads::wait_signaled
        )
    {
        htts2::payload<BaseClock>(this->payload_duration_ /* = p */);
        //++count_;
        return hpx::threads::thread_result_type(hpx::threads::terminated,
            hpx::threads::invalid_thread_id);
    }

    void stage_tasks(std::uint64_t target_osthread)
    {
        std::uint64_t const this_osthread = hpx::get_worker_thread_num();

        // This branch is very rarely taken (I've measured); this only occurs
        // if we are unlucky enough to be stolen from our intended queue.
        if (this_osthread != target_osthread)
        {
            // Reschedule in an attempt to correct.
            hpx::threads::register_work(
                hpx::util::bind(&hpx_driver::stage_tasks,
                    std::ref(*this), target_osthread)
              , nullptr // No HPX-thread name.
              , hpx::threads::pending
              , hpx::threads::thread_priority_normal
              , target_osthread // Place in the target OS-thread's queue.
                );
        }

        for (std::uint64_t i = 0; i < this->tasks_; ++i)
        {
            using hpx::util::placeholders::_1;
            hpx::threads::register_thread_plain(
                hpx::util::bind(&hpx_driver::payload_thread_function,
                    std::ref(*this), _1)
              , nullptr // No HPX-thread name.
              , hpx::threads::pending
              , false // Do not run immediately.
              , hpx::threads::thread_priority_normal
              , target_osthread // Place in the target OS-thread's queue.
                );
        }
    }

    void wait_for_tasks(hpx::lcos::local::barrier& finished)
    {
        std::uint64_t const pending_count =
            get_thread_count(hpx::threads::thread_priority_normal
                           , hpx::threads::pending);

        if (pending_count == 0)
        {
            std::uint64_t const all_count =
                get_thread_count(hpx::threads::thread_priority_normal);

            if (all_count != 1)
            {
                register_work(
                        hpx::util::bind(&hpx_driver::wait_for_tasks
                                  , std::ref(*this)
                                  , std::ref(finished)
                                   )
                      , nullptr, hpx::threads::pending
                      , hpx::threads::thread_priority_low);
                return;
            }
        }

        finished.wait();
    }

    typedef double results_type;

    results_type kernel()
    {
        ///////////////////////////////////////////////////////////////////////

        //count_ = 0;

        results_type results;

        std::uint64_t const this_osthread = hpx::get_worker_thread_num();

        htts2::timer<BaseClock> t;

        ///////////////////////////////////////////////////////////////////////
        // Warmup Phase
        for (std::uint64_t i = 0; i < this->osthreads_; ++i)
        {
            if (this_osthread == i) continue;

            hpx::threads::register_work(
                hpx::util::bind(&hpx_driver::stage_tasks, std::ref(*this), i)
              , nullptr // No HPX-thread name.
              , hpx::threads::pending
              , hpx::threads::thread_priority_normal
              , i // Place in the target OS-thread's queue.
                );
        }

        stage_tasks(this_osthread);

        ///////////////////////////////////////////////////////////////////////
        // Compute + Cooldown Phase

        // The use of an atomic and live waiting here does not add any noticable
        // overhead, as compared to the more complicated continuation-style
        // detection method that checks the threadmanager internal counters
        // (I've measured). Using this technique is preferable as it is more
        // comparable to the other implementations (especially qthreads).
//        do {
//            hpx::this_thread::suspend();
//        } while (count_ < (this->tasks_ * this->osthreads_));

        // Schedule a low-priority thread; when it is executed, it checks to
        // make sure all the tasks (which are normal priority) have been
        // executed, and then it
        hpx::lcos::local::barrier finished(2);

        register_work(hpx::util::bind(&hpx_driver::wait_for_tasks
                                , std::ref(*this)
                                , std::ref(finished)
                                 )
            , nullptr, hpx::threads::pending
            , hpx::threads::thread_priority_low);

        finished.wait();

        // w_M [nanoseconds]
        results = static_cast<double>(t.elapsed());

        return results;

        ///////////////////////////////////////////////////////////////////////
    }

    void print_results(results_type results) const
    {
        if (this->io_ == htts2::csv_with_headers)
            std::cout
                << "OS-threads (Independent Variable),"
                << "Tasks per OS-thread (Control Variable) [tasks/OS-threads],"
                << "Payload Duration (Control Variable) [nanoseconds],"
                << "Total Walltime [nanoseconds]"
                << "\n";

        hpx::util::format_to(std::cout, "{},{},{},{:.14g}\n",
            this->osthreads_,
            this->tasks_,
            this->payload_duration_,
            results
        );
    }

//    std::atomic<std::uint64_t> count_;
};

int main(int argc, char** argv)
{
    hpx_driver<> d(argc, argv);

    d.run();

    return 0;
}
