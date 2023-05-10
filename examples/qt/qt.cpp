//  Copyright (c) 2012-2014 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/util.hpp>
#include <hpx/modules/async_combinators.hpp>

#include <cstddef>
#include <vector>

using hpx::chrono::high_resolution_timer;

#include <QtGui/QApplication>
#include "widget.hpp"

double runner(double now)
{
    high_resolution_timer t(now);
    // do something cool here
    return t.elapsed();
}

HPX_PLAIN_ACTION(runner, runner_action)

void run(widget* w, std::size_t num_threads)
{
    std::vector<hpx::future<double>> futures(num_threads);

    for (std::size_t i = 0; i < num_threads; ++i)
    {
        runner_action a;
        futures[i] =
            hpx::async(a, hpx::find_here(), high_resolution_timer::now());
    }

    hpx::wait_each(
        [w](std::size_t i, auto&& f) { w->threadsafe_add_label(i, f.get()); },
        futures);
    w->threadsafe_run_finished();
}

void qt_main(int argc, char** argv)
{
    QApplication app(argc, argv);

    using hpx::placeholders::_1;
    using hpx::placeholders::_2;
    widget main(hpx::bind(run, _1, _2));
    main.show();

    app.exec();
}

int hpx_main(int argc, char* argv[])
{
    {
        // Get a reference to one of the main thread
        hpx::parallel::execution::main_pool_executor scheduler;
        // run an async function on the main thread to start the Qt application
        hpx::future<void> qt_application =
            hpx::async(scheduler, qt_main, argc, argv);

        // do something else while qt is executing in the background ...

        qt_application.wait();
    }
    return hpx::finalize();
}

int main(int argc, char** argv)
{
    return hpx::init(argc, argv);
}
