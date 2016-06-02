//  Copyright (c) 2012-2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/thread_executors.hpp>
#include <hpx/lcos/future_wait.hpp>
#include <hpx/util/bind.hpp>

#include <vector>

using hpx::util::high_resolution_timer;

#include <QtGui/QApplication>
#include <QtGui/QWidget>
#include "widget.hpp"

double runner(double now)
{
    high_resolution_timer t(now);
    // do something cool here
    return t.elapsed();
}

HPX_PLAIN_ACTION(runner, runner_action)

void run(widget * w, std::size_t num_threads)
{
    std::vector<hpx::lcos::future<double> > futures(num_threads);

    for(std::size_t i = 0; i < num_threads; ++i)
    {
        runner_action a;
        futures[i] = hpx::async(a, hpx::find_here(), high_resolution_timer::now());
    }

    hpx::lcos::wait(futures, [w](std::size_t i, double t){ w->add_label(i, t); });
    w->run_finished();
}

void qt_main(int argc, char ** argv)
{
    QApplication app(argc, argv);
	QApplication 
    using hpx::util::placeholders::_1;
    using hpx::util::placeholders::_2;
    widget main(hpx::util::bind(run, _1, _2));
    main.show();

    app.exec();
}

int hpx_main(int argc, char ** argv)
{
    {
        // Get a reference to one of the main thread
        hpx::threads::executors::main_pool_executor scheduler;
        // run an async function on the main thread to start the Qt application
        hpx::future<void> qt_application
            = hpx::async(scheduler, qt_main, argc, argv);

        // do something else while qt is executing in the background ...

        qt_application.wait();
    }
    return hpx::finalize();
}

int main(int argc, char **argv)
{
    return hpx::init(argc, argv);
}
