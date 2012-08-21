
//  Copyright (c) 2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>

#include <hpx/include/iostreams.hpp>

using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;

using hpx::util::high_resolution_timer;

using hpx::init;
using hpx::finalize;

using hpx::cout;
using hpx::flush;

#include <QtGui/QApplication>
#include "widget.hpp"
#include "hpx_qt.hpp"

double runner(double now)
{
    high_resolution_timer t(now);
    // TODO: do something cool here
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

    hpx::wait(futures, [w](std::size_t i, double t){ w->add_label(i, t); });
    w->run_finished();
}

int hpx_main(int argc, char ** argv)
{
    return finalize();
}

int main(int argc, char **argv)
{
    QApplication app(argc, argv);

    widget main(boost::bind(run, _1, _2));

    hpx::qt::runtime r(argc, argv);
    QObject::connect(&r, SIGNAL(hpx_started()), &main, SLOT(show()));

    app.exec();
}
