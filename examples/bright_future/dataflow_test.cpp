
//  Copyright (c) 2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/iostreams.hpp>

#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>
#include <hpx/runtime/components/client_base.hpp>
#include <hpx/util/high_resolution_timer.hpp>

#include <boost/lockfree/fifo.hpp>

#include <examples/bright_future/dataflow/dataflow.hpp>

using hpx::cout;
using hpx::flush;

using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;

using hpx::init;
using hpx::finalize;
using hpx::find_here;
using hpx::lcos::promise;
using hpx::lcos::dataflow;
using hpx::lcos::dataflow_base;
using hpx::util::high_resolution_timer;


using hpx::actions::plain_action0;
using hpx::actions::plain_result_action0;
using hpx::actions::plain_result_action1;
using hpx::actions::plain_result_action2;

int f(int i, int j)
{
    cout << "f\n" << flush;
    return i + j;
}
typedef plain_result_action2<int, int, int, &f> f_action;

HPX_REGISTER_PLAIN_ACTION(f_action);

int h(int i, int j)
{
    cout << "h\n" << flush;
    return i + j;
}
typedef plain_result_action2<int, int, int, &h> h_action;

HPX_REGISTER_PLAIN_ACTION(h_action);

int g()
{
    static int i = 9000;
    cout << "g\n" << flush;
    return ++i;
}
typedef plain_result_action0<int, &g> g_action;

HPX_REGISTER_PLAIN_ACTION(g_action);

void trigger()
{
    cout << "trigger!\n" << flush;
}

typedef plain_action0<&trigger> trigger_action;

HPX_REGISTER_PLAIN_ACTION(trigger_action);

boost::uint64_t id(boost::uint64_t i)
{
    return i;
}

typedef plain_result_action1<boost::uint64_t, boost::uint64_t, &id> id_action;

HPX_REGISTER_PLAIN_ACTION(id_action);

boost::uint64_t add(boost::uint64_t n1, boost::uint64_t n2)
{
    return n1 + n2;
}
typedef plain_result_action2<boost::uint64_t, boost::uint64_t, boost::uint64_t, &add> add_action;
HPX_REGISTER_PLAIN_ACTION(add_action);

dataflow<add_action> fib(boost::uint64_t n)
{
    if(n<2) return dataflow<add_action>(find_here(), n, 0);

    return dataflow<add_action>(find_here(), fib(n-1), fib(n-2));
}

#define M0(Z, N, D)\
void BOOST_PP_CAT(f, N)()\
{\
    cout << BOOST_PP_STRINGIZE(BOOST_PP_CAT(f, N)) << "\n" << flush;\
}\
typedef plain_action0<&BOOST_PP_CAT(f,N)> BOOST_PP_CAT(BOOST_PP_CAT(f, N), action);\
HPX_REGISTER_PLAIN_ACTION(BOOST_PP_CAT(BOOST_PP_CAT(f, N), action));

BOOST_PP_REPEAT(10, M0, _)


int hpx_main(variables_map & vm)
{
    {
        boost::uint64_t n = vm["n"].as<boost::uint64_t>();

        dataflow<g_action> a(find_here());
        dataflow<g_action> b(find_here());
        dataflow<f_action> c(find_here(), a, b);

        // blocks until the result is delivered! (constructs a promise and sets
        // this as the target of the dataflow)
        cout << c.get() << "\n" << flush;

        high_resolution_timer t;
        boost::uint64_t r = fib(n).get();
        double time = t.elapsed();
        cout << "fib(" << n << ") = " << r << " calculated in " << time << " seconds\n" << flush;

        cout << dataflow<h_action>(find_here(), a, 4).get() << "\n" << flush;
        cout << dataflow<h_action>(find_here(), 5, b).get() << "\n" << flush;
        cout
            << dataflow<g_action>(
                find_here()
              , dataflow<trigger_action>(
                    find_here()
                )
            ).get()
            << "\n" << flush;

        cout << "entering destruction test scope\n" << flush;
        {
            dataflow<f_action>(
                find_here()
              , dataflow<g_action>(find_here())
              , dataflow<g_action>(find_here())
            ).get();

            dataflow<f1action>(
                find_here()
              , dataflow<f2action>(
                    find_here()
                  , dataflow<f3action>(
                        find_here()
                      , dataflow<f4action>(
                            find_here()
                        )
                    )
                )
            ).get();
        }
        cout << "leaving destruction test scope\n" << flush;
    }
    cout << "end of hpx_main\n" << flush;
    finalize();

    return 0;
}

int main(int argc, char ** argv)
{
    options_description
        cmdline("usage: " HPX_APPLICATION_STRING " [options]");

    cmdline.add_options()
        ( "n" , value<boost::uint64_t>()->default_value(10),
            "n value for the Fibonacci function")
        ;

    return init(cmdline, argc, argv);
}
