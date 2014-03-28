//  Copyright (c) 2006, Giovanni P. Deretta
//  Copyright (c) 2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/include/util.hpp>
#include <hpx/runtime/threads/thread_data.hpp>

#include <iostream>
#include <iomanip>

#include <boost/function.hpp>

using namespace std;
using namespace hpx::threads;

using hpx::util::coroutines::coroutine;

///////////////////////////////////////////////////////////////////////////////
struct timer
{
    timer() : counter (gettimestamp()) {}

    double stop() { return gettimestamp() - counter; }

private:
    inline double gettimestamp()
    {
        return double(hpx::util::high_resolution_clock::now());//* 1e-9;
    }
    double counter;
};

///////////////////////////////////////////////////////////////////////////////
typedef boost::function<thread_function_type> function_test_type;
typedef hpx::threads::coroutine_type coroutine_test_type;

volatile int global_int = 0;

thread_state_enum foo(thread_state_ex_enum)
{
    global_int ^= 0xAAAA;
    return terminated;
}

// thread_state_enum ol_foo(thread_state_ex_enum)
// {
//     global_int ^= 0xAAAA;
//     return terminated;
// }

thread_state_enum foo_coro(thread_state_ex_enum s)
{
    while(true)
    {
        foo(s);
        s = get_self().yield(pending);
        if (s == wait_terminate)
            break;
    }
    return terminated;
}

struct foo_struct
{
    thread_state_enum operator()(thread_state_ex_enum s)
    {
        global_int ^= 0xAAAA;
        return terminated;
    }
};

struct foo_struct_coro
{
    typedef void result_type;

    thread_state_enum operator()(thread_state_ex_enum s)
    {
        while(true)
        {
            global_int ^= 0xAAAA;
            s = get_self().yield(pending);
            if (s == wait_terminate)
                break;
        }
        return terminated;
    }

    bool operator!() const { return true; }
    void clear() {}
};

template <typename F>
double test(F && f, int n)
{
    global_int = 5;
    timer t;
    while(n--)
        f(n != 0 ? wait_signaled : wait_terminate);
    return t.stop();
}

int main(int argc, char** argv)
{
    int const iterations = boost::lexical_cast<int>(argv[1]);

    foo_struct_coro t;
    coroutine_test_type coro_foo_struct(t, hpx::find_here());

    double foo_time = test(foo, iterations);
    double coro_time = test(coro_foo_struct, iterations);

    cout << "foo() time: " << foo_time/iterations << endl;
    cout << "coroutine() time: " << coro_time/iterations << endl;

   return 0;
}
