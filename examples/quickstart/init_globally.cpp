//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This example demonstrates several things:
//
// - How to initialize (and terminate) the HPX runtime from a global object
//   (see the type `manage_global_runtime' below)
// - How to register and unregister any (kernel) thread with the HPX runtime
// - How to launch an HPX thread from any (registered) kernel thread and
//   how to wait for the HPX thread to run to completion before continuing.
//   Any value returned from the HPX thread will be marshalled back to the
//   calling (kernel) thread.
//

#include <hpx/hpx.hpp>
#include <hpx/hpx_start.hpp>

#include <cstdlib>
#include <type_traits>
#include <thread>
#include <mutex>
#include <chrono>

///////////////////////////////////////////////////////////////////////////////
#if defined(linux) || defined(__linux) || defined(__linux__)

int __argc = 0;
char** __argv = 0;

void set_argv_argv(INT argc, char* argv[], char* env[])
{
    __argc = argc;
    __argv = argv;
}

__attribute__((section(".init_array")))
    void (*set_global_argc_argv)(int, char*[], char*[]) = &set_argv_argv;

#elif defined(__APPLE__)

#include <crt_externs.h>

inline int get_arraylen(char** argv)
{
    int count = 0;
    if (NULL != argv)
    {
        while(NULL != argv[count])
            ++count;
    }
    return count;
}

int __argc = get_arraylen(*_NSGetArgv());
char** __argv = *_NSGetArgv();

#endif

///////////////////////////////////////////////////////////////////////////////
struct manage_global_runtime
{
    manage_global_runtime()
      : rts_(0), running_(false)
    {
#if defined(HPX_WINDOWS)
        hpx::detail::init_winsocket();
#endif

        using hpx::util::placeholders::_1;
        using hpx::util::placeholders::_2;

        auto start_function =
            hpx::util::bind(&manage_global_runtime::hpx_main, this, _1, _2);

        if (!hpx::start(start_function, __argc, __argv, hpx::runtime_mode_console))
        {
            // Something went wrong while initializing the runtime.
            // This early we can't generate any output, just bail out.
            std::abort();
        }

        // wait for the main HPX thread to have started running
        std::unique_lock<std::mutex> lk(startup_mtx_);
        while (!running_)
            startup_cond_.wait(lk);
    }

    ~manage_global_runtime()
    {
        // notify hpx_main above to tear down the runtime
        {
            std::lock_guard<hpx::lcos::local::spinlock> lk(mtx_);
            rts_ = 0;               // reset pointer
            cond_.notify_one();     // signal exit
        }

        // wait for the runtime to exit
        hpx::stop();
    }

    // registration of external (to HPX) threads
    void register_thread(char const* name)
    {
        rts_->register_thread(name);
    }
    void unregister_thread()
    {
        rts_->unregister_thread();
    }

protected:
    // Main HPX thread, does nothing but wait for the application to exit
    int hpx_main(int argc, char* argv[])
    {
        // store a pointer to the runtime here
        rts_ = hpx::get_runtime_ptr();

        // signal to constructor that thread has started running
        {
            std::lock_guard<std::mutex> lk(startup_mtx_);
            running_ = true;
            startup_cond_.notify_one();
        }

        // now, wait for destructor to be called
        {
            std::unique_lock<hpx::lcos::local::spinlock> lk(mtx_);
            if (rts_ != 0)
                cond_.wait(lk);
        }

        // tell the runtime it's ok to exit
        return hpx::finalize();
    }

private:
    hpx::lcos::local::spinlock mtx_;
    hpx::lcos::local::condition_variable cond_;

    std::mutex startup_mtx_;
    std::condition_variable startup_cond_;
    bool running_;

    hpx::runtime* rts_;
};

manage_global_runtime init;

///////////////////////////////////////////////////////////////////////////////
struct thread_registration_wrapper
{
    thread_registration_wrapper(char const* name)
    {
        // Register this thread with HPX, this should be done once for
        // each external OS-thread intended to invoke HPX functionality.
        // Calling this function more than once will silently fail (will
        // return false).
        init.register_thread(name);
    }
    ~thread_registration_wrapper()
    {
        // Unregister the thread from HPX, this should be done once in the
        // end before the external thread exists.
        init.unregister_thread();
    }
};

///////////////////////////////////////////////////////////////////////////////
// These are helper functions which schedule an HPX thread that should run the
// given function and will wait for this HPX thread to exit before returning to
// the caller.

// overload for running functions which return a value
template <typename F, typename... Ts,
    typename Enable = typename std::enable_if<
        !std::is_void<typename std::result_of<F(Ts&&...)>::type>::value
    >::type>
typename std::result_of<F(Ts&&...)>::type
execute_hpx_thread(F const& f, Ts&&... ts)
{
    std::mutex mtx;
    std::condition_variable started_cond;
    bool running = false;

    std::condition_variable cond;

    typename std::result_of<F(Ts&&...)>::type retval;

    // Create an HPX thread
    hpx::threads::register_thread_nullary(
        // this lambda function will be scheduled to run as an HPX thread
        [&]()
        {
            // signal successful initialization
            {
                std::lock_guard<std::mutex> lk(mtx);
                running = true;
                started_cond.notify_all();
            }

            // execute the given function, forward all parameters, store result
            retval = f(std::forward<Ts>(ts)...);

            // now signal to the waiting thread that we're done
            cond.notify_one();
        });

    {
        // first wait for the HPX thread to have started running
        std::unique_lock<std::mutex> lk(mtx);
        while (!running)
            started_cond.wait(lk);

        // wait for the HPX thread to exit
        cond.wait(lk);
    }

    return retval;
}

// overload for running functions which return void
template <typename F, typename... Ts,
    typename Enable = typename std::enable_if<
        std::is_void<typename std::result_of<F(Ts&&...)>::type>::value
    >::type>
void execute_hpx_thread(F const& f, Ts&&... ts)
{
    std::mutex mtx;
    std::condition_variable started_cond;
    bool running = false;

    std::condition_variable cond;

    // Create an HPX thread
    hpx::threads::register_thread_nullary(
        // this lambda function will be scheduled to run as an HPX thread
        [&]()
        {
            // signal successful initialization
            {
                std::lock_guard<std::mutex> lk(mtx);
                running = true;
                started_cond.notify_all();
            }

            // execute the given function, forward all parameters
            f(std::forward<Ts>(ts)...);

            // now signal to the waiting thread that we're done
            cond.notify_one();
        });

    // first wait for the HPX thread to have started running
    std::unique_lock<std::mutex> lk(mtx);
    while (!running)
        started_cond.wait(lk);

    // wait for the HPX thread to exit
    cond.wait(lk);
}

///////////////////////////////////////////////////////////////////////////////
// These functions will be executed as an HPX thread
void hpx_thread_func1()
{
    // All of the HPX functionality is available here, including hpx::async,
    // hpx::future, and friends.

    // As an example, just sleep for one second.
    hpx::this_thread::sleep_for(std::chrono::seconds(1));
}

int hpx_thread_func2(int arg)
{
    // All of the HPX functionality is available here, including hpx::async,
    // hpx::future, and friends.

    // As an example, just sleep for one second.
    hpx::this_thread::sleep_for(std::chrono::seconds(1));

    // Simply return the argument.
    return arg;
}

///////////////////////////////////////////////////////////////////////////////
// This code will be executed by a system thread
void thread_func()
{
    // Register this (kernel) thread with the HPX runtime (unregister at exit).
    // Use a unique name for each of the created threads (could be derived from
    // std::this_thread::get_id()).
    thread_registration_wrapper register_thread("thread_func");

    // Now, a limited number of HPX API functions can be called.

    // Create an HPX thread (returning an int) and wait for it to run to
    // completion.
    int result = execute_hpx_thread(&hpx_thread_func2, 42);

    // Create an HPX thread (returning void) and wait for it to run to
    // completion.
    if (result == 42)
        execute_hpx_thread(&hpx_thread_func1);
}

///////////////////////////////////////////////////////////////////////////////
int main()
{
    // start a new (kernel) thread
    std::thread t(&thread_func);

    // The main thread was automatically registered with the HPX runtime
    // no explicit registration for this thread is necessary.
    execute_hpx_thread(&hpx_thread_func1);

    // wait for the (kernel) thread to run to completion
    t.join();

    return 0;
}


