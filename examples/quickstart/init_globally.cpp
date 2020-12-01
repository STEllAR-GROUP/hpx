//  Copyright (c) 2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This example demonstrates several things:
//
// - How to initialize (and terminate) the HPX runtime from a global object
//   (see the type `manage_global_runtime' below)
// - How to register and unregister any (kernel) thread with the HPX runtime
// - How to launch an HPX thread from any (registered) kernel thread and
//   how to wait for the HPX thread to run to completion before continuing.
//   Any value returned from the HPX thread will be marshaled back to the
//   calling (kernel) thread.
//
// This scheme is generally useful if HPX should be initialized from a shared
// library and the main executable might not even be aware of this.

#include <hpx/hpx.hpp>
#include <hpx/include/run_as.hpp>
#include <hpx/hpx_start.hpp>

#include <mutex>
#include <string>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
// Store the command line arguments in global variables to make them available
// to the startup code.

#if defined(linux) || defined(__linux) || defined(__linux__)

int __argc = 0;
char** __argv = nullptr;

void set_argc_argv(int argc, char* argv[], char*[])
{
    __argc = argc;
    __argv = argv;
}

__attribute__((section(".preinit_array")))
    void (*set_global_argc_argv)(int, char*[], char*[]) = &set_argc_argv;

#elif defined(__APPLE__)

#include <crt_externs.h>

inline int get_arraylen(char** argv)
{
    int count = 0;
    if (nullptr != argv)
    {
        while(nullptr != argv[count])
            ++count;
    }
    return count;
}

int __argc = get_arraylen(*_NSGetArgv());
char** __argv = *_NSGetArgv();

#endif

///////////////////////////////////////////////////////////////////////////////
// This class demonstrates how to initialize a console instance of HPX
// (locality 0). In order to create an HPX instance which connects to a running
// HPX application two changes have to be made:
//
//  - replace hpx::runtime_mode::default_ with hpx::runtime_mode::connect
//  - replace hpx::finalize() with hpx::disconnect()
//
// Note that the mode runtime_mode::default_ corresponds to runtime_mode::console
// if the distributed runtime is enabled, and runtime_mode::local otherwise.
struct manage_global_runtime
{
    manage_global_runtime()
      : running_(false), rts_(nullptr)
    {
#if defined(HPX_WINDOWS)
        hpx::detail::init_winsocket();
#endif

        std::vector<std::string> const cfg = {
            // make sure hpx_main is always executed
            "hpx.run_hpx_main!=1",
            // allow for unknown command line options
            "hpx.commandline.allow_unknown!=1",
            // disable HPX' short options
            "hpx.commandline.aliasing!=0"
        };

        using hpx::util::placeholders::_1;
        using hpx::util::placeholders::_2;
        hpx::util::function_nonser<int(int, char**)> start_function =
            hpx::util::bind(&manage_global_runtime::hpx_main, this, _1, _2);
        hpx::init_params init_args;
        init_args.cfg = cfg;
        init_args.mode = hpx::runtime_mode::default_;

        if (!hpx::start(start_function, __argc, __argv, init_args))
        {
            // Something went wrong while initializing the runtime.
            // This early we can't generate any output, just bail out.
            std::abort();
        }

        // Wait for the main HPX thread (hpx_main below) to have started running
        std::unique_lock<std::mutex> lk(startup_mtx_);
        while (!running_)
            startup_cond_.wait(lk);
    }

    ~manage_global_runtime()
    {
        // notify hpx_main above to tear down the runtime
        {
            std::lock_guard<hpx::lcos::local::spinlock> lk(mtx_);
            rts_ = nullptr;               // reset pointer
        }

        cond_.notify_one();     // signal exit

        // wait for the runtime to exit
        hpx::stop();
    }

    // registration of external (to HPX) threads
    void register_thread(char const* name)
    {
        hpx::register_thread(rts_, name);
    }
    void unregister_thread()
    {
        hpx::unregister_thread(rts_);
    }

protected:
    // Main HPX thread, does nothing but wait for the application to exit
    int hpx_main(int, char*[])
    {
        // Store a pointer to the runtime here.
        rts_ = hpx::get_runtime_ptr();

        // Signal to constructor that thread has started running.
        {
            std::lock_guard<std::mutex> lk(startup_mtx_);
            running_ = true;
        }

        startup_cond_.notify_one();

        // Here other HPX specific functionality could be invoked...

        // Now, wait for destructor to be called.
        {
            std::unique_lock<hpx::lcos::local::spinlock> lk(mtx_);
            if (rts_ != nullptr)
                cond_.wait(lk);
        }

        // tell the runtime it's ok to exit
        return hpx::finalize();
    }

private:
    hpx::lcos::local::spinlock mtx_;
    hpx::lcos::local::condition_variable_any cond_;

    std::mutex startup_mtx_;
    std::condition_variable startup_cond_;
    bool running_;

    hpx::runtime* rts_;
};

// This global object will initialize HPX in its constructor and make sure HPX
// stops running in its destructor.
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
// These functions will be executed as HPX threads.
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
// This code will be executed by a system thread.
void thread_func()
{
    // Register this (kernel) thread with the HPX runtime (unregister at exit).
    // Use a unique name for each of the created threads (could be derived from
    // std::this_thread::get_id()).
    thread_registration_wrapper register_thread("thread_func");

    // Now, a limited number of HPX API functions can be called.

    // Create an HPX thread (returning an int) and wait for it to run to
    // completion.
    int result = hpx::threads::run_as_hpx_thread(&hpx_thread_func2, 42);

    // Create an HPX thread (returning void) and wait for it to run to
    // completion.
    if (result == 42)
        hpx::threads::run_as_hpx_thread(&hpx_thread_func1);
}

///////////////////////////////////////////////////////////////////////////////
int main()
{
    // Start a new (kernel) thread to demonstrate thread registration with HPX.
    std::thread t(&thread_func);

    // The main thread was automatically registered with the HPX runtime,
    // no explicit registration for this thread is necessary.
    hpx::threads::run_as_hpx_thread(&hpx_thread_func1);

    // wait for the (kernel) thread to run to completion
    t.join();

    return 0;
}
