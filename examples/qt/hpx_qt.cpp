//  Copyright (c) 2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "hpx_qt.hpp"

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>

namespace hpx { namespace qt {
    
    struct runtime::impl
        : public QThread
    {
        impl(int argc, char ** argv, runtime * back, QObject * parent = 0)
            : QThread(parent)
              , argc(argc)
              , argv(argv)
              , back_(back)
        {}
        
        void run();
        
        hpx::lcos::promise<void> shutdown_promise;
        int argc;
        char ** argv;
        runtime * back_;
    };
}}

namespace {

    static inline hpx::threads::thread_state_enum thread_function_nullary(
        HPX_STD_FUNCTION<void()> const& func)
    {
        // execute the actual thread function
        func();

        // Verify that there are no more registered locks for this
        // OS-thread. This will throw if there are still any locks
        // held.
        hpx::util::force_error_on_lock();

        return hpx::threads::terminated;
    }

    static inline void shutdown_hpx_now(hpx::qt::runtime::impl * impl_)
    {
        impl_->shutdown_promise.set_value();
    }
}

namespace hpx { namespace qt {

    runtime::runtime(int argc, char ** argv, QObject * parent)
        : impl_(new impl(argc, argv, this, parent))
    {
        impl_->start();
    }

    runtime::~runtime()
    {
        apply(HPX_STD_BIND(::shutdown_hpx_now, impl_));
        impl_->wait();
        delete impl_;
    }


    hpx::threads::threadmanager_base * runtime::tm_ = 0;
    
    void runtime::qt_startup()
    {
        hpx::qt::runtime::tm_ = &hpx::threads::get_thread_manager();
        emit hpx_started();
    }

    void runtime::qt_shutdown(hpx::qt::runtime::impl * impl_)
    {
        impl_->shutdown_promise.get_future().get();
    }

    void runtime::impl::run()
    {
        using boost::program_options::options_description;
        options_description
            desc_commandline("usage: " HPX_APPLICATION_STRING " [options]");
        
        hpx::init(
            desc_commandline
          , argc
          , argv
          , HPX_STD_BIND(&runtime::qt_startup, back_)
          , HPX_STD_BIND(runtime::qt_shutdown, this)
          );
    }
            
    void runtime::apply(HPX_STD_FUNCTION<void()> const& f)
    {
        hpx::threads::thread_init_data data(
            HPX_STD_BIND(&::thread_function_nullary, f)
          , "widget_callback"
          , 0
          , hpx::threads::thread_priority_normal
          , std::size_t(-1)
          , 0x10000
        );
        tm_->register_thread(data, hpx::threads::pending, true);
    }
}}
