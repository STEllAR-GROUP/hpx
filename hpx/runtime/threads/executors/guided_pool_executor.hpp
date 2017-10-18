//  Copyright (c)      2017 John Biddiscombe
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_RUNTIME_THREADS_GUIDED_POOL_EXECUTOR
#define HPX_RUNTIME_THREADS_GUIDED_POOL_EXECUTOR

#include <hpx/async.hpp>
#include <hpx/runtime/threads/executors/pool_executor.hpp>
#include <hpx/runtime/threads/detail/thread_pool_base.hpp>
#include <hpx/util/thread_description.hpp>
#include <hpx/util/thread_specific_ptr.hpp>
#include <hpx/lcos/dataflow.hpp>
#include <hpx/util/invoke.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <iostream>

#include <hpx/config/warnings_prefix.hpp>

#include <typeinfo>

#ifdef __GNUG__
# include <cstdlib>
# include <cxxabi.h>
#endif

// ------------------------------------------------------------------
// helper to demangle type names
// ------------------------------------------------------------------
#ifdef __GNUG__
std::string demangle(const char* name)
{
    // some arbitrary value to eliminate the compiler warning
    int status = -4;
    std::unique_ptr<char, void(*)(void*)> res {
        abi::__cxa_demangle(name, NULL, NULL, &status),
                std::free
    };
    return (status==0) ? res.get() : name ;
}
#else
// does nothing if not g++
std::string demangle(const char* name) {
    return name;
}
#endif

// --------------------------------------------------------------------
// print type information
// --------------------------------------------------------------------
inline std::string print_type()
{
    return "\n";
}

template <typename T>
inline std::string print_type()
{
    return demangle(typeid(T).name());
}

template<typename T, typename... Args>
inline std::string print_type(T&& head, Args&&... tail)
{
    std::string temp = print_type<T>() + "\n\t";
    return temp + print_type(std::forward<Args>(tail)...);
}

// --------------------------------------------------------------------
// pool_numa_hint
// --------------------------------------------------------------------
namespace hpx { namespace threads { namespace executors
{
    struct bitmap_storage
    {
        struct tls_tag {};
        static hpx::util::thread_specific_ptr<hwloc_bitmap_ptr, tls_tag> bitmap_storage_;
    };

    // --------------------------------------------------------------------
    // Template type for a numa domain scheduling hint
    template <typename... Args>
    struct HPX_EXPORT pool_numa_hint {};

    // Template type for a core scheduling hint
    template <typename... Args>
    struct HPX_EXPORT pool_core_hint {};

    // --------------------------------------------------------------------
    struct async_helper_tag {};
    struct then_helper_tag {};

    template <typename Arg, typename Tag>
    struct async_helper {};

    template <typename Arg>
    struct async_helper<Arg, async_helper_tag> {
//        typedef std::function<int(Arg)> function_type;
        typedef int(*function_type)(Arg);
    };

    template <typename Arg>
    struct async_helper<Arg, then_helper_tag> {
        typedef int(*function_type)(hpx::future<Arg>);
    };

    // --------------------------------------------------------------------
    // helper : numa domain scheduling and then execution
    template <typename Executor, typename NumaFunction>
    struct pre_execution_async_domain_schedule
    {
        Executor     executor_;
        NumaFunction numa_function_;
        //
        template <typename F, typename ... Ts>
        auto operator()(F && f, Ts &&... ts) const
        {
            int domain = numa_function_(ts...);
            std::cout << "The numa domain is " << domain << "\n";

            std::cout << "Function : \n\t"  << print_type<F>();
            std::cout << "Arguments : \n\t" <<  print_type(ts...);

            // now we must forward the task on to the correct dispatch function
            typedef typename util::detail::invoke_deferred_result<F, Ts...>::type
                result_type;
            std::cout << "Result type : \n\t" << print_type<result_type>() << "\n";

            lcos::local::futures_factory<result_type()> p(
                const_cast<Executor&>(executor_),
                util::deferred_call(std::forward<F>(f), std::forward<Ts>(ts)...));

            p.apply(
                launch::async,
                threads::thread_priority_default,
                threads::thread_stacksize_default,
                threads::thread_schedule_hint(domain));

            return p.get_future();
        }
    };

    // --------------------------------------------------------------------
    // helper : numa domain scheduling and then execution
    template <typename Executor, typename NumaFunction>
    struct pre_execution_then_domain_schedule
    {
        Executor     executor_;
        NumaFunction numa_function_;
        //
        template <typename F, typename P, typename ... Ts>
        auto operator()(F && f, P && predecessor, Ts &&... ts) const
        {
            int domain = numa_function_(predecessor, ts...);
            std::cout << "The numa domain is " << domain << "\n";

            std::cout << "pre_execution_then_domain_schedule" << "\n";

            std::cout << "Function : \n\t" << print_type<F>() << "\n";
            std::cout << "Predecessor : \n\t" << print_type<P>() << "\n";
            std::cout << "Arguments : \n\t" << print_type(ts...) << "\n";


            // now we must forward the task on to the correct dispatch function
            typedef typename util::detail::invoke_deferred_result<F, hpx::future<P>, Ts...>::type
                result_type;
            std::cout << "Result type : \n\t" << print_type<result_type>() << "\n";

            auto fut = hpx::make_ready_future(std::move(predecessor));
            //
            lcos::local::futures_factory<result_type()> p(
                const_cast<Executor&>(executor_),
                util::deferred_call(std::forward<F>(f), std::move(fut), std::forward<Ts>(ts)...));

            std::cout << "setting up p \n";
            p.apply(
                launch::async,
                threads::thread_priority_default,
                threads::thread_stacksize_default,
                threads::thread_schedule_hint(domain));

            std::cout << "returning get_future \n";
            return p.get_future();
        }
    };

    // --------------------------------------------------------------------
    struct HPX_EXPORT guided_pool_executor_base {
    public:
        guided_pool_executor_base(const std::string& pool_name)
            : pool_executor_(pool_name)
        {}

        guided_pool_executor_base(const std::string& pool_name,
                             thread_stacksize stacksize)
            : pool_executor_(pool_name, stacksize)
        {}

        guided_pool_executor_base(const std::string& pool_name,
                             thread_priority priority,
                             thread_stacksize stacksize = thread_stacksize_default)
            : pool_executor_(pool_name, priority, stacksize)
        {}

    protected:
        pool_executor pool_executor_;
    };


    // --------------------------------------------------------------------
    template <typename... Args>
    struct HPX_EXPORT guided_pool_executor {};

    // --------------------------------------------------------------------
    // this is a guided pool executor templated over a function type
    // the function type should be the one used for async calls
    template <typename R, typename...Args>
    struct HPX_EXPORT guided_pool_executor<pool_numa_hint<R(*)(Args...)>>
        : guided_pool_executor_base
    {
    public:
        typedef guided_pool_executor<pool_numa_hint<R(*)(Args...)>> executor_type;
        using guided_pool_executor_base::guided_pool_executor_base;

        template <typename F, typename ... Ts>
        hpx::future<
            typename hpx::util::detail::invoke_deferred_result<F, Ts...>::type>
        async_execute(F && f, Ts &&... ts)
        {
            std::cout << "async_execute pool_numa_hint<R(*)(Args...)> : Function : \n\t" << print_type(f);
            std::cout << "async_execute pool_numa_hint<R(*)(Args...)> : Arguments : \n\t" << print_type(ts...);

typedef hpx::future<typename hpx::util::detail::invoke_deferred_result<F, Ts...>::type> res_type;
std::cout << "async_execute pool_numa_hint<R(*)(Args...)> : Result : \n\t" << print_type<res_type>() << "\n";
std::cout << "async_execute pool_numa_hint<R(*)(Args...)> : R : \n\t" << print_type<R>() << "\n";


            // hold onto the function until all futures have become ready
            // by using a dataflow operation, then call the scheduling hint
            // before passing the task onwards to the real executor
            return hpx::dataflow(
                util::unwrapping(
                    pre_execution_async_domain_schedule<pool_executor,
                        pool_numa_hint<R(*)(Args...)>>{
                            pool_executor_, hint_
                        }
                ),
                std::forward<F>(f), std::forward<Ts>(ts)...);
        };

    private:
        pool_numa_hint<R(*)(Args...)> hint_;
    };

    // --------------------------------------------------------------------
    // this is a guided pool executor templated over args only
    // the args should be the same as those that would be called
    // for an async function or continuation. This makes it possible to
    // guide a lambda rather than a full function.
    template <typename...Args>
    struct HPX_EXPORT guided_pool_executor<pool_numa_hint<Args...>>
        : guided_pool_executor_base
    {
    public:
        typedef guided_pool_executor<pool_numa_hint<Args...>> executor_type;
        using guided_pool_executor_base::guided_pool_executor_base;

        template <typename F, typename ... Ts>
        hpx::future<
            typename hpx::util::detail::invoke_deferred_result<F, Ts...>::type>
        async_execute(F && f, Ts &&... ts) const
        {
            std::cout << "async_execute pool_numa_hint<Args...> : Function : \n\t" << print_type<F>();
            std::cout << "async_execute pool_numa_hint<Args...> : Arguments : \n\t" << print_type(ts...);

            // hold onto the function until all futures have become ready
            // by using a dataflow operation, then call the scheduling hint
            // before passing the task onwards to the real executor
            return hpx::dataflow(
                util::unwrapping(
                    pre_execution_async_domain_schedule<pool_executor,
                        pool_numa_hint<Args...>> {
                            pool_executor_, hint_
                        }
                ),
                std::forward<F>(f), std::forward<Ts>(ts)...);
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename F, template <typename> typename Future, typename X, typename ... Ts>
        auto
        then_execute(F && f, Future<X> & predecessor, Ts &&... ts)
        ->  hpx::future<typename hpx::util::detail::invoke_deferred_result<
            F, Future<X>, Ts...>::type>
        {
            typedef typename hpx::util::detail::invoke_deferred_result<
                    F, Future<X>, Ts...>::type result_type;

            std::cout << "then_execute pool_numa_hint<Args...> : Function : \n\t" << print_type<F>() << "\n";
            std::cout << "then_execute pool_numa_hint<Args...> : Predecessor : \n\t" << print_type<Future<X>>() << "\n";;
            std::cout << "then_execute pool_numa_hint<Args...> : Future type : \n\t" << print_type<X>() << "\n";
            std::cout << "then_execute pool_numa_hint<Args...> : Arguments : \n\t" << print_type(ts...) << "\n";
            std::cout << "then_execute pool_numa_hint<Args...> : result_type : \n\t" << print_type<result_type>() << "\n";
            std::cout << "then_execute executor_type : executor_type : \n\t" << print_type<executor_type>() << "\n";

            // The Ts &&... args are not actually used in a continuation since only
            // the future becoming ready (predecessor) is actually passed on

            return hpx::dataflow(
                util::unwrapping(
                    [&](X && predecessor, Ts &&... ts)
                    {
                        std::cout << "then_execute dataflow : Received a value "
                                  << predecessor << std::endl;
                        //
                        pre_execution_then_domain_schedule<pool_executor,
                            pool_numa_hint<Args...>>
                            pre_exec { pool_executor_, hint_ };
                            std::cout << "then_execute Function : \n\t" << print_type<F>() << "\n";

                        return pre_exec.operator ()(std::forward<F>(f), std::forward<X>(predecessor));
                }),
                std::move(predecessor), std::forward<Ts>(ts)...);
        }

    private:
        pool_numa_hint<Args...> hint_;
    };

}}}

namespace hpx { namespace parallel { namespace execution
{
    template <typename Executor>
    struct executor_execution_category<
        threads::executors::guided_pool_executor<Executor> >
    {
        typedef parallel::execution::parallel_execution_tag type;
    };

    template <typename Executor>
    struct is_one_way_executor<
            threads::executors::guided_pool_executor<Executor> >
      : std::true_type
    {};

    template <typename Executor>
    struct is_two_way_executor<
            threads::executors::guided_pool_executor<Executor> >
      : std::true_type
    {};

    template <typename Executor>
    struct is_bulk_one_way_executor<
            threads::executors::guided_pool_executor<Executor> >
      : std::false_type
    {};

    template <typename Executor>
    struct is_bulk_two_way_executor<
            threads::executors::guided_pool_executor<Executor> >
      : std::false_type
    {};

}}}

#include <hpx/config/warnings_suffix.hpp>

#endif /*HPX_RUNTIME_THREADS_GUIDED_POOL_EXECUTOR*/
