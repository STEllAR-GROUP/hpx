//  Copyright (c) 2017-2018 John Biddiscombe
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/parallel_executors.hpp>
#include <hpx/include/threadmanager.hpp>
#define GUIDED_EXECUTOR_DEBUG 1
#include <hpx/futures/packaged_continuation.hpp>
#include <hpx/resource_partitioner/partitioner.hpp>
#include <hpx/thread_executors/guided_pool_executor.hpp>
//#include <hpx/topology/cpu_mask.hpp>
//#include <hpx/include/parallel_executors.hpp>
#include <hpx/modules/async_distributed.hpp>

// we should not need these
#include <hpx/thread_pools/scheduled_thread_pool_impl.hpp>

#include <hpx/async_combinators/when_all.hpp>
#include <hpx/async_distributed/dataflow.hpp>
#include <hpx/datastructures/tuple.hpp>
#include <hpx/debugging/demangle_helper.hpp>
#include <hpx/functional/deferred_call.hpp>
#include <hpx/functional/invoke.hpp>
#include <hpx/functional/invoke_fused.hpp>
#include <hpx/functional/invoke_result.hpp>
#include <hpx/pack_traversal/pack_traversal.hpp>
#include <hpx/type_support/decay.hpp>
//
#include <chrono>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>

#include <hpx/modules/testing.hpp>

// --------------------------------------------------------------------
// custom executor async/then/when/dataflow specialization example
// --------------------------------------------------------------------
using namespace hpx;

struct test_async_executor
{
    // --------------------------------------------------------------------
    // helper structs to make future<tuple<f1, f2, f3, ...>>>
    // detection of futures simpler
    // --------------------------------------------------------------------
    template <typename TupleOfFutures>
    struct is_tuple_of_futures;

    template <typename... Futures>
    struct is_tuple_of_futures<hpx::tuple<Futures...>>
      : util::all_of<
            traits::is_future<typename std::remove_reference<Futures>::type>...>
    {
    };

    template <typename Future>
    struct is_future_of_tuple_of_futures
      : std::integral_constant<bool,
            traits::is_future<Future>::value &&
                is_tuple_of_futures<typename traits::future_traits<
                    typename std::remove_reference<Future>::type>::
                        result_type>::value>
    {
    };

    // --------------------------------------------------------------------
    // For C++11 compatibility
    template <bool B, typename T = void>
    using enable_if_t = typename std::enable_if<B, T>::type;

    // --------------------------------------------------------------------
    // function that returns a const ref to the contents of a future
    // without calling .get() on the future so that we can use the value
    // and then pass the original future on to the intended destination.
    // --------------------------------------------------------------------
    struct future_extract_value
    {
        template <typename T, template <typename> class Future>
        const T& operator()(const Future<T>& el) const
        {
            typedef
                typename traits::detail::shared_state_ptr_for<Future<T>>::type
                    shared_state_ptr;
            shared_state_ptr const& state =
                traits::detail::get_shared_state(el);
            return *state->get_result();
        }
    };

    // --------------------------------------------------------------------
    // async execute specialized for simple arguments typical
    // of a normal async call with arbitrary arguments
    // --------------------------------------------------------------------
    template <typename F, typename... Ts>
    future<typename util::invoke_result<F, Ts...>::type> async_execute(
        F&& f, Ts&&... ts)
    {
        typedef typename util::detail::invoke_deferred_result<F, Ts...>::type
            result_type;

        using namespace hpx::util::debug;
        std::cout << "async_execute : Function    : " << print_type<F>()
                  << "\n";
        std::cout << "async_execute : Arguments   : "
                  << print_type<Ts...>(" | ") << "\n";
        std::cout << "async_execute : Result      : "
                  << print_type<result_type>() << "\n";

        // forward the task execution on to the real internal executor
        return hpx::parallel::execution::async_execute(executor_,
            util::annotated_function(std::forward<F>(f), "custom"),
            std::forward<Ts>(ts)...);
    }

    // --------------------------------------------------------------------
    // .then() execute specialized for a future<P> predecessor argument
    // note that future<> and shared_future<> are both supported
    // --------------------------------------------------------------------
    template <typename F, typename Future, typename... Ts,
        typename = enable_if_t<traits::is_future<
            typename std::remove_reference<Future>::type>::value>>
    auto then_execute(F&& f, Future&& predecessor, Ts&&... ts) -> future<
        typename util::detail::invoke_deferred_result<F, Future, Ts...>::type>
    {
        typedef typename util::detail::invoke_deferred_result<F, Future,
            Ts...>::type result_type;

        using namespace hpx::util::debug;
        std::cout << "then_execute : Function     : " << print_type<F>()
                  << "\n";
        std::cout << "then_execute : Predecessor  : " << print_type<Future>()
                  << "\n";
        std::cout
            << "then_execute : Future       : "
            << print_type<typename traits::future_traits<Future>::result_type>()
            << "\n";
        std::cout << "then_execute : Arguments    : "
                  << print_type<Ts...>(" | ") << "\n";
        std::cout << "then_execute : Result       : "
                  << print_type<result_type>() << "\n";

        return hpx::parallel::execution::then_execute(executor_,
            std::forward<F>(f), std::forward<Future>(predecessor),
            std::forward<Ts>(ts)...);
    }

    // --------------------------------------------------------------------
    // .then() execute specialized for a when_all dispatch for any future types
    // future< tuple< is_future<a>::type, is_future<b>::type, ...> >
    // --------------------------------------------------------------------
    template <typename F, template <typename> class OuterFuture,
        typename... InnerFutures, typename... Ts,
        typename = enable_if_t<is_future_of_tuple_of_futures<
            OuterFuture<hpx::tuple<InnerFutures...>>>::value>,
        typename = enable_if_t<
            is_tuple_of_futures<hpx::tuple<InnerFutures...>>::value>>
    auto then_execute(F&& f,
        OuterFuture<hpx::tuple<InnerFutures...>>&& predecessor, Ts&&... ts)
        -> future<typename util::detail::invoke_deferred_result<F,
            OuterFuture<hpx::tuple<InnerFutures...>>, Ts...>::type>
    {
        typedef typename util::detail::invoke_deferred_result<F,
            OuterFuture<hpx::tuple<InnerFutures...>>, Ts...>::type result_type;

        // get the tuple of futures from the predecessor future <tuple of futures>
        const auto& predecessor_value =
            future_extract_value().operator()(predecessor);

        // create a tuple of the unwrapped future values
        auto unwrapped_futures_tuple =
            util::map_pack(future_extract_value{}, predecessor_value);

        using namespace hpx::util::debug;
        std::cout << "when_all(fut) : Predecessor : "
                  << print_type<OuterFuture<hpx::tuple<InnerFutures...>>>()
                  << "\n";
        std::cout << "when_all(fut) : unwrapped   : "
                  << print_type<decltype(unwrapped_futures_tuple)>(" | ")
                  << "\n";
        std::cout << "when_all(fut) : Arguments   : "
                  << print_type<Ts...>(" | ") << "\n";
        std::cout << "when_all(fut) : Result      : "
                  << print_type<result_type>() << "\n";

        // invoke a function with the unwrapped tuple future types to demonstrate
        // that we can access them
        std::cout << "when_all(fut) : tuple       : ";
        util::invoke_fused(
            [](const auto&... ts) {
                std::cout << print_type<decltype(ts)...>(" | ") << "\n";
            },
            unwrapped_futures_tuple);

        // forward the task execution on to the real internal executor
        return hpx::parallel::execution::then_execute(executor_,
            util::annotated_function(std::forward<F>(f), "custom then"),
            std::forward<OuterFuture<hpx::tuple<InnerFutures...>>>(predecessor),
            std::forward<Ts>(ts)...);
    }

    // --------------------------------------------------------------------
    // execute specialized for a dataflow dispatch
    // dataflow unwraps the outer future for us but passes a dataflowframe
    // function type, result type and tuple of futures as arguments
    // --------------------------------------------------------------------
    template <typename F, typename... InnerFutures,
        typename = enable_if_t<
            traits::is_future_tuple<hpx::tuple<InnerFutures...>>::value>>
    auto async_execute(F&& f, hpx::tuple<InnerFutures...>&& predecessor)
        -> future<typename util::detail::invoke_deferred_result<F,
            hpx::tuple<InnerFutures...>>::type>
    {
        typedef typename util::detail::invoke_deferred_result<F,
            hpx::tuple<InnerFutures...>>::type result_type;

        auto unwrapped_futures_tuple =
            util::map_pack(future_extract_value{}, predecessor);

        using namespace hpx::util::debug;
        std::cout << "dataflow      : Predecessor : "
                  << print_type<hpx::tuple<InnerFutures...>>() << "\n";
        std::cout << "dataflow      : unwrapped   : "
                  << print_type<decltype(unwrapped_futures_tuple)>(" | ")
                  << "\n";
        std::cout << "dataflow-frame: Result      : "
                  << print_type<result_type>() << "\n";

        // invoke a function with the unwrapped tuple future types to demonstrate
        // that we can access them
        std::cout << "dataflow      : tuple       : ";
        util::invoke_fused(
            [](const auto&... ts) {
                std::cout << print_type<decltype(ts)...>(" | ") << "\n";
            },
            unwrapped_futures_tuple);

        // forward the task execution on to the real internal executor
        return hpx::parallel::execution::async_execute(executor_,
            util::annotated_function(std::forward<F>(f), "custom async"),
            std::forward<hpx::tuple<InnerFutures...>>(predecessor));
    }

private:
    parallel::execution::default_executor executor_;
};

// --------------------------------------------------------------------
// set traits for executor to say it is an async executor
// --------------------------------------------------------------------
namespace hpx { namespace parallel { namespace execution {
    template <>
    struct is_two_way_executor<test_async_executor> : std::true_type
    {
    };
}}}    // namespace hpx::parallel::execution

template <typename T>
T dummy_task(T val)
{
    // using std::thread here is intentional
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    return val;
}
// --------------------------------------------------------------------
// test various execution modes
// --------------------------------------------------------------------
template <typename Executor>
int test(const std::string& message, Executor& exec)
{
    // test 1
    std::cout << "============================" << std::endl;
    std::cout << message << std::endl;
    std::cout << "============================" << std::endl;
    std::cout << "Test 1 : async()" << std::endl;
    auto fa = async(
        exec,
        [](int a, double b, const char* c) {
            std::cout << "Inside async " << c << std::endl;
            HPX_TEST_EQ(a == 1 && b == 2.2 && std::string(c) == "Hello", true);
            return "async";
        },
        1, 2.2, "Hello");
    HPX_TEST_EQ(fa.get(), "async");
    std::cout << std::endl;

    // test 2a
    std::cout << "============================" << std::endl;
    std::cout << "Test 2a : .then()" << std::endl;
    int testval = 5;
    future<decltype(testval)> f =
        hpx::async(&dummy_task<decltype(testval)>, testval);
    //
    future<std::string> ft = f.then(exec, [testval](auto&& f) {
        std::cout << "Inside .then() " << std::endl;
        HPX_TEST_EQ_MSG(
            f.is_ready(), true, "Continuation run before future ready");
        decltype(testval) r = f.get();
        std::cout << "expected " << testval << " got " << r << std::endl;
        HPX_TEST_EQ(r, testval);
        return std::string("then");
    });
    HPX_TEST_EQ(ft.get(), "then");
    std::cout << std::endl;

    // test 2b
    std::cout << "============================" << std::endl;
    std::cout << "Test 2b : .then(shared)" << std::endl;
    auto fs = hpx::async(&dummy_task<decltype(testval)>, testval).share();
    //
    future<std::string> fts = fs.then(exec, [testval](auto&& f) {
        std::cout << "Inside .then(shared)" << std::endl;
        HPX_TEST_EQ_MSG(
            f.is_ready(), true, "Continuation run before future ready");
        decltype(testval) r = f.get();
        std::cout << "expected " << testval << " got " << r << std::endl;
        HPX_TEST_EQ(r, testval);
        return std::string("then(shared)");
    });
    HPX_TEST_EQ(fts.get(), "then(shared)");
    std::cout << std::endl;

    // test 3a
    std::cout << "============================" << std::endl;
    std::cout << "Test 3a : when_all()" << std::endl;
    int testval2 = 123;
    double testval3 = 4.567;
    auto fw1 = hpx::async(&dummy_task<decltype(testval2)>, testval2);
    auto fw2 = hpx::async(&dummy_task<decltype(testval3)>, testval3);
    //
    auto fw = when_all(fw1, fw2).then(exec,
        [testval2, testval3](
            future<hpx::tuple<future<int>, future<double>>>&& f) {
            std::cout << "Inside when_all : " << std::endl;
            HPX_TEST_EQ_MSG(
                f.is_ready(), true, "Continuation run before future ready");
            auto tup = f.get();
            auto cmplx = std::complex<double>(
                double(hpx::get<0>(tup).get()), hpx::get<1>(tup).get());
            auto cmplxe = std::complex<double>(double(testval2), testval3);
            std::cout << "expected " << cmplxe << " got " << cmplx << std::endl;
            HPX_TEST_EQ(cmplx, cmplxe);
            return std::string("when_all");
        });
    HPX_TEST_EQ(fw.get(), "when_all");
    std::cout << std::endl;

    // test 3b
    std::cout << "============================" << std::endl;
    std::cout << "Test 3b : when_all(shared)" << std::endl;
    std::uint64_t testval4 = 666;
    float testval5 = 876.5;
    auto fws1 = hpx::async(&dummy_task<decltype(testval4)>, testval4);
    auto fws2 = hpx::async(&dummy_task<decltype(testval5)>, testval5).share();
    //
    auto fws =
        when_all(fws1, fws2)
            .then(exec,
                [testval4, testval5](future<hpx::tuple<future<std::uint64_t>,
                        shared_future<float>>>&& f) {
                    std::cout << "Inside when_all(shared) : " << std::endl;
                    HPX_TEST_EQ_MSG(f.is_ready(), true,
                        "Continuation run before future ready");
                    auto tup = f.get();
                    auto cmplx =
                        std::complex<double>(double(hpx::get<0>(tup).get()),
                            double(hpx::get<1>(tup).get()));
                    auto cmplxe = std::complex<double>(
                        double(testval4), double(testval5));
                    std::cout << "expected " << cmplxe << " got " << cmplx
                              << std::endl;
                    HPX_TEST_EQ(cmplx, cmplxe);
                    return std::string("when_all(shared)");
                });
    HPX_TEST_EQ(fws.get(), "when_all(shared)");
    std::cout << std::endl;

    // test 4a
    std::cout << "============================" << std::endl;
    std::cout << "Test 4a : dataflow()" << std::endl;
    std::uint16_t testval6 = 333;
    double testval7 = 777.777;
    auto f1 = hpx::async(&dummy_task<decltype(testval6)>, testval6);
    auto f2 = hpx::async(&dummy_task<decltype(testval7)>, testval7);
    //
    auto fd = dataflow(
        exec,
        [testval6, testval7](future<std::uint16_t>&& f1, future<double>&& f2) {
            std::cout << "Inside dataflow : " << std::endl;
            HPX_TEST_EQ_MSG(f1.is_ready() && f2.is_ready(), true,
                "Continuation run before future ready");
            double r1 = f1.get();
            double r2 = f2.get();
            auto cmplx = std::complex<double>(r1, r2);
            auto cmplxe = std::complex<double>(double(testval6), testval7);
            std::cout << "expected " << cmplxe << " got " << cmplx << std::endl;
            HPX_TEST_EQ(cmplx, cmplxe);
            return std::string("dataflow");
        },
        f1, f2);
    HPX_TEST_EQ(fd.get(), "dataflow");
    std::cout << std::endl;

    // test 4b
    std::cout << "============================" << std::endl;
    std::cout << "Test 4b : dataflow(shared)" << std::endl;
    std::uint32_t testval8 = 987;
    double testval9 = 654.321;
    auto fs1 = hpx::async(&dummy_task<decltype(testval8)>, testval8);
    auto fs2 = hpx::async(&dummy_task<decltype(testval9)>, testval9);
    //
    auto fds = dataflow(
        exec,
        [testval8, testval9](
            future<std::uint32_t>&& f1, shared_future<double>&& f2) {
            std::cout << "Inside dataflow(shared) : " << std::endl;
            HPX_TEST_EQ_MSG(f1.is_ready() && f2.is_ready(), true,
                "Continuation run before future ready");
            double r1 = f1.get();
            double r2 = f2.get();
            auto cmplx = std::complex<double>(r1, r2);
            auto cmplxe = std::complex<double>(double(testval8), testval9);
            std::cout << "expected " << cmplxe << " got " << cmplx << std::endl;
            HPX_TEST_EQ(cmplx, cmplxe);
            return std::string("dataflow(shared)");
        },
        fs1, fs2);
    HPX_TEST_EQ(fds.get(), "dataflow(shared)");

    std::cout << "============================" << std::endl;
    std::cout << "Complete" << std::endl;
    std::cout << "============================" << std::endl << std::endl;
    return 0;
}

struct dummy_tag
{
};

namespace hpx { namespace parallel { namespace execution {
    template <>
    struct pool_numa_hint<dummy_tag>
    {
        int operator()() const
        {
            std::cout << "Hint 0 \n";
            return 0;
        }
        int operator()(const int, const double, const char*) const
        {
            std::cout << "Hint 1 \n";
            return 1;
        }
        int operator()(const int) const
        {
            std::cout << "Hint 2 \n";
            return 2;
        }
        int operator()(const hpx::tuple<future<int>, future<double>>&) const
        {
            std::cout << "Hint 3(a) \n";
            return 3;
        }
        int operator()(
            const hpx::tuple<future<std::uint64_t>, shared_future<float>>&)
            const
        {
            std::cout << "Hint 3(b) \n";
            return 3;
        }
        int operator()(const std::uint16_t, const double) const
        {
            std::cout << "Hint 4(a) \n";
            return 4;
        }
        int operator()(const std::uint32_t, const double&) const
        {
            std::cout << "Hint 4(b) \n";
            return 4;
        }
    };
}}}    // namespace hpx::parallel::execution

int hpx_main()
{
    bool ok = true;
    try
    {
        test_async_executor exec;
        test("Testing async custom executor", exec);
    }
    catch (std::exception& e)
    {
        std::cout << "Exception " << e.what() << std::endl;
        ok = false;
    }

    typedef hpx::parallel::execution::pool_numa_hint<dummy_tag> dummy_hint;
    try
    {
        hpx::parallel::execution::guided_pool_executor<dummy_hint> exec2(
            &hpx::resource::get_thread_pool("default"));
        test("Testing guided_pool_executor<dummy_hint>", exec2);
    }
    catch (std::exception& e)
    {
        std::cout << "Exception " << e.what() << std::endl;
        ok = false;
    }

    try
    {
        hpx::parallel::execution::guided_pool_executor_shim<dummy_hint> exec3(
            true, &hpx::resource::get_thread_pool("default"));
        test("Testing guided_pool_executor_shim<dummy_hint>", exec3);
    }
    catch (std::exception& e)
    {
        std::cout << "Exception " << e.what() << std::endl;
        ok = false;
    }

    std::cout << "Tests done \n";
    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    // Initialize and run HPX
    HPX_TEST_EQ_MSG(
        hpx::init(argc, argv), 0, "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
