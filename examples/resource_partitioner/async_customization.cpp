#include <hpx/hpx_main.hpp>
#include <hpx/include/parallel_executors.hpp>
#include <hpx/runtime/threads/executors/default_executor.hpp>
#include <hpx/async.hpp>

#include <hpx/lcos/dataflow.hpp>
#include <hpx/util/invoke.hpp>
#include <hpx/util/invoke_fused.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/result_of.hpp>
#include <hpx/util/tuple.hpp>
#include <hpx/util/deferred_call.hpp>
#include <hpx/util/pack_traversal.hpp>
#include <hpx/util/demangle_helper.hpp>

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

    template <typename...Futures>
    struct is_tuple_of_futures<util::tuple<Futures...>>
        : util::detail::all_of<traits::is_future<
            typename std::remove_reference<Futures>::type>...>
    {};

    template <typename Future>
    struct is_future_of_tuple_of_futures
        : std::integral_constant<bool,
            traits::is_future<Future>::value &&
            is_tuple_of_futures<
                typename traits::future_traits<
                    typename std::remove_reference<Future>::type
                >::result_type>::value>
    {};

    // --------------------------------------------------------------------
    // function that returns a const ref to the contents of a future
    // without calling .get() on the future so that we can use the value
    // and then pass the original future on to the intended destination.
    // --------------------------------------------------------------------
    struct future_extract_value
    {
        template<typename T, template <typename> typename Future>
        const T& operator()(const Future<T> &el) const
        {
            typedef typename traits::detail::shared_state_ptr_for<Future<T>>::type
                shared_state_ptr;
            shared_state_ptr const& state = traits::detail::get_shared_state(el);
            return *state->get_result();
        }
    };

    // --------------------------------------------------------------------
    // async execute specialized for simple arguments typical
    // of a normal async call with arbitrary arguments
    // --------------------------------------------------------------------
    template <typename F, typename ... Ts>
    future<typename util::invoke_result<F, Ts...>::type>
    async_execute(F && f, Ts &&... ts)
    {
        typedef typename util::detail::invoke_deferred_result<F, Ts...>::type
            result_type;

        std::cout << "async_execute : Function    : "
                  << debug::print_type<F>() << "\n";
        std::cout << "async_execute : Arguments   : "
                  << debug::print_type<Ts...>(" | ") << "\n";
        std::cout << "async_execute : Result      : "
                  << debug::print_type<result_type>() << "\n";

        // forward the task execution on to the real internal executor
        lcos::local::futures_factory<result_type()> p(
            executor_,
            util::deferred_call(std::forward<F>(f),
                                     std::forward<Ts>(ts)...)
        );

        p.apply(
            launch::async,
            threads::thread_priority_default,
            threads::thread_stacksize_default);

        return p.get_future();
    }

    // --------------------------------------------------------------------
    // .then() execute specialized for a future<P> predecessor argument
    // note that future<> and shared_future<> are both supported
    // --------------------------------------------------------------------
    template <typename F,
              typename Future,
              typename ... Ts,
              typename = typename std::enable_if_t<traits::is_future<
              typename std::remove_reference<Future>::type>::value>>
    auto
    then_execute(F && f, Future&& predecessor, Ts &&... ts)
    ->  future<typename util::detail::invoke_deferred_result<
        F, Future, Ts...>::type>
    {
        typedef typename util::detail::invoke_deferred_result<
                F, Future, Ts...>::type result_type;

        std::cout << "then_execute : Function     : "
                  << debug::print_type<F>() << "\n";
        std::cout << "then_execute : Predecessor  : "
                  << debug::print_type<Future>() << "\n";
        std::cout << "then_execute : Future       : "
                  << debug::print_type<typename
                     traits::future_traits<Future>::result_type>() << "\n";
        std::cout << "then_execute : Arguments    : "
                  << debug::print_type<Ts...>(" | ") << "\n";
        std::cout << "then_execute : Result       : "
                  << debug::print_type<result_type>() << "\n";

        // forward the task on to the 'real' underlying executor
        lcos::local::futures_factory<result_type()> p(
            executor_,
            util::deferred_call(std::forward<F>(f),
                                std::forward<Future>(predecessor),
                                std::forward<Ts>(ts)...)
        );

        p.apply(
            launch::async,
            threads::thread_priority_default,
            threads::thread_stacksize_default);

        return p.get_future();
    }

    // --------------------------------------------------------------------
    // .then() execute specialized for a when_all dispatch for any future types
    // future< tuple< is_future<a>::type, is_future<b>::type, ...> >
    // --------------------------------------------------------------------
    template <typename F,
              template <typename> typename  OuterFuture,
              typename ... InnerFutures,
              typename ... Ts,
              typename = typename std::enable_if_t<is_future_of_tuple_of_futures<
                OuterFuture<util::tuple<InnerFutures...>>>::value>,
              typename = typename std::enable_if_t<is_tuple_of_futures<
                util::tuple<InnerFutures...>>::value>
              >
    auto
    then_execute(F && f,
                 OuterFuture<util::tuple<InnerFutures... > >&& predecessor,
                 Ts &&... ts)
    ->  future<typename util::detail::invoke_deferred_result<
        F, OuterFuture<util::tuple<InnerFutures... >>, Ts...>::type>
    {
        typedef typename util::detail::invoke_deferred_result<
            F, OuterFuture<util::tuple<InnerFutures... >>, Ts...>::type
                result_type;

        // get the tuple of futures from the predecessor future <tuple of futures>
        const auto & predecessor_value = future_extract_value().operator()(predecessor);

        // create a tuple of the unwrapped future values
        auto unwrapped_futures_tuple = util::map_pack(
            future_extract_value{},
            predecessor_value
        );

        std::cout << "when_all(fut) : Predecessor : "
                  << debug::print_type<OuterFuture<util::tuple<InnerFutures...>>>()
                  << "\n";
        std::cout << "when_all(fut) : unwrapped   : "
                  << debug::print_type<decltype(unwrapped_futures_tuple)>(" | ") << "\n";
        std::cout << "when_all(fut) : Arguments   : "
                  << debug::print_type<Ts...>(" | ") << "\n";
        std::cout << "when_all(fut) : Result      : "
                  << debug::print_type<result_type>() << "\n";

        // invoke a function with the unwrapped tuple future types to demonstrate
        // that we can access them
        std::cout << "when_all(fut) : tuple       : ";
        util::invoke_fused([](const auto & ...ts) {
            std::cout << debug::print_type<decltype(ts)...>(" | ") << "\n";
        }, unwrapped_futures_tuple);

        // forward the task execution on to the real internal executor
        lcos::local::futures_factory<result_type()> p(
            executor_,
            util::deferred_call(
                std::forward<F>(f),
                std::forward<OuterFuture<util::tuple<InnerFutures...>>>(predecessor),
                std::forward<Ts>(ts)...)
        );

        p.apply(
            launch::async,
            threads::thread_priority_default,
            threads::thread_stacksize_default);

        return p.get_future();
    }

    // --------------------------------------------------------------------
    // .then() execute specialized for a dataflow dispatch
    // dataflow unwraps the outer future for us but passes a dataflowframe
    // function type, result type and tuple of futures as arguments
    // --------------------------------------------------------------------
    template <typename F,
              typename DataFlowFrame,
              typename Result,
              typename ... InnerFutures,
              typename = typename std::enable_if_t<
                  is_tuple_of_futures<util::tuple<InnerFutures...>>::value>
              >
    auto
    async_execute(F && f,
                  DataFlowFrame && df,
                  Result && r,
                  util::tuple<InnerFutures... > && predecessor)
    ->  future<typename util::detail::invoke_deferred_result<
        F, DataFlowFrame, Result, util::tuple<InnerFutures... >>::type>
    {
        typedef typename util::detail::invoke_deferred_result<
            F, DataFlowFrame, Result, util::tuple<InnerFutures... >>::type
                result_type;

        auto unwrapped_futures_tuple = util::map_pack(
            future_extract_value{},
            predecessor
        );

        std::cout << "dataflow      : Predecessor : "
                  << debug::print_type<util::tuple<InnerFutures...>>()
                  << "\n";
        std::cout << "dataflow      : unwrapped   : "
                  << debug::print_type<decltype(unwrapped_futures_tuple)>(" | ") << "\n";
        std::cout << "dataflow-frame: Result      : "
                  << debug::print_type<Result>() << "\n";

        // invoke a function with the unwrapped tuple future types to demonstrate
        // that we can access them
        std::cout << "dataflow      : tuple       : ";
        util::invoke_fused([](const auto & ...ts) {
            std::cout << debug::print_type<decltype(ts)...>(" | ") << "\n";
        }, unwrapped_futures_tuple);

        // forward the task execution on to the real internal executor
        lcos::local::futures_factory<result_type()> p(
            executor_,
            util::deferred_call(
                std::forward<F>(f),
                std::forward<DataFlowFrame>(df),
                std::forward<Result>(r),
                std::forward<util::tuple<InnerFutures...>>(predecessor)
            )
        );

        p.apply(
            launch::async,
            threads::thread_priority_default,
            threads::thread_stacksize_default);

        return p.get_future();
    }

private:
    threads::executors::default_executor executor_;
};

// --------------------------------------------------------------------
// set traits for executor to say it is an async executor
// --------------------------------------------------------------------
namespace hpx { namespace parallel { namespace execution
{
    template <>
    struct is_two_way_executor<test_async_executor>
      : std::true_type
    {};
}}}

// --------------------------------------------------------------------
// test various execution modes
// --------------------------------------------------------------------
int main()
{
    test_async_executor exec;

    // test 1
    std::cout << "============================" << std::endl;
    std::cout << "Test 1 : async()" << std::endl;
    auto fa = async(exec, [](int a, double b, const char *c)
    {
        std::cout << "Inside async " << c << std::endl;
        return 2.1415f;
    }, 1, 2.2, "Hello");
    fa.get();
    std::cout << std::endl;

    // test 2a
    std::cout << "============================" << std::endl;
    std::cout << "Test 2a : .then()" << std::endl;
    future<int> f = make_ready_future(5);
    //
    future<std::string> ft = f.then(exec,
        [](future<int> && f)
        {
            std::cout << "Inside .then()" << std::endl;
            return std::string("then");
        });
    ft.get();
    std::cout << std::endl;

    // test 2b
    std::cout << "============================" << std::endl;
    std::cout << "Test 2b : .then(shared)" << std::endl;
    auto fs = make_ready_future(5).share();
    //
    future<std::string> fts = fs.then(exec,
        [](shared_future<int> && f)
        {
            std::cout << "Inside .then(shared)" << std::endl;
            return std::string("then(shared)");
        });
    fts.get();
    std::cout << std::endl;

    // test 3a
    std::cout << "============================" << std::endl;
    std::cout << "Test 3a : when_all()" << std::endl;
    future<int>    fw1 = make_ready_future(123);
    future<double> fw2 = make_ready_future(4.567);
    //
    auto fw = when_all(fw1, fw2).then(exec,
        [](future<util::tuple<future<int>, future<double>>> && f)
        {
            auto tup = f.get();
            auto cmplx = std::complex<double>(
                util::get<0>(tup).get(), util::get<1>(tup).get());
            std::cout << "Inside when_all : " << cmplx << std::endl;
            return std::string("when_all");
        }
    );
    fw.get();
    std::cout << std::endl;

    // test 3b
    std::cout << "============================" << std::endl;
    std::cout << "Test 3b : when_all(shared)" << std::endl;
    future<uint64_t>     fws1 = make_ready_future(uint64_t(42));
    shared_future<float> fws2 = make_ready_future(3.1415f).share();
    //
    auto fws = when_all(fws1, fws2).then(exec,
        [](future<util::tuple<future<uint64_t>, shared_future<float>>> && f)
        {
            auto tup = f.get();
            auto cmplx = std::complex<double>(
                util::get<0>(tup).get(), util::get<1>(tup).get());
            std::cout << "Inside when_all(shared) : " << cmplx << std::endl;
            return cmplx;
        }
    );
    fws.get();
    std::cout << std::endl;

    // test 4a
    std::cout << "============================" << std::endl;
    std::cout << "Test 4a : dataflow()" << std::endl;
    future<uint16_t> f1 = make_ready_future(uint16_t(255));
    future<double>   f2 = make_ready_future(127.890);
    //
    auto fd = dataflow(exec,
        [](future<uint16_t> && f1, future<double> && f2)
        {
            auto cmplx = std::complex<uint64_t>(f1.get(), f2.get());
            std::cout << "Inside dataflow : " << cmplx << std::endl;
            return cmplx;
        }
        , f1, f2
    );
    fd.get();
    std::cout << std::endl;

    // test 4b
    std::cout << "============================" << std::endl;
    std::cout << "Test 4b : dataflow(shared)" << std::endl;
    future<uint16_t>      fs1 = make_ready_future(uint16_t(255));
    shared_future<double> fs2 = make_ready_future(127.890).share();
    //
    auto fds = dataflow(exec,
        [](future<uint16_t> && f1, shared_future<double> && f2)
        {
            auto cmplx = std::complex<uint64_t>(f1.get(), f2.get());
            std::cout << "Inside dataflow(shared) : " << cmplx << std::endl;
            return cmplx;
        }
        , fs1, fs2
    );
    fds.get();
    std::cout << std::endl;
    return 0;
}
