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
// print type information
// --------------------------------------------------------------------
template<typename T=void, typename... Args>
inline std::string print_type(const char *delim="");

template<>
inline std::string print_type<>(const char *delim)
{
    return "";
}

template<typename T, typename... Args>
inline std::string print_type(const char *delim)
{
    std::string temp(hpx::util::demangle_helper<T>().type_id());
    if constexpr (sizeof...(Args)>0) {
        return temp + delim + print_type<Args...>(delim);
    }
    return temp;
}

// --------------------------------------------------------------------
//
// --------------------------------------------------------------------
struct tuple_futures_extract_value
{
    template<typename T, template <typename> typename Future>
    const T& operator()(const Future<T> &el) const
    {
        typedef typename hpx::traits::detail::shared_state_ptr_for<Future<T>>::type
            shared_state_ptr;
        shared_state_ptr const& state = hpx::traits::detail::get_shared_state(el);
        const T *future_value = state->get_result();
        return *future_value;
    }
};

// --------------------------------------------------------------------
// async test
// --------------------------------------------------------------------
using namespace hpx;

struct test_async_executor
{
    typedef parallel::execution::parallel_execution_tag execution_category;

    test_async_executor() : executor_() {};

    // --------------------------------------------------------------------
    // async execute specialized for simple arguments typical
    // of a normal async call with arbitrary arguments
    // --------------------------------------------------------------------
    template <typename F, typename ... Ts>
    future<typename util::invoke_result<F, Ts...>::type>
    async_execute(F && f, Ts &&... ts)
    {
        typedef typename util::detail::invoke_deferred_result<F, Ts...>::type result_type;

        std::cout << "async_execute : Function   : "
                  << print_type<F>(" | ") << "\n";
        std::cout << "async_execute : Arguments  : "
                  << print_type<Ts...>(" | ") << "\n";
        std::cout << "async_execute : Result     : "
                  << print_type<result_type>(" | ") << "\n";

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
    template <typename F, template <typename> typename Future,
              typename P, typename ... Ts,
              typename = typename std::enable_if_t<traits::is_future<Future<P>>::value>>
    auto
    then_execute(F && f, Future<P> & predecessor, Ts &&... ts)
    ->  future<typename util::detail::invoke_deferred_result<
        F, Future<P>, Ts...>::type>
    {
        typedef typename util::detail::invoke_deferred_result<
                F, Future<P>, Ts...>::type result_type;

        std::cout << "then_execute : Function    : "
                  << print_type<F>(" | ") << "\n";
        std::cout << "then_execute : Predecessor : "
                  << print_type<Future<P>>(" | ") << "\n";
        std::cout << "then_execute : is_future   : "
                  << print_type<typename traits::is_future<Future<P>>::type>(" | ") << "\n";
        std::cout << "then_execute : Arguments   : "
                  << print_type<Ts...>(" | ") << "\n";
        std::cout << "then_execute : Result      : "
                  << print_type<result_type>(" | ") << "\n";

        // forward the task on to the 'real' underlying executor
        lcos::local::futures_factory<result_type()> p(
            executor_,
            util::deferred_call(std::forward<F>(f),
                                std::move(predecessor),
                                std::forward<Ts>(ts)...)
        );

        p.apply(
            launch::async,
            threads::thread_priority_default,
            threads::thread_stacksize_default);

        return p.get_future();
    }
/*
    // --------------------------------------------------------------------
    // .then() execute specialized for a when_all dispatch
    // a future< tuple< future<a>, future<b>, ...> > is expected and unwrapped
    // --------------------------------------------------------------------
    template <typename F, template <typename> typename Future,
              typename ... Ps, typename ... Ts>
    auto
    then_execute(F && f, Future<util::tuple<lcos::future<Ps>...>> & predecessor, Ts &&... ts)
    ->  future<typename util::detail::invoke_deferred_result<
        F, Future<util::tuple<lcos::future<Ps>...>>, Ts...>::type>
    {
        typedef typename util::detail::invoke_deferred_result<
            F, Future<util::tuple<lcos::future<Ps>...>>, Ts...>::type result_type;

        // get the arguments from the predecessor future tuple of futures
        typedef typename traits::detail::shared_state_ptr_for<Future<util::tuple<lcos::future<Ps>...>>>::type
            shared_state_ptr;
        shared_state_ptr const& state = traits::detail::get_shared_state(predecessor);
        util::tuple<lcos::future<Ps>...> *predecessor_value = state->get_result();

        std::cout << "Tuple contents : ";
        auto unwrapped_futures_tuple = hpx::util::map_pack(
            tuple_futures_extract_value{},
            *predecessor_value
        );
        std::cout << "\n";

        hpx::util::invoke_fused([](const auto & ...ts) {
            std::cout << print_args(ts...) << std::endl;
        }, unwrapped_futures_tuple);

        std::cout << "when_all : Predecessor     : "
                  << print_type<Future<util::tuple<lcos::future<Ps>...>>>() << "\n";
        std::cout << "when_all : unwrapped       : "
                  << print_type<Ps...>() << "\n";
        std::cout << "when_all : Result          : "
                  << print_type<result_type>() << "\n";

        // forward the task execution on to the real internal executor
        lcos::local::futures_factory<result_type()> p(
            executor_,
            util::deferred_call(
                std::forward<F>(f),
                std::forward<Future<util::tuple<lcos::future<Ps>...>>>(predecessor),
                std::forward<Ts>(ts)...)
        );

        p.apply(
            launch::async,
            threads::thread_priority_default,
            threads::thread_stacksize_default);

        return p.get_future();
    }
*/
    template <typename TupleOfFutures>
    struct is_tuple_of_futures;

    template <typename...Futures>
    struct is_tuple_of_futures<util::tuple<Futures...>>
        : util::detail::all_of<traits::is_future<Futures>...>
    {};

    template <typename Future>
    struct is_future_of_tuple_of_futures
        : std::integral_constant<bool,
            traits::is_future<Future>::value &&
            is_tuple_of_futures<typename traits::future_traits<Future>::result_type>::value>
    {};

    // --------------------------------------------------------------------
    // .then() execute specialized for a when_all dispatch for any future types
    // future< tuple< is_future<a>::type, is_future<b>::type, ...> >
    // --------------------------------------------------------------------
    template <typename F,
              template <typename> typename  OuterFuture,
              typename ... InnerFutures,
              typename ... Ts,
              typename = typename std::enable_if_t<is_future_of_tuple_of_futures<OuterFuture<util::tuple<InnerFutures...>>>::value>,
              typename = typename std::enable_if_t<is_tuple_of_futures<util::tuple<InnerFutures...>>::value>
              >
    auto
    then_execute(F && f, OuterFuture<util::tuple<InnerFutures... > > & predecessor, Ts &&... ts)
    ->  future<typename util::detail::invoke_deferred_result<
        F, OuterFuture<util::tuple<InnerFutures... >>, Ts...>::type>
    {

        typedef typename util::detail::invoke_deferred_result<
            F, OuterFuture<util::tuple<InnerFutures... >>, Ts...>::type
                result_type;

        // get the arguments from the predecessor future tuple of futures
        typedef typename traits::detail::shared_state_ptr_for<
                OuterFuture<util::tuple<InnerFutures...>>>::type
                shared_state_ptr;

        shared_state_ptr const& state = traits::detail::get_shared_state(predecessor);
        util::tuple<InnerFutures...> *predecessor_value = state->get_result();

        auto unwrapped_futures_tuple = hpx::util::map_pack(
            tuple_futures_extract_value{},
            *predecessor_value
        );

        std::cout << "when_all(fut) : Predecessor: "
                  << print_type<OuterFuture<util::tuple<InnerFutures...>>>(" | ") << "\n";
        std::cout << "when_all(fut) : unwrapped  : "
                  << print_type<decltype(unwrapped_futures_tuple)>(" | ") << "\n";
        std::cout << "when_all(fut) : Result     : "
                  << print_type<result_type>(" | ") << "\n";
        std::cout << "when_all(fut) : tuple      : ";
        hpx::util::invoke_fused([](const auto & ...ts) {
            std::cout << print_type<decltype(ts)...>(" | ") << std::endl;
        }, unwrapped_futures_tuple);
        std::cout << "\n";


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

private:
    threads::executors::default_executor executor_;
};

namespace hpx { namespace parallel { namespace execution
{
    template <>
    struct is_two_way_executor<test_async_executor>
      : std::true_type
    {};
}}}

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
            std::cout << "Inside .then() \n" << std::endl;
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
            std::cout << "Inside .then(shared) \n" << std::endl;
            return std::string("then(shared)");
        });
    fts.get();
    std::cout << std::endl;

    // test 3
    std::cout << "============================" << std::endl;
    std::cout << "Test 3 : dataflow()" << std::endl;
    future<int>    f1 = make_ready_future(5);
    future<double> f2 = make_ready_future(5.0);
    //
    auto fd = dataflow(exec,
            [](future<int> && f1, future<double> && f2)
            {
                std::cout << "Inside dataflow \n" << std::endl;
                return std::complex<double>(f1.get(), f2.get());
            }
        , f1, f2
    );
    fd.get();
    std::cout << std::endl;

    // test 4a
    std::cout << "============================" << std::endl;
    std::cout << "Test 4a : when_all()" << std::endl;
    future<int>    fw1 = make_ready_future(42);
    future<double> fw2 = make_ready_future(3.1415);
    //
    auto fw = when_all(fw1, fw2).then(exec,
        [](future<util::tuple<future<int> , future<double>>> &&)
        {
            std::cout << "Inside when_all \n" << std::endl;
            return std::string("when_all");
        }
    );
    fw.get();
    std::cout << std::endl;

    // test 4b
    std::cout << "============================" << std::endl;
    std::cout << "Test 4b : when_all(shared)" << std::endl;
    future<uint64_t>     fws1 = make_ready_future(uint64_t(42));
    shared_future<float> fws2 = make_ready_future(3.1415f).share();
    //
    auto fws = when_all(fws1, fws2).then(exec,
        [](future<util::tuple<future<uint64_t>, shared_future<float>>> && f)
        {
            std::cout << "Inside when_all(shared) \n" << std::endl;
            auto tup = f.get();
            return std::complex<double>(util::get<0>(tup).get(), util::get<1>(tup).get());
        }
    );
    fws.get();
    std::cout << std::endl;

    return 0;
}
