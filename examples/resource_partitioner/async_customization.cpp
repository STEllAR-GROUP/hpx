#include <hpx/hpx_main.hpp>
#include <hpx/include/parallel_executors.hpp>
#include <hpx/async.hpp>

#include <hpx/lcos/dataflow.hpp>
#include <hpx/util/invoke.hpp>
#include <hpx/util/invoke_fused.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/result_of.hpp>
#include <hpx/util/tuple.hpp>
#include <hpx/util/deferred_call.hpp>

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
inline std::string print_type() { return ""; }

template <class T>
inline std::string print_type()
{
    return demangle(typeid(T).name());
}

template<typename T, typename... Args>
inline std::string print_type(T&& head, Args&&... tail)
{
    std::string temp = print_type<T>();
    std::cout << "\t" << temp << std::endl;
    return print_type(std::forward<Args>(tail)...);
}

// --------------------------------------------------------------------
// async test
// --------------------------------------------------------------------
struct then_continuation_tag {};

struct test_async_executor
{
    typedef hpx::parallel::execution::parallel_execution_tag execution_category;

    template <typename F, typename ... Ts>
    static hpx::future<typename hpx::util::invoke_result<F, Ts...>::type>
    async_execute(F && f, Ts &&... ts)
    {
        std::cout << "async_execute : Function : \n";
        print_type(f);
        std::cout << "async_execute : Arguments : \n";
        print_type(ts...);

        typedef typename hpx::util::detail::invoke_deferred_result<F, Ts...>::type T2;
        print_type<T2>();

        return hpx::async(hpx::launch::async, std::forward<F>(f),
            std::forward<Ts>(ts)...);
    }

    ///////////////////////////////////////////////////////////////////////
    template <typename F, typename Future, typename ... Ts>
    auto
    then_execute(F && f, Future& predecessor, Ts &&... ts) const
    ->  hpx::future<typename hpx::util::detail::invoke_deferred_result<
        F, Future, Ts...>::type>
    {
        typedef typename hpx::util::detail::invoke_deferred_result<
                F, Future, Ts...>::type result_type;

        std::cout << "then_execute : Function : \n";
        print_type(f);
        std::cout << "then_execute : Predecessor : \n";
        print_type(predecessor);
        std::cout << "then_execute : Arguments : \n";
        print_type(ts...);

        auto func = hpx::util::bind(
            hpx::util::one_shot(std::forward<F>(f)),
                    hpx::util::placeholders::_1,
                    std::forward<Ts>(ts)...);

        typename hpx::traits::detail::shared_state_ptr<result_type>::type
            p = hpx::lcos::detail::make_continuation_exec<result_type>(
                    predecessor, *this,
                    std::move(func));

        return hpx::traits::future_access<
                hpx::lcos::future<result_type>
            >::create(std::move(p));
    }

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
/*
    std::cout << "Test 1 " << std::endl;
    hpx::async(exec, [](int a, double b, const char *c)
    {
        std::cout << "Test " << c << std::endl;
    }, 1, 2.2, "Hello").get();

    std::cout << std::endl;
*/
    // test 2
    std::cout << "Test 2 " << std::endl;
    hpx::future<int> f = hpx::make_ready_future(5);
    //
    hpx::future<std::string> f2 = f.then(exec,
        [](hpx::future<int> && f)
        {
            std::cout << "inside continuation 1 \n" << std::endl;
            return std::string("Hello");
        });
    f2.get();
    std::cout << std::endl;

    return 0;
}
