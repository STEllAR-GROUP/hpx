//  Copyright (c)      2013 Thomas Heller
//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This is a purely local version demonstrating the proposed extension to
// C++ implementing resumable functions (see N3564, 
// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2013/n3564.pdf). The
// necessary transformations are performed by hand.

#include <hpx/hpx_init.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/util.hpp>
#include <hpx/include/lcos.hpp>

#include <iostream>

#include <boost/cstdint.hpp>
#include <boost/format.hpp>

#if defined(BOOST_MSVC)
#define HPX_NO_INLINE __declspec(noinline)
#else
#define HPX_NO_INLINE
#endif

namespace hpx { namespace lcos { namespace local {
    namespace detail {
        template <typename Func, typename F1, typename F2>
        struct dataflow_frame
            : boost::enable_shared_from_this<dataflow_frame<Func, F1, F2> >
        {
            typedef
                decltype(std::declval<Func>()(std::declval<F1>().get(), std::declval<F2>().get()))
                result_type;

            typedef hpx::future<result_type> future_result_type;
            typedef hpx::lcos::local::promise<result_type> promise_result_type;

            typedef typename hpx::util::detail::remove_reference<Func>::type func_type;
            typedef typename hpx::util::detail::remove_reference<F1>::type f1_type;
            typedef typename hpx::util::detail::remove_reference<F2>::type f2_type;
            
            typedef decltype(std::declval<F1>().get()) f1_result_type;
            typedef decltype(std::declval<F2>().get()) f2_result_type;

            template <typename FFunc, typename FF1, typename FF2>
            dataflow_frame(BOOST_FWD_REF(FFunc) func, BOOST_FWD_REF(FF1) f1, BOOST_FWD_REF(FF2) f2)
              : func_(func)
              , f1_(f1)
              , f2_(f2)
              , state_(0)
            {}

            void await()
            {
                switch (state_)
                {
                case 1:
                    goto L1;
                case 2:
                    goto L2;
                }

                if(!f1_.is_ready())
                {
                    state_ = 1;
                    if(!result_.valid())
                    {
                        result_ = result_promise_.get_future();
                    }
                    f1_.then(boost::bind(&dataflow_frame::await, this->shared_from_this()));
                    return;
                }
L1:
                f1_result_ = f1_.get();

                if(!f2_.is_ready())
                {
                    state_ = 2;
                    if(!result_.valid())
                    {
                        result_ = result_promise_.get_future();
                    }
                    f2_.then(boost::bind(&dataflow_frame::await, this->shared_from_this()));
                    return;
                }
L2:
                f2_result_ = f2_.get();

                if(state_ == 0)
                {
                    result_ = hpx::make_ready_future(func_(f1_result_, f2_result_));
                }
                else
                {
                    result_promise_.set_value(func_(f1_result_, f2_result_));
                }
            }

            func_type func_;
            f1_type f1_;
            f2_type f2_;
            f1_result_type f1_result_;
            f2_result_type f2_result_;
            future_result_type result_;
            promise_result_type result_promise_;
            int state_;
        };
    }

    template <typename Func, typename F1, typename F2>
    BOOST_FORCEINLINE
    typename detail::dataflow_frame<Func, F1, F2>::future_result_type
    dataflow(BOOST_FWD_REF(Func) func, BOOST_FWD_REF(F1) f1, BOOST_FWD_REF(F2) f2)
    {
        typedef detail::dataflow_frame<Func, F1, F2> frame_type;

        boost::shared_ptr<frame_type> frame =
            boost::make_shared<frame_type>(
                boost::forward<Func>(func)
              , boost::forward<F1>(f1)
              , boost::forward<F2>(f2)
            );

        frame->await();

        return frame->result_;
    }

}}}

///////////////////////////////////////////////////////////////////////////////
boost::uint64_t threshold = 2;

///////////////////////////////////////////////////////////////////////////////
HPX_NO_INLINE boost::uint64_t fibonacci_serial(boost::uint64_t n)
{
    if (n < 2)
        return n;
    return fibonacci_serial(n-1) + fibonacci_serial(n-2);
}

///////////////////////////////////////////////////////////////////////////////
hpx::future<boost::uint64_t> fibonacci(boost::uint64_t n)
{
    if (n < 2) return hpx::make_ready_future(n);
    if (n < threshold) return hpx::make_ready_future(fibonacci_serial(n));

    hpx::future<boost::uint64_t> lhs_future = hpx::async(&fibonacci, n-1).unwrap();
    hpx::future<boost::uint64_t> rhs_future = fibonacci(n-2);

    return
        hpx::lcos::local::dataflow(
            [](boost::uint64_t lhs, boost::uint64_t rhs)
            {
                return lhs + rhs;
            }
          , lhs_future, rhs_future
        );
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map& vm)
{
    // extract command line argument, i.e. fib(N)
    boost::uint64_t n = vm["n-value"].as<boost::uint64_t>();
    std::string test = vm["test"].as<std::string>();
    boost::uint64_t max_runs = vm["n-runs"].as<boost::uint64_t>();

    if (max_runs == 0) {
        std::cerr << "fibonacci_futures: wrong command line argument value for "
            "option 'n-runs', should not be zero" << std::endl;
        return hpx::finalize(); // Handles HPX shutdown
    }

    threshold = vm["threshold"].as<unsigned int>();
    if (threshold < 2 || threshold > n) {
        std::cerr << "fibonacci_futures: wrong command line argument value for "
            "option 'threshold', should be in between 2 and n-value"
            ", value specified: " << threshold << std::endl;
        return hpx::finalize(); // Handles HPX shutdown
    }

    bool executed_one = false;
    boost::uint64_t r = 0;

    if (test == "all" || test == "0")
    {
        // Keep track of the time required to execute.
        boost::uint64_t start = hpx::util::high_resolution_clock::now();

        for (boost::uint64_t i = 0; i != max_runs; ++i)
        {
            // serial execution
            r = fibonacci_serial(n);
        }

//        double d = double(hpx::util::high_resolution_clock::now() - start) / 1.e9;
        boost::uint64_t d = hpx::util::high_resolution_clock::now() - start;
        char const* fmt = "fibonacci_serial(%1%) == %2%,"
            "elapsed time:,%3%,[s]\n";
        std::cout << (boost::format(fmt) % n % r % (d / max_runs));

        executed_one = true;
    }

    if (test == "all" || test == "1")
    {
        // Keep track of the time required to execute.
        boost::uint64_t start = hpx::util::high_resolution_clock::now();

        for (boost::uint64_t i = 0; i != max_runs; ++i)
        {
            // Create a future for the whole calculation, execute it locally,
            // and wait for it.
            r = fibonacci(n).get();
        }

//        double d = double(hpx::util::high_resolution_clock::now() - start) / 1.e9;
        boost::uint64_t d = hpx::util::high_resolution_clock::now() - start;
        char const* fmt = "fibonacci_await(%1%) == %2%,"
            "elapsed time:,%3%,[s]\n";
        std::cout << (boost::format(fmt) % n % r % (d / max_runs));

        executed_one = true;
    }

    if (!executed_one)
    {
        std::cerr << "fibonacci_futures: wrong command line argument value for "
            "option 'tests', should be either 'all' or a number between zero "
            "and 1, value specified: " << test << std::endl;
    }

    return hpx::finalize(); // Handles HPX shutdown
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options
    boost::program_options::options_description
       desc_commandline("Usage: " HPX_APPLICATION_STRING " [options]");

    using boost::program_options::value;
    desc_commandline.add_options()
        ( "n-value", value<boost::uint64_t>()->default_value(10),
          "n value for the Fibonacci function")
        ( "n-runs", value<boost::uint64_t>()->default_value(1),
          "number of runs to perform")
        ( "threshold", value<unsigned int>()->default_value(2),
          "threshold for switching to serial code")
        ( "test", value<std::string>()->default_value("all"),
          "select tests to execute (0-1, default: all)")
    ;

    // Initialize and run HPX
    return hpx::init(desc_commandline, argc, argv);
}
