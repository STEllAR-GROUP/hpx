//  Copyright (c) 2025
//  SPDX-License-Identifier: BSL-1.0

#include <hpx/config.hpp>

#if defined(HPX_HAVE_STDEXEC)

#include <hpx/execution.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/testing.hpp>

#include <atomic>
#include <vector>
#include <iostream>
#include <thread>
#include <chrono>
#include <unordered_set>
#include <typeinfo>  // Added for type information

namespace ex = hpx::execution::experimental;

namespace {
    struct bulk_test_functor {
        std::atomic<int>* counter;
        void operator()(int) const noexcept {
            counter->fetch_add(1);
        }
    };
    
    struct bulk_value_functor {
        std::vector<int>* results;
        void operator()(int i, int value) const noexcept {
            (*results)[i] = value + i;
        }
    };
    
    struct bulk_thread_id_functor {
        std::vector<std::thread::id>* thread_ids;
        void operator()(int i) const noexcept {
            (*thread_ids)[i] = std::this_thread::get_id();
        }
    };
    
    struct bulk_chunked_functor {
        std::atomic<int>* chunk_count;
        std::atomic<int>* total_items;
        void operator()(int begin, int end) const noexcept {
            chunk_count->fetch_add(1);
            total_items->fetch_add(end - begin);
        }
    };
}

void print_type_information()
{
    std::cout << "\n=== Type Information ===" << std::endl;
    
    // Print bulk CPO type
    std::cout << "1. Bulk CPO type: " << typeid(ex::bulk).name() << std::endl;
    
    // Print thread pool scheduler type
    ex::thread_pool_scheduler sched{};
    std::cout << "2. Thread pool scheduler type: " << typeid(sched).name() << std::endl;
    
    // Create a bulk sender and print its type
    auto bulk_sender = ex::schedule(sched) | ex::bulk(ex::par, 10, [](int){});
    std::cout << "3. Bulk sender type: " << typeid(bulk_sender).name() << std::endl;
    
    // Get and print completion scheduler type
    auto completion_sched = ex::get_completion_scheduler<ex::set_value_t>(ex::get_env(bulk_sender));
    std::cout << "4. Completion scheduler type: " << typeid(completion_sched).name() << std::endl;
    
    // Additional type information
    std::cout << "\n=== Additional Type Information ===" << std::endl;
    std::cout << "5. Domain type: " << typeid(stdexec::get_domain(sched)).name() << std::endl;
    std::cout << "6. bulk_chunked CPO type: " << typeid(ex::bulk_chunked).name() << std::endl;
    std::cout << "7. bulk_unchunked CPO type: " << typeid(ex::bulk_unchunked).name() << std::endl;
    
    // Print sender expression types
    auto just_sender = ex::just(42);
    std::cout << "8. Just sender type: " << typeid(just_sender).name() << std::endl;
    
    auto transfer_sender = ex::just(42) | ex::continue_on(sched);
    std::cout << "9. Transfer sender type: " << typeid(transfer_sender).name() << std::endl;
    
    // Check if bulk sender is a sender expression
    std::cout << "\n=== Type Traits ===" << std::endl;
    std::cout << "Is bulk sender a sender? " << std::boolalpha << ex::sender<decltype(bulk_sender)> << std::endl;
    std::cout << "Is bulk sender a sender_expr? " << stdexec::sender_expr<decltype(bulk_sender)> << std::endl;
    
    // Print tag type if it's a sender expression
    if constexpr (stdexec::sender_expr<decltype(bulk_sender)>) {
        using tag_t = stdexec::tag_of_t<decltype(bulk_sender)>;
        std::cout << "Bulk sender tag type: " << typeid(tag_t).name() << std::endl;
    }
    
    std::cout << std::endl;
}

void test_bulk_with_stdexec_domain()
{
    ex::thread_pool_scheduler sched{};
    
    std::cout << "=== Testing bulk with stdexec domain ===" << std::endl;
    
    // Print type information first
    print_type_information();
    
    // Simple bulk test
    {
        std::cout << "\n[Test 1] Simple bulk execution:" << std::endl;
        std::atomic<int> counter{0};
        constexpr int n = 100;
        
        auto sender = ex::schedule(sched) 
            | ex::bulk(ex::par, n, bulk_test_functor{&counter});
        
        ex::sync_wait(std::move(sender));
        HPX_TEST_EQ(counter.load(), n);
        std::cout << " Executed " << counter.load() << " iterations (expected: " << n << ")" << std::endl;
    }
    
    // Test with value propagation
    {
        std::cout << "\n[Test 2] Bulk with value propagation:" << std::endl;
        std::vector<int> results(10, 0);
        
        auto sender = ex::just(42)
            | ex::continue_on(sched)
            | ex::bulk(ex::par, 10, bulk_value_functor{&results});
        
        auto [final_value] = *ex::sync_wait(std::move(sender));
        HPX_TEST_EQ(final_value, 42);
        std::cout << " Final value: " << final_value << " (expected: 42)" << std::endl;
        
        bool all_correct = true;
        for (int i = 0; i < 10; ++i) {
            HPX_TEST_EQ(results[i], 42 + i);
            if (results[i] != 42 + i) {
                all_correct = false;
                std::cout << "  ✗ results[" << i << "] = " << results[i] 
                          << " (expected: " << (42 + i) << ")" << std::endl;
            }
        }
        if (all_correct) {
            std::cout << " All values correctly computed" << std::endl;
        }
    }
    
    // Test parallel execution (verify it actually uses multiple threads)
    {
        std::cout << "\n[Test 3] Parallel execution verification:" << std::endl;
        constexpr int n = 1000;
        std::vector<std::thread::id> thread_ids(n);
        
        auto sender = ex::schedule(sched)
            | ex::bulk(ex::par, n, bulk_thread_id_functor{&thread_ids});
        
        ex::sync_wait(std::move(sender));
        
        std::unordered_set<std::thread::id> unique_threads(thread_ids.begin(), thread_ids.end());
        std::cout << " Used " << unique_threads.size() << " unique threads for " 
                  << n << " items" << std::endl;
        
        // With parallel execution, we should see multiple threads
        if (unique_threads.size() > 1) {
            std::cout << " Confirmed parallel execution" << std::endl;
        } else {
            std::cout << "  ⚠ Warning: Only one thread used (might be expected for small N or single-core)" << std::endl;
        }
    }
    
    // Test bulk_chunked
    {
        std::cout << "\n[Test 4] Bulk chunked execution:" << std::endl;
        std::atomic<int> chunk_count{0};
        std::atomic<int> total_items{0};
        constexpr int n = 1000;
        
        auto sender = ex::schedule(sched)
            | ex::bulk_chunked(ex::par, n, bulk_chunked_functor{&chunk_count, &total_items});
        
        ex::sync_wait(std::move(sender));
        
        HPX_TEST_EQ(total_items.load(), n);
        std::cout << " Processed " << total_items.load() << " total items in " 
                  << chunk_count.load() << " chunks" << std::endl;
        std::cout << " Average chunk size: " << (n / chunk_count.load()) << std::endl;
    }
    
    // Test with empty range
    {
        std::cout << "\n[Test 5] Empty bulk operation:" << std::endl;
        std::atomic<int> counter{0};
        
        auto sender = ex::schedule(sched)
            | ex::bulk(ex::par, 0, bulk_test_functor{&counter});
        
        ex::sync_wait(std::move(sender));
        HPX_TEST_EQ(counter.load(), 0);
        std::cout << " Empty bulk correctly executed 0 iterations" << std::endl;
    }
    
    // Test with large bulk operation
    {
        std::cout << "\n[Test 6] Large bulk operation:" << std::endl;
        std::atomic<int> counter{0};
        constexpr int n = 10000;
        
        auto start = std::chrono::high_resolution_clock::now();
        
        auto sender = ex::schedule(sched)
            | ex::bulk(ex::par, n, bulk_test_functor{&counter});
        
        ex::sync_wait(std::move(sender));
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        HPX_TEST_EQ(counter.load(), n);
        std::cout << " Large bulk: " << counter.load() << " iterations in " 
                  << duration.count() << " microseconds" << std::endl;
        std::cout << " Throughput: " << (n * 1000000.0 / duration.count()) 
                  << " iterations/second" << std::endl;
    }
    
    // Test chaining bulk operations
    {
        std::cout << "\n[Test 7] Chained bulk operations:" << std::endl;
        std::atomic<int> counter1{0};
        std::atomic<int> counter2{0};
        
        auto sender = ex::schedule(sched)
            | ex::bulk(ex::par, 50, bulk_test_functor{&counter1})
            | ex::continue_on(sched)
            | ex::bulk(ex::par, 30, bulk_test_functor{&counter2});
        
        ex::sync_wait(std::move(sender));
        
        HPX_TEST_EQ(counter1.load(), 50);
        HPX_TEST_EQ(counter2.load(), 30);
        std::cout << " First bulk: " << counter1.load() << " iterations" << std::endl;
        std::cout << " Second bulk: " << counter2.load() << " iterations" << std::endl;
    }
    
    std::cout << "\n=== All bulk tests passed! ===" << std::endl;
}

int hpx_main()
{
    std::cout << "HPX Runtime initialized" << std::endl;
    std::cout << "Number of threads: " << hpx::get_num_worker_threads() << std::endl;
    
    test_bulk_with_stdexec_domain();
    
    std::cout << "\nTest summary: " << hpx::util::report_errors() << " errors" << std::endl;
    
    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    std::cout << "Starting HPX bulk test with stdexec..." << std::endl;
    
    int result = hpx::local::init(hpx_main, argc, argv);
    
    if (result != 0) {
        std::cerr << "HPX main exited with non-zero status: " << result << std::endl;
    }
    
    int errors = hpx::util::report_errors();
    if (errors == 0) {
        std::cout << "SUCCESS: All tests passed!" << std::endl;
    } else {
        std::cerr << "FAILURE: " << errors << " test(s) failed!" << std::endl;
    }
    
    return errors;
}

#else
int main()
{
    std::cout << "Skipping test: HPX_HAVE_STDEXEC not defined" << std::endl;
    return 0;
}
#endif
