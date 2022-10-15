#include <iostream> 
#include <string> 
#include <vector>

#include <hpx/hpx_init.hpp> 
#if defined(HPX_HAVE_SYCL)

// Check compiler compatability
#if defined(SYCL_LANGUAGE_VERSION)
#pragma message("OKAY: Sycl compiler detected...")
#if defined(__INTEL_LLVM_COMPILER)
#pragma message("OKAY: Intel dpcpp detected!")
#elif defined(__HIPSYCL__)
#warning("HIPSycl syclcc compiler detected!")
// TODO Fix compilation with hipsycl
// Technically it should be possible to substitute -fno-sycl flag with --hipsycl-platform=cpu
// and keep the rest the same
// See: syclcc --help
// and https://github.com/illuhad/hipSYCL/blob/develop/doc/macros.md
#error("Support for hipsycl not yet implemented!")
#else
#warning("Non-Intel compiler have not yet been tested with the HPX Sycl integration...")
#endif
#else
//#error("Compiler does not seem to support SYCL! SYCL_LANGUAGE_VERSION is undefined!")
#endif

// Check for separate compiler host and device passes
#if defined(__SYCL_SINGLE_SOURCE__)
#error("Sycl single source compiler not supported! Use one with multiple passes")
#else
#pragma message("OKAY: Sycl compiler with two or more compile passes detected")
#endif


#if defined(__SYCL_DEVICE_ONLY__)
#pragma message("Sycl device pass...")
#else
#pragma message("Sycl host pass...")
#endif

#if defined(__HIPSYCL__) 
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-copy"
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wdouble-promotion"
#pragma clang diagnostic ignored "-Wunused-local-typedef"
#pragma clang diagnostic ignored "-Wsign-compare"
#pragma clang diagnostic ignored "-Wgcc-compat"
#pragma clang diagnostic ignored "-Wembedded-directive"
#pragma clang diagnostic ignored "-Wmismatched-tags"
#pragma clang diagnostic ignored "-Wreorder-ctor"
#endif

#include <hpx/async_sycl/sycl_event_pool.hpp> 
#include <CL/sycl.hpp> 




#if defined(__HIPSYCL__)
using namespace cl::sycl;
#else
using namespace sycl;
#endif

constexpr size_t vector_size = 1000000;

void VectorAdd(queue& q, const std::vector<int>& a_vector,
    const std::vector<int>& b_vector, std::vector<int>& add_parallel)
{
    event my_kernel_event;
    event my_kernel_event2;
    if (!hpx::sycl::experimental::sycl_event_pool::get_event_pool().pop(my_kernel_event))
    {
        HPX_THROW_EXCEPTION(
            hpx::invalid_status, "add_event_callback", "could not get an event");
    }
    // input range
    range<1> num_items{a_vector.size()};
    {
      // buffers from host vectors
      buffer a_buf(a_vector.data(), num_items);
      buffer b_buf(b_vector.data(), num_items);
      buffer add_buf(add_parallel.data(), num_items);

      my_kernel_event = q.submit([&](handler& h) {
          // Tell sycl we'd like to access our buffers here
          accessor a(a_buf, h, read_only);
          accessor b(b_buf, h, read_only);
          accessor add(add_buf, h, write_only, no_init);
          // run Add kernel
          h.parallel_for(num_items, [=](auto i) { add[i] = a[i] + b[i]; });
          // Note: destruction of the accessors should not cause a device->host
          // memcpy (I think...)
      });
      // simulate get_future
      my_kernel_event2 = q.submit([&](handler& h) {
          h.parallel_for(range<1>{1}, [=](auto i) { });
          });
      // should be running
      const auto event_status =
          my_kernel_event.get_info<info::event::command_execution_status>();
      const auto event_status2 =
          my_kernel_event2.get_info<info::event::command_execution_status>();
      if (event_status != info::event_command_status::complete &&
          event_status2 != info::event_command_status::complete)
          std::cerr << "OKAY: Kernel not yet done" << std::endl;
      else
        std::cerr << "ERROR: Kernel already done" << std::endl;
      // according to the sycl specification (2020) section 3.9.8, the entire
      // thing will synchronize here, due to the buffers being destroyed
    }
    // should be done
    const auto event_status =
        my_kernel_event.get_info<info::event::command_execution_status>();
    const auto event_status2 =
        my_kernel_event2.get_info<info::event::command_execution_status>();
    if (event_status != info::event_command_status::complete ||
        event_status2 != info::event_command_status::complete)
        std::cerr << "ERROR: Kernel still not done" << std::endl;
    else
      std::cerr << "OKAY: Kernel done after end of scope" << std::endl;
}

int hpx_main(int, char**)
{
    std::cout << "Starting HPX main" << std::endl;

    // Select default sycl device
    default_selector d_selector;

    // input vectors
    std::vector<int> a(vector_size), b(vector_size),
        add_sequential(vector_size), add_parallel(vector_size);
    for (size_t i = 0; i < a.size(); i++)
    {
        a.at(i) = i;
        b.at(i) = i;
    }

    try
    {
        // TODO Insert executor once finished
        queue q(d_selector, property::queue::in_order{});
        /* queue q(d_selector); */
        std::cout << "SYCL language version: " << SYCL_LANGUAGE_VERSION << "\n";
        std::cout << "Running on device: "
                  << q.get_device().get_info<info::device::name>() << "\n";
        std::cout << "Vector size: " << a.size() << "\n";
        // TODO Launch with executor
        VectorAdd(q, a, b, add_parallel);
        // TODO Add check for asynchronous launch
        // TODO Synchronize executor
    }
    catch (exception const& e)
    {
        std::cout << "An exception is caught for vector add.\n";
        std::terminate();
    }

    // Check results
    for (size_t i = 0; i < add_sequential.size(); i++)
        add_sequential.at(i) = a.at(i) + b.at(i);
    for (size_t i = 0; i < add_sequential.size(); i++)
    {
        if (add_parallel.at(i) != add_sequential.at(i))
        {
            std::cout << "Vector add failed on device.\n ";
            return -1;
        }
    }

    static_assert(vector_size >= 6, "vector_size unreasonably small");
    for (size_t i = 0; i < 3; i++)
    {
        std::cout << "[" << i << "]: " << a[i] << " + " << b[i] << " = "
                  << add_parallel[i] << "\n";
    }
    std::cout << "...\n";
    for (size_t i = 3; i > 0; i--)
    {
        std::cout << "[" << vector_size - i << "]: " << a[vector_size - i]
                  << " + " << b[vector_size - i] << " = "
                  << add_parallel[vector_size - i] << "\n";
    }

    a.clear();
    b.clear();
    add_sequential.clear();
    add_parallel.clear();

    std::cout << "Vector add successful.\n";
    std::cout << "Finalizing HPX main" << std::endl;
    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    std::cout << "Starting main" << std::endl;
    return hpx::init(argc, argv);
}
#endif
