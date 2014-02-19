#include <hpx/hpx_init.hpp>
#include <hpx/include/runtime.hpp>
#include <hpx/lcos/when_all.hpp>
#include <hpx/include/iostreams.hpp>
//
#include <random>

#define TEST_SUCCESS 1
#define TEST_FAIL    0
//
#define FAILURE_RATE_PERCENT 5 
#define SAMPLES_PER_LOOP     10
#define TEST_LOOPS           1000
//
std::random_device rseed;
std::mt19937 gen(rseed());
std::uniform_int_distribution<int> dist(0,99); // interval [0,100)

#define USE_LAMBDA

//----------------------------------------------------------------------------
int reduce(hpx::unique_future<std::vector<hpx::unique_future<int>>> &&futvec)
{
  int res = TEST_SUCCESS;
  std::vector<hpx::unique_future<int> > vfs = futvec.get();  
  for (hpx::unique_future<int>& f: vfs) {
    if (f.get() == TEST_FAIL) return TEST_FAIL;
  }
  return res;
}

//----------------------------------------------------------------------------
int generate_one()
{
  // generate roughly x% fails
  int result = TEST_SUCCESS;
  if (dist(gen)>=(100-FAILURE_RATE_PERCENT)) {
    result = TEST_FAIL;
  }
  return result;
}

//----------------------------------------------------------------------------
hpx::unique_future<int> test_reduce()
{
  std::vector<hpx::unique_future<int>> req_futures;
  //
  for (int i=0; i<SAMPLES_PER_LOOP; i++) {
    // generate random sequence of pass/fails using % fail rate per incident
    hpx::unique_future<int> result = hpx::async(generate_one);
    req_futures.push_back(std::move(result));
  }

  hpx::unique_future<std::vector<hpx::unique_future<int>>> all_ready = hpx::when_all(req_futures);

#ifdef USE_LAMBDA
  hpx::unique_future<int> result = all_ready.then(
    [](hpx::unique_future<std::vector<hpx::unique_future<int>>> &&futvec) -> int {
      // futvec is ready or the lambda would not be called
      std::vector<hpx::unique_future<int>> vfs = futvec.get();
      // all futures in v are ready as fut is ready
      int res = TEST_SUCCESS;
      for (hpx::unique_future<int>& f: vfs) {
        if (f.get() == TEST_FAIL) return TEST_FAIL;
      }
      return res;
  });
#else
  hpx::unique_future<int> result = all_ready.then(reduce);
#endif
  //
  return result;
}

//----------------------------------------------------------------------------
int hpx_main()
{
  hpx::util::high_resolution_timer htimer;
  // run N times and see if we get approximately the right amount of fails 
  int count = 0;
  for (int i=0; i<TEST_LOOPS; i++) {
    int result = test_reduce().get();
    count += result;
  }
  double pr_pass  = std::pow(1.0 - FAILURE_RATE_PERCENT/100.0, SAMPLES_PER_LOOP);
  double exp_pass = TEST_LOOPS*pr_pass;
  hpx::cout << "From " << TEST_LOOPS << " tests, we got " 
    << "\n " << count << " passes" 
    << "\n " << exp_pass << " expected \n" 
    << "\n " << htimer.elapsed() << " seconds \n" << hpx::flush;
  // Initiate shutdown of the runtime system.
  return hpx::finalize(); 
}

//----------------------------------------------------------------------------
int main(int argc, char* argv[])
{
  // Initialize and run HPX.
  return hpx::init(argc, argv);
}

