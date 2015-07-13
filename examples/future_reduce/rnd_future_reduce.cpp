//  Copyright (c) 2014 John Biddiscombe
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/runtime.hpp>
#include <hpx/lcos/when_all.hpp>
#include <hpx/include/iostreams.hpp>
//
#include <random>

#include  <boost/nondet_random.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int_distribution.hpp>

//
// This is a simple example which generates random numbers and returns
// pass or fail from a routine.
// When called by many threads returning a vector of futures - if the user wants to
// reduce the vector of pass/fails into a single pass fail based on a simple
// any fail = !pass rule, then this example shows how to do it.
// The user can experiment with the failure rate to see if the statistics match
// their expectations.
// Also. Routine can use either a lambda, or a function under control of USE_LAMBDA

#define TEST_SUCCESS 1
#define TEST_FAIL    0
//
#define FAILURE_RATE_PERCENT 5
#define SAMPLES_PER_LOOP     10
#define TEST_LOOPS           1000
//
boost::random::random_device rseed;
boost::random::mt19937 gen(rseed());
boost::random::uniform_int_distribution<int> dist(0,99); // interval [0,100)

#define USE_LAMBDA

//----------------------------------------------------------------------------
int reduce(hpx::future<std::vector<hpx::future<int> > > &&futvec)
{
  int res = TEST_SUCCESS;
  std::vector<hpx::future<int> > vfs = futvec.get();
  for (hpx::future<int>& f: vfs) {
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
hpx::future<int> test_reduce()
{
  std::vector<hpx::future<int> > req_futures;
  //
  for (int i=0; i<SAMPLES_PER_LOOP; i++) {
    // generate random sequence of pass/fails using % fail rate per incident
    hpx::future<int> result = hpx::async(generate_one);
    req_futures.push_back(std::move(result));
  }

  hpx::future<std::vector<hpx::future<int> > > all_ready = hpx::when_all(req_futures);

#ifdef USE_LAMBDA
  hpx::future<int> result = all_ready.then(
    [](hpx::future<std::vector<hpx::future<int> > > &&futvec) -> int {
      // futvec is ready or the lambda would not be called
      std::vector<hpx::future<int> > vfs = futvec.get();
      // all futures in v are ready as fut is ready
      int res = TEST_SUCCESS;
      for (hpx::future<int>& f: vfs) {
        if (f.get() == TEST_FAIL) return TEST_FAIL;
      }
      return res;
  });
#else
  hpx::future<int> result = all_ready.then(reduce);
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

