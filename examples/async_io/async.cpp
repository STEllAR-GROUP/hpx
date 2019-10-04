#include <hpx/lcos/future.hpp>
#include <hpx/async.hpp>
#include <hpx/hpx_main.hpp>

using namespace hpx;
int main() {
  future<int> f = async([]() -> int { return 91; });
  int val = f.get();
  return hpx::finalize();
}

