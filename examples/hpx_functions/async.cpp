#include <hpx/lcos/future.hpp>
using namespace hpx;
int main() {
  future<int> f = async([]() -> int { return 91; });
  int val = f.get();
}

