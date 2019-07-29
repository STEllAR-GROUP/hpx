#include <hpx/lcos/future.hpp>

int main() {
  future<int> f = async([]() -> int { return 91; });
  future<string> fB = fA.then([](future<int> f)) {
  return f.get().to_string();
  };
}
