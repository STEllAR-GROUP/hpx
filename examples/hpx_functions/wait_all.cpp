#include <hpx/lcos/future.hpp>
int main() {
hpx::vector<hpx::future<void>> results;
for (int i = 0; i != NUM; ++i) {
    results.push_back(hpx::async(...));
    hpx::wait_all(results);
  }
}
