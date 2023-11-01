#include <hpx/hpx_main.hpp>
#include <hpx/synchronization/mutex.hpp>
#include <hpx/synchronization/range_lock.hpp>

int main()
{
    hpx::synchronization::RangeLock<hpx::mutex, std::lock_guard> rl;
    std::size_t x = rl.lock(0, 10);
    rl.unlock(x);
}
