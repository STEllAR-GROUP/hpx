#include <hpx/hpx_main.hpp>
#include <hpx/synchronization/mutex.hpp>
#include <hpx/synchronization/range_lock.hpp>

int main()
{
    hpx::synchronization::ByteLock<hpx::mutex, std::lock_guard> bl;
    std::size_t x = bl.lock(0, 10);
    bl.unlock(x);
}
