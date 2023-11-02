#include <hpx/hpx_main.hpp>
#include <hpx/synchronization/mutex.hpp>
#include <hpx/synchronization/range_lock.hpp>

int main()
{
    {
        hpx::synchronization::range_lock rl;
        std::size_t x = rl.lock(0, 10);
        rl.unlock(x);
        return 0;
    }

    {
        hpx::synchronization::range_lock rl;

        hpx::synchronization::range_guard<hpx::synchronization::range_lock> rg(
            rl, 0, 10);
    }

    {
        hpx::synchronization::range_lock rl;

        hpx::synchronization::range_unique_lock<
            hpx::synchronization::range_lock>
            rg(rl, 0, 10);
    }
}
