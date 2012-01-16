
#include <hpx/hpx_fwd.hpp>
#include "grid.hpp"
#include <hpx/runtime/actions/plain_action.hpp>

HPX_REGISTER_PLAIN_ACTION(touch_mem_action);

HPX_EXPORT std::size_t touch_mem(std::size_t desired, std::size_t ps, std::size_t l, std::size_t n)
{
    std::size_t current = hpx::threads::threadmanager_base::get_thread_num();

    if (current == desired)
    {
        // Yes! The PX-thread is run by the designated OS-thread.
        char * p = reinterpret_cast<char *>(ps);
        for(int i = desired * l; i < std::min((desired+1) * l, n); ++i)
        {
            p[i] = 0;
        }
        return desired;
    }

    // this PX-thread is run by the wrong OS-thread, make the foreman retry
    return std::size_t(-1);
}
