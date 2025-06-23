#include <hpx/experimental/run_on_all.hpp>
#include <hpx/hpx_main.hpp>
#include <hpx/modules/runtime_local.hpp>
int main()
{
    hpx::experimental::run_on_all([](std::size_t thread_index) {
        std::cout << "Hola desde tarea " << thread_index
                  << " ejecutandose en el hilo HPX " << hpx::get_worker_thread_num()
                  << std::endl;
    });
    
    return 0;
}
