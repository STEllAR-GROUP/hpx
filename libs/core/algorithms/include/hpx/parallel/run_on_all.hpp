
#pragma once

#include <hpx/hpx.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/lcos.hpp>
#include <iostream>
#include <vector>

namespace hpx::experimental {

    // Ejecuta una función N veces en paralelo usando hpx::async
    template <typename F>
    void run_on_all(std::size_t num_tasks, F&& f)
    {
        std::vector<hpx::future<void>> futures;

        for (std::size_t i = 0; i < num_tasks; ++i)
        {
            futures.push_back(hpx::async([=]() {
                f(i);    // Pasamos el índice como argumento
            }));
        }

        hpx::wait_all(futures);    // Esperamos a que terminen todas las tareas
    }

    // Sobrecarga para usar todos los hilos disponibles automáticamente
    template <typename F>
    void run_on_all(F&& f)
    {
        std::size_t num_threads = hpx::get_num_worker_threads();
        run_on_all(num_threads, HPX_FORWARD(F, f));
    }

}

