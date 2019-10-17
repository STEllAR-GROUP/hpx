
#include <iostream>

#include <mpi.h>

void plain_ring()
{
    int rank = 0;
    int size = 0;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int dst_rank = rank == size - 1 ? 0 : rank + 1;
    int src_rank = rank == 0 ? size - 1 : rank - 1;

    int result = 0;
    if (rank == 0)
    {
        MPI_Send(&result, 1, MPI_INT, dst_rank, 0, MPI_COMM_WORLD);
        MPI_Recv(&result, 1, MPI_INT, src_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::cout << "Result: " << result << "\n";
        return;
    }

    MPI_Recv(&result, 1, MPI_INT, src_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    result += rank;
    MPI_Send(&result, 1, MPI_INT, dst_rank, 0, MPI_COMM_WORLD);
}

#include <hpx/lcos/future.hpp>

namespace hpx { namespace mpi
{
    struct mpi_future_data : hpx::lcos::detail::task_base<void>
    {
        HPX_NON_COPYABLE(mpi_future_data);

        using init_no_addref = hpx::lcos::detail::task_base<void>::init_no_addref;

        mpi_future_data() = default;

        mpi_future_data(init_no_addref no_addref)
          : hpx::lcos::detail::task_base<void>(no_addref)
        {}

        void do_run()
        {
            boost::intrusive_ptr<mpi_future_data> this_(this);
            std::thread progress([this_]()
            {
                MPI_Status status;
                MPI_Wait(&this_->request, &status);
                this_->set_data(hpx::util::unused);
            });
            progress.detach();
        }

        MPI_Request request;
    };

    template <typename F, typename ...Ts>
    hpx::future<void> invoke(F f, Ts &&...ts)
    {
        boost::intrusive_ptr<mpi_future_data> data =
            new mpi_future_data(mpi_future_data::init_no_addref{});
        f(std::forward<Ts>(ts)..., &data->request);

        using traits::future_access;
        return future_access<hpx::future<void>>::create(std::move(data));
    }
}}

void invoke_ring()
{
    int rank = 0;
    int size = 0;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int dst_rank = rank == size - 1 ? 0 : rank + 1;
    int src_rank = rank == 0 ? size - 1 : rank - 1;

    int result = 0;

    if (rank == 0)
    {
        MPI_Send(&result, 1, MPI_INT, dst_rank, 0, MPI_COMM_WORLD);
    }

    auto f = hpx::mpi::invoke(MPI_Irecv, &result, 1, MPI_INT, src_rank, 0, MPI_COMM_WORLD);

    f.then([&](auto f) mutable
    {
        f.get(); // propagate exceptions...
        if (rank == 0)
        {
            std::cout << "Result: " << result << "\n";
            return hpx::make_ready_future();
        }

        result += rank;

        return hpx::mpi::invoke(MPI_Isend, &result, 1, MPI_INT, dst_rank, 0, MPI_COMM_WORLD);
    }).get();
}

#include <hpx/basic_execution/default_agent.hpp>
#include <hpx/basic_execution/default_context.hpp>
#include <exception>
namespace hpx { namespace openmp {
    struct context : hpx::basic_execution::default_context
    {
        void post(hpx::util::unique_function_nonser<void()> f) const override;
    };

    struct agent : hpx::basic_execution::default_agent
    {
        hpx::openmp::context const& context() const override
        {
            return context_;
        }

        static hpx::openmp::context context_;
    };

    struct frame
    {
        hpx::util::unique_function_nonser<void()> f;
    };

    void context::post(hpx::util::unique_function_nonser<void()> f) const
    {
        auto ff = new frame{std::move(f)};
#pragma omp task firstprivate(ff)
        {
            hpx::openmp::agent agnt;
            hpx::basic_execution::this_thread::reset_agent r(agnt);
            std::exception_ptr eptr;
            try
            {
                ff->f();
            }
            catch (...)
            {
                eptr = std::current_exception();
            }
            delete ff;
            if (eptr) std::rethrow_exception(eptr);
        }
    }

    hpx::openmp::context agent::context_;
}}

#include <hpx/async.hpp>

void openmp()
{
    hpx::future<int> f;
#pragma omp parallel
    {
        hpx::openmp::agent agnt;
        hpx::basic_execution::this_thread::reset_agent r(agnt);

        f = hpx::async([]()
        {
            std::cout << "Hello" << " World\n";
            return 42;
        });
    }

    std::cout << "Result: " << f.get();
}

int main(int argc, char**argv)
{
//     MPI_Init(&argc, &argv);

//     plain_ring();
//     invoke_ring();
    openmp();

//     MPI_Finalize();
}
