
//  Copyright (c) 2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "create_grid_dim.hpp"
#include <mpi.h>

#include "grid.hpp"
#include <cmath>

#include <hpx/hpx.hpp>
#include <hpx/util/high_resolution_timer.hpp>


#include <iostream>

using std::cout;
using std::flush;

using hpx::util::high_resolution_timer;

using bright_future::range_type;
using bright_future::jacobi_kernel_simple;

typedef bright_future::grid<double> grid_type;

void copy_send_buf(int dir, int disp, std::size_t y_start, std::size_t y_end, std::size_t x_start, std::size_t x_end, std::size_t src, std::vector<grid_type> const & u, std::vector<double> & send_buf, int * local_dim)
{
    // copy send buffer:
    switch (dir)
    {
        // x direction:
        case 0:
            switch(disp)
            {
                // left
                case -1:
                    {
                        std::size_t i = 0;
                        for(std::size_t y = y_start; y < y_end; ++y)
                        {
                            send_buf[i++] = u[src](0, y);
                        }
                    }
                    break;
                // right
                case 1:
                    {
                        std::size_t i = 0;
                        for(std::size_t y = y_start; y < y_end; ++y)
                        {
                            send_buf[i++] = u[src](local_dim[0], y);
                        }
                    }
                    break;
            }
            break;
        // y direction:
        case 1:
            switch(disp)
            {
                // bottom
                case -1:
                    {
                        std::size_t i = 0;
                        for(std::size_t x = x_start; x < x_end; ++x)
                        {
                            send_buf[i++] = u[src](x, 0);
                        }
                    }
                    break;
                // top
                case 1:
                    {
                        std::size_t i = 0;
                        for(std::size_t x = x_start; x < x_end; ++x)
                        {
                            send_buf[i++] = u[src](x, local_dim[1]);
                        }
                    }
                    break;
            }
            break;
    }
}

void copy_recv_buf(int dir, int disp, std::size_t y_start, std::size_t y_end, std::size_t x_start, std::size_t x_end, std::size_t src, std::vector<grid_type> & u, std::vector<double> const & recv_buf, int * local_dim)
{
    // copy send buffer:
    switch (dir)
    {
        // x direction:
        case 0:
            switch(disp)
            {
                // left
                case -1:
                    {
                        std::size_t i = 0;
                        for(std::size_t y = y_start; y < y_end; ++y)
                        {
                            u[src](0, y) = recv_buf[i++];
                        }
                    }
                    break;
                // right
                case 1:
                    {
                        std::size_t i = 0;
                        for(std::size_t y = y_start; y < y_end; ++y)
                        {
                             u[src](local_dim[0], y) = recv_buf[i++];
                        }
                    }
                    break;
            }
            break;
        // y direction:
        case 1:
            switch(disp)
            {
                // bottom
                case -1:
                    {
                        std::size_t i = 0;
                        for(std::size_t x = x_start; x < x_end; ++x)
                        {
                             u[src](x, 0) = recv_buf[i++];
                        }
                    }
                    break;
                // top
                case 1:
                    {
                        std::size_t i = 0;
                        for(std::size_t x = x_start; x < x_end; ++x)
                        {
                             u[src](x, local_dim[1]) = recv_buf[i++];
                        }
                    }
                    break;
            }
            break;
    }
}


void gs(
    std::size_t n_x
  , std::size_t n_y
  , double hx_
  , double hy_
  , double k_
  , double relaxation_
  , unsigned max_iterations
  , unsigned iteration_block
  , unsigned block_size
  , std::size_t cache_block
  , std::string const & output
)
{
    int myrank = 0;
    int numprocs = 0;

    std::size_t spat_dim[] = {n_x-1, n_y-1};
    int proc_dim[] = {0,0};

    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

    MPI_Dims_create(numprocs, 2, proc_dim);

    if(myrank == 0)
    {
        std::cout << "Grid: " << proc_dim[0] << "x" << proc_dim[1] << "\n";
        std::cout << "(" << n_x-1 << "x" << n_y-1 << ")\n";
    }

    MPI_Comm grid_comm = MPI_COMM_NULL;
    {
        int pbc_check[] = {false, false};
        int reorder = true;
        MPI_Cart_create(MPI_COMM_WORLD, 2, proc_dim, pbc_check, reorder, &grid_comm);
    }

    if(grid_comm == MPI_COMM_NULL)
    {
        return;
    }

    int myrank_grid = 0;
    int numprocs_grid = 0;

    MPI_Comm_rank(grid_comm, &myrank_grid);
    MPI_Comm_size(grid_comm, &numprocs_grid);

    int local_dim[] = {0, 0};
    int my_coord[] = {0, 0};

    MPI_Cart_coords(grid_comm, myrank_grid, 2, my_coord);

    for(int i = 0; i < 2; ++i)
    {
        local_dim[i] = spat_dim[i] / proc_dim[i];
        if(static_cast<std::size_t>(my_coord[i]) < spat_dim[i] % proc_dim[i])
            local_dim[i] = local_dim[0] + 1;
    }

    std::size_t x_start = 0; std::size_t x_end = local_dim[0] + 1;
    std::size_t y_start = 0; std::size_t y_end = local_dim[1] + 1;

    /*
    std::cout
        << "grid " << my_coord[0] << "x" << my_coord[1] << "(" << myrank_grid << "):\n " << x_start << "x" << x_end << "\n"
        << " " << y_start << "x" << y_end << "\n"
        << " " << local_dim[0] << "x" << local_dim[1] << "\n";
        */

    std::vector<grid_type> u(2, grid_type(local_dim[0] + 1 , local_dim[1] + 1, block_size, 1));


    std::size_t max_buf_len = std::max(local_dim[0] + 1, local_dim[1] + 1);

    std::vector<double> send_buf(max_buf_len);
    std::vector<double> recv_buf(max_buf_len);


    high_resolution_timer t;
    std::size_t src = 0;
    std::size_t dst = 1;
    if(myrank == 0)
        t.restart();
    for(std::size_t iter = 0; iter < max_iterations; ++iter)
    {
        int disp_value[2] = {-1,1};
        int dir_value[2] = {0,1};

        for(std::size_t disp_i = 0; disp_i < 2; ++disp_i)
        {
            int disp = disp_value[disp_i];
            for(std::size_t dir_i = 0; dir_i < 2; ++dir_i)
            {
                int dir = dir_value[dir_i];

                int source = 0;
                int dest = 0;

                MPI_Request req;
                MPI_Cart_shift(grid_comm, dir, disp, &source, &dest);

                if(source != MPI_PROC_NULL)
                {
                    MPI_Irecv(&recv_buf[0], local_dim[dir] + 1, MPI_DOUBLE_PRECISION, source, 0, grid_comm, &req);
                }

                if(dest != MPI_PROC_NULL)
                {
                    copy_send_buf(dir, disp, y_start, y_end, x_start, x_end, src, u, send_buf, local_dim);
                    MPI_Send(&send_buf[0], local_dim[dir] + 1, MPI_DOUBLE_PRECISION, dest, 0, grid_comm);
                }

                if(source != MPI_PROC_NULL)
                {
                    MPI_Wait(&req, MPI_STATUS_IGNORE);
                    copy_recv_buf(dir, disp, y_start, y_end, x_start, x_end, src, u, recv_buf, local_dim);
                }
            }
        }
        
#pragma omp parallel for shared(u) schedule(static)
        for(std::size_t y_block = 1; y_block < static_cast<std::size_t>(local_dim[1]); y_block += block_size)
        {
            std::size_t y_end = (std::min)(y_block + block_size, static_cast<std::size_t>(local_dim[1]));
            for(std::size_t x_block = 1; x_block < static_cast<std::size_t>(local_dim[0]); x_block += block_size)
            {
                std::size_t x_end = (std::min)(x_block + block_size, static_cast<std::size_t>(local_dim[0]));
                jacobi_kernel_simple(
                    u
                  , range_type(x_block, x_end)
                  , range_type(y_block, y_end)
                  , src, dst
                  , cache_block
                );
            }
        }
        std::swap(src, dst);
    }
    
    if(myrank == 0)
    {
        double time_elapsed = t.elapsed();
        cout << n_x-1 << "x" << n_y-1 << " " << ((((n_x-2)*(n_y-2) * max_iterations)/1e6)/time_elapsed) << " MLUP/S\n" << flush;
    }

}
