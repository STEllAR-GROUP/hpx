//  Copyright (c) 2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <examples/mini_ghost/stepper.hpp>
#include <examples/mini_ghost/stencils.hpp>

#include <hpx/lcos/wait_all.hpp>

typedef hpx::components::managed_component<
    mini_ghost::stepper<float>
> stepper_float_type;

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(stepper_float_type, stepper_float);

typedef hpx::components::managed_component<
    mini_ghost::stepper<double>
> stepper_double_type;

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(stepper_double_type, stepper_double);

namespace mini_ghost {

    template <typename Real>
    stepper<Real>::stepper()
      : gen(0)
      , random(0.0, 1.0)
      , rank(-1)
    {}

    template <typename Real>
    void stepper<Real>::init(params<Real> & p)
    {
        nx = p.nx;
        ny = p.ny;
        nz = p.nz;
        npx = p.npx;
        npy = p.npy;
        npz = p.npz;
        rank = p.rank;
        stencil = p.stencil;
        num_vars = p.num_vars;
        report_diffusion = p.report_diffusion;
        error_tol = p.error_tol;
        source_total.resize(p.num_vars, 0.0);
        global_sums.resize(p.num_vars);
        flux_out.resize(p.num_vars, 0.0);
        setup_communication_parameter(p);
        setup_global_indices(p);
        setup_spikes(p);
        //FIXME: setup performance captures...
        src = 0;
        dst = 1;
        setup_grids(p);

        // First round of boundaries ...
        for(std::size_t var = 0; var != num_vars; ++var)
        {
            send_buffer_norths[var](grids[src][var], 1, var);
            send_buffer_souths[var](grids[src][var], 1, var);
            send_buffer_easts [var](grids[src][var], 1, var);
            send_buffer_wests [var](grids[src][var], 1, var);
            send_buffer_fronts[var](grids[src][var], 1, var);
            send_buffer_backs [var](grids[src][var], 1, var);
        }

        if(p.rank == 0)
        {
            print_header(p);
        }
    }

    template <typename Real>
    void stepper<Real>::run(std::size_t num_spikes, std::size_t num_tsteps)
    {
        hpx::util::high_resolution_timer timer_all;
        for(std::size_t spike = 0; spike != num_spikes; ++spike)
        {
            insert_spike(spike);

            std::vector<hpx::future<void> > sum_futures(num_vars);
            //std::fill(sum_futures.begin(), sum_futures.end(), hpx::make_ready_future());
            std::size_t generation = 0;
            for(std::size_t step = 1; step != num_tsteps+1; ++step)
            {
                // Report sum results for step-1
                for(std::size_t var = 0; var != num_vars; ++var)
                {
                    hpx::util::high_resolution_timer timer;
                    // Wait on the sum reduction ...
                    if(sum_futures[var].valid())
                        sum_futures[var].wait();

                    // Receive boundaries ...
                    recv_buffer_souths[var](grids[src][var], step);
                    recv_buffer_norths[var](grids[src][var], step);
                    recv_buffer_wests [var](grids[src][var], step);
                    recv_buffer_easts [var](grids[src][var], step);
                    recv_buffer_backs [var](grids[src][var], step);
                    recv_buffer_fronts[var](grids[src][var], step);

                    flux_accumulate(var);
                    //FIXME: do stencil update ...
                    switch (stencil)
                    {
                        case STENCIL_NONE:
                            stencils<STENCIL_NONE>::call(
                                grids[dst][var], grids[src][var]);
                            break;
                        case STENCIL_2D5PT:
                            stencils<STENCIL_2D5PT>::call(
                                grids[dst][var], grids[src][var]);
                            break;
                        case STENCIL_2D9PT:
                            stencils<STENCIL_2D9PT>::call(
                                grids[dst][var], grids[src][var]);
                            break;
                        case STENCIL_3D7PT:
                            stencils<STENCIL_3D7PT>::call(
                                grids[dst][var], grids[src][var]);
                            break;
                        case STENCIL_3D27PT:
                            stencils<STENCIL_3D27PT>::call(
                                grids[dst][var], grids[src][var]);
                            break;
                        default:
                            std::cerr << "Unkown stencil\n";
                            hpx::terminate();
                            break;
                    }

                    // Send boundaries ...
                    send_buffer_norths[var](grids[dst][var], step+1, var);
                    send_buffer_souths[var](grids[dst][var], step+1, var);
                    send_buffer_easts [var](grids[dst][var], step+1, var);
                    send_buffer_wests [var](grids[dst][var], step+1, var);
                    send_buffer_fronts[var](grids[dst][var], step+1, var);
                    send_buffer_backs [var](grids[dst][var], step+1, var);

                    if(grids_to_sum[var])
                    {
                        hpx::util::high_resolution_timer time_reduction;
                        // Sum grid ...
                        Real sum = 0;
                        for(Real & value : grids[dst][var].data_)
                        {
                            sum += value;
                        }
                        sum += flux_out[var];

                        sum_futures[var] =
                            global_sums[var].add(stepper_ids.size(), generation).then(
                                hpx::launch::sync
                              , [this, var, step](hpx::future<Real> value)
                                {
                                    if(rank == 0)
                                    {
                                        Real error_iter
                                            = std::abs(source_total[var] - value.get()) / source_total[var];
                                        bool terminate = error_iter > error_tol;

                                        if(terminate || (step % report_diffusion == 0))
                                        {
                                            std::cout
                                                << "Time step " << step << " "
                                                << "for variable " << var << " "
                                                << "the error is " << error_iter << "; "
                                                << "error tolerance is " << error_tol << "."
                                                << std::endl;
                                        }
                                        if(terminate) hpx::terminate();
                                    }
                                }
                            );
                        for(hpx::id_type id : stepper_ids)
                        {
                            hpx::apply(set_global_sum_action(), id, var, generation, rank, sum);
                        }
                    }
                    std::swap(src, dst);
                }
            }
        }
    }

    template <typename Real>
    void stepper<Real>::set_north_zone(buffer_type buffer, std::size_t step, std::size_t var)
    {
        recv_buffer_norths[var].set_buffer(buffer, step);
    }

    template <typename Real>
    void stepper<Real>::set_south_zone(buffer_type buffer, std::size_t step, std::size_t var)
    {
        recv_buffer_souths[var].set_buffer(buffer, step);
    }

    template <typename Real>
    void stepper<Real>::set_east_zone(buffer_type buffer, std::size_t step, std::size_t var)
    {
        recv_buffer_easts[var].set_buffer(buffer, step);
    }

    template <typename Real>
    void stepper<Real>::set_west_zone(buffer_type buffer, std::size_t step, std::size_t var)
    {
        recv_buffer_wests[var].set_buffer(buffer, step);
    }

    template <typename Real>
    void stepper<Real>::set_front_zone(buffer_type buffer, std::size_t step, std::size_t var)
    {
        recv_buffer_fronts[var].set_buffer(buffer, step);
    }

    template <typename Real>
    void stepper<Real>::set_back_zone(buffer_type buffer, std::size_t step, std::size_t var)
    {
        recv_buffer_backs[var].set_buffer(buffer, step);
    }

    template <typename Real>
    void stepper<Real>::set_global_sum(std::size_t idx, std::size_t generation, std::size_t which, Real value)
    {
        HPX_ASSERT(which < stepper_ids.size());
        global_sums[idx].set_data(generation, which, value);
    }

    template <typename Real>
    void stepper<Real>::setup_communication_parameter(params<Real> &p)
    {
        // register our own name
        const std::string base_name = "/mini_ghost/stepper/";
        std::string name = base_name;
        name += boost::lexical_cast<std::string>(p.rank);
        hpx::agas::register_name(name, this->get_gid());
        stepper_ids.resize(p.nranks);

        send_buffer_norths.resize(p.num_vars);
        send_buffer_souths.resize(p.num_vars);
        send_buffer_easts.resize (p.num_vars);
        send_buffer_wests.resize (p.num_vars);
        send_buffer_fronts.resize(p.num_vars);
        send_buffer_backs.resize (p.num_vars);

        recv_buffer_norths.resize(p.num_vars);
        recv_buffer_souths.resize(p.num_vars);
        recv_buffer_easts.resize (p.num_vars);
        recv_buffer_wests.resize (p.num_vars);
        recv_buffer_fronts.resize(p.num_vars);
        recv_buffer_backs.resize (p.num_vars);

        stepper_ids[p.rank] = this->get_gid();
        for(std::size_t rank = 0; rank != p.nranks; ++rank)
        {
            name = base_name;
            name += boost::lexical_cast<std::string>(rank);
            while(!stepper_ids[rank])
            {
                stepper_ids[rank] = hpx::agas::resolve_name_sync(name);
            }
        }

        // Set position in 3D processor grid
        std::size_t myrank_xy = p.rank % (p.npx*p.npy);
        std::size_t remainder = myrank_xy % p.npx;
        my_py = myrank_xy / p.npx;
        my_px = 0;
        if(remainder != 0)
        {
            my_px = remainder;
        }
        else
        {
            my_px = 0;
        }
        my_pz = p.rank / (p.npx*p.npz);

        // Set neighbors
        num_neighs = 0;
        if(my_py != 0)
        {
            for(std::size_t var = 0; var != num_vars; ++var)
            {
                send_buffer_souths[var].dest_
                    = stepper_ids[p.rank - p.npx];
                recv_buffer_souths[var].valid_ = true;
            }
            ++num_neighs;
        }
        if(my_py != p.npy-1)
        {
            for(std::size_t var = 0; var != num_vars; ++var)
            {
                send_buffer_norths[var].dest_
                    = stepper_ids[p.rank + p.npx];
                recv_buffer_norths[var].valid_ = true;
            }
            ++num_neighs;
        }
        if(my_px != 0)
        {
            for(std::size_t var = 0; var != num_vars; ++var)
            {
                send_buffer_wests[var].dest_
                    = stepper_ids[p.rank - 1];
                recv_buffer_wests[var].valid_ = true;
            }
            ++num_neighs;
        }
        if(my_px != p.npx-1)
        {
            for(std::size_t var = 0; var != num_vars; ++var)
            {
                send_buffer_easts[var].dest_
                    = stepper_ids[p.rank + 1];
                recv_buffer_easts[var].valid_ = true;
            }
            ++num_neighs;
        }
        if(my_pz != 0)
        {
            for(std::size_t var = 0; var != num_vars; ++var)
            {
                send_buffer_backs[var].dest_
                    = stepper_ids[p.rank - (p.npx*p.npy)];
                recv_buffer_backs[var].valid_ = true;
            }
            ++num_neighs;
        }
        if(my_pz != p.npz-1)
        {
            for(std::size_t var = 0; var != num_vars; ++var)
            {
                send_buffer_fronts[var].dest_
                    = stepper_ids[p.rank + (p.npx*p.npy)];
                recv_buffer_fronts[var].valid_ = true;
            }
            ++num_neighs;
        }
    }

    template <typename Real>
    void stepper<Real>::setup_global_indices(params<Real> & p)
    {
        // Determine global indices (excluding ghost)
        global_nx = p.nx * p.npx;
        global_ny = p.ny * p.npy;
        global_nz = p.nz * p.npz;

        my_global_nx.first = p.nx * my_px + 1;
        my_global_ny.first = p.ny * my_py + 1;
        my_global_nz.first = p.nz * my_pz + 1;

        my_global_nx.second = my_global_nx.first + p.nx - 1;
        my_global_ny.second = my_global_ny.first + p.ny - 1;
        my_global_nz.second = my_global_nz.first + p.nz - 1;
    }

    template <typename Real>
    void stepper<Real>::setup_spikes(params<Real> & p)
    {
        spikes.resize(p.num_vars, p.num_spikes);
        spike_loc.resize(4, p.num_spikes);

        std::size_t global_n = global_nx * global_ny * global_nz;
        for(Real & v : spikes.data_)
        {
            v = random(gen) * global_n;
        }

        spikes(1,0) = static_cast<Real>(global_n);

        spike_loc(0, 0) = std::size_t(-1);
        spike_loc(1, 0) = global_nx / 2;
        spike_loc(2, 0) = global_nx / 2;
        spike_loc(3, 0) = global_nx / 2;

        grid<Real> rspike_loc(3, p.num_spikes);
        for(Real & v : rspike_loc.data_)
        {
            v = random(gen);
        }

        for(std::size_t i = 1; i < p.num_spikes; ++i)
        {
            spike_loc(0, i) = std::size_t(-1);
            spike_loc(1, i) = static_cast<std::size_t>(rspike_loc(0, i) * global_nx);
            spike_loc(2, i) = static_cast<std::size_t>(rspike_loc(1, i) * global_ny);
            spike_loc(3, i) = static_cast<std::size_t>(rspike_loc(2, i) * global_nz);
        }

        for(std::size_t i = 0; i < p.num_spikes; ++i)
        {
            std::size_t xloc = spike_loc(1, i);
            std::size_t yloc = spike_loc(2, i);
            std::size_t zloc = spike_loc(3, i);
            if((my_global_nx.first <= xloc) && (xloc <= my_global_nx.second) &&
               (my_global_ny.first <= yloc) && (yloc <= my_global_ny.second) &&
               (my_global_nz.first <= zloc) && (zloc <= my_global_nz.second))
            {
                spike_loc(0, i) = p.rank;
                spike_loc(1, i) = spike_loc(1, i) - my_global_nx.first;
                spike_loc(2, i) = spike_loc(2, i) - my_global_ny.first;
                spike_loc(3, i) = spike_loc(3, i) - my_global_nz.first;
            }
            else
            {
                spike_loc(0, i) = std::size_t(-1);
                spike_loc(1, i) = std::size_t(-1);
                spike_loc(2, i) = std::size_t(-1);
                spike_loc(3, i) = std::size_t(-1);
            }
        }
    }

    template <typename Real>
    void stepper<Real>::setup_grids(params<Real> & p)
    {
        std::vector<hpx::future<void> > sum_futures;
        grids[src].resize(p.num_vars);
        grids[dst].resize(p.num_vars);
        if(p.debug_grid)
        {
            for(auto & grid : grids[src])
            {
                grid.resize(p.nx+1, p.ny+1, p.nz+1);
                for(Real & value : grid.data_)
                {
                    value = 0.0;
                }
            }
            for(auto & grid : grids[dst])
            {
                grid.resize(p.nx+1, p.ny+1, p.nz+1);
            }
        }
        else
        {
            std::size_t generation = 0;
            std::size_t i = 0;
            sum_futures.reserve(p.num_vars);
            for(auto & grid : grids[src])
            {
                grid.resize(p.nx+1, p.ny+1, p.nz+1);
                Real sum = 0;
                for(Real & value : grid.data_)
                {
                    value = random(gen);
                    sum += value;
                }
                sum_futures.push_back(
                    global_sums[i].add(p.nranks, generation).then(
                        [this, i](hpx::future<Real> value)
                        {
                            source_total[i] = value.get();
                        }
                    )
                );
                for(hpx::id_type id : stepper_ids)
                {
                    hpx::apply(set_global_sum_action(), id, i, generation, p.rank, sum);
                }
                ++i;
            }
            for(auto & grid : grids[dst])
            {
                grid.resize(p.nx+1, p.ny+1, p.nz+1);
            }
        }

        // Buff Init
        /////////////////////////////////////////////////////////////////////////
        num_sum_grid = 0;
        grids_to_sum.resize(p.num_vars, false);
        if(p.percent_sum == 100)
        {
            std::fill(grids_to_sum.begin(), grids_to_sum.end(), true);
            num_sum_grid = p.num_vars;
        }
        else if(p.percent_sum > 0)
        {
            double percent_sum = p.percent_sum / 100.0;
            for(std::size_t i = 0; i < p.num_vars; ++i)
            {
                double num = random(gen);
                if(num < percent_sum)
                {
                    grids_to_sum[i] = true;
                    ++num_sum_grid;
                }
            }
        }
        // Buff Init done ...
        /////////////////////////////////////////////////////////////////////////
        hpx::lcos::wait_all(sum_futures);
    }

    template <typename Real>
    void stepper<Real>::insert_spike(std::size_t spike)
    {
        if(rank == spike_loc(0, spike))
        {
            std::size_t ix = spike_loc(1, spike);
            std::size_t iy = spike_loc(2, spike);
            std::size_t iz = spike_loc(3, spike);

            std::size_t idx = 0;
            for(auto & grid : grids[src])
            {
                grid(ix, iy, iz) = spikes(idx++, spike);
            }
        }

        if(rank == 0)
        {
            for(std::size_t var = 0; var != num_vars; ++var)
            {
                source_total[var] = source_total[var] + spikes(var, spike);
            }
        }
    }

    template <typename Real>
    void stepper<Real>::flux_accumulate(std::size_t var)
    {
        Real divisor = 0.0;
        switch (stencil)
        {
            case STENCIL_2D5PT:
                divisor = 1.0f/5.0f;
                break;
            case STENCIL_2D9PT:
                divisor = 1.0f/9.0f;
                break;
            case STENCIL_3D7PT:
                divisor = 1.0f/7.0f;
                break;
            case STENCIL_3D27PT:
                divisor = 1.0f/27.0f;
                break;
        }

        if(my_px == 0)
        {
            for(std::size_t z = 1; z != nz; ++z)
            {
                for(std::size_t y = 1; y != ny; ++y)
                {
                    flux_out[var] += grids[src][var](1, y, z) * divisor;
                }
            }
        }

        if(my_px == npx - 1)
        {
            for(std::size_t z = 1; z != nz; ++z)
            {
                for(std::size_t y = 1; y != ny; ++y)
                {
                    flux_out[var] += grids[src][var](nx, y, z) * divisor;
                }
            }
        }

        if(my_py == 0)
        {
            for(std::size_t z = 1; z != nz; ++z)
            {
                for(std::size_t x = 1; x != nx; ++x)
                {
                    flux_out[var] += grids[src][var](x, 1, z) * divisor;
                }
            }
        }

        if(my_py == npy - 1)
        {
            for(std::size_t z = 1; z != nz; ++z)
            {
                for(std::size_t x = 1; x != nx; ++x)
                {
                    flux_out[var] += grids[src][var](x, ny, z) * divisor;
                }
            }
        }

        if(my_pz == 0)
        {
            for(std::size_t y = 1; y != ny; ++y)
            {
                for(std::size_t x = 1; x != nx; ++x)
                {
                    flux_out[var] += grids[src][var](x, y, 1) * divisor;
                }
            }
        }

        if(my_py == npy - 1)
        {
            for(std::size_t y = 1; y != ny; ++y)
            {
                for(std::size_t x = 1; x != nx; ++x)
                {
                    flux_out[var] += grids[src][var](x, y, nz) * divisor;
                }
            }
        }
    }

    template <typename Real>
    void stepper<Real>::print_header(params<Real> & p)
    {
        std::cout
        << "\n"
        << "======================================================================\n"
        << "\n"
        << "        Mantevo miniapp MiniGhost experiment\n"
        << "        HPX port\n"
        << "\n"
        << "======================================================================\n"
        << "\n";
        switch (p.stencil)
        {
            case STENCIL_NONE:
                std::cout << "No computation inserted\n";
                break;
            case STENCIL_2D5PT:
                std::cout << "Computation: 5 pt difference stencil on a 2D grid (STENCIL_2D5PT)\n";
                break;
            case STENCIL_2D9PT:
                std::cout << "Computation: 9 pt difference stencil on a 2D grid (STENCIL_2D9PT)\n";
                break;
            case STENCIL_3D7PT:
                std::cout << "Computation: 7 pt difference stencil on a 3D grid (STENCIL_3D27PT)\n";
                break;
            case STENCIL_3D27PT:
                std::cout << "Computation: 27 pt difference stencil on a 3D grid (STENCIL_3D27PT)\n";
                break;
            default:
                std::cout << "** Warning ** Unkown compuation\n";
                break;
        }
        std::cout << std::endl;

        std::cout << "        Global Grid Dimension: "
            << p.nx * p.npx << ", " << p.ny * p.npy << ", " << p.nz * p.npz << "\n";
        std::cout << "        Local Grid Dimension : "
            << p.nx << ", " << p.ny << ", " << p.nz << "\n";
        std::cout << std::endl;

        std::cout << "Number of variables: " << p.num_vars << "\n";
        std::cout << std::endl;

        std::cout
            << "Error reported every " << p.report_diffusion << " time steps. "
            << "Tolerance is " << p.error_tol << "\n";
        std::cout
            << "Number of variables reduced each time step: " << num_sum_grid
            << "; requested " << p.percent_sum << "%\n";
        std::cout << std::endl;

        std::cout << "        Time Steps: " << p.num_tsteps << "\n";
        std::cout << "        Task grid : " << p.npx << ", " << p.npy << ", " << p.npz << "\n";
        std::cout << std::endl;

        switch (p.scaling)
        {
            case SCALING_WEAK:
                std::cout << "HPX version, weak scaling\n";
                break;
            case SCALING_STRONG:
                std::cout << "HPX version, weak scaling\n";
                break;
            default:
                std::cout << "HPX version, unkown scaling\n";
                hpx::terminate();
                break;
        }

        if(p.nranks == 1)
        {
            std::cout << "1 process executing\n";
        }
        else
        {
            std::cout << p.nranks << " processes executing\n";
        }
        std::cout << std::endl;

        using namespace boost::posix_time;
        using namespace boost::gregorian;
        boost::posix_time::ptime now = boost::posix_time::second_clock::local_time();
        std::cout << "Program execution date " << to_simple_string(now) << "\n";
        std::cout << std::endl;
    }

    template struct stepper<float>;
    template struct stepper<double>;
}
