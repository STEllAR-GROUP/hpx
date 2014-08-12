//  Copyright (c) 2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_EXAMPLES_MINI_GHOST_SPIKES_HPP
#define HPX_EXAMPLES_MINI_GHOST_SPIKES_HPP

#include <examples/mini_ghost/grid.hpp>

namespace mini_ghost {
    template <typename Real>
    struct spikes
    {
        spikes() {}

        template <typename Random>
        spikes(
            params<Real> & p
          , std::size_t my_px
          , std::size_t my_py
          , std::size_t my_pz
          , Random const & random
        )
          : num_spikes_(p.num_spikes)
          , spikes_(p.num_vars, p.num_spikes)
          , spike_loc_(4, p.num_spikes)
        {
            // Determine global indices (excluding ghost)
            std::size_t first_nx = p.nx * my_px + 1;
            std::size_t first_ny = p.ny * my_py + 1;
            std::size_t first_nz = p.nz * my_pz + 1;

            typedef std::pair<std::size_t, std::size_t> range_type;

            range_type my_global_nx(first_nx, first_nx + p.nx + 1);
            range_type my_global_ny(first_ny, first_ny + p.ny + 1);
            range_type my_global_nz(first_nz, first_nz + p.nz + 1);

            std::size_t global_nx = p.nx * p.npx;
            std::size_t global_ny = p.ny * p.npy;
            std::size_t global_nz = p.nz * p.npz;
            std::size_t global_n = global_nx * global_ny * global_nz;

            for(Real & v : spikes_.data_)
            {
                v = random() * global_n;
            }

            spikes_(0,0) = static_cast<Real>(global_n);

            spike_loc_(0, 0) = std::size_t(-1);
            spike_loc_(1, 0) = global_nx / 2;
            spike_loc_(2, 0) = global_ny / 2;
            spike_loc_(3, 0) = global_nz / 2;

            grid<Real> rspike_loc(3, p.num_spikes);
            for(Real & v : rspike_loc.data_)
            {
                v = random();
            }

            for(std::size_t i = 1; i < p.num_spikes; ++i)
            {
                spike_loc_(0, i) = std::size_t(-1);
                spike_loc_(1, i) = static_cast<std::size_t>(rspike_loc(0, i) * global_nx);
                spike_loc_(2, i) = static_cast<std::size_t>(rspike_loc(1, i) * global_ny);
                spike_loc_(3, i) = static_cast<std::size_t>(rspike_loc(2, i) * global_nz);
            }

            for(std::size_t i = 0; i < p.num_spikes; ++i)
            {
                std::size_t xloc = spike_loc_(1, i);
                std::size_t yloc = spike_loc_(2, i);
                std::size_t zloc = spike_loc_(3, i);

                if((my_global_nx.first <= xloc) && (xloc <= my_global_nx.second) &&
                   (my_global_ny.first <= yloc) && (yloc <= my_global_ny.second) &&
                   (my_global_nz.first <= zloc) && (zloc <= my_global_nz.second))
                {
                    spike_loc_(0, i) = p.rank;
                    spike_loc_(1, i) = spike_loc_(1, i) - my_global_nx.first;
                    spike_loc_(2, i) = spike_loc_(2, i) - my_global_ny.first;
                    spike_loc_(3, i) = spike_loc_(3, i) - my_global_nz.first;
                }
                else
                {
                    spike_loc_(0, i) = std::size_t(-1);
                    spike_loc_(1, i) = std::size_t(-1);
                    spike_loc_(2, i) = std::size_t(-1);
                    spike_loc_(3, i) = std::size_t(-1);
                }
            }
        }

        Real insert(grid<Real> & g, std::size_t var, std::size_t rank, std::size_t spike) const
        {
            Real value = spikes_(var, spike);
            if(rank == spike_loc_(0, spike))
            {
                std::size_t ix = spike_loc_(1, spike);
                std::size_t iy = spike_loc_(2, spike);
                std::size_t iz = spike_loc_(3, spike);

                g(ix, iy, iz) = value;
            }

            return value;
        }

        std::size_t num_spikes_;
        grid<Real> spikes_;
        grid<std::size_t> spike_loc_;
    };
}

#endif
