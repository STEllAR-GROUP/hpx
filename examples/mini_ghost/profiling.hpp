//  Copyright (c) 2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_EXAMPLES_MINI_PROFILING_HPP
#define HPX_EXAMPLES_MINI_PROFILING_HPP

#include <examples/mini_ghost/params.hpp>

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/count.hpp>
#include <boost/accumulators/statistics/min.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/max.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/variance.hpp>
#include <boost/accumulators/statistics/moment.hpp>

#include <boost/serialization/vector.hpp>

namespace mini_ghost
{
    struct profiling
    {
        struct profiling_data
        {
            std::vector<std::size_t> num_adds_;
            void num_adds(std::size_t value)
            {
                num_adds_[hpx::get_worker_thread_num()] += value;
            }
            std::size_t num_adds() const
            {
                std::size_t res = 0;
                for(auto v : num_adds_) { res += v; }
                return res;
            }

            std::vector<std::size_t> num_divides_;
            void num_divides(std::size_t value)
            {
                num_divides_[hpx::get_worker_thread_num()] += value;
            }
            std::size_t num_divides() const
            {
                std::size_t res = 0;
                for(auto v : num_divides_) { res += v; }
                return res;
            }

            std::vector<std::size_t> num_sumgrid_;
            void num_sumgrid(std::size_t value)
            {
                num_sumgrid_[hpx::get_worker_thread_num()] += value;
            }
            std::size_t num_sumgrid() const
            {
                std::size_t res = 0;
                for(auto v : num_sumgrid_) { res += v; }
                return res;
            }

            std::vector<double> time_init_;
            void time_init(double value)
            {
                time_init_[hpx::get_worker_thread_num()] += value;
            }
            double time_init() const
            {
                double res = 0;
                for(auto v : time_init_) { res += v; }
                return res;
            }

            std::vector<double> time_hpx_;
            void time_hpx(double value)
            {
                time_hpx_[hpx::get_worker_thread_num()] += value;
            }
            double time_hpx() const
            {
                double res = 0;
                for(auto v : time_hpx_) { res += v; }
                return res;
            }

            std::vector<double> time_pack_;
            void time_pack(double value)
            {
                time_pack_[hpx::get_worker_thread_num()] += value;
            }
            double time_pack() const
            {
                double res = 0;
                for(auto v : time_pack_) { res += v; }
                return res;
            }

            std::vector<double> time_recv_;
            void time_recv(double value)
            {
                time_recv_[hpx::get_worker_thread_num()] += value;
            }
            double time_recv() const
            {
                double res = 0;
                for(auto v : time_recv_) { res += v; }
                return res;
            }

            std::vector<double> time_send_;
            void time_send(double value)
            {
                time_send_[hpx::get_worker_thread_num()] += value;
            }
            double time_send() const
            {
                double res = 0;
                for(auto v : time_send_) { res += v; }
                return res;
            }

            std::vector<double> time_stencil_;
            void time_stencil(double value)
            {
                time_stencil_[hpx::get_worker_thread_num()] += value;
            }
            double time_stencil() const
            {
                double res = 0;
                for(auto v : time_stencil_) { res += v; }
                return res;
            }

            std::vector<double> time_unpack_;
            void time_unpack(double value)
            {
                time_unpack_[hpx::get_worker_thread_num()] += value;
            }
            double time_unpack() const
            {
                double res = 0;
                for(auto v : time_unpack_) { res += v; }
                return res;
            }

            std::vector<double> time_wait_;
            void time_wait(double value)
            {
                time_wait_[hpx::get_worker_thread_num()] += value;
            }
            double time_wait() const
            {
                double res = 0;
                for(auto v : time_wait_) { res += v; }
                return res;
            }

            std::vector<double> time_wall_;
            void time_wall(double value)
            {
                time_wall_[hpx::get_worker_thread_num()] += value;
            }
            double time_wall() const
            {
                double res = 0;
                for(auto v : time_wall_) { res += v; }
                return res;
            }


            std::vector<double> time_pack_x_;
            void time_pack_x(double value)
            {
                time_pack_x_[hpx::get_worker_thread_num()] += value;
            }
            double time_pack_x() const
            {
                double res = 0;
                for(auto v : time_pack_x_) { res += v; }
                return res;
            }

            std::vector<double> time_send_x_;
            void time_send_x(double value)
            {
                time_send_x_[hpx::get_worker_thread_num()] += value;
            }
            double time_send_x() const
            {
                double res = 0;
                for(auto v : time_send_x_) { res += v; }
                return res;
            }

            std::vector<double> time_wait_x_;
            void time_wait_x(double value)
            {
                time_wait_x_[hpx::get_worker_thread_num()] += value;
            }
            double time_wait_x() const
            {
                double res = 0;
                for(auto v : time_wait_x_) { res += v; }
                return res;
            }

            std::vector<double> time_recv_x_;
            void time_recv_x(double value)
            {
                time_recv_x_[hpx::get_worker_thread_num()] += value;
            }
            double time_recv_x() const
            {
                double res = 0;
                for(auto v : time_recv_x_) { res += v; }
                return res;
            }

            std::vector<double> time_unpack_x_;
            void time_unpack_x(double value)
            {
                time_unpack_x_[hpx::get_worker_thread_num()] += value;
            }
            double time_unpack_x() const
            {
                double res = 0;
                for(auto v : time_unpack_x_) { res += v; }
                return res;
            }


            std::vector<double> time_pack_y_;
            void time_pack_y(double value)
            {
                time_pack_y_[hpx::get_worker_thread_num()] += value;
            }
            double time_pack_y() const
            {
                double res = 0;
                for(auto v : time_pack_y_) { res += v; }
                return res;
            }

            std::vector<double> time_send_y_;
            void time_send_y(double value)
            {
                time_send_y_[hpx::get_worker_thread_num()] += value;
            }
            double time_send_y() const
            {
                double res = 0;
                for(auto v : time_send_y_) { res += v; }
                return res;
            }

            std::vector<double> time_wait_y_;
            void time_wait_y(double value)
            {
                time_wait_y_[hpx::get_worker_thread_num()] += value;
            }
            double time_wait_y() const
            {
                double res = 0;
                for(auto v : time_wait_y_) { res += v; }
                return res;
            }

            std::vector<double> time_recv_y_;
            void time_recv_y(double value)
            {
                time_recv_y_[hpx::get_worker_thread_num()] += value;
            }
            double time_recv_y() const
            {
                double res = 0;
                for(auto v : time_recv_y_) { res += v; }
                return res;
            }

            std::vector<double> time_unpack_y_;
            void time_unpack_y(double value)
            {
                time_unpack_y_[hpx::get_worker_thread_num()] += value;
            }
            double time_unpack_y() const
            {
                double res = 0;
                for(auto v : time_unpack_y_) { res += v; }
                return res;
            }


            std::vector<double> time_pack_z_;
            void time_pack_z(double value)
            {
                time_pack_z_[hpx::get_worker_thread_num()] += value;
            }
            double time_pack_z() const
            {
                double res = 0;
                for(auto v : time_pack_z_) { res += v; }
                return res;
            }

            std::vector<double> time_send_z_;
            void time_send_z(double value)
            {
                time_send_z_[hpx::get_worker_thread_num()] += value;
            }
            double time_send_z() const
            {
                double res = 0;
                for(auto v : time_send_z_) { res += v; }
                return res;
            }

            std::vector<double> time_wait_z_;
            void time_wait_z(double value)
            {
                time_wait_z_[hpx::get_worker_thread_num()] += value;
            }
            double time_wait_z() const
            {
                double res = 0;
                for(auto v : time_wait_z_) { res += v; }
                return res;
            }

            std::vector<double> time_recv_z_;
            void time_recv_z(double value)
            {
                time_recv_z_[hpx::get_worker_thread_num()] += value;
            }
            double time_recv_z() const
            {
                double res = 0;
                for(auto v : time_recv_z_) { res += v; }
                return res;
            }

            std::vector<double> time_unpack_z_;
            void time_unpack_z(double value)
            {
                time_unpack_z_[hpx::get_worker_thread_num()] += value;
            }
            double time_unpack_z() const
            {
                double res = 0;
                for(auto v : time_unpack_z_) { res += v; }
                return res;
            }


            std::vector<double> time_sumgrid_;
            void time_sumgrid(double value)
            {
                time_sumgrid_[hpx::get_worker_thread_num()] += value;
            }
            double time_sumgrid() const
            {
                double res = 0;
                for(auto v : time_sumgrid_) { res += v; }
                return res;
            }

            std::vector<double> time_sumgrid_comp_;
            void time_sumgrid_comp(double value)
            {
                time_sumgrid_comp_[hpx::get_worker_thread_num()] += value;
            }
            double time_sumgrid_comp() const
            {
                double res = 0;
                for(auto v : time_sumgrid_comp_) { res += v; }
                return res;
            }

            std::vector<double> time_sumgrid_comm_;
            void time_sumgrid_comm(double value)
            {
                time_sumgrid_comm_[hpx::get_worker_thread_num()] += value;
            }
            double time_sumgrid_comm() const
            {
                double res = 0;
                for(auto v : time_sumgrid_comm_) { res += v; }
                return res;
            }


            profiling_data()
              : /*num_copy_(hpx::get_os_thread_count(), 0)
              ,*/ num_adds_(hpx::get_os_thread_count(), 0)
              , num_divides_(hpx::get_os_thread_count(), 0)

            /*
              , num_sends_(hpx::get_os_thread_count(), 0)
              , send_count_(hpx::get_os_thread_count(), 0)
              , send_count_max_(hpx::get_os_thread_count(), 0)
              , send_count_min_(hpx::get_os_thread_count(), 1000000000)
              , num_recvs_(hpx::get_os_thread_count(), 0)
              , recv_count_(hpx::get_os_thread_count(), 0)
              , recv_count_max_(hpx::get_os_thread_count(), 0)
              , recv_count_min_(hpx::get_os_thread_count(), 1000000000)
              , num_bcasts_(hpx::get_os_thread_count(), 0)
              , bcast_count_(hpx::get_os_thread_count(), 0)
              , bcast_count_max_(hpx::get_os_thread_count(), 0)
              , bcast_count_min_(hpx::get_os_thread_count(), 1000000000)
              , num_allreduces_(hpx::get_os_thread_count(), 0)
              , allreduce_count_(hpx::get_os_thread_count(), 0)
              , allreduce_count_max_(hpx::get_os_thread_count(), 0)
              , allreduce_count_min_(hpx::get_os_thread_count(), 1000000000)
            */

              , num_sumgrid_(hpx::get_os_thread_count(), 0)

              , time_init_(hpx::get_os_thread_count(), 0.0)
              , time_hpx_(hpx::get_os_thread_count(), 0.0)
              , time_pack_(hpx::get_os_thread_count(), 0.0)
              , time_recv_(hpx::get_os_thread_count(), 0.0)
              , time_send_(hpx::get_os_thread_count(), 0.0)
              , time_stencil_(hpx::get_os_thread_count(), 0.0)
              , time_unpack_(hpx::get_os_thread_count(), 0.0)
              , time_wait_(hpx::get_os_thread_count(), 0.0)
              , time_wall_(hpx::get_os_thread_count(), 0.0)

              , time_pack_x_(hpx::get_os_thread_count(), 0.0)
              , time_send_x_(hpx::get_os_thread_count(), 0.0)
              , time_wait_x_(hpx::get_os_thread_count(), 0.0)
              , time_recv_x_(hpx::get_os_thread_count(), 0.0)
              , time_unpack_x_(hpx::get_os_thread_count(), 0.0)

              , time_pack_y_(hpx::get_os_thread_count(), 0.0)
              , time_send_y_(hpx::get_os_thread_count(), 0.0)
              , time_wait_y_(hpx::get_os_thread_count(), 0.0)
              , time_recv_y_(hpx::get_os_thread_count(), 0.0)
              , time_unpack_y_(hpx::get_os_thread_count(), 0.0)

              , time_pack_z_(hpx::get_os_thread_count(), 0.0)
              , time_send_z_(hpx::get_os_thread_count(), 0.0)
              , time_wait_z_(hpx::get_os_thread_count(), 0.0)
              , time_recv_z_(hpx::get_os_thread_count(), 0.0)
              , time_unpack_z_(hpx::get_os_thread_count(), 0.0)

              , time_sumgrid_(hpx::get_os_thread_count(), 0.0)
              , time_sumgrid_comp_(hpx::get_os_thread_count(), 0.0)
              , time_sumgrid_comm_(hpx::get_os_thread_count(), 0.0)
            {
            }

            template <typename Archive>
            void serialize(Archive & ar, unsigned)
            {
                ar & num_adds_;
                ar & num_divides_;

                ar & num_sumgrid_;

                ar & time_init_;
                ar & time_pack_;
                ar & time_recv_;
                ar & time_send_;
                ar & time_stencil_;
                ar & time_unpack_;
                ar & time_wait_;
                ar & time_wall_;

                ar & time_pack_x_;
                ar & time_send_x_;
                ar & time_wait_x_;
                ar & time_recv_x_;
                ar & time_unpack_x_;

                ar & time_pack_y_;
                ar & time_send_y_;
                ar & time_wait_y_;
                ar & time_recv_y_;
                ar & time_unpack_y_;

                ar & time_pack_z_;
                ar & time_send_z_;
                ar & time_wait_z_;
                ar & time_recv_z_;
                ar & time_unpack_z_;

                ar & time_sumgrid_;
                ar & time_sumgrid_comp_;
                ar & time_sumgrid_comm_;
            }
        };

        static profiling_data & data()
        {
            hpx::util::static_<profiling_data> d;

            return d.get();
        }

        template <typename Real>
        static void report(std::ostream & os, std::vector<profiling_data> const & data, params<Real> & p)
        {
            using namespace boost::accumulators;
            typedef
                accumulator_set<
                    double
                  , features<
                        tag::min
                      , tag::max
                      , tag::count
                      , tag::mean
                      , tag::sum
                      , stats<
                            tag::variance(immediate)
                          , tag::moment<1>
                        >
                    >
                >
                accumulator_type;

            const double giga = 1000000000.;

            accumulator_type time_init;
            accumulator_type time_wall;
            accumulator_type time_stencil;
            accumulator_type time_comm;
            accumulator_type time_pack;
            accumulator_type time_send;
            accumulator_type time_wait;
            accumulator_type time_recv;
            accumulator_type time_unpack;

            accumulator_type time_x_comm;
            accumulator_type time_x_pack;
            accumulator_type time_x_send;
            accumulator_type time_x_wait;
            accumulator_type time_x_recv;
            accumulator_type time_x_unpack;

            accumulator_type time_y_comm;
            accumulator_type time_y_pack;
            accumulator_type time_y_send;
            accumulator_type time_y_wait;
            accumulator_type time_y_recv;
            accumulator_type time_y_unpack;

            accumulator_type time_z_comm;
            accumulator_type time_z_pack;
            accumulator_type time_z_send;
            accumulator_type time_z_wait;
            accumulator_type time_z_recv;
            accumulator_type time_z_unpack;

            accumulator_type time_sumgrid;
            accumulator_type time_sumgrid_comp;
            accumulator_type time_sumgrid_comm;

            std::size_t gnum_adds = 0;
            std::size_t gnum_divides = 0;
            std::size_t gnum_sumgrid = 0;

            for(auto & d : data)
            {
                time_init(d.time_init());
                time_wall(d.time_wall());
                time_stencil(d.time_stencil());

                double t = 0.0;
                t += d.time_pack();
                t += d.time_send();
                t += d.time_wait();
                t += d.time_recv();
                t += d.time_unpack();
                time_comm(t);
                time_pack(d.time_pack());
                time_send(d.time_send());
                time_wait(d.time_wait());
                time_recv(d.time_recv());
                time_unpack(d.time_unpack());

                t = 0.0;
                t += d.time_pack_x();
                t += d.time_send_x();
                t += d.time_wait_x();
                t += d.time_recv_x();
                t += d.time_unpack_x();
                time_x_comm(t);
                time_x_pack(d.time_pack_x());
                time_x_send(d.time_send_x());
                time_x_wait(d.time_wait_x());
                time_x_recv(d.time_recv_x());
                time_x_unpack(d.time_unpack_x());

                t = 0.0;
                t += d.time_pack_y();
                t += d.time_send_y();
                t += d.time_wait_y();
                t += d.time_recv_y();
                t += d.time_unpack_y();
                time_y_comm(t);
                time_y_pack(d.time_pack_y());
                time_y_send(d.time_send_y());
                time_y_wait(d.time_wait_y());
                time_y_recv(d.time_recv_y());
                time_y_unpack(d.time_unpack_y());

                t = 0.0;
                t += d.time_pack_z();
                t += d.time_send_z();
                t += d.time_wait_z();
                t += d.time_recv_z();
                t += d.time_unpack_z();
                time_z_comm(t);
                time_z_pack(d.time_pack_z());
                time_z_send(d.time_send_z());
                time_z_wait(d.time_wait_z());
                time_z_recv(d.time_recv_z());
                time_z_unpack(d.time_unpack_z());

                time_sumgrid(d.time_sumgrid());
                time_sumgrid_comp(d.time_sumgrid_comp());
                time_sumgrid_comm(d.time_sumgrid_comm());

                gnum_adds += d.num_adds();
                gnum_divides += d.num_divides();
                gnum_sumgrid += d.num_sumgrid();
            }

            os
                << "code: miniGhost (HPX)\n"
                << "version: 0.0.1\n"
                << "compiled_on: FIXME\n"
                << "directory: FIXME\n"
                << "compiler: FIXME\n"
                << "arch: FIXME\n"
                << "hostname: FIXME\n"
                << "module_list: FIXME\n"
                << "Comm_strategy: HPX based constrained based programming (no explicit message passing)\n"
            ;
            switch (p.stencil)
            {
                case 20: //STENCIL_NONE:
                    os << "No computation inserted\n";
                    break;
                case 21: //STENCIL_2D5PT:
                    os << "Computation: 5 pt difference stencil on a 2D grid (STENCIL_2D5PT)\n";
                    break;
                case 22: //STENCIL_2D9PT:
                    os << "Computation: 9 pt difference stencil on a 2D grid (STENCIL_2D9PT)\n";
                    break;
                case 23: //STENCIL_3D7PT:
                    os << "Computation: 7 pt difference stencil on a 3D grid (STENCIL_3D27PT)\n";
                    break;
                case 24: //STENCIL_3D27PT:
                    os << "Computation: 27 pt difference stencil on a 3D grid (STENCIL_3D27PT)\n";
                    break;
                default:
                    os << "Computation: none (STENCIL_NONE)\n";
                    break;
            }
            os
                << "Global_Grid_X: " << p.nx * p.npx << "\n"
                << "Global_Grid_Y: " << p.ny * p.npy << "\n"
                << "Global_Grid_Z: " << p.nz * p.npz << "\n"
                << "Local_Grid_X: " << p.nx << "\n"
                << "Local_Grid_Y: " << p.ny << "\n"
                << "Local_Grid_Z: " << p.nz << "\n"
                << "Number_variables: " << p.num_vars << "\n"
                << "Number_reduced: FIXME\n"// << p.num_vars << "\n"
                << "Percent_reduced: " << p.percent_sum << "\n"
                << "Time_steps: " << p.num_tsteps << "\n"
                << "Task_Grid_X: " << p.npx << "\n"
                << "Task_Grid_Y: " << p.npy << "\n"
                << "Task_Grid_Z: " << p.npz << "\n"
            ;
            switch (p.scaling)
            {
                case SCALING_WEAK:
                    os << "Scaling: weak\n";
                    break;
                case SCALING_STRONG:
                    os << "Scaling: strong\n";
                    break;
                default:
                    os << "Scaling: unknown\n";
                    break;
            }
            os
                << "Processes: " << p.nranks << "\n"
                << "Threads: " << p.nranks * hpx::get_os_thread_count() << "\n"
                << "Threads_Per_Locality: " << hpx::get_os_thread_count() << "\n"
                << "machine: " << hpx::get_locality_name() << "\n"
                << "Program_execution_date: FIXME" << "\n"
                << "Total_time: " << (max)(time_wall) << "\n"
                << "clock_resolution: FIXME\n"
                << "Error_tolerance: " << p.error_tol << "\n"
                << "Init_time_ave: " << moment<1>(time_init) << "\n"
                << "Init_time_min: " << (min)(time_init) << "\n"
                << "Init_time_max: " << (max)(time_init) << "\n"
                << "Init_time_mean: " << mean(time_init) << "\n"
                << "Init_time_variance: " << std::sqrt(variance(time_init)) << "\n"
                << "Comp_time_ave: " << moment<1>(time_stencil) << "\n"
                << "Comp_time_min: " << (min)(time_stencil) << "\n"
                << "Comp_time_max: " << (max)(time_stencil) << "\n"
                << "Comp_time_mean: " << mean(time_stencil) << "\n"
                << "Comp_time_variance: " << std::sqrt(variance(time_stencil)) << "\n"
            ;

            std::cout << hpx::get_os_thread_count() << " Total_time: " << (max)(time_wall) << "\n";

            double gflops_total  = double(gnum_adds + gnum_divides) / (max)(time_wall) / giga;
            double flops_total   = double(gnum_adds + gnum_divides);
            double flops_sums    = double(gnum_adds);
            double flops_divides = double(gnum_divides);
            os
                << "GFLOPS_Total: "   << gflops_total  << "\n"
                << "FLOPS_Total: "    << flops_total   << "\n"
                << "FLOPS_Sums: "     << flops_sums    << "\n"
                << "FLOPS_Divides: "  << flops_divides << "\n"
                << "Number_spikes: "  << p.num_spikes  << "\n"

                << "Comm_total_ave: " << moment<1>(time_comm) << "\n"
                << "Comm_total_max: " << (max)(time_comm) << "\n"
                << "Comm_total_min: " << (min)(time_comm) << "\n"
                << "Comm_total_mean: " << mean(time_comm) << "\n"
                << "Comm_total_variance: " << std::sqrt(variance(time_comm)) << "\n"

                << "Comm_pack_ave: " << moment<1>(time_pack) << "\n"
                << "Comm_pack_max: " << (max)(time_pack) << "\n"
                << "Comm_pack_min: " << (min)(time_pack) << "\n"
                << "Comm_pack_mean: " << mean(time_pack) << "\n"
                << "Comm_pack_variance: " << std::sqrt(variance(time_pack)) << "\n"

                << "Comm_send_ave: " << moment<1>(time_send) << "\n"
                << "Comm_send_max: " << (max)(time_send) << "\n"
                << "Comm_send_min: " << (min)(time_send) << "\n"
                << "Comm_send_mean: " << mean(time_send) << "\n"
                << "Comm_send_variance: " << std::sqrt(variance(time_send)) << "\n"

                << "Comm_wait_ave: " << moment<1>(time_wait) << "\n"
                << "Comm_wait_max: " << (max)(time_wait) << "\n"
                << "Comm_wait_min: " << (min)(time_wait) << "\n"
                << "Comm_wait_mean: " << mean(time_wait) << "\n"
                << "Comm_wait_variance: " << std::sqrt(variance(time_wait)) << "\n"

                << "Comm_recv_ave: " << moment<1>(time_recv) << "\n"
                << "Comm_recv_max: " << (max)(time_recv) << "\n"
                << "Comm_recv_min: " << (min)(time_recv) << "\n"
                << "Comm_recv_mean: " << mean(time_recv) << "\n"
                << "Comm_recv_variance: " << std::sqrt(variance(time_recv)) << "\n"

                << "Comm_unpack_ave: " << moment<1>(time_unpack) << "\n"
                << "Comm_unpack_max: " << (max)(time_unpack) << "\n"
                << "Comm_unpack_min: " << (min)(time_unpack) << "\n"
                << "Comm_unpack_mean: " << mean(time_unpack) << "\n"
                << "Comm_unpack_variance: " << std::sqrt(variance(time_unpack)) << "\n"

                << "Comm_x_total_ave: " << moment<1>(time_x_comm) << "\n"
                << "Comm_x_total_max: " << (max)(time_x_comm) << "\n"
                << "Comm_x_total_min: " << (min)(time_x_comm) << "\n"
                << "Comm_x_total_mean: " << mean(time_x_comm) << "\n"
                << "Comm_x_total_variance: " << std::sqrt(variance(time_x_comm)) << "\n"

                << "Comm_x_pack_ave: " << moment<1>(time_x_pack) << "\n"
                << "Comm_x_pack_max: " << (max)(time_x_pack) << "\n"
                << "Comm_x_pack_min: " << (min)(time_x_pack) << "\n"
                << "Comm_x_pack_mean: " << mean(time_x_pack) << "\n"
                << "Comm_x_pack_variance: " << std::sqrt(variance(time_x_pack)) << "\n"

                << "Comm_x_send_ave: " << moment<1>(time_x_send) << "\n"
                << "Comm_x_send_max: " << (max)(time_x_send) << "\n"
                << "Comm_x_send_min: " << (min)(time_x_send) << "\n"
                << "Comm_x_send_mean: " << mean(time_x_send) << "\n"
                << "Comm_x_send_variance: " << std::sqrt(variance(time_x_send)) << "\n"

                << "Comm_x_wait_ave: " << moment<1>(time_x_wait) << "\n"
                << "Comm_x_wait_max: " << (max)(time_x_wait) << "\n"
                << "Comm_x_wait_min: " << (min)(time_x_wait) << "\n"
                << "Comm_x_wait_mean: " << mean(time_x_wait) << "\n"
                << "Comm_x_wait_variance: " << std::sqrt(variance(time_x_wait)) << "\n"

                << "Comm_x_recv_ave: " << moment<1>(time_x_recv) << "\n"
                << "Comm_x_recv_max: " << (max)(time_x_recv) << "\n"
                << "Comm_x_recv_min: " << (min)(time_x_recv) << "\n"
                << "Comm_x_recv_mean: " << mean(time_x_recv) << "\n"
                << "Comm_x_recv_variance: " << std::sqrt(variance(time_x_recv)) << "\n"

                << "Comm_x_unpack_ave: " << moment<1>(time_x_unpack) << "\n"
                << "Comm_x_unpack_max: " << (max)(time_x_unpack) << "\n"
                << "Comm_x_unpack_min: " << (min)(time_x_unpack) << "\n"
                << "Comm_x_unpack_mean: " << mean(time_x_unpack) << "\n"
                << "Comm_x_unpack_variance: " << std::sqrt(variance(time_x_unpack)) << "\n"

                << "Comm_y_total_ave: " << moment<1>(time_y_comm) << "\n"
                << "Comm_y_total_max: " << (max)(time_y_comm) << "\n"
                << "Comm_y_total_min: " << (min)(time_y_comm) << "\n"
                << "Comm_y_total_mean: " << mean(time_y_comm) << "\n"
                << "Comm_y_total_variance: " << std::sqrt(variance(time_y_comm)) << "\n"

                << "Comm_y_pack_ave: " << moment<1>(time_y_pack) << "\n"
                << "Comm_y_pack_max: " << (max)(time_y_pack) << "\n"
                << "Comm_y_pack_min: " << (min)(time_y_pack) << "\n"
                << "Comm_y_pack_mean: " << mean(time_y_pack) << "\n"
                << "Comm_y_pack_variance: " << std::sqrt(variance(time_y_pack)) << "\n"

                << "Comm_y_send_ave: " << moment<1>(time_y_send) << "\n"
                << "Comm_y_send_max: " << (max)(time_y_send) << "\n"
                << "Comm_y_send_min: " << (min)(time_y_send) << "\n"
                << "Comm_y_send_mean: " << mean(time_y_send) << "\n"
                << "Comm_y_send_variance: " << std::sqrt(variance(time_y_send)) << "\n"

                << "Comm_y_wait_ave: " << moment<1>(time_y_wait) << "\n"
                << "Comm_y_wait_max: " << (max)(time_y_wait) << "\n"
                << "Comm_y_wait_min: " << (min)(time_y_wait) << "\n"
                << "Comm_y_wait_mean: " << mean(time_y_wait) << "\n"
                << "Comm_y_wait_variance: " << std::sqrt(variance(time_y_wait)) << "\n"

                << "Comm_y_recv_ave: " << moment<1>(time_y_recv) << "\n"
                << "Comm_y_recv_max: " << (max)(time_y_recv) << "\n"
                << "Comm_y_recv_min: " << (min)(time_y_recv) << "\n"
                << "Comm_y_recv_mean: " << mean(time_y_recv) << "\n"
                << "Comm_y_recv_variance: " << std::sqrt(variance(time_y_recv)) << "\n"

                << "Comm_y_unpack_ave: " << moment<1>(time_y_unpack) << "\n"
                << "Comm_y_unpack_max: " << (max)(time_y_unpack) << "\n"
                << "Comm_y_unpack_min: " << (min)(time_y_unpack) << "\n"
                << "Comm_y_unpack_mean: " << mean(time_y_unpack) << "\n"
                << "Comm_y_unpack_variance: " << std::sqrt(variance(time_y_unpack)) << "\n"

                << "Comm_z_total_ave: " << moment<1>(time_z_comm) << "\n"
                << "Comm_z_total_max: " << (max)(time_z_comm) << "\n"
                << "Comm_z_total_min: " << (min)(time_z_comm) << "\n"
                << "Comm_z_total_mean: " << mean(time_z_comm) << "\n"
                << "Comm_z_total_variance: " << std::sqrt(variance(time_z_comm)) << "\n"

                << "Comm_z_pack_ave: " << moment<1>(time_z_pack) << "\n"
                << "Comm_z_pack_max: " << (max)(time_z_pack) << "\n"
                << "Comm_z_pack_min: " << (min)(time_z_pack) << "\n"
                << "Comm_z_pack_mean: " << mean(time_z_pack) << "\n"
                << "Comm_z_pack_variance: " << std::sqrt(variance(time_z_pack)) << "\n"

                << "Comm_z_send_ave: " << moment<1>(time_z_send) << "\n"
                << "Comm_z_send_max: " << (max)(time_z_send) << "\n"
                << "Comm_z_send_min: " << (min)(time_z_send) << "\n"
                << "Comm_z_send_mean: " << mean(time_z_send) << "\n"
                << "Comm_z_send_variance: " << std::sqrt(variance(time_z_send)) << "\n"

                << "Comm_z_wait_ave: " << moment<1>(time_z_wait) << "\n"
                << "Comm_z_wait_max: " << (max)(time_z_wait) << "\n"
                << "Comm_z_wait_min: " << (min)(time_z_wait) << "\n"
                << "Comm_z_wait_mean: " << mean(time_z_wait) << "\n"
                << "Comm_z_wait_variance: " << std::sqrt(variance(time_z_wait)) << "\n"

                << "Comm_z_recv_ave: " << moment<1>(time_z_recv) << "\n"
                << "Comm_z_recv_max: " << (max)(time_z_recv) << "\n"
                << "Comm_z_recv_min: " << (min)(time_z_recv) << "\n"
                << "Comm_z_recv_mean: " << mean(time_z_recv) << "\n"
                << "Comm_z_recv_variance: " << std::sqrt(variance(time_z_recv)) << "\n"

                << "Comm_z_unpack_ave: " << moment<1>(time_z_unpack) << "\n"
                << "Comm_z_unpack_max: " << (max)(time_z_unpack) << "\n"
                << "Comm_z_unpack_min: " << (min)(time_z_unpack) << "\n"
                << "Comm_z_unpack_mean: " << mean(time_z_unpack) << "\n"
                << "Comm_z_unpack_variance: " << std::sqrt(variance(time_z_unpack)) << "\n"

                << "Number_gridsum: " << gnum_sumgrid << "\n"

                << "Gridsum_Time_ave: " << moment<1>(time_sumgrid) << "\n"
                << "Gridsum_Time_max: " << (max)(time_sumgrid) << "\n"
                << "Gridsum_Time_min: " << (min)(time_sumgrid) << "\n"
                << "Gridsum_Time_mean: " << mean(time_sumgrid) << "\n"
                << "Gridsum_Time_variance: " << std::sqrt(variance(time_sumgrid)) << "\n"

                << "Gridsum_Time_Comp_ave: " << moment<1>(time_sumgrid_comp) << "\n"
                << "Gridsum_Time_Comp_max: " << (max)(time_sumgrid_comp) << "\n"
                << "Gridsum_Time_Comp_min: " << (min)(time_sumgrid_comp) << "\n"
                << "Gridsum_Time_Comp_mean: " << mean(time_sumgrid_comp) << "\n"
                << "Gridsum_Time_Comp_variance: " << std::sqrt(variance(time_sumgrid_comp)) << "\n"

                << "Gridsum_Time_Comm_ave: " << moment<1>(time_sumgrid_comm) << "\n"
                << "Gridsum_Time_Comm_max: " << (max)(time_sumgrid_comm) << "\n"
                << "Gridsum_Time_Comm_min: " << (min)(time_sumgrid_comm) << "\n"
                << "Gridsum_Time_Comm_mean: " << mean(time_sumgrid_comm) << "\n"
                << "Gridsum_Time_Comm_variance: " << std::sqrt(variance(time_sumgrid_comm)) << "\n"
            ;

            os << std::flush;
        }
    };
}

#endif
