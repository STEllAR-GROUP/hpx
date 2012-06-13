
//  Copyright (c) 2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "stencil_iterator.hpp"

namespace jacobi
{
    namespace server
    {
        void stencil_iterator::run(std::size_t max_iterations)
        {
            //std::cout << "beginning to run ...\n";

            hpx::apply<next_action>(
                this->get_gid()
              , 0
              , max_iterations
            );
            
            //for(std::size_t iter = 0; iter < max_iterations; ++iter)
            {
                for(std::size_t x = 1; x < nx-1; x += line_block)
                {
                    std::size_t x_end = std::min(nx-1, x + line_block);
                    get_dep(max_iterations-1, x, x_end).get_future().get();
                    //hpx::cout << iter << ": (" << x << " " << y << "): finished\n" << hpx::flush;
                }
            }
        }
            
        hpx::lcos::dataflow_base<void> stencil_iterator::get_dep(std::size_t iter, std::size_t begin, std::size_t end)
        {
            BOOST_ASSERT(y > 0);
            BOOST_ASSERT(y < ny-1);
            std::pair<std::size_t, std::size_t> range(begin, end);
            bool calc_iter_dep = false;
            {
                hpx::util::spinlock::scoped_lock l(mtx);
                iteration_deps_type::mapped_type::iterator dep
                    = iteration_deps[iter].find(range);

                if(dep == iteration_deps[iter].end())
                {
                    auto calculated_iter = calculating_dep[iter].find(range);
                    if(calculated_iter == calculating_dep[iter].end())
                    {
                        calculating_dep[iter].insert(range);
                        calc_iter_dep = true;
                    }
                }
                else
                {
                    return dep->second;
                }
            }
            if(calc_iter_dep)
            {
                BOOST_ASSERT(this->get_gid());
                hpx::lcos::dataflow_base<void> d;
                if(iter>0)
                {
                    d =
                        hpx::lcos::dataflow<update_action>(
                            this->get_gid()
                          , center.get(begin, end)
                          , center.get(begin, end)
                          , top.get(iter, begin, end)
                          , bottom.get(iter, begin, end)
                          , get_dep(iter-1, begin, end)
                        );
                }
                else
                {
                    d =
                        hpx::lcos::dataflow<update_action>(
                            this->get_gid()
                          , center.get(begin, end)
                          , center.get(begin, end)
                          , top.get(iter, begin, end)
                          , bottom.get(iter, begin, end)
                        );
                }
                std::pair<iteration_deps_type::mapped_type::iterator, bool> iter_pair;
                {
                    hpx::util::spinlock::scoped_lock l(mtx);
                    iter_pair =
                        iteration_deps[iter].insert(std::make_pair(range, d));

                    BOOST_ASSERT(iter_pair.second);

                    calculating_dep[iter].erase(range);
                    /*
                    BOOST_FOREACH(hpx::threads::thread_id_type & id, iteration_deps_wait_list[iter][range])
                    {
                        hpx::threads::set_thread_state(id, hpx::threads::pending);
                    }
                    {
                        std::vector<hpx::threads::thread_id_type> tmp;
                        std::swap(iteration_deps_wait_list[iter][range], tmp);
                    }
                    */
		    return d;
                }
            }
            else
            {
                /*
                {
                    hpx::util::spinlock::scoped_lock l(mtx);
                    iteration_deps_wait_list[iter][range].push_back(hpx::threads::get_self().get_thread_id());
                }
                hpx::this_thread::suspend(boost::posix_time::milliseconds(1), "example::jacobi::server::stencil_iterator::get_dep");
                */
                //hpx::this_thread::suspend(boost::posix_time::milliseconds(1));
                return get_dep(iter, begin, end);
            }
            BOOST_ASSERT(false);
            return hpx::lcos::dataflow_base<void>();
        }

        void stencil_iterator::next(
            std::size_t iter
          , std::size_t max_iterations
        )
        {
            if(iter == max_iterations)
            {
                return;
            }
            BOOST_ASSERT(this->get_gid());
            BOOST_ASSERT(center.id);
            BOOST_ASSERT(top.id);
            BOOST_ASSERT(bottom.id);
            for(std::size_t x = 1, x_dep = 0; x < nx-1; x += line_block, ++x_dep)
            {
                std::size_t end = std::min(nx - 1, x + line_block);

                get_dep(iter, x, end);
            }

            hpx::apply<next_action>(
                this->get_gid()
              , ++iter
              , max_iterations
            );
        }
            
        hpx::lcos::dataflow_base<row_range> stencil_iterator::get(std::size_t iter, std::size_t begin, std::size_t end)
        {
            BOOST_ASSERT(this->get_gid());
            BOOST_ASSERT(center.id);
            if(y > 0 && y < ny-1 && iter > 0)
            {
                //get_dep(iter-1, begin, end).get_future().get();
		    return
			hpx::lcos::dataflow<server::row::get_action>(
			    center.id
			  , begin
			  , end
			  , get_dep(iter-1, begin, end)
			);
            }

            return
                hpx::lcos::dataflow<server::row::get_action>(
                    center.id
                  , begin
                  , end
                );
        }
    }
}

typedef hpx::components::managed_component<
    jacobi::server::stencil_iterator
> stencil_iterator_type;

HPX_REGISTER_MINIMAL_GENERIC_COMPONENT_FACTORY(stencil_iterator_type, stencil_iterator);


HPX_REGISTER_ACTION_EX(
    jacobi::server::stencil_iterator::init_action
  , jacobi_server_stencil_iterator_init_action
)

HPX_REGISTER_ACTION_EX(
    jacobi::server::stencil_iterator::setup_boundary_action
  , jacobi_server_stencil_iterator_setup_boundary_action
)

HPX_REGISTER_ACTION_EX(
    jacobi::server::stencil_iterator::run_action
  , jacobi_server_stencil_iterator_run_action
)

HPX_REGISTER_ACTION_EX(
    jacobi::server::stencil_iterator::update_action
  , jacobi_server_stencil_iterator_update_action
)

HPX_REGISTER_ACTION_EX(
    jacobi::server::stencil_iterator::next_action
  , jacobi_server_stencil_iterator_next_action
)
            
HPX_REGISTER_ACTION_EX(
    jacobi::server::stencil_iterator::get_action
  , jacobi_server_stencil_iterator_get_action
)
