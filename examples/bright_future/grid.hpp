//  Copyright (c) 2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_EXAMPLES_GRID_HPP
#define HPX_EXAMPLES_GRID_HPP

#ifdef OPENMP_GRID
#include <omp.h>
#else
#include <hpx/hpx.hpp>
#include <hpx/lcos/async.hpp>
#include <hpx/lcos/async_future_wait.hpp>
#endif

#include <vector>
#include <iostream>

#include <boost/config.hpp>
#include <boost/serialization/access.hpp>
#include <boost/assert.hpp>
#include <boost/move/move.hpp>

#if defined(BOOST_NO_VARIADIC_TEMPLATES)
#include <boost/preprocessor/dec.hpp>
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/comma_if.hpp>
#include <boost/preprocessor/enum_params.hpp>
#include <boost/preprocessor/repeat.hpp>
#endif

#ifndef OPENMP_GRID
HPX_ALWAYS_EXPORT std::size_t
    touch_mem(std::size_t, std::size_t, std::size_t, std::size_t);

typedef
    hpx::actions::plain_result_action4<
        std::size_t
      , std::size_t
      , std::size_t
      , std::size_t
      , std::size_t
      , touch_mem
    >
    touch_mem_action;
HPX_REGISTER_PLAIN_ACTION_DECLARATION(touch_mem_action);
#endif

#if defined(BOOST_NO_NOEXCEPT)
#define noexcept throw()
#endif

namespace bright_future
{
    template <typename T>
    struct numa_allocator;

    template <>
    struct numa_allocator<void>
    {
        typedef void * pointer;
        typedef const void * const_pointer;
        typedef void value_type;

        template <typename U> struct rebind { typedef numa_allocator<U> other; };
    };

    template <typename T>
    struct numa_allocator
    {
        typedef std::size_t size_type;
        typedef ptrdiff_t difference_type;
        typedef T* pointer;
        typedef const T* const_pointer;
        typedef T& reference;
        typedef const T& const_reference;
        typedef T value_type;

        std::size_t block_size;

        template <typename U> struct rebind { typedef numa_allocator<U> other; };

        numa_allocator() noexcept {}
        explicit numa_allocator(std::size_t block_size) noexcept  : block_size(block_size){}
        numa_allocator(numa_allocator const & n) noexcept  : block_size(n.block_size) {}
        template <typename U>
        numa_allocator(numa_allocator<U> const & n) noexcept : block_size(n.block_size)  {}
        ~numa_allocator() {}

        numa_allocator & operator=(numa_allocator const & n) noexcept
        {
            block_size = n.block_size;
        }

        template <typename U>
        numa_allocator & operator=(numa_allocator<U> const & n) noexcept
        {
            block_size = n.block_size;
        }

        pointer allocate(size_type n, numa_allocator<void>::const_pointer locality_hint = 0)
        {
            size_type len = n * sizeof(value_type);
            char *p = static_cast<char *>(std::malloc(len));

            if(p == 0) throw std::bad_alloc();

#ifdef OPENMP_GRID
            if(!omp_in_parallel())
            {
#pragma omp parallel for schedule(static)
                for(size_type i_block = 0; i_block < len; i_block += block_size)
                {
                    size_type i_end = (std::min)(i_block + block_size, len);
                    for(size_type i = i_block; i < i_end; i += sizeof(value_type))
                    {
                        for(size_type j = 0; j < sizeof(value_type); ++j)
                        {
                            p[i+j] = 0;
                        }
                    }
                }
            }
#else
            std::size_t const os_threads = hpx::get_os_thread_count();
            hpx::naming::id_type const prefix = hpx::find_here();
            std::size_t num_thread = 0;
            std::set<std::size_t> attendance;

            if(block_size == 0)
            {
                block_size = len / os_threads + 1;
            }

            for(size_type i_block = 0; i_block < len; i_block += block_size)
            {
                attendance.insert(num_thread);
                num_thread = (num_thread+1) % os_threads;
            }

            while(!attendance.empty())
            {
                std::vector<hpx::lcos::promise<std::size_t> > futures;
                futures.reserve(attendance.size());
                std::size_t start = 0;
                BOOST_FOREACH(std::size_t os_thread, attendance)
                {
                    futures.push_back(
                        hpx::lcos::async<touch_mem_action>(
                            prefix
                          , os_thread
                          , reinterpret_cast<std::size_t>(p)
                          , start
                          , (std::min)(start + block_size, len)
                        )
                    );
                    start += block_size;
                }

                hpx::lcos::wait(
                    futures
                  , [&](std::size_t, std::size_t t)
                    {if(std::size_t(-1) != t) attendance.erase(t); }
                );
            }
            /*
            std::size_t const os_threads = hpx::get_os_thread_count();
            hpx::naming::id_type const prefix = hpx::find_here();
            std::set<std::size_t> attendance;
            for(std::size_t os_thread = 0; os_thread < os_threads; ++os_thread)
                attendance.insert(os_thread);

            std::size_t len_per_thread = len / os_threads + 1;

            while(!attendance.empty())
            {
                std::vector<hpx::lcos::promise<std::size_t> > futures;
                futures.reserve(attendance.size());
                BOOST_FOREACH(std::size_t os_thread, attendance)
                {
                    futures.push_back(
                        hpx::lcos::async<touch_mem_action>(
                            prefix
                          , os_thread
                          , reinterpret_cast<std::size_t>(p)
                          , len_per_thread
                          , len
                        )
                    );
                }

                hpx::lcos::wait(
                    futures
                  , [&](std::size_t, std::size_t t)
                    {if(std::size_t(-1) != t) attendance.erase(t); }
                );
            }
            */
#endif
            return reinterpret_cast<pointer>(p);
        }

        void deallocate(pointer p, size_type)
        {
            std::free(p);
        }

        size_type max_size() const noexcept
        {
            return std::allocator<T>().max_size();
        }

        void construct(pointer p, const value_type& x)
        {
            new (p) value_type(x);
        }

#if defined(BOOST_NO_VARIADIC_TEMPLATES)
#define HPX_GRID_FWD_ARGS(z, n, _)                                            \
        BOOST_PP_COMMA() BOOST_FWD_REF(BOOST_PP_CAT(A, n)) BOOST_PP_CAT(a, n) \
    /**/
#define HPX_GRID_FORWARD_ARGS(z, n, _)                                        \
        BOOST_PP_COMMA_IF(n)                                                  \
            boost::forward<BOOST_PP_CAT(A, n)>(BOOST_PP_CAT(a, n))            \
    /**/
#define HPX_GRID_CONSTRUCT(z, n, _)                                           \
        template <typename U BOOST_PP_COMMA_IF(n)                             \
            BOOST_PP_ENUM_PARAMS(n, typename A) >                             \
        void construct(U* p BOOST_PP_REPEAT(n, HPX_GRID_FWD_ARGS, _))         \
        {                                                                     \
            new (p) U(BOOST_PP_REPEAT(n, HPX_GRID_FORWARD_ARGS, _));          \
        }                                                                     \
    /**/

    BOOST_PP_REPEAT(HPX_ACTION_ARGUMENT_LIMIT, HPX_GRID_CONSTRUCT, _)

#undef HPX_GRID_CONSTRUCT
#undef HPX_GRID_FORWARD_ARGS
#undef HPX_GRID_FWD_ARGS

#else
        template <typename U, typename... Args>
        void construct(U* p, BOOST_FWD_REF(Args)... args)
        {
            new (p) U(boost::forward<Args>(args)...);
        }
#endif

        template <typename U>
        void destroy(U * p)
        {
            p->~U();
        }
    };

    template <typename T1, typename T2>
    bool operator==(const numa_allocator<T1>&, const numa_allocator<T2>&) noexcept
    {
        return true;
    }

    template <typename T1, typename T2>
    bool operator!=(const numa_allocator<T1>&, const numa_allocator<T2>&) noexcept
    {
        return false;
    }

    template <typename T>
    struct grid
    {
        typedef std::vector<T, numa_allocator<T> > vector_type;
        typedef typename vector_type::size_type size_type;
        typedef typename vector_type::value_type value_type;
        typedef typename vector_type::reference reference_type;
        typedef typename vector_type::const_reference const_reference_type;
        typedef typename vector_type::iterator iterator;
        typedef typename vector_type::const_iterator const_iterator;
        typedef value_type result_type;

        grid()
        {}

        grid(size_type x_size, size_type y_size)
            : n_x(x_size)
            , n_y(y_size)
            , data(x_size * y_size)
        {}

        grid(size_type x_size, size_type y_size, size_type block_size, T const & init)
            : n_x(x_size)
            , n_y(y_size)
            , data(x_size * y_size, init, numa_allocator<T>(block_size))
        {}

        grid(grid const & g)
            : n_x(g.n_x)
            , n_y(g.n_y)
            , data(g.data)
        {}

        grid &operator=(grid const & g)
        {
            grid tmp(g);
            std::swap(*this, tmp);
            return *this;
        }

        template <typename F>
        void init(F f)
        {
            for(size_type y = 0; y < n_y; ++y)
            {
                for(size_type x = 0; x < n_x; ++x)
                {
                    (*this)(x, y) = f(x, y);
                }
            }
        }

        reference_type operator()(size_type x, size_type y)
        {
            BOOST_ASSERT(x < n_x);
            BOOST_ASSERT(y < n_y);
            return data[x + y * n_x];
        }

        const_reference_type operator()(size_type x, size_type y) const
        {
            BOOST_ASSERT(x < n_x);
            BOOST_ASSERT(y < n_y);
            return data[x + y * n_x];
        }

        reference_type operator[](size_type i)
        {
            return data[i];
        }

        const_reference_type operator[](size_type i) const
        {
            return data[i];
        }

        iterator begin()
        {
            return data.begin();
        }

        const_iterator begin() const
        {
            return data.begin();
        }

        iterator end()
        {
            return data.end();
        }

        const_iterator end() const
        {
            return data.end();
        }

        size_type size() const
        {
            return data.size();
        }

        size_type x() const
        {
            return n_x;
        }

        size_type y() const
        {
            return n_y;
        }

        vector_type const & data_handle() const
        {
            return data;
        }

    private:
        size_type n_x;
        size_type n_y;
        vector_type data;

        // serialization support
        friend class boost::serialization::access;

        template<class Archive>
        void serialize(Archive & ar, const unsigned int version)
        {
            ar & n_x & n_y & data;
        }
    };

    template <typename T>
    std::ostream & operator<<(std::ostream & os, grid<T> const & g)
    {
        typedef typename grid<T>::size_type size_type;

        for(size_type y = 0; y < g.y(); ++y)
        {
            for(size_type x = 0; x < g.x(); ++x)
            {
                os << g(x, y) << " ";
            }
            os << "\n";
        }

        return os;
    }

    typedef std::pair<std::size_t, std::size_t> range_type;

    inline void jacobi_kernel_simple(
        std::vector<grid<double> > & u
      , range_type const & x_range
      , range_type const & y_range
      , std::size_t old
      , std::size_t new_
      , std::size_t cache_block
    )
    {
        grid<double> & u_new = u[new_];
        grid<double> & u_old = u[old];
        for(std::size_t y_block = y_range.first; y_block < y_range.second; y_block += cache_block)
        {
            std::size_t y_end = (std::min)(y_block + cache_block, y_range.second);
            for(std::size_t x_block = x_range.first; x_block < x_range.second; x_block += cache_block)
            {
                std::size_t x_end = (std::min)(x_block + cache_block, x_range.second);
                for(std::size_t y = y_block; y < y_end; ++y)
                {
                    for(std::size_t x = x_block; x < x_end; ++x)
                    {
                        u_new(x, y)
                            =(
                                u_old(x+1,y) + u_old(x-1,y)
                              + u_old(x,y+1) + u_old(x,y-1)
                            ) * 0.25;
                    }
                }
            }
        }
    }

    template <typename T>
    inline T update(
        grid<T> const & u
      , grid<T> const & rhs
      , typename grid<T>::size_type x
      , typename grid<T>::size_type y
      , T hx_sq
      , T hy_sq
      , T div
      , T relaxation
    )
    {
        return
            u(x, y)
            + (
                (
                    (
                        (u(x - 1, y) + u(x - 1, y)) / hx_sq
                      + (u(x, y - 1) + u(x, y + 1)) / hy_sq
                      + rhs(x, y)
                    )
                    / div
                )
                - u(x, y)
            )
            * relaxation
            ;
    }

    template <typename T>
    inline T update_residuum(
        grid<T> const & u
      , grid<T> const & rhs
      , typename grid<T>::size_type x
      , typename grid<T>::size_type y
      , T hx_sq
      , T hy_sq
      , T k
    )
    {
        return
              rhs(x,y)
            + (u(x-1, y) + u(x+1, y) - 2.0 * u(x,y))/hx_sq
            + (u(x, y-1) + u(x, y+1) - 2.0 * u(x,y))/hy_sq
            - (u(x, y) * k * k)
            ;
    }
}
#endif
