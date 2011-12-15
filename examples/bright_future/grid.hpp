//  Copyright (c) 2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)


#ifndef HPX_EXAMPLES_GRID_HPP
#define HPX_EXAMPLES_GRID_HPP

#include <vector>
#include <iostream>

#include <boost/serialization/access.hpp>
#include <boost/assert.hpp>

namespace bright_future
{
    template <typename T>
    struct grid
    {
        typedef std::vector<T> vector_type;
        typedef typename vector_type::size_type size_type;
        typedef typename vector_type::value_type value_type;
        typedef typename vector_type::reference reference_type;
        typedef typename vector_type::const_reference const_reference_type;
        typedef typename vector_type::iterator iterator;
        typedef typename vector_type::const_iterator const_iterator;
        typedef value_type result_type;

        grid(size_type x_size, size_type y_size)
            : n_x(x_size)
            , n_y(y_size)
            , data(x_size * y_size)
        {}
        
        grid(size_type x_size, size_type y_size, T const & init)
            : n_x(x_size)
            , n_y(y_size)
            , data(x_size * y_size, init)
        {}

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
            return data.at(x + y * n_x);
        }

        const_reference_type operator()(size_type x, size_type y) const
        {
            BOOST_ASSERT(x < n_x);
            BOOST_ASSERT(y < n_y);
            return data.at(x + y * n_x);
        }

        reference_type operator[](size_type i)
        {
            return data.at(i);
        }

        const_reference_type operator[](size_type i) const
        {
            return data.at(i);
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
