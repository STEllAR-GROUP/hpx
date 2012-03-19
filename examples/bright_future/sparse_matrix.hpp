//  Copyright (c) 2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_EXAMPLES_BRIGHT_FUTURE_CRS_MATRIX_HPP
#define HPX_EXAMPLES_BRIGHT_FUTURE_CRS_MATRIX_HPP

namespace bright_future
{
    template <typename T>
    struct crs_matrix
    {
        typedef std::vector<T> values_type;
        typedef typename values_type::reference reference;
        typedef typename values_type::const_reference const_reference;
        typedef std::vector<std::size_t> indices_type;

        std::size_t row_begin(std::size_t i) const
        {
            return rows[i];
        }

        std::size_t row_end(std::size_t i) const
        {
            return rows[i+1];
        }

        std::size_t col(std::size_t j)
        {
            return indices[j];
        }

        T & operator[](std::size_t j)
        {
            return values[j];
        }

        T const & operator[](std::size_t j) const
        {
            return values[j];
        }

        template <typename Archive>
        void serialize(Archive & ar, unsigned)
        {
            ar & values;
            ar & indices;
            ar & rows;
        }

        std::vector<T>           values;
        std::vector<std::size_t> indices;
        std::vector<std::size_t> rows;
    };

    inline void jacobi_kernel_nonuniform(
        crs_matrix<double> const & A
      , std::vector<std::vector<double> > & x
      , std::vector<double> const & b
      , std::pair<std::size_t, std::size_t> range
      , std::size_t src
      , std::size_t dst
    )
    {
        for(std::size_t i = range.first; i < range.second; ++i)
        {
            double result = b[i];
            double div = 1.0;
            const std::size_t begin = A.row_begin(i);
            const std::size_t end = A.row_end(i);
            for(std::size_t j = begin; j < end; ++j)
            {
                if(j == i) div = div/A.values[j];
                else result -= A.values[j] * x[src][A.indices[j]];
            }
            x[dst][i] = result * div;
        }
    }
}

#endif
