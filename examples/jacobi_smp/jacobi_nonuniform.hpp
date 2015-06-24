
//  Copyright (c) 2011-2013 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <vector>

namespace jacobi_smp {
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

        std::vector<T>           values;
        std::vector<std::size_t> indices;
        std::vector<std::size_t> rows;
    };


    void jacobi(
        crs_matrix<double> const & A
      , std::vector<double> const & b
      , std::size_t iterations
      , std::size_t block_size
    );

    struct range
    {
        range() : begin_(0), end_(0) {}
        range(std::size_t begin, std::size_t end) : begin_(begin), end_(end) {}

        std::size_t begin() const { return begin_; }
        std::size_t end() const { return end_; }

        std::size_t begin_;
        std::size_t end_;
    };

    void jacobi_kernel_nonuniform(
        crs_matrix<double> const & A
      , std::vector<double> & dst
      , std::vector<double> const & src
      , std::vector<double> const & b
      , std::size_t row
    );
}
