
//  Copyright (c) 2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef JACOBI_ROW_HPP
#define JACOBI_ROW_HPP

#include <hpx/components/dataflow/dataflow_object.hpp>

#include <boost/smart_ptr/shared_array.hpp>

namespace jacobi
{
    struct row_range
    {
        std::ptrdiff_t begin_;
        std::ptrdiff_t end_;

        boost::shared_array<double> values_;

        row_range()
        {}

        row_range(boost::shared_array<double> values, std::ptrdiff_t b, std::ptrdiff_t e)
            : begin_(b)
            , end_(e)
            , values_(values)
        {}

        double * begin()
        {
            BOOST_ASSERT(values_);
            return &values_[begin_];
        }
        
        double const * begin() const
        {
            BOOST_ASSERT(values_);
            return &values_[begin_];
        }

        double * end()
        {
            BOOST_ASSERT(values_);
            return &values_[begin_] + end_;
        }
        
        double const * end() const
        {
            BOOST_ASSERT(values_);
            return &values_[begin_] + end_;
        }

        template <typename Archive>
        void load(Archive & ar, unsigned)
        {
            BOOST_ASSERT(false);
        }

        template <typename Archive>
        void save(Archive & ar, unsigned) const
        {
            BOOST_ASSERT(false);
        }

        BOOST_SERIALIZATION_SPLIT_MEMBER()

    };

    inline double * range_begin(row_range & r)
    {
        return r.begin();
    }

    inline double * range_end(row_range & r)
    {
        return r.end();
    }

    inline double const * range_begin(row_range const & r)
    {
        return r.begin();
    }

    inline double const * range_end(row_range const & r)
    {
        return r.end();
    }

    struct row
    {
        explicit row(std::size_t nx, double init = 0.0);

        typedef boost::shared_array<double> values_type;

        values_type values;

        struct get
        {
            typedef row_range result_type;

            std::ptrdiff_t begin_;
            std::ptrdiff_t end_;

            get() {}

            get(std::ptrdiff_t begin, std::ptrdiff_t end)
                : begin_(begin)
                , end_(end)
            {}

            result_type operator()(row & r) const
            {
                return result_type(r.values, begin_, end_);
            }

            template <typename Archive>
            void serialize(Archive & ar, unsigned)
            {
                ar & begin_;
                ar & end_;
            }
        };
    };
}

namespace boost
{
    template <>
    struct range_mutable_iterator<jacobi::row_range>
    {
        typedef double * type;
    };
    
    template <>
    struct range_const_iterator<jacobi::row_range>
    {
        typedef double const * type;
    };
}

#endif
