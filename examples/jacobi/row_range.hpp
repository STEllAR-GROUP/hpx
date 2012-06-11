
//  Copyright (c) 2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef JACOBI_ROW_RANGE_HPP
#define JACOBI_ROW_RANGE_HPP

#include <boost/serialization/split_member.hpp>
#include <boost/smart_ptr/shared_array.hpp>
#include <boost/range/iterator.hpp>
#include <boost/range/mutable_iterator.hpp>

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
        {
            BOOST_ASSERT(end_ > begin_);
        }

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
            std::vector<double> tmp;
            ar & tmp;
            values_.reset(new double[tmp.size()]);
            for(std::size_t x = 0; x < tmp.size(); ++x)
            {
                values_[x] = tmp[x];
            }
            begin_ = 0;
            end_ = tmp.size();
            BOOST_ASSERT(end_ > begin_);
        }

        template <typename Archive>
        void save(Archive & ar, unsigned) const
        {
            BOOST_ASSERT(values_);
            std::vector<double> tmp;
            tmp.reserve(end_-begin_);
            for(std::ptrdiff_t x = begin_; x < end_; ++x)
            {
                tmp.push_back(values_[x]);
            }
            ar & tmp;
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
