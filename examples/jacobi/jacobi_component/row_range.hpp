
//  Copyright (c) 2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef JACOBI_ROW_RANGE_HPP
#define JACOBI_ROW_RANGE_HPP

#include <boost/serialization/split_member.hpp>
#include <boost/smart_ptr/intrusive_ptr.hpp>
#include <boost/detail/atomic_count.hpp>
#include <boost/range/iterator.hpp>
#include <boost/range/mutable_iterator.hpp>

#include <vector>

namespace jacobi
{
    struct value_holder
    {
        value_holder()
            : count_(0)
        {}

        value_holder(std::size_t n, double init = 0.0)
            : v_(n, 0.0)
            , count_(0)
        {}

        double & operator[](std::size_t i)
        {
            BOOST_ASSERT(i < v_.size());
            return v_[i];
        }

        double const & operator[](std::size_t i) const
        {
            BOOST_ASSERT(i < v_.size());
            return v_[i];
        }

        std::vector<double> v_;
        boost::detail::atomic_count count_;

        friend void intrusive_ptr_add_ref(value_holder * p)
        {
            ++p->count_;
        }
        friend void intrusive_ptr_release(value_holder * p)
        {
            if (0 == --p->count_)
                delete p;
        }

        template <typename Archive>
        void serialize(Archive & ar, unsigned)
        {
            ar & v_;
        }
    };

    struct row_range
    {
        std::ptrdiff_t begin_;
        std::ptrdiff_t end_;

        boost::intrusive_ptr<value_holder> values_;

        row_range()
        {}

        row_range(boost::intrusive_ptr<value_holder> values, std::ptrdiff_t b, std::ptrdiff_t e)
            : begin_(b)
            , end_(e)
            , values_(values)
        {
            BOOST_ASSERT(end_ > begin_);
        }

        std::vector<double>::iterator begin()
        {
            BOOST_ASSERT(values_);
            return values_->v_.begin() + begin_;
        }
        
        std::vector<double>::const_iterator begin() const
        {
            BOOST_ASSERT(values_);
            return values_->v_.begin() + begin_;
        }

        std::vector<double>::iterator end()
        {
            BOOST_ASSERT(values_);
            return values_->v_.begin() + end_;
        }
        
        std::vector<double>::const_iterator end() const
        {
            BOOST_ASSERT(values_);
            return values_->v_.begin() + end_;
        }

        template <typename Archive>
        void load(Archive & ar, unsigned)
        {
            values_.reset(new value_holder());
            ar & values_->v_;
            begin_ = 0;
            end_ = values_->v_.size();
            BOOST_ASSERT(end_ > begin_);
        }

        template <typename Archive>
        void save(Archive & ar, unsigned) const
        {
            BOOST_ASSERT(values_);
            std::vector<double> tmp(values_->v_.begin() + begin_, values_->v_.begin() + end_);
            ar & tmp;
        }

        BOOST_SERIALIZATION_SPLIT_MEMBER()

    };

    inline std::vector<double>::iterator range_begin(row_range & r)
    {
        return r.begin();
    }

    inline std::vector<double>::iterator range_end(row_range & r)
    {
        return r.end();
    }

    inline std::vector<double>::const_iterator range_begin(row_range const & r)
    {
        return r.begin();
    }

    inline std::vector<double>::const_iterator range_end(row_range const & r)
    {
        return r.end();
    }
}

namespace boost
{
    template <>
    struct range_mutable_iterator<jacobi::row_range>
    {
        typedef std::vector<double>::iterator type;
    };
    
    template <>
    struct range_const_iterator<jacobi::row_range>
    {
        typedef std::vector<double>::const_iterator type;
    };
}

#endif
