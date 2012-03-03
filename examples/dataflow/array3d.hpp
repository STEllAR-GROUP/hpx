//  Copyright (c) 2009-2011 Matt Anderson
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_7F247232_9948_4D16_B08B_CEB968B5D660)
#define HPX_7F247232_9948_4D16_B08B_CEB968B5D660

#include <vector>

#include <boost/config.hpp>

#include <boost/serialization/vector.hpp>
#include <boost/serialization/access.hpp>

namespace hpx { namespace components { namespace amr
{

struct array3d
{
  private:
    std::size_t width_;
    std::size_t height_;
    std::vector<int> data_;

    friend class boost::serialization::access;

    template <typename Archive>
    void serialize(Archive &ar, const unsigned int)
    { ar & width_ & height_ & data_; }

  public:
    array3d(std::size_t x, std::size_t y, std::size_t z, int init = 0)
      : width_(x), height_(y), data_(x * y * z, init)
      {}

    int& operator() (std::size_t x, std::size_t y, std::size_t z)
    {
        return data_.at(x + y * width_ + z * width_ * height_);
    }

    int operator()(std::size_t x, std::size_t y, std::size_t z) const
    {
        return data_.at(x + y * width_ + z * width_ * height_);
    }
};

}}}

#endif

