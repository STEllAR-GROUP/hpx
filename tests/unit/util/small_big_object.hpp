////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2013 Shuangyang Yang
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#ifndef HPX_SMALL_BIG_OBJECT_HPP
#define HPX_SMALL_BIG_OBJECT_HPP

#include <hpx/runtime/serialization/serialize.hpp>

#define ENABLE_DEBUG false

///////////////////////////////////////////////////////////////////////////////
struct small_object
{
  private:
    boost::uint64_t x_;

    friend class hpx::serialization::access;

    template <typename Archive>
    void serialize(Archive& ar, unsigned const)
    {
        ar & x_;

        if(ENABLE_DEBUG)
        {
            std::cout << "small_object: serialize(" << x_ << ")\n";
        }
    }

  public:
    small_object() : x_(0)
    {
        if(ENABLE_DEBUG)
        {
            std::cout << "small_object: default ctor\n";
        }
    }

    small_object(boost::uint64_t x) : x_(x)
    {
        if(ENABLE_DEBUG)
        {
            std::cout << "small_object: ctor(" << x << ")\n";
        }
    }

    small_object(small_object const& o) : x_(o.x_)
    {
        if(ENABLE_DEBUG)
        {
            std::cout << "small_object: copy(" << o.x_ << ")\n";
        }
    }

    small_object& operator=(small_object const& o)
    {
        x_ = o.x_;
        if(ENABLE_DEBUG)
        {
            std::cout << "small_object: assign(" << o.x_ << ")\n";
        }
        return *this;
    }

    bool operator==(small_object const& o) const
    {
        if(ENABLE_DEBUG)
        {
            std::cout << "small_object: equal(" << o.x_ << ")\n";
        }
        return x_ == o.x_;
    }

    ~small_object()
    {
        if(ENABLE_DEBUG)
        {
            std::cout << "small_object: dtor(" << x_ << ")\n";
        }
    }

    boost::uint64_t operator()(boost::uint64_t const& z_)
    {
        if(ENABLE_DEBUG)
        {
            std::cout << "small_object: call(" << x_ << ", " << z_ << ")\n";
        }
        return x_ + z_;
    }


    friend inline std::istream& operator>> (std::istream& in, small_object& obj)
    {
        in >> obj.x_;
        if(ENABLE_DEBUG)
        {
            std::cout << "small_object: istream ("<< obj.x_ << ")\n";
        }

        return in;
    }

    friend inline std::ostream& operator<< (std::ostream& out, small_object const& obj)
    {
        out << obj.x_;
        if(ENABLE_DEBUG)
        {
            std::cout << "small_object: ostream ("<< obj.x_ << ")\n";
        }

        return out;
    }
};

///////////////////////////////////////////////////////////////////////////////
struct big_object
{
  private:
    boost::uint64_t x_;
    boost::uint64_t y_;

    friend class hpx::serialization::access;

    template <typename Archive>
    void serialize(Archive& ar, unsigned const)
    {
        ar & x_;
        ar & y_;
        if(ENABLE_DEBUG)
        {
            std::cout << "big_object: serialize(" << x_ << ", " << y_ << ")\n";
        }
    }

  public:
    big_object() : x_(0), y_(0)
    {
        if(ENABLE_DEBUG)
        {
            std::cout << "big_object: default ctor\n";
        }
    }

    big_object(boost::uint64_t x, boost::uint64_t y)
      : x_(x), y_(y)
    {
        if(ENABLE_DEBUG)
        {
            std::cout << "big_object: ctor(" << x << ", " << y << ")\n";
        }
    }

    big_object(big_object const& o)
      : x_(o.x_), y_(o.y_)
    {
        if(ENABLE_DEBUG)
        {
            std::cout << "big_object: copy(" << o.x_ << ", " << o.y_ << ")\n";
        }
    }

    big_object& operator=(big_object const& o)
    {
        x_ = o.x_;
        y_ = o.y_;
        if(ENABLE_DEBUG)
        {
            std::cout << "big_object: assign(" << o.x_ << ", " << o.y_ << ")\n";
        }
        return *this;
    }

    bool operator==(big_object const& o) const
    {
        if(ENABLE_DEBUG)
        {
            std::cout << "big_object: equal(" << o.x_ << ", " << o.y_ << ")\n";
        }
        return ((x_ == o.x_) && (y_ == o.y_));
    }

    ~big_object()
    {
        if(ENABLE_DEBUG)
        {
            std::cout << "big_object: dtor(" << x_ << ", " << y_ << ")\n";
        }
    }

    boost::uint64_t operator()(
        boost::uint64_t const& z_
      , boost::uint64_t const& w_
        )
    {
        if(ENABLE_DEBUG)
        {
            std::cout << "big_object: call(" << x_ << ", " << y_
                  << ", " << z_ << ", " << w_ << ")\n";
        }
        return x_ + y_ + z_ + w_;
    }

    friend inline std::istream&
    operator>> (std::istream& in, big_object& obj)
    {
        in >> obj.x_;
        in >> obj.y_;
        if(ENABLE_DEBUG)
        {
            std::cout << "big_object: istream ("<< obj.x_ <<", "<< obj.y_ << ")\n";
        }

        return in;
    }

    friend inline std::ostream&
    operator<< (std::ostream& out, big_object const& obj)
    {
        out << obj.x_;
        out << obj.y_;
        if(ENABLE_DEBUG)
        {
            std::cout << "big_object: ostream ("<< obj.x_ <<", "<< obj.y_ << ")\n";
        }

        return out;
    }
};

#undef ENABLE_DEBUG

#endif
