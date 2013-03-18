////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2013 Shuangyang Yang
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/hpx_init.hpp>
#include <hpx/util/detail/hold_any.hpp>

#include <boost/serialization/access.hpp>

using boost::program_options::variables_map;
using boost::program_options::options_description;

using hpx::util::hold_any;
using hpx::util::hold_any_nonser;
using hpx::util::any_cast;

using hpx::init;
using hpx::finalize;

///////////////////////////////////////////////////////////////////////////////
struct small_object
{
  private:
    boost::uint64_t x_;

    friend class boost::serialization::access;

    template <
        typename Archive
    >
    void serialize(
        Archive& ar
      , unsigned const
        )
    {
        ar & x_;
        std::cout << "small_object: serialize(" << x_ << ")\n";
    }

  public:
    small_object()
      : x_(0)
    {
        std::cout << "small_object: default ctor\n";
    }

    small_object(
        boost::uint64_t x
        )
      : x_(x)
    {
        std::cout << "small_object: ctor(" << x << ")\n";
    }

    small_object(
        small_object const& o
        )
      : x_(o.x_)
    {
        std::cout << "small_object: copy(" << o.x_ << ")\n";
    }

    small_object& operator=(
        small_object const& o
        )
    {
        x_ = o.x_;
        std::cout << "small_object: assign(" << o.x_ << ")\n";
        return *this;
    }

    ~small_object()
    {
        std::cout << "small_object: dtor(" << x_ << ")\n";
    }

    boost::uint64_t operator()(
        boost::uint64_t const& z_
        )
    {
        std::cout << "small_object: call(" << x_ << ", " << z_ << ")\n";
        return x_ + z_;
    }


    friend inline std::istream& 
    operator>> (std::istream& in, small_object& obj)
    {
        in >> obj.x_;
        std::cout << "small_object: istream ("<< obj.x_ << ")\n";

        return in;
    }

    friend inline std::ostream& 
    operator<< (std::ostream& out, small_object const& obj)
    {
        out << obj.x_;
        std::cout << "small_object: ostream ("<< obj.x_ << ")\n";

        return out;
    }
};

///////////////////////////////////////////////////////////////////////////////
struct big_object
{
  private:
    boost::uint64_t x_;
    boost::uint64_t y_;

    friend class boost::serialization::access;

    template <
        typename Archive
    >
    void serialize(
        Archive& ar
      , unsigned const
        )
    {
        ar & x_;
        ar & y_;
        std::cout << "big_object: serialize(" << x_ << ", " << y_ << ")\n";
    }

  public:
    big_object()
      : x_(0)
      , y_(0)
    {
        std::cout << "big_object: default ctor\n";
    }

    big_object(
        boost::uint64_t x
      , boost::uint64_t y
        )
      : x_(x)
      , y_(y)
    {
        std::cout << "big_object: ctor(" << x << ", " << y << ")\n";
    }

    big_object(
        big_object const& o
        )
      : x_(o.x_)
      , y_(o.y_)
    {
        std::cout << "big_object: copy(" << o.x_ << ", " << o.y_ << ")\n";
    }

    big_object& operator=(
        big_object const& o
        )
    {
        x_ = o.x_;
        y_ = o.y_;
        std::cout << "big_object: assign(" << o.x_ << ", " << o.y_ << ")\n";
        return *this;
    }

    ~big_object()
    {
        std::cout << "big_object: dtor(" << x_ << ", " << y_ << ")\n";
    }

    boost::uint64_t operator()(
        boost::uint64_t const& z_
      , boost::uint64_t const& w_
        )
    {
        std::cout << "big_object: call(" << x_ << ", " << y_
                  << z_ << ", " << w_ << ")\n";
        return x_ + y_ + z_ + w_;
    }

    friend inline std::istream& 
    operator>> (std::istream& in, big_object& obj)
    {
        in >> obj.x_;
        in >> obj.y_;
        std::cout << "big_object: istream ("<< obj.x_ <<", "<< obj.y_ << ")\n";

        return in;
    }

    friend inline std::ostream& 
    operator<< (std::ostream& out, big_object const& obj)
    {
        out << obj.x_;
        out << obj.y_;
        std::cout << "big_object: ostream ("<< obj.x_ <<", "<< obj.y_ << ")\n";

        return out;
    }
};

struct foo
{
    void operator()() {}
};

//template <typename Archive>
//void serialize(Archive & ar, foo & f, unsigned) {}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(variables_map& vm)
{
    {
        {
            if (sizeof(small_object) <= sizeof(void*))
                std::cout << "object is small\n";
            else
                std::cout << "object is large\n";

            small_object const f(17);

            hold_any any1(f);
            hold_any any2(any1);
            hold_any any3;
            any3 = any1;

            (any_cast<small_object>(any1)) (7);
            (any_cast<small_object>(any2)) (9);
            (any_cast<small_object>(any3)) (11);
        }

        {
            if (sizeof(big_object) <= sizeof(void*))
                std::cout << "object is small\n";
            else
                std::cout << "object is large\n";

            big_object const f(5, 12);

            hold_any any1(f);
            hold_any any2(any1);
            hold_any any3 = any1;

            (any_cast<big_object>(any1)) (0, 1);
            (any_cast<big_object>(any2)) (1, 0);
            (any_cast<big_object>(any3)) (1, 1);
        }
    }
    // non serializable version
    {
        {
            if (sizeof(small_object) <= sizeof(void*))
                std::cout << "object is small\n";
            else
                std::cout << "object is large\n";

            small_object const f(17);

            hold_any_nonser any1_nonser(f);
            hold_any_nonser any2_nonser(any1_nonser);
            hold_any_nonser any3_nonser = any1_nonser;

            (any_cast<small_object>(any1_nonser)) (2);
            (any_cast<small_object>(any2_nonser)) (4);
            (any_cast<small_object>(any3_nonser)) (6);

        }

        {
            if (sizeof(big_object) <= sizeof(void*))
                std::cout << "object is small\n";
            else
                std::cout << "object is large\n";

            big_object const f(5, 12);

            hold_any_nonser any1_nonser(f);
            hold_any_nonser any2_nonser(any1_nonser);
            hold_any_nonser any3_nonser = any1_nonser;

            (any_cast<big_object>(any1_nonser)) (3,4);
            (any_cast<big_object>(any2_nonser)) (5,6);
            (any_cast<big_object>(any3_nonser)) (7,8);
        }
    }

    finalize();

    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options
    options_description cmdline("Usage: " HPX_APPLICATION_STRING " [options]");

    // Initialize and run HPX
    return init(cmdline, argc, argv);
}

