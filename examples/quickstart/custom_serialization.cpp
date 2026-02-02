//  Copyright (c) 2022 John Sorial
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This example is meant for inclusion in the documentation.

#include <hpx/format.hpp>
#include <hpx/future.hpp>
#include <hpx/hpx_main.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/runtime.hpp>
#include <hpx/include/util.hpp>
#include <hpx/serialization.hpp>

#include <iostream>
#include <memory>

//[point_member_serialization
struct point_member_serialization
{
    int x{0};
    int y{0};

    // Required when defining the serialization function as private
    // In this case it isn't
    // Provides serialization access to HPX
    friend class hpx::serialization::access;

    // Second argument exists solely for compatibility with boost serialize
    // it is NOT processed by HPX in any way.
    template <typename Archive>
    void serialize(Archive& ar, unsigned int const)
    {
        // clang-format off
        ar & x & y;
        // clang-format on
    }
};

// Allow bitwise serialization
HPX_IS_BITWISE_SERIALIZABLE(point_member_serialization)
//]

//[rectangle_member_serialization
struct rectangle_member_serialization
{
    point_member_serialization top_left;
    point_member_serialization lower_right;

    template <typename Archive>
    void serialize(Archive& ar, unsigned int const)
    {
        // clang-format off
        ar & top_left & lower_right;
        // clang-format on
    }
};
//]

//[rectangle_free
struct rectangle_free
{
    point_member_serialization top_left;
    point_member_serialization lower_right;
};

template <typename Archive>
void serialize(Archive& ar, rectangle_free& pt, unsigned int const)
{
    // clang-format off
    ar & pt.lower_right & pt.top_left;
    // clang-format on
}
//]

//[point_class
class point_class
{
public:
    point_class(int x, int y)
      : x(x)
      , y(y)
    {
    }

    point_class() = default;

    [[nodiscard]] int get_x() const noexcept
    {
        return x;
    }

    [[nodiscard]] int get_y() const noexcept
    {
        return y;
    }

private:
    int x;
    int y;
};

template <typename Archive>
void load(Archive& ar, point_class& pt, unsigned int const)
{
    int x, y;
    ar >> x >> y;
    pt = point_class(x, y);
}

template <typename Archive>
void save(Archive& ar, point_class const& pt, unsigned int const)
{
    ar << pt.get_x() << pt.get_y();
}

// This tells HPX that you have spilt your serialize function into
// load and save
HPX_SERIALIZATION_SPLIT_FREE(point_class)
//]

//[SendRectangle
void send_rectangle_struct(rectangle_free rectangle)
{
    hpx::util::format_to(std::cout,
        "Rectangle(Point(x={1},y={2}),Point(x={3},y={4}))\n",
        rectangle.top_left.x, rectangle.top_left.y, rectangle.lower_right.x,
        rectangle.lower_right.y);
}
//]

HPX_PLAIN_ACTION(send_rectangle_struct)

//[planet_weight_calculator
class planet_weight_calculator
{
public:
    explicit planet_weight_calculator(double g)
      : g(g)
    {
    }

    template <class Archive>
    friend void save_construct_data(
        Archive&, planet_weight_calculator const*, unsigned int);

    [[nodiscard]] double get_g() const
    {
        return g;
    }

private:
    // Provides serialization access to HPX
    friend class hpx::serialization::access;
    template <class Archive>
    void serialize(Archive&, unsigned int const)
    {
        // Serialization will be done in the save_construct_data
        // Still needs to be defined
    }

    double g;
};
//]

//[save_construct_data
template <class Archive>
inline void save_construct_data(Archive& ar,
    planet_weight_calculator const* weight_calc, unsigned int const)
{
    ar << weight_calc->g;    // Do all of your serialization here
}

template <class Archive>
inline void load_construct_data(
    Archive& ar, planet_weight_calculator* weight_calc, unsigned int const)
{
    double g;
    ar >> g;

    // ::new(ptr) construct new object at given address
    hpx::construct_at(weight_calc, g);
}
//]

void send_gravity(planet_weight_calculator gravity)
{
    std::cout << "gravity.g = " << gravity.get_g() << std::endl;
}

HPX_PLAIN_ACTION(send_gravity)

//[Main
int main()
{
    // Needs at least two localities to run
    // When sending to your current locality, no serialization is done
    send_rectangle_struct_action rectangle_action;
    auto rectangle = rectangle_free{{0, 0}, {0, 5}};
    hpx::async(rectangle_action, hpx::find_here(), rectangle).get();

    send_gravity_action gravityAction;
    auto gravity = planet_weight_calculator(9.81);

    auto remote_localities = hpx::find_remote_localities();
    if (!remote_localities.empty())
    {
        hpx::async(gravityAction, remote_localities[0], gravity).get();
    }
    else
    {
        hpx::async(gravityAction, hpx::find_here(), gravity).get();
    }
    return 0;
}
//]
