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
#include <hpx/serialization.hpp>
#include <iostream>

//[PointMemberSerialization
struct PointMemberSerialization
{
    int x{0};
    int y{0};

    // Required when defining the serialization function as private
    // In this case it isn't
    // Provides serialization access to HPX
    friend class hpx::serialization::access;

    // Second argument exists soley for compatibity with boost serialize
    // it is NOT procced by HPX in any way.
    template <typename Archive>
    void serialize(Archive& ar, const unsigned int)
    {
        // clang-format off
        ar & x & y;
        // clang-format on
    }
};
// Allow bitwise serialization
HPX_IS_BITWISE_SERIALIZABLE(PointMemberSerialization)
//]

//[RectangleMemberSerialization
struct RectangleMemberSerialization
{
    PointMemberSerialization top_left;
    PointMemberSerialization lower_right;

    template <typename Archive>
    void serialize(Archive& ar, const unsigned int)
    {
        // clang-format off
        ar & top_left & lower_right;
        // clang-format on
    }
};
//]

//[RectangleFREE
struct RectangleFREE
{
    PointMemberSerialization top_left;
    PointMemberSerialization lower_right;
};

template <typename Archive>
void serialize(Archive& ar, RectangleFREE& pt, const unsigned int)
{
    // clang-format off
    ar & pt.lower_right & pt.top_left;
    // clang-format one

}
//]

//[PointClass
class PointClass
{
public:
    PointClass(int x, int y)
      : x(x)
      , y(y)
    {
    }

    PointClass() = default;

    [[nodiscard]] int getX() const noexcept
    {
        return x;
    }

    [[nodiscard]] int getY() const noexcept
    {
        return y;
    }

private:
    int x;
    int y;
};

template <typename Archive>
void load(Archive& ar, PointClass& pt, const unsigned int)
{
    int x, y;
    ar >> x >> y;
    pt = PointClass(x, y);
}

template <typename Archive>
void save(Archive& ar, PointClass const& pt, const unsigned int)
{
    ar << pt.getX() << pt.getY();
}
// This tells HPX that you have spilt your serialize function into
// load and save
HPX_SERIALIZATION_SPLIT_FREE(PointClass)
//]

//[SendRectangle
void send_rectangle_struct(RectangleFREE rectangle)
{
    hpx::util::format_to(std::cout,
        "Rectangle(Point(x={1},y={2}),Point(x={3},y={4}))\n",
        rectangle.top_left.x, rectangle.top_left.y, rectangle.lower_right.x,
        rectangle.lower_right.y);
}

HPX_PLAIN_ACTION(send_rectangle_struct);

//[PlanetWeightCalculator
class PlanetWeightCalculator
{
public:
    explicit PlanetWeightCalculator(double g)
      : g(g)
    {
    }

    template <class Archive>
    friend void save_construct_data(
        Archive&, PlanetWeightCalculator const*, unsigned int);

    [[nodiscard]] double getG() const
    {
        return g;
    }

private:
    // Provides serialization access to HPX
    friend class hpx::serialization::access;
    template <class Archive>
    void serialize(Archive&, const unsigned int)
    {
        // Serialization will be done in the save_construct_data
        // Still needs to be defined
    }

    double g;
};
//]

//[save_construct_data
template <class Archive>
inline void save_construct_data(
    Archive& ar, PlanetWeightCalculator const* weight_calc, const unsigned int)
{
    ar << weight_calc->g;    // Do all of your serialization here
}

template <class Archive>
inline void load_construct_data(
    Archive& ar, PlanetWeightCalculator* weight_calc, const unsigned int)
{
    double g;
    ar >> g;
    ::new (weight_calc) PlanetWeightCalculator(
        g);    // ::new(ptr) construct new object at given address
}
//]

void send_gravity(PlanetWeightCalculator gravity)
{
    std::cout << "gravity.g = " << gravity.getG() << std::flush;
}

HPX_PLAIN_ACTION(send_gravity);

//[Main
int main()
{
    // Needs at least two localities to run
    // When sending to your current locality, no serialization is done
    send_rectangle_struct_action rectangle_action;
    auto rectangle = RectangleFREE{{0, 0}, {0, 5}};
    hpx::async(rectangle_action, hpx::find_here(), rectangle).wait();
    send_gravity_action gravityAction;
    auto gravity = PlanetWeightCalculator(9.81);
    hpx::async(gravityAction, hpx::find_here(), gravity).wait();
}
//]
