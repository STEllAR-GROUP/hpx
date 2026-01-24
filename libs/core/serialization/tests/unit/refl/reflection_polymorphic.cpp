//  Copyright (c) 2026 Ujjwal Shekhar
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/modules/serialization.hpp>
#include <hpx/modules/testing.hpp>

#include <memory>
#include <string>

// Base class with one member
// (No serialize function)
struct Base
{
    int a = 0;

    // A virtual destructor is required for polymorphic types
    virtual ~Base() = default;

    // Comparison for testing
    virtual bool equals(Base const& rhs) const
    {
        return a == rhs.a;
    }
};

HPX_TRAITS_NONINTRUSIVE_POLYMORPHIC(Base)

// Derived class with its own member
// (No serialize function)
struct Derived : Base
{
    std::string b;

    // Comparison for testing
    bool equals(Base const& rhs) const override
    {
        auto const* d = dynamic_cast<Derived const*>(&rhs);
        return d && Base::equals(rhs) && b == d->b;
    }
};

HPX_SERIALIZATION_REGISTER_CLASS(Derived)

int main()
{
    // Serialize
    std::vector<char> buffer;

    // Create a Derived object but store it in a Base pointer
    std::unique_ptr<Base> input_data = std::make_unique<Derived>();
    static_cast<Derived*>(input_data.get())->a = 123;
    static_cast<Derived*>(input_data.get())->b = "polymorphism!";

    {
        hpx::serialization::output_archive oarchive(buffer);
        // Serialize the base pointer. This will trigger your
        // refl_serialize for 'Derived' and its base 'Base'.
        oarchive << input_data;
    }

    // Deserialize
    std::unique_ptr<Base> output_data;

    {
        hpx::serialization::input_archive iarchive(buffer);
        // Deserialize into the base pointer. HPX should
        // correctly create a 'Derived' object.
        iarchive >> output_data;
    }

    HPX_TEST(nullptr != output_data.get());

    // Check if it's the correct dynamic type
    auto* d = dynamic_cast<Derived*>(output_data.get());
    HPX_TEST(nullptr != d);

    // Check values
    HPX_TEST(input_data->equals(*output_data));

    return hpx::util::report_errors();
}