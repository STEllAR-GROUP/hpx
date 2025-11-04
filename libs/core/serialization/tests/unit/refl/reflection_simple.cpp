//  Copyright (c) 2025 Ujjwal Shekhar
//  SPDX-License-Identifier: BSL-1.0

#include <hpx/config.hpp>
#include <hpx/modules/serialization.hpp>
#include <hpx/modules/testing.hpp>

#include <string>
#include <vector>

// A simple class that is NOT bitwise serializable.
// We will NOT provide a serialize() function for this.
class A
{
private:
    int a;
    std::string b;
    std::vector<int> c;

public:
    A() = default;

    A(int a, std::string const& b, std::vector<int> const& c)
      : a(a)
      , b(b)
      , c(c)
    {
    }

    // getters to test
    int get_a() const
    {
        return a;
    }
    std::string get_b() const
    {
        return b;
    }
    std::vector<int> get_c() const
    {
        return c;
    }

    // equality operator for testing
    bool operator==(A const& rhs) const
    {
        return a == rhs.a && b == rhs.b && c == rhs.c;
    }
};

int main()
{
    using simple_test_struct = A;

    // 1. Create the object to serialize
    std::vector<char> buffer;
    hpx::serialization::output_archive oarchive(buffer);

    simple_test_struct input_data(42, "hello reflection", {1, 2, 3, 4, 5});

    // 2. Serialize
    // This will fail to compile if reflection is not working,
    // as it will hit the static_assert in access.hpp
    oarchive << input_data;

    // 3. Deserialize
    hpx::serialization::input_archive iarchive(buffer);
    simple_test_struct output_data;

    iarchive >> output_data;

    // 4. Verify
    HPX_TEST_EQ(input_data.get_a(), output_data.get_a());
    HPX_TEST_EQ(input_data.get_b(), output_data.get_b());
    HPX_TEST_EQ(input_data.get_c().size(), output_data.get_c().size());
    HPX_TEST(input_data == output_data);

    return hpx::util::report_errors();
}