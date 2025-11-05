// //  Copyright (c) 2025 Ujjwal Shekhar
// //  SPDX-License-Identifier: BSL-1.0
// //  Distributed under the Boost Software License, Version 1.0. (See accompanying
// //  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// #include <hpx/modules/serialization.hpp>
// #include <hpx/modules/testing.hpp>

// #include <vector>

// TBD

// // This is a POD (Plain Old Data) struct.
// // We are NOT adding any serialization macros.
// // HPX should automatically detect this as bitwise serializable.
// // This test confirms our reflection patch did not break this fast path.
// struct POD_Struct
// {
//     int a;
//     double b;

//     bool operator==(POD_Struct const& rhs) const
//     {
//         return a == rhs.a && b == rhs.b;
//     }
// };

// int main()
// {
//     std::vector<char> buffer;
//     hpx::serialization::output_archive oarchive(buffer);

//     POD_Struct input_data = {42, 3.14159};

//     oarchive << input_data;

//     hpx::serialization::input_archive iarchive(buffer);
//     POD_Struct output_data;
//     iarchive >> output_data;

//     // Verify
//     HPX_TEST(input_data == output_data);
//     HPX_TEST_EQ(input_data.a, 42);
//     HPX_TEST_EQ(input_data.b, 3.14159);

//     return hpx::util::report_errors();
// }