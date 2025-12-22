//  Copyright (c) 2025 Hartmut Kaiser
//  Copyright (c) 2025 Mamidi Surya Teja
//
// SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0. (See
// accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/modules/memory.hpp>
#include <hpx/modules/testing.hpp>

#include <cstddef>
#include <utility>

namespace n_element_type {

    void test()
    {
        using T = hpx::memory::shared_array<int>::element_type;
        T value = 42;
        HPX_TEST_EQ(value, 42);
    }

}    // namespace n_element_type

namespace n_default_constructor {

    void test()
    {
        hpx::memory::shared_array<int> sa;
        HPX_TEST(sa.get() == nullptr);
        HPX_TEST_EQ(sa.use_count(), 0);
    }

}    // namespace n_default_constructor

namespace n_pointer_constructor {
    void test()
    {
        {
            int* p = new int[5]{1, 2, 3, 4, 5};
            hpx::memory::shared_array<int> sa(p);
            HPX_TEST_EQ(sa.get(), p);
            HPX_TEST_EQ(sa.use_count(), 1);
            HPX_TEST_EQ(sa[0], 1);
            HPX_TEST_EQ(sa.get()[4], 5);
        }

        {
            double* p = new double[3]{1.1, 2.2, 3.3};
            hpx::memory::shared_array<double> sa(p);
            HPX_TEST_EQ(sa.get(), p);
            HPX_TEST_EQ(sa.use_count(), 1);
        }
    }
}    // namespace n_pointer_constructor

namespace n_copy_constructor {
    void test()
    {
        {
            hpx::memory::shared_array<int> sa1;
            hpx::memory::shared_array<int> sa2(sa1);
            HPX_TEST_EQ(sa2.get(), sa1.get());
        }

        {
            int* p = new int[3]{10, 20, 30};
            hpx::memory::shared_array<int> sa1(p);
            HPX_TEST_EQ(sa1.use_count(), 1);

            hpx::memory::shared_array<int> sa2(sa1);
            HPX_TEST_EQ(sa2.get(), sa1.get());
            HPX_TEST_EQ(sa2.get(), p);
            HPX_TEST_EQ(sa1.use_count(), 2);
            HPX_TEST_EQ(sa2.use_count(), 2);
        }

        {
            int* p = new int[2]{100, 200};
            hpx::memory::shared_array<int> sa1(p);
            hpx::memory::shared_array<int> sa2(sa1);
            hpx::memory::shared_array<int> sa3(sa2);
            HPX_TEST_EQ(sa1.use_count(), 3);
            HPX_TEST_EQ(sa2.use_count(), 3);
            HPX_TEST_EQ(sa3.use_count(), 3);
            HPX_TEST_EQ(sa1.get(), p);
            HPX_TEST_EQ(sa2.get(), p);
            HPX_TEST_EQ(sa3.get(), p);
        }
    }
}    // namespace n_copy_constructor

namespace n_move_constructor {
    void test()
    {
        {
            int* p = new int[3]{1, 2, 3};
            hpx::memory::shared_array<int> sa1(p);
            HPX_TEST_EQ(sa1.use_count(), 1);

            hpx::memory::shared_array<int> sa2(std::move(sa1));
            HPX_TEST_EQ(sa2.get(), p);
            HPX_TEST(
                sa1.get() == nullptr);    // NOLINT(bugprone-use-after-move)
        }

        {
            double* p = new double[2]{1.5, 2.5};
            hpx::memory::shared_array<double> sa1(p);
            hpx::memory::shared_array<double> sa2(sa1);
            HPX_TEST_EQ(sa1.use_count(), 2);

            hpx::memory::shared_array<double> sa3(std::move(sa1));
            HPX_TEST_EQ(sa3.get(), p);
            HPX_TEST(
                sa1.get() == nullptr);    // NOLINT(bugprone-use-after-move)
            HPX_TEST_EQ(sa2.get(), p);
        }
    }
}    // namespace n_move_constructor

namespace n_swap {
    void test()
    {
        {
            hpx::memory::shared_array<int> sa1;
            hpx::memory::shared_array<int> sa2;

            sa1.swap(sa2);

            HPX_TEST(sa1.get() == nullptr);
            HPX_TEST(sa2.get() == nullptr);
        }

        {
            int* p = new int[3]{1, 2, 3};
            hpx::memory::shared_array<int> sa1(p);
            hpx::memory::shared_array<int> sa2;

            sa1.swap(sa2);

            HPX_TEST(sa1.get() == nullptr);
            HPX_TEST_EQ(sa2.get(), p);
            HPX_TEST_EQ(sa2.use_count(), 1);
        }

        {
            int* p1 = new int[2]{10, 20};
            int* p2 = new int[2]{30, 40};
            hpx::memory::shared_array<int> sa1(p1);
            hpx::memory::shared_array<int> sa2(p2);

            sa1.swap(sa2);

            HPX_TEST_EQ(sa1.get(), p2);
            HPX_TEST_EQ(sa2.get(), p1);
            HPX_TEST_EQ(sa1.use_count(), 1);
            HPX_TEST_EQ(sa2.use_count(), 1);
        }

        {
            int* p1 = new int[2]{1, 2};
            int* p2 = new int[2]{3, 4};
            hpx::memory::shared_array<int> sa1(p1);
            hpx::memory::shared_array<int> sa1_copy(sa1);
            hpx::memory::shared_array<int> sa2(p2);

            HPX_TEST_EQ(sa1.use_count(), 2);

            sa1.swap(sa2);

            HPX_TEST_EQ(sa1.get(), p2);
            HPX_TEST_EQ(sa2.get(), p1);
            HPX_TEST_EQ(sa1_copy.get(), p1);
            HPX_TEST_EQ(sa1.use_count(), 1);
            HPX_TEST_EQ(sa2.use_count(), 2);
            HPX_TEST_EQ(sa1_copy.use_count(), 2);
        }
    }
}    // namespace n_swap

namespace n_use_count {
    void test()
    {
        {
            hpx::memory::shared_array<int> sa;
            HPX_TEST_EQ(sa.use_count(), 0);
        }

        {
            int* p = new int[1]{42};
            hpx::memory::shared_array<int> sa(p);
            HPX_TEST_EQ(sa.use_count(), 1);

            {
                hpx::memory::shared_array<int> sa2(sa);
                HPX_TEST_EQ(sa.use_count(), 2);
                HPX_TEST_EQ(sa2.use_count(), 2);

                {
                    hpx::memory::shared_array<int> sa3(sa);
                    HPX_TEST_EQ(sa.use_count(), 3);
                    HPX_TEST_EQ(sa2.use_count(), 3);
                    HPX_TEST_EQ(sa3.use_count(), 3);
                }

                HPX_TEST_EQ(sa.use_count(), 2);
                HPX_TEST_EQ(sa2.use_count(), 2);
            }

            HPX_TEST_EQ(sa.use_count(), 1);
        }
    }
}    // namespace n_use_count

namespace n_get {
    void test()
    {
        {
            hpx::memory::shared_array<int> sa;
            HPX_TEST(sa.get() == nullptr);
        }

        {
            int* p = new int[5]{1, 2, 3, 4, 5};
            hpx::memory::shared_array<int> sa(p);
            HPX_TEST_EQ(sa.get(), p);
            HPX_TEST_EQ(sa.get()[0], 1);
            HPX_TEST_EQ(sa.get()[2], 3);
            HPX_TEST_EQ(sa.get()[4], 5);
        }

        {
            int* p = new int[3]{10, 20, 30};
            hpx::memory::shared_array<int> sa1(p);
            hpx::memory::shared_array<int> sa2(sa1);

            HPX_TEST_EQ(sa1.get(), sa2.get());
            HPX_TEST_EQ(sa1.get(), p);
        }

        {
            int* p = new int[2]{100, 200};
            hpx::memory::shared_array<int> const sa(p);
            HPX_TEST_EQ(sa.get(), p);
            HPX_TEST_EQ(sa.get()[0], 100);
        }
    }
}    // namespace n_get

namespace n_reset {
    void test()
    {
        {
            int* p = new int[3]{1, 2, 3};
            hpx::memory::shared_array<int> sa(p);
            HPX_TEST_EQ(sa.use_count(), 1);
            HPX_TEST_EQ(sa.get(), p);

            sa.reset();
            HPX_TEST(sa.get() == nullptr);
        }

        {
            int* p1 = new int[2]{10, 20};
            int* p2 = new int[2]{30, 40};
            hpx::memory::shared_array<int> sa(p1);
            HPX_TEST_EQ(sa.get(), p1);
            HPX_TEST_EQ(sa.use_count(), 1);

            sa.reset(p2);
            HPX_TEST_EQ(sa.get(), p2);
            HPX_TEST_EQ(sa.use_count(), 1);
        }

        {
            hpx::memory::shared_array<int> sa;
            HPX_TEST(sa.get() == nullptr);

            int* p = new int[3]{5, 6, 7};
            sa.reset(p);
            HPX_TEST_EQ(sa.get(), p);
            HPX_TEST_EQ(sa.use_count(), 1);
        }
    }
}    // namespace n_reset

namespace n_destructor {
    void test()
    {
        {
            int* p = new int[3]{1, 2, 3};
            {
                hpx::memory::shared_array<int> sa(p);
                HPX_TEST_EQ(sa.use_count(), 1);
            }
        }

        {
            int* p = new int[2]{10, 20};
            hpx::memory::shared_array<int> sa1(p);

            {
                hpx::memory::shared_array<int> sa2(sa1);
                HPX_TEST_EQ(sa1.use_count(), 2);
                HPX_TEST_EQ(sa2.use_count(), 2);
            }

            HPX_TEST_EQ(sa1.use_count(), 1);
            HPX_TEST_EQ(sa1.get(), p);
        }
    }
}    // namespace n_destructor

namespace n_multiple_types {
    void test()
    {
        {
            char* p = new char[5]{'h', 'e', 'l', 'l', 'o'};
            hpx::memory::shared_array<char> sa(p);
            HPX_TEST_EQ(sa.get(), p);
            HPX_TEST_EQ(sa.get()[0], 'h');
            HPX_TEST_EQ(sa.get()[4], 'o');
            HPX_TEST_EQ(sa.use_count(), 1);
        }

        {
            long* p = new long[4]{100L, 200L, 300L, 400L};
            hpx::memory::shared_array<long> sa(p);
            HPX_TEST_EQ(sa.get(), p);
            HPX_TEST_EQ(sa.get()[0], 100L);
            HPX_TEST_EQ(sa.get()[3], 400L);
        }

        {
            float* p = new float[3]{1.0f, 2.0f, 3.0f};
            hpx::memory::shared_array<float> sa(p);
            HPX_TEST_EQ(sa.get(), p);
        }

        {
            std::size_t* p = new std::size_t[2]{1000, 2000};
            hpx::memory::shared_array<std::size_t> sa(p);
            HPX_TEST_EQ(sa.get(), p);
            HPX_TEST_EQ(sa.get()[0], std::size_t(1000));
            HPX_TEST_EQ(sa.get()[1], std::size_t(2000));
        }
    }
}    // namespace n_multiple_types

namespace n_custom_deleter {
    static int deleter_call_count = 0;

    void custom_array_deleter(int* p) noexcept
    {
        ++deleter_call_count;
        delete[] p;
    }

    void test()
    {
        deleter_call_count = 0;

        {
            int* p = new int[3]{1, 2, 3};
            hpx::memory::shared_array<int> sa(p, custom_array_deleter);
            HPX_TEST_EQ(sa.get(), p);
            HPX_TEST_EQ(sa.use_count(), 1);
            HPX_TEST_EQ(sa[0], 1);
            HPX_TEST_EQ(sa[2], 3);
            HPX_TEST_EQ(deleter_call_count, 0);
        }

        HPX_TEST_EQ(deleter_call_count, 1);

        deleter_call_count = 0;

        {
            int* p1 = new int[2]{10, 20};
            hpx::memory::shared_array<int> sa1(p1, custom_array_deleter);
            hpx::memory::shared_array<int> sa2(sa1);
            HPX_TEST_EQ(sa1.use_count(), 2);
            HPX_TEST_EQ(sa2.use_count(), 2);
            HPX_TEST_EQ(deleter_call_count, 0);
        }

        HPX_TEST_EQ(deleter_call_count, 1);

        deleter_call_count = 0;

        {
            int* p = new int[2]{5, 6};
            hpx::memory::shared_array<int> sa(p, [](int* ptr) noexcept {
                ++deleter_call_count;
                delete[] ptr;
            });
            HPX_TEST_EQ(sa.get(), p);
            HPX_TEST_EQ(sa[0], 5);
        }

        HPX_TEST_EQ(deleter_call_count, 1);
    }
}    // namespace n_custom_deleter

namespace n_reset_with_deleter {
    static int deleter_call_count = 0;

    void test()
    {
        deleter_call_count = 0;
        {
            int* p1 = new int[2]{1, 2};
            int* p2 = new int[2]{3, 4};

            hpx::memory::shared_array<int> sa(p1, [](int* ptr) noexcept {
                ++deleter_call_count;
                delete[] ptr;
            });
            HPX_TEST_EQ(sa.get(), p1);
            HPX_TEST_EQ(deleter_call_count, 0);

            sa.reset(p2, [](int* ptr) noexcept {
                ++deleter_call_count;
                delete[] ptr;
            });
            HPX_TEST_EQ(sa.get(), p2);
            HPX_TEST_EQ(deleter_call_count, 1);
        }

        HPX_TEST_EQ(deleter_call_count, 2);
    }
}    // namespace n_reset_with_deleter

namespace n_no_delete {
    static int deleter_call_count = 0;

    void test()
    {
        deleter_call_count = 0;
        int arr[3] = {10, 20, 30};
        {
            hpx::memory::shared_array<int> sa(
                arr, [](int*) noexcept { ++deleter_call_count; });
            HPX_TEST_EQ(sa.get(), arr);
            HPX_TEST_EQ(sa[0], 10);
            HPX_TEST_EQ(sa[2], 30);
        }

        HPX_TEST_EQ(deleter_call_count, 1);
        HPX_TEST_EQ(arr[0], 10);
    }
}    // namespace n_no_delete

namespace n_nullptr_constructor {
    void test()
    {
        {
            hpx::memory::shared_array<int> sa(nullptr);
            HPX_TEST(sa.get() == nullptr);
            HPX_TEST_EQ(sa.use_count(), 0);
            HPX_TEST(!sa);
        }

        {
            hpx::memory::shared_array<int> sa1;
            hpx::memory::shared_array<int> sa2(nullptr);
            HPX_TEST(sa1 == sa2);
            HPX_TEST(sa1 == nullptr);
            HPX_TEST(nullptr == sa2);
        }
    }
}    // namespace n_nullptr_constructor

namespace n_comparison {
    void test()
    {
        {
            hpx::memory::shared_array<int> sa1;
            hpx::memory::shared_array<int> sa2;
            HPX_TEST(sa1 == sa2);
            HPX_TEST(!(sa1 != sa2));
        }

        {
            int* p = new int[2]{1, 2};
            hpx::memory::shared_array<int> sa1(p);
            hpx::memory::shared_array<int> sa2(sa1);
            HPX_TEST(sa1 == sa2);
            HPX_TEST(!(sa1 != sa2));
        }

        {
            int* p1 = new int[2]{1, 2};
            int* p2 = new int[2]{3, 4};
            hpx::memory::shared_array<int> sa1(p1);
            hpx::memory::shared_array<int> sa2(p2);
            HPX_TEST(sa1 != sa2);
            HPX_TEST(!(sa1 == sa2));
        }

        {
            int* p = new int[2]{1, 2};
            hpx::memory::shared_array<int> sa(p);
            HPX_TEST(sa != nullptr);
            HPX_TEST(nullptr != sa);
            HPX_TEST(!(sa == nullptr));
        }
    }
}    // namespace n_comparison

namespace n_unique {
    void test()
    {
        {
            hpx::memory::shared_array<int> sa;
            HPX_TEST(!sa.unique());
        }

        {
            int* p = new int[2]{1, 2};
            hpx::memory::shared_array<int> sa(p);
            HPX_TEST(sa.unique());

            {
                hpx::memory::shared_array<int> sa2(sa);
                HPX_TEST(!sa.unique());
                HPX_TEST(!sa2.unique());
            }

            HPX_TEST(sa.unique());
        }
    }
}    // namespace n_unique

int main()
{
    n_element_type::test();
    n_default_constructor::test();
    n_pointer_constructor::test();
    n_copy_constructor::test();
    n_move_constructor::test();
    n_swap::test();
    n_use_count::test();
    n_get::test();
    n_reset::test();
    n_destructor::test();
    n_multiple_types::test();
    n_custom_deleter::test();
    n_reset_with_deleter::test();
    n_no_delete::test();
    n_nullptr_constructor::test();
    n_comparison::test();
    n_unique::test();

    return hpx::util::report_errors();
}
