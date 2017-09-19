// Copyright (c) 2017 Adrian Serio
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// This example tests the functionality of save_checkpoint and 
// restore_checkpoint.
//

#include <hpx/hpx_main.hpp>
#include <hpx/util/checkpoint.hpp>
#include <hpx/util/lightweight_test.hpp>

using hpx::util::checkpoint;
using hpx::util::save_checkpoint;
using hpx::util::restore_checkpoint;

// Main
int main()
{
    //[check_test_1
    char character = 'd';
    int integer = 10;
    float flt = 10.01;
    bool boolean = true;
    std::string str = "I am a string of characters";
    std::vector<char> vec(str.begin(), str.end());
    checkpoint archive;

    //Test 1
    // test basic functionality
    hpx::shared_future<checkpoint> f_archive = save_checkpoint(std::move(archive)
                                         , character
                                         , integer
                                         , flt
                                         , boolean
                                         , str
                                         , vec);
    //]
    //[check_test_2 
    char character2;
    int integer2;
    float flt2;
    bool boolean2;
    std::string str2;
    std::vector<char> vec2;

    restore_checkpoint(
        f_archive.get(), character2, integer2, flt2, boolean2, str2, vec2);
    //]

    HPX_TEST_EQ(character, character2);
    HPX_TEST_EQ(integer, integer2);
    HPX_TEST_EQ(flt, flt2);
    HPX_TEST_EQ(boolean, boolean2);
    HPX_TEST_EQ(str, str2);
    HPX_TEST(vec==vec2);

    //Test 2
    // test asignment operator
    char character3;
    int integer3;
    float flt3;
    bool boolean3;
    std::string str3;
    std::vector<char> vec3;
    checkpoint archive2;

    archive2 = f_archive.get();
    restore_checkpoint(
        archive2, character3, integer3, flt3, boolean3, str3, vec3);

    HPX_TEST_EQ(character, character3);
    HPX_TEST_EQ(integer, integer3);
    HPX_TEST_EQ(flt, flt3);
    HPX_TEST_EQ(boolean, boolean3);
    HPX_TEST_EQ(str, str3);
    HPX_TEST(vec==vec3);

    //Test 3
    // test move semantics
    hpx::future<checkpoint> archive3=
                               save_checkpoint(checkpoint(), vec, integer);
    restore_checkpoint(archive3.get(), vec2, integer2);

    HPX_TEST(vec==vec2);
    HPX_TEST_EQ(integer, integer2);

    //Test 4
    // test sync policy
    std::vector<int> test_vec;
    for (int c = 0; c < 101; c++)
    {
        test_vec.push_back(c);
    }
    checkpoint archive4 = save_checkpoint(hpx::launch::sync
                                        , checkpoint()
                                        , test_vec);
    std::vector<int> test_vec2;
    restore_checkpoint(archive4, test_vec2);
    
    HPX_TEST(test_vec==test_vec2);

    //Test 5
    // test creation of a checkpoint from a checkpoint
    // test proper handling of futures
    checkpoint archive5(archive2);
    hpx::future<std::vector<int>> test_vec2_future =
        hpx::make_ready_future(test_vec2);
    hpx::future<checkpoint> f_check =
        save_checkpoint(std::move(archive5), test_vec2_future);
    hpx::future<std::vector<int>> test_vec3_future;
    restore_checkpoint(f_check.get(), test_vec3_future);
    
    HPX_TEST(test_vec2 == test_vec3_future.get());

    //Test 6
    // test the operator= constructor
    checkpoint archive6;
    archive6 = std::move(archive5);

    HPX_TEST(archive6==archive5);

    //Test 7
    // test writing to a file
    // test .load()
    std::ofstream test_file_7("checkpoint_test_file.txt");
    std::vector<float> vec7{1.02, 1.03, 1.04, 1.05};
    hpx::future<checkpoint> fut_7=save_checkpoint(vec7);
    checkpoint archive7 = fut_7.get();
    test_file_7.write(archive7.data.data(),archive7.size());
    test_file_7.close();

    std::vector<float> vec7_1;
    checkpoint archive7_1;
    archive7_1.load("checkpoint_test_file.txt");
    restore_checkpoint(archive7_1, vec7_1);

    HPX_TEST(vec7==vec7_1);
    
    //Test 8
    // test policies
    int a8=10, b8=20, c8=30;
    hpx::future<checkpoint>f_8=save_checkpoint(hpx::launch::async, a8, b8, c8);
    int a8_1, b8_1, c8_1;
    restore_checkpoint(f_8.get(), a8_1, b8_1, c8_1);
    checkpoint archive8 = save_checkpoint(hpx::launch::sync, a8_1, b8_1, c8_1);
    int a8_2, b8_2, c8_2;
    restore_checkpoint(archive8, a8_2, b8_2, c8_2);

    HPX_TEST(a8==a8_2);
    HPX_TEST(b8==b8_2);
    HPX_TEST(c8==c8_2);

    return 0;
}

