// (C) Copyright Dave Abrahams and Thomas Becker 2003.
//
//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/util/lightweight_test.hpp>
#include <hpx/util/tuple.hpp>
#include <hpx/util/transform_iterator.hpp>
#include <hpx/util/zip_iterator.hpp>

#include <cstddef>
#include <functional>
#include <list>
#include <set>
#include <string>
#include <type_traits>
#include <vector>

// Tests for https://svn.boost.org/trac/boost/ticket/1517
int to_value(std::list<int>::const_iterator v)
{
    return *v;
}

void category_test()
{
    std::list<int> rng1;
    std::string rng2;

    hpx::util::make_zip_iterator(
        hpx::util::make_tuple(
            // BidirectionalInput
            hpx::util::make_transform_iterator(rng1.begin(), &to_value),
            rng2.begin() // RandomAccess
        )
    );
}
//

int main(void)
{
    category_test();

//     size_t num_successful_tests = 0;
//     size_t num_failed_tests = 0;

    /////////////////////////////////////////////////////////////////////////////
    //
    // Zip iterator construction and dereferencing
    //
    /////////////////////////////////////////////////////////////////////////////

    std::vector<double> vect1(3);
    vect1[0] = 42.;
    vect1[1] = 43.;
    vect1[2] = 44.;

    std::set<int> intset;
    intset.insert(52);
    intset.insert(53);
    intset.insert(54);
    //

    typedef hpx::util::zip_iterator<
            std::set<int>::iterator, std::vector<double>::iterator
        > zit_mixed;

    zit_mixed zip_it_mixed =
        zit_mixed(hpx::util::make_tuple(intset.begin(), vect1.begin()));

    hpx::util::tuple<int, double> val_tuple(*zip_it_mixed);

    hpx::util::tuple<const int&, double&> ref_tuple(*zip_it_mixed);

    double dblOldVal = hpx::util::get<1>(ref_tuple);
    hpx::util::get<1>(ref_tuple) -= 41.;

    HPX_TEST(52 == hpx::util::get<0>(val_tuple) &&
        42. == hpx::util::get<1>(val_tuple) &&
        52 == hpx::util::get<0>(ref_tuple) &&
        1. == hpx::util::get<1>(ref_tuple) && 1. == *vect1.begin());

    // Undo change to vect1
    hpx::util::get<1>(ref_tuple) = dblOldVal;

    /////////////////////////////////////////////////////////////////////////////
    //
    // Zip iterator with 12 components
    //
    /////////////////////////////////////////////////////////////////////////////

    // Declare 12 containers
    //
    std::list<int> li1;
    li1.push_back(1);
    std::set<int> se1;
    se1.insert(2);
    std::vector<int> ve1;
    ve1.push_back(3);
    //
    std::list<int> li2;
    li2.push_back(4);
    std::set<int> se2;
    se2.insert(5);
    std::vector<int> ve2;
    ve2.push_back(6);
    //
    std::list<int> li3;
    li3.push_back(7);
    std::set<int> se3;
    se3.insert(8);
    std::vector<int> ve3;
    ve3.push_back(9);
    //
    std::list<int> li4;
    li4.push_back(10);
    std::set<int> se4;
    se4.insert(11);
    std::vector<int> ve4;
    ve4.push_back(12);

    // typedefs for cons lists of iterators.
    typedef hpx::util::detail::tuple_cat_result<
            hpx::util::tuple<
                std::set<int>::iterator
            >,
            hpx::util::tuple<
                std::vector<int>::iterator, std::list<int>::iterator,
                std::set<int>::iterator, std::vector<int>::iterator,
                std::list<int>::iterator, std::set<int>::iterator,
                std::vector<int>::iterator, std::list<int>::iterator,
                std::set<int>::iterator, std::vector<int>::const_iterator
            >
        >::type cons_11_its_type;
    //
    typedef hpx::util::detail::tuple_cat_result<
            hpx::util::tuple<
                std::list<int>::const_iterator
            >,
            cons_11_its_type
        >::type cons_12_its_type;

    // typedefs for cons lists for dereferencing the zip iterator
    // made from the cons list above.
    typedef hpx::util::detail::tuple_cat_result<
            hpx::util::tuple<
                const int&
            >,
            hpx::util::tuple<
                int&, int&, const int&, int&, int&, const int&,
                int&, int&, const int&, const int&
            >
        >::type cons_11_refs_type;
    //
    typedef hpx::util::detail::tuple_cat_result<
            hpx::util::tuple<
                const int&
            >,
            cons_11_refs_type
        >::type cons_12_refs_type;

    // typedef for zip iterator with 12 elements
    typedef hpx::util::zip_iterator<cons_12_its_type> zip_it_12_type;

    // Declare a 12-element zip iterator.
    zip_it_12_type zip_it_12(
        li1.begin(), se1.begin(), ve1.begin(), li2.begin(), se2.begin(),
        ve2.begin(), li3.begin(), se3.begin(), ve3.begin(), li4.begin(),
        se4.begin(), ve4.begin());

    // Dereference, mess with the result a little.
    cons_12_refs_type zip_it_12_dereferenced(*zip_it_12);
    hpx::util::get<9>(zip_it_12_dereferenced) = 42;

    // Make a copy and move it a little to force some instantiations.
    zip_it_12_type zip_it_12_copy(zip_it_12);
    ++zip_it_12_copy;

    HPX_TEST(
        hpx::util::get<11>(zip_it_12.get_iterator_tuple()) == ve4.begin() &&
        hpx::util::get<11>(zip_it_12_copy.get_iterator_tuple()) ==
            ve4.end() &&
        1 == hpx::util::get<0>(zip_it_12_dereferenced) &&
        12 == hpx::util::get<11>(zip_it_12_dereferenced) &&
        42 == *(li4.begin()));

    /////////////////////////////////////////////////////////////////////////////
    //
    // Zip iterator incrementing and dereferencing
    //
    /////////////////////////////////////////////////////////////////////////////

    std::vector<double> vect2(3);
    vect2[0] = 2.2;
    vect2[1] = 3.3;
    vect2[2] = 4.4;

    hpx::util::zip_iterator<hpx::util::tuple<
        std::vector<double>::const_iterator,
        std::vector<double>::const_iterator
    > > zip_it_begin(hpx::util::make_tuple(vect1.begin(), vect2.begin()));

    hpx::util::zip_iterator<hpx::util::tuple<
        std::vector<double>::const_iterator,
        std::vector<double>::const_iterator
    > > zip_it_run(hpx::util::make_tuple(vect1.begin(), vect2.begin()));

    hpx::util::zip_iterator<hpx::util::tuple<
        std::vector<double>::const_iterator,
        std::vector<double>::const_iterator
    > > zip_it_end(hpx::util::make_tuple(vect1.end(), vect2.end()));

    HPX_TEST(
        zip_it_run == zip_it_begin &&
        42. == hpx::util::get<0>(*zip_it_run) &&
        2.2 == hpx::util::get<1>(*zip_it_run) &&
        43. == hpx::util::get<0>(*(++zip_it_run)) &&
        3.3 == hpx::util::get<1>(*zip_it_run) &&
        44. == hpx::util::get<0>(*(++zip_it_run)) &&
        4.4 == hpx::util::get<1>(*zip_it_run) && zip_it_end == ++zip_it_run);

    /////////////////////////////////////////////////////////////////////////////
    //
    // Zip iterator decrementing and dereferencing
    //
    /////////////////////////////////////////////////////////////////////////////

    HPX_TEST(
        zip_it_run == zip_it_end && zip_it_end == zip_it_run-- &&
        44. == hpx::util::get<0>(*zip_it_run) &&
        4.4 == hpx::util::get<1>(*zip_it_run) &&
        43. == hpx::util::get<0>(*(--zip_it_run)) &&
        3.3 == hpx::util::get<1>(*zip_it_run) &&
        42. == hpx::util::get<0>(*(--zip_it_run)) &&
        2.2 == hpx::util::get<1>(*zip_it_run) && zip_it_begin == zip_it_run);

    /////////////////////////////////////////////////////////////////////////////
    //
    // Zip iterator copy construction and equality
    //
    /////////////////////////////////////////////////////////////////////////////

    hpx::util::zip_iterator<
        hpx::util::tuple<std::vector<double>::const_iterator,
            std::vector<double>::const_iterator>>
        zip_it_run_copy(zip_it_run);

    HPX_TEST(zip_it_run == zip_it_run && zip_it_run == zip_it_run_copy);

    /////////////////////////////////////////////////////////////////////////////
    //
    // Zip iterator inequality
    //
    /////////////////////////////////////////////////////////////////////////////

    HPX_TEST(
        !(zip_it_run != zip_it_run_copy) &&
        zip_it_run != ++zip_it_run_copy);

    /////////////////////////////////////////////////////////////////////////////
    //
    // Zip iterator less than
    //
    /////////////////////////////////////////////////////////////////////////////

    // Note: zip_it_run_copy == zip_it_run + 1
    //
    HPX_TEST(
        zip_it_run < zip_it_run_copy && !(zip_it_run < --zip_it_run_copy) &&
        zip_it_run == zip_it_run_copy);

    /////////////////////////////////////////////////////////////////////////////
    //
    // Zip iterator less than or equal
    //
    /////////////////////////////////////////////////////////////////////////////

    // Note: zip_it_run_copy == zip_it_run
    //
    ++zip_it_run;
    zip_it_run_copy += 2;

    HPX_TEST(
        zip_it_run <= zip_it_run_copy && zip_it_run <= --zip_it_run_copy &&
        !(zip_it_run <= --zip_it_run_copy) && zip_it_run <= zip_it_run);

    /////////////////////////////////////////////////////////////////////////////
    //
    // Zip iterator greater than
    //
    /////////////////////////////////////////////////////////////////////////////

    // Note: zip_it_run_copy == zip_it_run - 1
    //
    HPX_TEST(
        zip_it_run > zip_it_run_copy && !(zip_it_run > ++zip_it_run_copy) &&
        zip_it_run == zip_it_run_copy);

    /////////////////////////////////////////////////////////////////////////////
    //
    // Zip iterator greater than or equal
    //
    /////////////////////////////////////////////////////////////////////////////

    ++zip_it_run;

    // Note: zip_it_run == zip_it_run_copy + 1
    //
    HPX_TEST(
        zip_it_run >= zip_it_run_copy && --zip_it_run >= zip_it_run_copy &&
        !(zip_it_run >= ++zip_it_run_copy));

    /////////////////////////////////////////////////////////////////////////////
    //
    // Zip iterator + int
    //
    /////////////////////////////////////////////////////////////////////////////

    // Note: zip_it_run == zip_it_run_copy - 1
    //
    zip_it_run = zip_it_run + 2;
    ++zip_it_run_copy;

    HPX_TEST(zip_it_run == zip_it_run_copy && zip_it_run == zip_it_begin + 3);

    /////////////////////////////////////////////////////////////////////////////
    //
    // Zip iterator - int
    //
    /////////////////////////////////////////////////////////////////////////////

    // Note: zip_it_run == zip_it_run_copy, and both are at end position
    //
    zip_it_run = zip_it_run - 2;
    --zip_it_run_copy;
    --zip_it_run_copy;

    HPX_TEST(zip_it_run == zip_it_run_copy && (zip_it_run - 1) == zip_it_begin);

    /////////////////////////////////////////////////////////////////////////////
    //
    // Zip iterator +=
    //
    /////////////////////////////////////////////////////////////////////////////

    // Note: zip_it_run == zip_it_run_copy, and both are at begin + 1
    //
    zip_it_run += 2;
    HPX_TEST(zip_it_run == zip_it_begin + 3);

    /////////////////////////////////////////////////////////////////////////////
    //
    // Zip iterator -=
    //
    /////////////////////////////////////////////////////////////////////////////

    // Note: zip_it_run is at end position, zip_it_run_copy is at
    // begin plus one.
    //
    zip_it_run -= 2;
    HPX_TEST(zip_it_run == zip_it_run_copy);

    /////////////////////////////////////////////////////////////////////////////
    //
    // Zip iterator getting member iterators
    //
    /////////////////////////////////////////////////////////////////////////////

    // Note: zip_it_run and zip_it_run_copy are both at
    // begin plus one.
    //
    HPX_TEST(
        hpx::util::get<0>(zip_it_run.get_iterator_tuple()) == vect1.begin() + 1 &&
        hpx::util::get<1>(zip_it_run.get_iterator_tuple()) == vect2.begin() + 1);

    /////////////////////////////////////////////////////////////////////////////
    //
    // Making zip iterators
    //
    /////////////////////////////////////////////////////////////////////////////

    std::vector<hpx::util::tuple<double, double>> vect_of_tuples(3);

    std::copy(
        hpx::util::make_zip_iterator(
            hpx::util::make_tuple(vect1.begin(), vect2.begin())
        ),
        hpx::util::make_zip_iterator(
            hpx::util::make_tuple(vect1.end(), vect2.end())
        ),
        vect_of_tuples.begin());

    HPX_TEST(
        42. == hpx::util::get<0>(*vect_of_tuples.begin()) &&
        2.2 == hpx::util::get<1>(*vect_of_tuples.begin()) &&
        43. == hpx::util::get<0>(*(vect_of_tuples.begin() + 1)) &&
        3.3 == hpx::util::get<1>(*(vect_of_tuples.begin() + 1)) &&
        44. == hpx::util::get<0>(*(vect_of_tuples.begin() + 2)) &&
        4.4 == hpx::util::get<1>(*(vect_of_tuples.begin() + 2)));

    /////////////////////////////////////////////////////////////////////////////
    //
    // Zip iterator non-const --> const conversion
    //
    /////////////////////////////////////////////////////////////////////////////

    hpx::util::zip_iterator<hpx::util::tuple<
        std::set<int>::const_iterator,
        std::vector<double>::const_iterator
    > > zip_it_const(hpx::util::make_tuple(intset.begin(), vect2.begin()));
    //
    hpx::util::zip_iterator<hpx::util::tuple<
        std::set<int>::iterator,
        std::vector<double>::const_iterator
    > > zip_it_half_const(hpx::util::make_tuple(intset.begin(), vect2.begin()));
    //
    hpx::util::zip_iterator<hpx::util::tuple<
        std::set<int>::iterator,
        std::vector<double>::iterator
    > > zip_it_non_const(hpx::util::make_tuple(intset.begin(), vect2.begin()));

    zip_it_half_const = ++zip_it_non_const;
    zip_it_const = zip_it_half_const;
    ++zip_it_const;

    // Error: can't convert from const to non-const
    //  zip_it_non_const = ++zip_it_const;

    HPX_TEST(
        54 == hpx::util::get<0>(*zip_it_const) &&
        4.4 == hpx::util::get<1>(*zip_it_const) &&
        53 == hpx::util::get<0>(*zip_it_half_const) &&
        3.3 == hpx::util::get<1>(*zip_it_half_const));

    /////////////////////////////////////////////////////////////////////////////
    //
    // Zip iterator categories
    //
    /////////////////////////////////////////////////////////////////////////////

    // The big iterator of the previous test has vector, list, and set iterators.
    // Therefore, it must be bidirectional, but not random access.
    bool bBigItIsBidirectionalIterator =
        std::is_convertible<
            hpx::util::zip_iterator_category<zip_it_12_type>::type,
            std::bidirectional_iterator_tag>::value;

    bool bBigItIsRandomAccessIterator =
        std::is_convertible<
            hpx::util::zip_iterator_category<zip_it_12_type>::type,
            std::random_access_iterator_tag>::value;

    // A combining iterator with all vector iterators must have random access
    // traversal.
    //
    typedef hpx::util::zip_iterator<
        hpx::util::tuple<std::vector<double>::const_iterator,
            std::vector<double>::const_iterator>>
        all_vects_type;

    bool bAllVectsIsRandomAccessIterator =
        std::is_convertible<
            hpx::util::zip_iterator_category<all_vects_type>::type,
            std::random_access_iterator_tag>::value;

    // The big test.
    HPX_TEST(
        bBigItIsBidirectionalIterator && !bBigItIsRandomAccessIterator &&
        bAllVectsIsRandomAccessIterator);

    return hpx::util::report_errors();
}
