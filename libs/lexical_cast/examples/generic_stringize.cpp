// Copyright Antony Polukhin, 2013-2019.

//  SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0.
// (See the accompanying file LICENSE_1_0.txt
// or a copy at <http://www.boost.org/LICENSE_1_0.txt>.)

#include <hpx/config.hpp>
#ifdef HPX_MSVC_WARNING_PRAGMA
#pragma warning(disable : 4512)    // generic_stringize.cpp(37) : warning C4512:
//'stringize_functor' : assignment operator could not be generated
#endif

//[lexical_cast_stringize
/*`
    In this example we'll make a `stringize` method that accepts a sequence, converts
    each element of the sequence into string and appends that string to the result.

    Example is based on the example from the
    [@http://www.packtpub.com/boost-cplusplus-application-development-cookbook/book
    Boost C++ Application Development Cookbook]
    by Antony Polukhin, ISBN 9781849514880.

    Step 1: Making a functor that converts any type to a string and remembers result:
*/

#include <hpx/lexical_cast.hpp>
#include <string>
#include <utility>

struct stringize_functor
{
private:
    std::string& result;

public:
    explicit stringize_functor(std::string& res)
        : result(res)
    {
    }

    template <class T>
    void operator()(const T& v) const
    {
        result += hpx::util::lexical_cast<std::string>(v);
    }
};

//` Step 2: Applying `stringize_functor` to each element in sequence:
#include <boost/fusion/include/for_each.hpp>
template <class Sequence>
std::string stringize(const Sequence& seq)
{
    std::string result;
    boost::fusion::for_each(seq, stringize_functor(result));
    return result;
}

//` Step 3: Using the `stringize` with different types:
#include <boost/fusion/adapted/boost_tuple.hpp>
#include <boost/fusion/adapted/std_pair.hpp>
#include <hpx/assertion.hpp>

int main()
{
    boost::tuple<char, int, char, int> decim('-', 10, 'e', 5);
    HPX_ASSERT(stringize(decim) == "-10e5");

    std::pair<int, std::string> value_and_type(270, "Kelvin");
    HPX_ASSERT(stringize(value_and_type) == "270Kelvin");
}

//] [/lexical_cast_stringize]
