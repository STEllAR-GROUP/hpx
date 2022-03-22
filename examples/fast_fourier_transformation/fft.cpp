//  Copyright (c) 2022 Deepak Suresh

//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>

#include <iostream>
#include <math.h>
#include <vector>

// For storing complex values of nth roots of unity
using cd = std::complex<double>;

// Function for fast fourier transformation
std::vector<cd> fft(const std::vector<cd>& a)
{
    uint64_t n = a.size();

    // if input contains just one element
    if (n == 1)
        return std::vector<cd>(1, a[0]);

    // For storing n complex nth roots of unity
    std::vector<cd> w(n);
    hpx::experimental::for_loop(hpx::execution::par, 0, n, [&](auto i) {
        double alpha = exp(-2 * M_PI * i / n);
        w[i] = cd(cos(alpha), sin(alpha));
    });

    std::vector<cd> A0(n / 2), A1(n / 2);

    hpx::experimental::for_loop(hpx::execution::par, 0, n / 2, [&](auto j) {
        // even indexed coefficients
        A0[j] = a[j * 2];

        // odd indexed coefficients
        A1[j] = a[j * 2 + 1];
    });

    // Recursive call for even indexed coefficients
    std::vector<cd> y0 = fft(A0);

    // Recursive call for odd indexed coefficients
    std::vector<cd> y1 = fft(A1);

    // for storing values of y0, y1, y2, ..., yn-1.
    std::vector<cd> y(n);

    hpx::experimental::for_loop(hpx::execution::par, 0, n / 2, [&](auto k) {
        y[k] = y0[k] + w[k] * y1[k];
        y[k + n / 2] = y0[k] - w[k] * y1[k];
    });

    return y;
}

int hpx_main()
{
    // 4 element DFT
    std::vector<cd> a = {11, 12, 13, 14};

    std::vector<cd> b = fft(a);

    for (uint64_t i = 0; i < a.size(); i++)
        std::cout << b[i] << std::endl;

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    return hpx::init(argc, argv);
}
