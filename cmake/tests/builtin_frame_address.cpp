//  Copyright (c) 2024 Isidoros Tsaousis-Seiras
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// test for availability of __builtin_frame_address(0)
// This is needed because circle (build_226) defines __GNUC__
// but does not provide the builtin

int main()
{
    (void)__builtin_frame_address(0);
}
