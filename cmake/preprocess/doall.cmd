rem Copyright (c) 2007-2012 Hartmut Kaiser
rem
rem Distributed under the Boost Software License, Version 1.0. (See accompanying
rem file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

start wave -o- -DHPX_LIMIT=5 preprocess_hpx.cpp --license=preprocess_license.hpp
start wave -o- -DHPX_LIMIT=10 preprocess_hpx.cpp --license=preprocess_license.hpp
start wave -o- -DHPX_LIMIT=20 preprocess_hpx.cpp --license=preprocess_license.hpp 
rem start wave -o- -DHPX_LIMIT=30 preprocess_hpx.cpp --license=preprocess_license.hpp
rem start wave -o- -DHPX_LIMIT=40 preprocess_hpx.cpp --license=preprocess_license.hpp
rem start wave -o- -DHPX_LIMIT=50 preprocess_hpx.cpp --license=preprocess_license.hpp
