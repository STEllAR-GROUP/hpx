//  Copyright (c) 2023 Isidoros Tsaousis-Seiras
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/init.hpp>
#include <hpx/execution_base/stdexec_forward.hpp>

#include <hpx/modules/testing.hpp>

#include <utility>

#ifdef HPX_HAVE_STDEXEC
using namespace hpx::execution::experimental;

int main() {
	auto x = just(42);

	auto [a] = sync_wait(std::move(x)).value();

  HPX_TEST(a == 42);

  return hpx::util::report_errors();
}
#else
int main() {}
#endif