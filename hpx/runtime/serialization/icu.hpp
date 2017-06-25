#ifndef __HPXICU_H__
#define __HPXICU_H__ 1

#include <hpx/config.hpp>
#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/traits/is_bitwise_serializable.hpp>
#include <hpx/include/components.hpp>

#include <unicode/unistr.h>

namespace hpx { namespace serialization {

void serialize(
  hpx::serialization::input_archive &ar,
  icu::UnicodeString &str,
  int) {

  ar >> s;
}

void serialize(
  hpx::serialization::output_archive &ar,
  const icu::UnicodeString &str,
  int) {

  ar << s;
}

}}

#endif

