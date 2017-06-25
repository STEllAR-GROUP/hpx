#ifndef __HPXVALARRAY_H__
#define __HPXVALARRAY_H__ 1

#include <hpx/config.hpp>
#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/traits/is_bitwise_serializable.hpp>
#include <hpx/include/components.hpp>

#include <valarray>

namespace hpx { namespace serialization {

template<typename T>
void serialize(
  hpx::serialization::input_archive &ar,
  std::valarray<T> &arr,
  int) {

  std::size_t sz = 0;
  ar & sz;
  arr.resize(sz);

  if(sz < 1) { return; }

  for(int i = 0; i < sz; ++i) {
    ar >> arr[i];
  }
}

template<typename T>
void serialize(
  hpx::serialization::output_archive &ar,
  const std::valarray<T> arr,
  int) {

  const std::size_t sz = s.size();
  ar & sz;
  for(auto v : arr) {
    ar << v;
  }
}


} }

#endif

