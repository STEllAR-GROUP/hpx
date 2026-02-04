/* Copyright 2020 Stanford University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef CORE_RANDOM_H
#define CORE_RANDOM_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

double random_uniform(const void *input, size_t input_bytes);

#ifdef __cplusplus
}
#endif

#endif
