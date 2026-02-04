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
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "core_random.h"

#include "siphash.h"

#ifdef TEST_HARNESS
#include <assert.h>
#include <math.h>
#include <stdio.h>
#endif

void gen_bits(void const* input, size_t input_bytes, void* output)
{
    uint8_t const k[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    siphash(input, input_bytes, k, output, sizeof(uint64_t));
}

double random_uniform(void const* input, size_t input_bytes)
{
    uint64_t bits;
    gen_bits(input, input_bytes, &bits);

    return ((double) bits) * 0x1.p-64;
}

#ifdef TEST_HARNESS
int main()
{
    constexpr size_t num_buckets = 1024;
    size_t histogram[num_buckets] = {0};

    constexpr size_t num_samples = 1 << 20;
    for (size_t sample = 0; sample < num_samples; sample++)
    {
        double v = random_uniform(&sample, sizeof(sample));
        size_t bucket = size_t(floor(v * num_buckets));
        assert(bucket >= 0 && bucket < num_buckets);
        histogram[bucket]++;
    }

    for (size_t bucket = 0; bucket < num_buckets; ++bucket)
    {
        printf("%lu\n", histogram[bucket]);
    }

    return 0;
}
#endif
