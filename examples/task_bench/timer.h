/* Copyright 2020 Stanford University
 * Copyright 2020 Los Alamos National Laboratory
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
#ifndef TIMER_H
#define TIMER_H

#include <cstddef>

#include <sys/time.h>

struct Timer
{
public:
    static double time_elapsed;

    static inline double get_cur_time()
    {
        struct timeval tv;
        double t;

        gettimeofday(&tv, NULL);
        t = tv.tv_sec + tv.tv_usec / 1e6;
        return t;
    }

    static inline double time_start()
    {
        time_elapsed = get_cur_time();
        return time_elapsed;
    }

    static inline double time_end()
    {
        time_elapsed = get_cur_time() - time_elapsed;
        return time_elapsed;
    }
};

#endif
