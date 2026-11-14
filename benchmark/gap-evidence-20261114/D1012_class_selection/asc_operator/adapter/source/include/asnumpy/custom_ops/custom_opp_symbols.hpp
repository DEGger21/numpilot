/******************************************************************************
 * Copyright (c) 2026 AISS Group at Harbin Institute of Technology. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *****************************************************************************/

#pragma once

#include <dlfcn.h>

#include <stdexcept>
#include <string>

#ifndef ASNUMPY_CUSTOM_OPP_LIBRARY
#error "ASNUMPY_CUSTOM_OPP_LIBRARY must name the vendor libcust_opapi.so"
#endif

namespace asnumpy {
namespace custom_opp {

// Custom operators extend CANN, and some of them share their ACLNN entry-point
// names with built-in ops: ``aclnnFlip``, for example, also exists in CANN's
// ``libopapi.so`` with a different signature.  A plain link therefore does not
// guarantee that a call reaches the vendor library -- the dynamic linker may
// bind the reference to the built-in symbol first, and that call crashes.
// Resolving each entry point from the vendor library handle is deterministic
// regardless of link order or of which other CANN libraries are loaded.
template <typename Fn>
Fn LoadSymbol(const char* name) {
    static void* handle = nullptr;
    if (handle == nullptr) {
        handle = ::dlopen(ASNUMPY_CUSTOM_OPP_LIBRARY, RTLD_NOW | RTLD_LOCAL);
        if (handle == nullptr) {
            throw std::runtime_error(
                std::string("cannot load custom OPP library: ") + ::dlerror());
        }
    }
    ::dlerror();
    void* symbol = ::dlsym(handle, name);
    if (symbol == nullptr) {
        throw std::runtime_error(
            std::string("custom OPP library lacks symbol ") + name);
    }
    return reinterpret_cast<Fn>(symbol);
}

} // namespace custom_opp
} // namespace asnumpy
