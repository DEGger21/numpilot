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

#include <asnumpy/custom_ops/flip.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace asnumpy {

void bind_custom_ops(py::module_& module) {
    module.def(
        "flip",
        &NumpyFlip,
        py::arg("a").noconvert(),
        py::arg("axis") = 1,
        "Compute ``asnumpy.flip`` on the NPU.\n\n"
        "Admitted profile: rank 2 float32, axis = 1 (or its rank-2 alias -1). Unsupported dtypes, ranks, shapes, "
        "axes and keywords are rejected explicitly. No implicit cast or CPU "
        "fallback is provided.");
}

} // namespace asnumpy

// Defensive top-level forwarding keeps the binding usable with either a
// global or namespace-qualified declaration in module.cpp.
void bind_custom_ops(py::module_& module) {
    asnumpy::bind_custom_ops(module);
}
