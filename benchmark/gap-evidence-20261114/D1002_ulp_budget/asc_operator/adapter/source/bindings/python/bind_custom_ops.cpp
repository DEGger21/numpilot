#include <asnumpy/custom_ops/spacing.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace asnumpy {
void bind_custom_ops(py::module_& module) {
    module.def("spacing", &NumpySpacing, py::arg("a").noconvert(),
               "NumPy-compatible asnumpy.spacing on the NPU.");
}
} // namespace asnumpy

void bind_custom_ops(py::module_& module) {
    asnumpy::bind_custom_ops(module);
}
