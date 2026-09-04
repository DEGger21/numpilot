# This file is part of the NumPilot project at Harbin Institute of Technology (HIT).
#
# Copyright (c) 2026 AISS Group, Harbin Institute of Technology (HIT).
# SPDX-License-Identifier: Apache-2.0
#
# Authors (accounts):
# - Su Tonghua <@sutonghua>
# - Li Xing <@li-xingHIT>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Bounded-source checks for the real mini CANNBot generation helper."""

import pytest

import numpilot.demo_llm_operator_generation as generation_tool


def _generation_tool():
    return generation_tool


def _ascendc_fragment():
    return """```ascendc
// UNVERIFIED_DRAFT
__aicore__ inline void add_one(
    AscendC::LocalTensor<float> dst,
    const AscendC::LocalTensor<float> src,
    uint32_t count) {
  AscendC::Adds(dst, src, 1.0f, count);
}
```
"""


def test_cannbot_mini_accepts_only_bounded_ascendc_adds_fragment():
    tool = _generation_tool()
    code = tool._extract_cpp(_ascendc_fragment(), cann_bot_mini=True)
    assert "__aicore__ inline void add_one" in code
    assert "AscendC::Adds(dst, src, 1.0f, count);" in code
    assert "UNVERIFIED_DRAFT" in code


def test_cannbot_mini_rejects_old_scalar_cpp_function():
    tool = _generation_tool()
    response = """```cpp
// UNVERIFIED_DRAFT
float add_one(float x) {
  float y = x + 1.0f;
  return y;
}
```
"""
    with pytest.raises(ValueError, match="Ascend C helper"):
        tool._extract_cpp(response, cann_bot_mini=True)


def test_cannbot_mini_rejects_fragment_without_exact_adds_contract():
    tool = _generation_tool()
    response = _ascendc_fragment().replace(
        "AscendC::Adds(dst, src, 1.0f, count);",
        "AscendC::Add(dst, src, count);",
    )
    with pytest.raises(ValueError, match="exactly one AscendC::Adds"):
        tool._extract_cpp(response, cann_bot_mini=True)


def test_generic_demo_lane_keeps_its_existing_scalar_contract():
    tool = _generation_tool()
    response = """```cpp
// UNVERIFIED_DRAFT
float DemoAddOne(float x) {
  return x + 1.0f;
}
```
"""
    code = tool._extract_cpp(response, cann_bot_mini=False)
    assert "return x + 1.0f" in code
