/* Copyright 2020 Alexey Chernov <4ernov@gmail.com>
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

#include "../onnx_importer.h"

#include <cassert>

#include <hlir/graph.h>
#include <hlir/ops/binary.h>
#include <hlir/ops/constant.h>
#include <hlir/ops/reduce.h>
#include <hlir/ops/unary.h>
#include <hlir/ir_types.h>

using namespace std;

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::hlir;

using namespace onnx;

void onnx_importer::convert_op_Relu(const NodeProto &node)
{
    assert(node.input().size() == 1);
    assert(node.output().size() == 1);

    const auto &input { node.input()[0] };
    const auto &output { node.output()[0] };

    auto&& in_shape = get_shape(input);

    auto zero { graph_.emplace<constant>(0.f) };
    auto max { graph_.emplace<binary>(binary_max, move(in_shape), zero->output().shape(), value_range<float>::full()) };

    max->input_b().connect(zero->output());

    input_tensors_.emplace(&max->input_a(), input);
    output_tensors_.emplace(output, &max->output());
}

void onnx_importer::convert_op_Sigmoid(const NodeProto &node)
{
    assert(node.input().size() == 1);
    assert(node.output().size() == 1);

    const auto &input { node.input()[0] };
    const auto &output { node.output()[0] };

    auto in_shape = get_shape(input);

    auto neg = graph_.emplace<unary>(unary_neg, in_shape);
    auto exp = graph_.emplace<unary>(unary_exp, neg->output().shape());
    auto one = graph_.emplace<constant>(1.f);
    auto plus = graph_.emplace<binary>(binary_add, one->output().shape(), exp->output().shape(), value_range<float>::full());
    auto div = graph_.emplace<binary>(binary_div, one->output().shape(), plus->output().shape(), value_range<float>::full());

    exp->input().connect(neg->output());
    plus->input_a().connect(one->output());
    plus->input_b().connect(exp->output());
    div->input_a().connect(one->output());
    div->input_b().connect(plus->output());

    input_tensors_.emplace(&neg->input(), input);
    output_tensors_.emplace(output, &div->output());
}

void onnx_importer::convert_op_Clip(const NodeProto &node)
{
    //fprintf(stderr, "clip!\n");
    assert(node.input().size() > 1);
    assert(node.output().size() == 1);

    const auto &input { node.input()[0] };

    if (node.input_size() == 2) {
        //fprintf(stderr, "min only!\n");
        const auto &min_v { node.input()[1] };
        const auto &output { node.output()[0] };

        auto&& in_shape = get_shape(input);
        auto max { graph_.emplace<binary>(binary_max, move(in_shape), get_shape(min_v), value_range<float>::full()) };
        
        input_tensors_.emplace(&max->input_b(), min_v);
        input_tensors_.emplace(&max->input_a(), input);
        output_tensors_.emplace(output, &max->output());
    } else {
        //fprintf(stderr, "min - max!\n");
        const auto &min_v { node.input()[1] };
        const auto &max_v { node.input()[2] };
        const auto &output { node.output()[0] };

        auto&& in_shape = get_shape(input);
        //fprintf(stderr, "input shape is %s", hlir::to_string(in_shape).c_str());
        auto max { graph_.emplace<binary>(binary_max, move(in_shape), move(get_shape(min_v)), value_range<float>::full()) };
        auto min { graph_.emplace<binary>(binary_min, max->output().shape(), move(get_shape(max_v)), value_range<float>::full()) };
        //fprintf(stderr, "after op\n");
        min->input_a().connect(max->output());
        //fprintf(stderr, "connected!\n");

        input_tensors_.emplace(&min->input_b(), max_v);
        input_tensors_.emplace(&max->input_b(), min_v);
        input_tensors_.emplace(&max->input_a(), input);
        //fprintf(stderr, "adding output for %s\n", output.c_str());
        output_tensors_.emplace(output, &min->output());
    }
}

void onnx_importer::convert_op_LeakyRelu(const NodeProto &node)
{
    assert(node.input().size() == 1);
    assert(node.output().size() == 1);

    const auto &input { node.input()[0] };
    const auto &output { node.output()[0] };
    auto in_shape1 = get_shape(input);
    auto in_shape2 = get_shape(input);

    const auto alpha_value { get_attribute<float>(node, "alpha").value() };
    const auto& alpha { graph_.emplace<constant>(alpha_value) };
    //fprintf(stderr, "cname = %s\n", alpha->name().c_str());

    auto mul = graph_.emplace<binary>(binary_mul, move(in_shape1), alpha->output().shape(), value_range<float>::full());
    auto max = graph_.emplace<binary>(binary_max, move(in_shape2), mul->output().shape(), value_range<float>::full());

    mul->input_b().connect(alpha->output());
    max->input_b().connect(mul->output());

    input_tensors_.emplace(&mul->input_a(), input);
    input_tensors_.emplace(&max->input_a(), input);
    output_tensors_.emplace(output, &max->output());
}
