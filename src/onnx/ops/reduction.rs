// Reduction operators: ReduceMean, ReduceSum, ReduceMax, ReduceMin

use crate::ast::Node;
use crate::onnx::convert::{sanitize_identifier, OnnxError};
use crate::onnx::ops::{ConversionContext, ConversionResult, OpHandler};
use crate::protos::onnx::NodeProto;

pub struct ReductionHandler;

impl OpHandler for ReductionHandler {
    fn supports(&self, op_type: &str) -> bool {
        matches!(
            op_type,
            "ReduceMean" | "ReduceSum" | "ReduceMax" | "ReduceMin" | "CumSum"
        )
    }

    fn convert(
        &self,
        node: &NodeProto,
        context: &ConversionContext,
    ) -> Result<ConversionResult, OnnxError> {
        let op_type = node.op_type.as_str();
        let node_name = if !node.name.is_empty() {
            node.name.as_str().to_string()
        } else {
            "unnamed".to_string()
        };

        match op_type {
            "ReduceMean" => self.convert_reduce(node, &node_name, "reduceMean", context),
            "ReduceSum" => self.convert_reduce(node, &node_name, "reduceSum", context),
            "ReduceMax" => self.convert_reduce(node, &node_name, "reduceMax", context),
            "ReduceMin" => self.convert_reduce(node, &node_name, "reduceMin", context),
            "CumSum" => self.convert_cumsum(node, &node_name, context),
            _ => Err(OnnxError::UnsupportedOp {
                op: op_type.to_string(),
                node: node_name,
            }),
        }
    }
}

impl ReductionHandler {
    /// Convert ONNX reduce operations to WebNN reduce operations
    fn convert_reduce(
        &self,
        node: &NodeProto,
        node_name: &str,
        webnn_op: &str,
        context: &ConversionContext,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.is_empty() {
            return Err(OnnxError::InvalidShape(format!(
                "{} expects at least 1 input",
                webnn_op
            )));
        }

        // Extract attributes
        let mut axes: Option<Vec<i64>> = None;
        let mut keepdims = 1i64; // ONNX default is 1 (keep dimensions)

        for attr in node.attribute.as_slice() {
            match attr.name.as_str() {
                "axes" => {
                    axes = Some(attr.ints.clone());
                }
                "keepdims" => {
                    if attr.i != 0 {
                        keepdims = attr.i;
                    }
                }
                _ => {}
            }
        }

        // In opset >= 18, axes can be a second input (initializer)
        if axes.is_none() && inputs.len() >= 2 {
            let axes_input = &inputs[1];
            // Try to read from initializers
            if let Some(tensor) = context.initializers.get(axes_input) {
                let raw = tensor.raw_data.as_slice();
                if !raw.is_empty() {
                    if tensor.data_type == crate::protos::onnx::TensorProto_DataType::Int64 as i32 {
                        axes = Some(
                            raw.chunks_exact(8)
                                .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                                .collect()
                        );
                    } else if tensor.data_type == crate::protos::onnx::TensorProto_DataType::Int32 as i32 {
                        axes = Some(
                            raw.chunks_exact(4)
                                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as i64)
                                .collect()
                        );
                    }
                } else if !tensor.int64_data.as_slice().is_empty() {
                    axes = Some(tensor.int64_data.as_slice().to_vec());
                } else if !tensor.int32_data.as_slice().is_empty() {
                    axes = Some(tensor.int32_data.as_slice().iter().map(|&v| v as i64).collect());
                }
            }
            // Try to read from const_values (from constant folding)
            if axes.is_none() {
                if let Some(values) = context.const_values.get(axes_input) {
                    axes = Some(values.clone());
                }
            }
        }

        let output_name = if node.output.as_slice().is_empty() {
            format!("{}_output", node_name)
        } else {
            sanitize_identifier(&node.output.as_slice()[0].to_string())
        };

        let input0 = context.resolve_input(&inputs[0]);

        let mut options = crate::onnx::ops::create_options_with_label(&node_name);

        // Add axes if specified, normalizing negative indices
        if let Some(axes_values) = axes {
            // Get input tensor rank to normalize negative axes
            let normalized_axes = if let Some(shape) = context.value_shapes.get(&inputs[0]) {
                let rank = shape.len() as i64;
                axes_values
                    .iter()
                    .map(|&axis| if axis < 0 { rank + axis } else { axis })
                    .collect::<Vec<i64>>()
            } else {
                // If shape unknown, pass through as-is (shouldn't happen in practice)
                axes_values
            };
            options.insert("axes".to_string(), serde_json::json!(normalized_axes));
        }

        // Add keepDims option (WebNN uses keepDimensions)
        options.insert(
            "keepDimensions".to_string(),
            serde_json::json!(keepdims != 0),
        );

        let mut result = ConversionResult::new(vec![Node {
            id: output_name.clone(),
            op: webnn_op.to_string(),
            inputs: vec![input0],
            options,
            outputs: None,
        }]);

        if let Some(output) = node.output.as_slice().first() {
            result
                .output_mappings
                .insert(output.to_string(), output_name.clone());
        }

        Ok(result)
    }

    /// Convert ONNX CumSum to WebNN cumulativeSum
    /// ONNX: CumSum(x, axis) with exclusive/reverse attributes
    /// WebNN: cumulativeSum(input, axis, options)
    fn convert_cumsum(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &ConversionContext,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.len() != 2 {
            return Err(OnnxError::InvalidShape(format!(
                "CumSum expects 2 inputs (x, axis), got {}",
                inputs.len()
            )));
        }

        let output_name = if node.output.as_slice().is_empty() {
            format!("{}_output", node_name)
        } else {
            sanitize_identifier(&node.output.as_slice()[0].to_string())
        };

        let input0 = context.resolve_input(&inputs[0]);

        // Extract axis value from second input (must be a constant scalar)
        let axis_value = self.extract_axis_value(&inputs[1], context)?;

        // Extract attributes
        let mut exclusive = false;
        let mut reversed = false;

        for attr in node.attribute.as_slice() {
            match attr.name.as_str() {
                "exclusive" => {
                    exclusive = attr.i != 0;
                }
                "reverse" => {
                    reversed = attr.i != 0;
                }
                _ => {}
            }
        }

        let mut options = crate::onnx::ops::create_options_with_label(&node_name);
        if exclusive {
            options.insert("exclusive".to_string(), serde_json::json!(true));
        }
        if reversed {
            options.insert("reversed".to_string(), serde_json::json!(true));
        }

        // WebNN cumulativeSum takes axis as direct parameter, not as input
        // We need to handle this specially in emit_js.rs
        options.insert("axis".to_string(), serde_json::json!(axis_value));

        let mut result = ConversionResult::new(vec![Node {
            id: output_name.clone(),
            op: "cumulativeSum".to_string(),
            inputs: vec![input0],
            options,
            outputs: None,
        }]);

        if let Some(output) = node.output.as_slice().first() {
            result
                .output_mappings
                .insert(output.to_string(), output_name.clone());
        }

        Ok(result)
    }

    /// Extract axis value from a tensor input (must be a constant scalar)
    fn extract_axis_value(
        &self,
        input_name: &str,
        context: &ConversionContext,
    ) -> Result<i64, OnnxError> {
        // Check if it's an initializer (constant)
        if let Some(tensor) = context.initializers.get(input_name) {
            let data_type = tensor.data_type;

            // Extract scalar value based on data type
            match data_type {
                x if x == crate::protos::onnx::TensorProto_DataType::Int64 as i32 => {
                    if !tensor.int64_data.is_empty() {
                        return Ok(tensor.int64_data[0]);
                    } else if tensor.raw_data.len() >= 8 {
                        let bytes = &tensor.raw_data[0..8];
                        return Ok(i64::from_le_bytes([
                            bytes[0], bytes[1], bytes[2], bytes[3],
                            bytes[4], bytes[5], bytes[6], bytes[7],
                        ]));
                    }
                }
                x if x == crate::protos::onnx::TensorProto_DataType::Int32 as i32 => {
                    if !tensor.int32_data.is_empty() {
                        return Ok(tensor.int32_data[0] as i64);
                    } else if tensor.raw_data.len() >= 4 {
                        let bytes = &tensor.raw_data[0..4];
                        return Ok(i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as i64);
                    }
                }
                _ => {}
            }
        }

        Err(OnnxError::InvalidShape(format!(
            "CumSum axis must be a constant scalar tensor, got: {}",
            input_name
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protos::onnx::{AttributeProto, NodeProto};

    fn create_test_node(op_type: &str, inputs: Vec<&str>, outputs: Vec<&str>) -> NodeProto {
        NodeProto {
            op_type: op_type.to_string(),
            name: format!("test_{}", op_type.to_lowercase()),
            input: inputs.iter().map(|s| s.to_string()).collect(),
            output: outputs.iter().map(|s| s.to_string()).collect(),
            ..Default::default()
        }
    }

    fn add_int_attribute(node: &mut NodeProto, name: &str, value: i64) {
        let attr = AttributeProto {
            name: name.to_string(),
            i: value,
            ..Default::default()
        };
        node.attribute.push(attr);
    }

    fn add_ints_attribute(node: &mut NodeProto, name: &str, values: Vec<i64>) {
        let attr = AttributeProto {
            name: name.to_string(),
            ints: values,
            ..Default::default()
        };
        node.attribute.push(attr);
    }

    #[test]
    fn test_reduction_handler_supports() {
        let handler = ReductionHandler;
        assert!(handler.supports("ReduceMean"));
        assert!(handler.supports("ReduceSum"));
        assert!(handler.supports("ReduceMax"));
        assert!(handler.supports("ReduceMin"));
        assert!(!handler.supports("Add"));
    }

    #[test]
    fn test_convert_reduce_mean() {
        let handler = ReductionHandler;
        let mut node = create_test_node("ReduceMean", vec!["x"], vec!["y"]);
        add_ints_attribute(&mut node, "axes", vec![1, 2]);
        add_int_attribute(&mut node, "keepdims", 1);
        let initializers = std::collections::HashMap::new();
        let value_shapes = std::collections::HashMap::new();
        let const_values = std::collections::HashMap::new();
        let value_ids = std::collections::HashMap::new();
        let value_types = std::collections::HashMap::new();
        let context = ConversionContext {
            initializers: &initializers,
            value_shapes: &value_shapes,
            const_values: &const_values,
            value_ids: &value_ids,
            value_types: &value_types,
        };

        let result = handler.convert(&node, &context).unwrap();
        assert_eq!(result.nodes.len(), 1);
        assert_eq!(result.nodes[0].op, "reduceMean");
        assert_eq!(result.nodes[0].inputs, vec!["x"]);
        assert!(result.nodes[0].options.contains_key("axes"));
        assert!(result.nodes[0].options.contains_key("keepDimensions"));
    }

    #[test]
    fn test_convert_reduce_sum() {
        let handler = ReductionHandler;
        let mut node = create_test_node("ReduceSum", vec!["x"], vec!["y"]);
        add_ints_attribute(&mut node, "axes", vec![0]);
        let initializers = std::collections::HashMap::new();
        let value_shapes = std::collections::HashMap::new();
        let const_values = std::collections::HashMap::new();
        let value_ids = std::collections::HashMap::new();
        let value_types = std::collections::HashMap::new();
        let context = ConversionContext {
            initializers: &initializers,
            value_shapes: &value_shapes,
            const_values: &const_values,
            value_ids: &value_ids,
            value_types: &value_types,
        };

        let result = handler.convert(&node, &context).unwrap();
        assert_eq!(result.nodes.len(), 1);
        assert_eq!(result.nodes[0].op, "reduceSum");
    }
}
