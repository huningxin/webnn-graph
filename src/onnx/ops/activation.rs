// Activation and unary math operators: Relu, Gelu, Tanh, Sigmoid, Sqrt, Exp, Log, Abs, Neg, Erf

use crate::ast::Node;
use crate::onnx::convert::{sanitize_identifier, OnnxError};
use crate::onnx::ops::{ConversionContext, ConversionResult, OpHandler};
use crate::protos::onnx::NodeProto;
use serde_json::Map;

pub struct ActivationHandler;

impl OpHandler for ActivationHandler {
    fn supports(&self, op_type: &str) -> bool {
        matches!(
            op_type,
            "Relu"
                | "Gelu"
                | "Tanh"
                | "Sigmoid"
                | "Sqrt"
                | "Exp"
                | "Log"
                | "Abs"
                | "Neg"
                | "Erf"
                | "Cos"
                | "Sin"
                | "Clip"
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

        // Special handling for Clip - needs to extract min/max values
        if op_type == "Clip" {
            return self.convert_clip(node, &node_name, context);
        }

        // Map ONNX operator to WebNN operation name
        let webnn_op = match op_type {
            "Relu" => "relu",
            "Gelu" => "gelu",
            "Tanh" => "tanh",
            "Sigmoid" => "sigmoid",
            "Sqrt" => "sqrt",
            "Exp" => "exp",
            "Log" => "log",
            "Abs" => "abs",
            "Neg" => "neg",
            "Erf" => "erf",
            "Cos" => "cos",
            "Sin" => "sin",
            _ => {
                return Err(OnnxError::UnsupportedOp {
                    op: op_type.to_string(),
                    node: node_name,
                })
            }
        };

        self.convert_unary(node, &node_name, webnn_op, context)
    }
}

impl ActivationHandler {
    /// Convert ONNX unary/activation operation to WebNN
    fn convert_unary(
        &self,
        node: &NodeProto,
        node_name: &str,
        webnn_op: &str,
        context: &ConversionContext,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.len() != 1 {
            return Err(OnnxError::InvalidShape(format!(
                "{} expects 1 input, got {}",
                webnn_op,
                inputs.len()
            )));
        }

        let output_name = if node.output.as_slice().is_empty() {
            format!("{}_output", node_name)
        } else {
            sanitize_identifier(&node.output.as_slice()[0].to_string())
        };

        let input0 = context.resolve_input(&inputs[0]);

        let options = Map::new();

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

    /// Convert ONNX Clip to WebNN clamp
    /// ONNX Clip: 1-3 inputs (input, min, max)
    /// WebNN clamp: 1 input with minValue/maxValue options
    fn convert_clip(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &ConversionContext,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.is_empty() || inputs.len() > 3 {
            return Err(OnnxError::InvalidShape(format!(
                "Clip expects 1-3 inputs, got {}",
                inputs.len()
            )));
        }

        let output_name = if node.output.as_slice().is_empty() {
            format!("{}_output", node_name)
        } else {
            sanitize_identifier(&node.output.as_slice()[0].to_string())
        };

        let input0 = context.resolve_input(&inputs[0]);

        let mut options = Map::new();

        // Extract min value (second input if present)
        if inputs.len() >= 2 && !inputs[1].is_empty() {
            if let Some(min_value) = self.extract_scalar_value(&inputs[1], context)? {
                options.insert("minValue".to_string(), min_value);
            }
        }

        // Extract max value (third input if present)
        if inputs.len() >= 3 && !inputs[2].is_empty() {
            if let Some(max_value) = self.extract_scalar_value(&inputs[2], context)? {
                options.insert("maxValue".to_string(), max_value);
            }
        }

        let mut result = ConversionResult::new(vec![Node {
            id: output_name.clone(),
            op: "clamp".to_string(),
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

    /// Extract scalar value from a tensor (for min/max in Clip)
    fn extract_scalar_value(
        &self,
        input_name: &str,
        context: &ConversionContext,
    ) -> Result<Option<serde_json::Value>, OnnxError> {
        // Check if it's an initializer (constant)
        if let Some(tensor) = context.initializers.get(input_name) {
            let data_type = tensor.data_type;

            // Extract scalar value based on data type
            match data_type {
                x if x == crate::protos::onnx::TensorProto_DataType::Float as i32 => {
                    if !tensor.float_data.is_empty() {
                        return Ok(Some(serde_json::json!(tensor.float_data[0])));
                    } else if !tensor.raw_data.is_empty() {
                        let bytes = &tensor.raw_data[0..4];
                        let value = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                        return Ok(Some(serde_json::json!(value)));
                    }
                }
                x if x == crate::protos::onnx::TensorProto_DataType::Double as i32 => {
                    if !tensor.double_data.is_empty() {
                        return Ok(Some(serde_json::json!(tensor.double_data[0])));
                    } else if tensor.raw_data.len() >= 8 {
                        let bytes = &tensor.raw_data[0..8];
                        let value = f64::from_le_bytes([
                            bytes[0], bytes[1], bytes[2], bytes[3],
                            bytes[4], bytes[5], bytes[6], bytes[7],
                        ]);
                        return Ok(Some(serde_json::json!(value)));
                    }
                }
                x if x == crate::protos::onnx::TensorProto_DataType::Int64 as i32 => {
                    if !tensor.int64_data.is_empty() {
                        return Ok(Some(serde_json::json!(tensor.int64_data[0])));
                    } else if tensor.raw_data.len() >= 8 {
                        let bytes = &tensor.raw_data[0..8];
                        let value = i64::from_le_bytes([
                            bytes[0], bytes[1], bytes[2], bytes[3],
                            bytes[4], bytes[5], bytes[6], bytes[7],
                        ]);
                        return Ok(Some(serde_json::json!(value)));
                    }
                }
                x if x == crate::protos::onnx::TensorProto_DataType::Int32 as i32 => {
                    if !tensor.int32_data.is_empty() {
                        return Ok(Some(serde_json::json!(tensor.int32_data[0])));
                    } else if tensor.raw_data.len() >= 4 {
                        let bytes = &tensor.raw_data[0..4];
                        let value = i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                        return Ok(Some(serde_json::json!(value)));
                    }
                }
                _ => {}
            }
        }

        // If not found or not a scalar constant, return None
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protos::onnx::NodeProto;

    fn create_test_node(op_type: &str, inputs: Vec<&str>, outputs: Vec<&str>) -> NodeProto {
        NodeProto {
            op_type: op_type.to_string(),
            name: format!("test_{}", op_type.to_lowercase()),
            input: inputs.iter().map(|s| s.to_string()).collect(),
            output: outputs.iter().map(|s| s.to_string()).collect(),
            ..Default::default()
        }
    }

    #[test]
    fn test_activation_handler_supports() {
        let handler = ActivationHandler;
        assert!(handler.supports("Relu"));
        assert!(handler.supports("Gelu"));
        assert!(handler.supports("Tanh"));
        assert!(handler.supports("Sigmoid"));
        assert!(handler.supports("Sqrt"));
        assert!(handler.supports("Exp"));
        assert!(handler.supports("Log"));
        assert!(handler.supports("Abs"));
        assert!(handler.supports("Neg"));
        assert!(handler.supports("Erf"));
        assert!(handler.supports("Cos"));
        assert!(handler.supports("Sin"));
        assert!(!handler.supports("Add"));
    }

    #[test]
    fn test_convert_relu() {
        let handler = ActivationHandler;
        let node = create_test_node("Relu", vec!["x"], vec!["y"]);
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
        assert_eq!(result.nodes[0].op, "relu");
        assert_eq!(result.nodes[0].inputs, vec!["x"]);
    }

    #[test]
    fn test_convert_sqrt() {
        let handler = ActivationHandler;
        let node = create_test_node("Sqrt", vec!["x"], vec!["y"]);
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
        assert_eq!(result.nodes[0].op, "sqrt");
        assert_eq!(result.nodes[0].inputs, vec!["x"]);
    }

    #[test]
    fn test_convert_gelu() {
        let handler = ActivationHandler;
        let node = create_test_node("Gelu", vec!["x"], vec!["y"]);
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
        assert_eq!(result.nodes[0].op, "gelu");
    }

    #[test]
    fn test_convert_cos() {
        let handler = ActivationHandler;
        let node = create_test_node("Cos", vec!["x"], vec!["y"]);
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
        assert_eq!(result.nodes[0].op, "cos");
    }

    #[test]
    fn test_convert_sin() {
        let handler = ActivationHandler;
        let node = create_test_node("Sin", vec!["x"], vec!["y"]);
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
        assert_eq!(result.nodes[0].op, "sin");
    }
}
