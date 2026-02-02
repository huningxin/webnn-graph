// Type conversion and constant operators: Cast, Constant

use crate::ast::Node;
use crate::onnx::convert::{sanitize_identifier, OnnxError};
use crate::onnx::ops::{ConversionContext, ConversionResult, OpHandler};
use crate::protos::onnx::NodeProto;

pub struct ConversionHandler;

impl OpHandler for ConversionHandler {
    fn supports(&self, op_type: &str) -> bool {
        matches!(op_type, "Cast" | "Constant" | "DequantizeLinear")
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
            "Cast" => self.convert_cast(node, &node_name, context),
            "Constant" => self.convert_constant(node, &node_name),
            "DequantizeLinear" => self.convert_dequantize_linear(node, &node_name, context),
            _ => Err(OnnxError::UnsupportedOp {
                op: op_type.to_string(),
                node: node_name,
            }),
        }
    }
}

impl ConversionHandler {
    /// Convert ONNX Cast to WebNN cast
    /// ONNX Cast converts tensor data type
    fn convert_cast(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &ConversionContext,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.len() != 1 {
            return Err(OnnxError::InvalidShape(format!(
                "Cast expects 1 input, got {}",
                inputs.len()
            )));
        }

        // Extract 'to' attribute (target data type)
        let mut to_type: Option<i64> = None;
        for attr in node.attribute.as_slice() {
            if attr.name.as_str() == "to" && attr.i != 0 {
                to_type = Some(attr.i);
            }
        }

        if to_type.is_none() {
            return Err(OnnxError::MissingAttribute {
                attr: "to".to_string(),
                op: "Cast".to_string(),
            });
        }

        let output_name = if node.output.as_slice().is_empty() {
            format!("{}_output", node_name)
        } else {
            sanitize_identifier(&node.output.as_slice()[0].to_string())
        };

        let input0 = context.resolve_input(&inputs[0]);

        // Map ONNX type to WebNN DataType
        let target_type = crate::onnx::convert::map_onnx_data_type(to_type.unwrap() as i32)?;

        // WebNN cast expects lowercase type name as direct parameter
        let target_type_str = format!("{:?}", target_type).to_lowercase();

        let mut options = crate::onnx::ops::create_options_with_label(&node_name);
        options.insert(
            "to".to_string(),
            serde_json::json!(target_type_str),
        );

        let mut result = ConversionResult::new(vec![Node {
            id: output_name.clone(),
            op: "cast".to_string(),
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

    /// Convert ONNX Constant to WebNN constant
    /// ONNX Constant creates an inline constant tensor
    fn convert_constant(
        &self,
        node: &NodeProto,
        node_name: &str,
    ) -> Result<ConversionResult, OnnxError> {
        let output_name = if node.output.as_slice().is_empty() {
            format!("{}_output", node_name)
        } else {
            sanitize_identifier(&node.output.as_slice()[0].to_string())
        };

        // Extract 'value' attribute (TensorProto)
        let mut value_tensor = None;
        for attr in node.attribute.as_slice() {
            if attr.name.as_str() == "value" && attr.t.is_some() {
                value_tensor = Some(attr.t.as_ref().unwrap());
            }
        }

        if value_tensor.is_none() {
            return Err(OnnxError::MissingAttribute {
                attr: "value".to_string(),
                op: "Constant".to_string(),
            });
        }

        let tensor = value_tensor.unwrap();
        let onnx_type = tensor.data_type;
        let data_type = crate::onnx::convert::map_onnx_data_type(onnx_type)?;

        let shape: Vec<i64> = tensor.dims.as_slice().to_vec();
        let raw_data = tensor.raw_data.as_slice().to_vec();

        // WebNN expects lowercase type names
        let data_type_str = format!("{:?}", data_type).to_lowercase();

        let mut options = crate::onnx::ops::create_options_with_label(&node_name);
        options.insert(
            "dataType".to_string(),
            serde_json::json!(data_type_str),
        );
        options.insert("shape".to_string(), serde_json::json!(shape));

        // Store raw bytes as base64 for now (WebNN implementation can decode)
        let b64_data =
            base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &raw_data);
        options.insert("data".to_string(), serde_json::json!(b64_data));

        let mut result = ConversionResult::new(vec![Node {
            id: output_name.clone(),
            op: "constant".to_string(),
            inputs: vec![],
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

    /// Convert ONNX DequantizeLinear to WebNN dequantizeLinear
    /// Formula: y = (x - x_zero_point) * x_scale
    /// Supports per-tensor, per-axis, and blocked quantization
    fn convert_dequantize_linear(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &ConversionContext,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.len() < 2 || inputs.len() > 3 {
            return Err(OnnxError::InvalidShape(format!(
                "DequantizeLinear expects 2 or 3 inputs, got {}",
                inputs.len()
            )));
        }

        let output_name = if node.output.as_slice().is_empty() {
            format!("{}_output", node_name)
        } else {
            sanitize_identifier(&node.output.as_slice()[0].to_string())
        };

        // Resolve inputs
        let x = context.resolve_input(&inputs[0]);
        let x_scale = context.resolve_input(&inputs[1]);
        
        // WebNN requires zeroPoint as mandatory parameter
        // If ONNX doesn't provide it, we need to create a zero-filled constant
        let (x_zero_point, extra_nodes) = if inputs.len() == 3 {
            (context.resolve_input(&inputs[2]), vec![])
        } else {
            // Create a zero constant tensor with the same shape as x_scale
            let zero_point_name = format!("{}_zero_point", node_name);
            
            // Get scale shape from context if available, otherwise assume scalar
            let zero_shape = if let Some(scale_shape) = context.value_shapes.get(&inputs[1]) {
                scale_shape.clone()
            } else {
                vec![] // Scalar
            };
            
            // Get data type from input x
            let zero_dtype = if let Some(input_type) = context.value_types.get(&inputs[0]) {
                format!("{:?}", input_type).to_lowercase()
            } else {
                "int8".to_string() // Default to int8 for quantized inputs
            };
            
            let mut zero_options = crate::onnx::ops::create_options_with_label(&format!("{}_zero_point", node_name));
            zero_options.insert("dataType".to_string(), serde_json::json!(zero_dtype));
            zero_options.insert("shape".to_string(), serde_json::json!(zero_shape));
            zero_options.insert("value".to_string(), serde_json::json!(0));
            
            let zero_node = Node {
                id: zero_point_name.clone(),
                op: "constant".to_string(),
                inputs: vec![],
                options: zero_options,
                outputs: None,
            };
            
            (zero_point_name.clone(), vec![zero_node])
        };

        // Extract attributes
        let mut axis: i64 = 1; // Default axis
        let mut block_size: i64 = 0; // Default: not blocked

        for attr in node.attribute.as_slice() {
            match attr.name.as_str() {
                "axis" => {
                    if attr.i != 0 {
                        axis = attr.i;
                    }
                }
                "block_size" => {
                    if attr.i != 0 {
                        block_size = attr.i;
                    }
                }
                _ => {}
            }
        }

        let mut options = crate::onnx::ops::create_options_with_label(&node_name);
        
        // Add axis if not default
        if axis != 1 {
            options.insert("axis".to_string(), serde_json::json!(axis));
        }
        
        // Add block_size if blocked quantization
        if block_size > 0 {
            options.insert("blockSize".to_string(), serde_json::json!(block_size));
        }

        // Build inputs array: [x, x_scale, x_zero_point] (all three are required by WebNN)
        let node_inputs = vec![x, x_scale, x_zero_point];

        // Combine extra nodes (zero constant if needed) with main node
        let mut all_nodes = extra_nodes;
        all_nodes.push(Node {
            id: output_name.clone(),
            op: "dequantizeLinear".to_string(),
            inputs: node_inputs,
            options,
            outputs: None,
        });

        let mut result = ConversionResult::new(all_nodes);

        if let Some(output) = node.output.as_slice().first() {
            result
                .output_mappings
                .insert(output.to_string(), output_name.clone());
        }

        Ok(result)
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

    #[test]
    fn test_conversion_handler_supports() {
        let handler = ConversionHandler;
        assert!(handler.supports("Cast"));
        assert!(!handler.supports("Add"));
    }

    #[test]
    fn test_convert_cast() {
        let handler = ConversionHandler;
        let mut node = create_test_node("Cast", vec!["x"], vec!["y"]);
        add_int_attribute(&mut node, "to", 7); // INT64
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
        assert_eq!(result.nodes[0].op, "cast");
        assert_eq!(result.nodes[0].inputs, vec!["x"]);
        assert!(result.nodes[0].options.contains_key("to"));
    }
}
