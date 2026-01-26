use crate::ast::{DataType, GraphJson};

fn dt_to_js(dt: &DataType) -> &'static str {
    match dt {
        DataType::Float32 => "float32",
        DataType::Float16 => "float16",
        DataType::Int4 => "int4",
        DataType::Uint4 => "uint4",
        DataType::Int32 => "int32",
        DataType::Uint32 => "uint32",
        DataType::Int64 => "int64",
        DataType::Uint64 => "uint64",
        DataType::Int8 => "int8",
        DataType::Uint8 => "uint8",
    }
}

/// Emit the WeightsFile helper class for loading weights
pub fn emit_weights_loader_js() -> &'static str {
    r#"/**
 * Helper class for loading and managing WebNN graph weights
 */
export class WeightsFile {
  constructor(buffer, manifest) {
    this.buffer = buffer;
    this.manifest = manifest;
  }

  /**
   * Load weights from URL paths
   * @param {string} weightsPath - Path to .weights binary file
   * @param {string} manifestPath - Path to .manifest.json file
   * @returns {Promise<WeightsFile>}
   */
  static async load(weightsPath, manifestPath) {
    const [weightsResponse, manifestResponse] = await Promise.all([
      fetch(weightsPath),
      fetch(manifestPath)
    ]);

    if (!weightsResponse.ok) {
      throw new Error(`Failed to load weights: ${weightsResponse.statusText}`);
    }
    if (!manifestResponse.ok) {
      throw new Error(`Failed to load manifest: ${manifestResponse.statusText}`);
    }

    const buffer = await weightsResponse.arrayBuffer();
    const manifest = await manifestResponse.json();

    // Validate manifest format
    if (manifest.format !== 'wg-weights-manifest') {
      throw new Error(`Invalid manifest format: ${manifest.format}`);
    }
    if (manifest.version !== 1) {
      throw new Error(`Unsupported manifest version: ${manifest.version}`);
    }

    // Validate weights file header
    const view = new DataView(buffer);
    const magic = new TextDecoder().decode(new Uint8Array(buffer, 0, 4));
    if (magic !== 'WGWT') {
      throw new Error(`Invalid weights file magic: ${magic}`);
    }
    const version = view.getUint32(4, true); // little-endian
    if (version !== 1) {
      throw new Error(`Unsupported weights file version: ${version}`);
    }

    return new WeightsFile(buffer, manifest);
  }

  /**
   * Get a slice descriptor for a named tensor
   * @param {string} name - Tensor name
   * @returns {Object} Tensor metadata with byteOffset and byteLength
   */
  getSlice(name) {
    const tensor = this.manifest.tensors[name];
    if (!tensor) {
      throw new Error(`Tensor not found in manifest: ${name}`);
    }
    return tensor;
  }

  /**
   * Get the raw data for a named tensor
   * @param {string} name - Tensor name
   * @returns {ArrayBuffer} Tensor data
   */
  getData(name) {
    const tensor = this.getSlice(name);
    return this.buffer.slice(tensor.byteOffset, tensor.byteOffset + tensor.byteLength);
  }

  /**
   * List all available tensor names
   * @returns {string[]}
   */
  getTensorNames() {
    return Object.keys(this.manifest.tensors);
  }
}
"#
}

pub fn emit_builder_js(g: &GraphJson) -> String {
    let mut s = String::new();
    s.push_str("/**\n");
    s.push_str(" * Build a WebNN MLGraph from the graph definition\n");
    s.push_str(" * @param {MLContext} context - WebNN context\n");
    s.push_str(" * @param {WeightsFile} weights - Loaded weights file\n");
    s.push_str(" * @returns {Promise<MLGraph>}\n");
    s.push_str(" */\n");
    s.push_str("export async function buildGraph(context, weights) {\n");
    s.push_str("  const builder = new MLGraphBuilder(context);\n");
    s.push_str("  const env = new Map();\n\n");

    for (name, d) in &g.inputs {
        let shape = format!("{:?}", d.shape);
        s.push_str(&format!(
            "  env.set({name:?}, builder.input({name:?}, {{ dataType: {dt:?}, shape: {shape} }}));\n",
            name = name,
            dt = dt_to_js(&d.data_type),
            shape = shape,
        ));
    }
    s.push('\n');

    for (name, c) in &g.consts {
        match &c.init {
            crate::ast::ConstInit::Weights { r#ref } => {
                let shape = format!("{:?}", c.shape);
                s.push_str(&format!(
"  {{\n    const sl = weights.getSlice({r:?});\n    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);\n    env.set({name:?}, builder.constant({{ dataType: {dt:?}, shape: {shape} }}, buf));\n  }}\n",
                    r = r#ref,
                    name = name,
                    dt = dt_to_js(&c.data_type),
                    shape = shape,
                ));
            }
            crate::ast::ConstInit::Scalar { value } => {
                s.push_str(&format!(
                    "  env.set({name:?}, builder.constant({dt:?}, {val}));\n",
                    name = name,
                    dt = dt_to_js(&c.data_type),
                    val = value,
                ));
            }
            crate::ast::ConstInit::InlineBytes { bytes } => {
                let shape = format!("{:?}", c.shape);
                s.push_str(&format!(
                    "  env.set({name:?}, builder.constant({{ dataType: {dt:?}, shape: {shape} }}, new Uint8Array({bytes:?}).buffer));\n",
                    name = name,
                    dt = dt_to_js(&c.data_type),
                    shape = shape,
                    bytes = bytes
                ));
            }
        }
    }
    s.push('\n');

    for n in &g.nodes {
        // Special handling for constant operation with inline base64 data
        if n.op == "constant" {
            if let Some(data) = n.options.get("data").and_then(|v| v.as_str()) {
                let data_type = n.options.get("dataType")
                    .and_then(|v| v.as_str())
                    .unwrap_or("float32");
                let shape = n.options.get("shape")
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "[]".to_string());
                
                // Decode base64 data to ArrayBuffer
                s.push_str(&format!(
                    "  env.set({id:?}, builder.constant({{ dataType: {dt:?}, shape: {shape} }}, Uint8Array.from(atob({data:?}), c => c.charCodeAt(0)).buffer));\n",
                    id = n.id,
                    dt = data_type,
                    shape = shape,
                    data = data
                ));
                continue;
            }
        }

        // Special handling for cast operation - extract 'to' parameter as direct argument
        if n.op == "cast" {
            let input_ref = &n.inputs[0];
            if let Some(to_type) = n.options.get("to").and_then(|v| v.as_str()) {
                // WebNN cast: builder.cast(input, dataType) not builder.cast(input, {to: dataType})
                s.push_str(&format!(
                    "  env.set({id:?}, builder.cast(env.get({input:?}), {to_type:?}));\n",
                    id = n.id,
                    input = input_ref,
                    to_type = to_type
                ));
                continue;
            }
        }

        // Special handling for reshape operation - extract 'newShape' parameter as direct argument
        if n.op == "reshape" {
            let input_ref = &n.inputs[0];
            if let Some(new_shape) = n.options.get("newShape") {
                // WebNN reshape: builder.reshape(input, newShape) not builder.reshape(input, {newShape: [...]})
                s.push_str(&format!(
                    "  env.set({id:?}, builder.reshape(env.get({input:?}), {new_shape}));\n",
                    id = n.id,
                    input = input_ref,
                    new_shape = new_shape
                ));
                continue;
            }
        }

        // Special handling for slice operation - extract starts and sizes as direct arguments
        if n.op == "slice" {
            let input_ref = &n.inputs[0];
            if let (Some(starts), Some(ends)) = (n.options.get("starts"), n.options.get("ends")) {
                // Calculate sizes from starts and ends
                // WebNN slice: builder.slice(input, starts, sizes, {strides: ...})
                let starts_arr = starts.as_array().unwrap();
                let ends_arr = ends.as_array().unwrap();
                let sizes: Vec<i64> = starts_arr.iter().zip(ends_arr.iter())
                    .map(|(s, e)| e.as_i64().unwrap() - s.as_i64().unwrap())
                    .collect();
                
                let mut options = serde_json::Map::new();
                if let Some(steps) = n.options.get("steps") {
                    options.insert("strides".to_string(), steps.clone());
                }
                let opts_str = serde_json::Value::Object(options).to_string();
                
                s.push_str(&format!(
                    "  env.set({id:?}, builder.slice(env.get({input:?}), {starts}, {sizes:?}, {opts}));\n",
                    id = n.id,
                    input = input_ref,
                    starts = starts,
                    sizes = sizes,
                    opts = opts_str
                ));
                continue;
            }
        }

        // Special handling for softmax operation - extract 'axis' parameter as direct argument
        if n.op == "softmax" {
            let input_ref = &n.inputs[0];
            if let Some(axis) = n.options.get("axis") {
                // WebNN softmax: builder.softmax(input, axis) not builder.softmax(input, {axis: value})
                s.push_str(&format!(
                    "  env.set({id:?}, builder[\"softmax\"](env.get({input:?}), {axis}));\n",
                    id = n.id,
                    input = input_ref,
                    axis = axis
                ));
                continue;
            }
        }

        // Regular operation handling
        let ins = n
            .inputs
            .iter()
            .map(|x| format!("env.get({:?})", x))
            .collect::<Vec<_>>()
            .join(", ");
        let opts = serde_json::Value::Object(n.options.clone()).to_string();
        if let Some(outs) = &n.outputs {
            s.push_str(&format!(
                "  {{\n    const tmp = builder[{op:?}]({ins}, {opts});\n",
                op = n.op,
                ins = ins,
                opts = opts
            ));
            for (i, o) in outs.iter().enumerate() {
                s.push_str(&format!("    env.set({o:?}, tmp[{i}]);\n", o = o, i = i));
            }
            s.push_str("  }\n");
        } else {
            s.push_str(&format!(
                "  env.set({id:?}, builder[{op:?}]({ins}, {opts}));\n",
                id = n.id,
                op = n.op,
                ins = ins,
                opts = opts
            ));
        }
    }

    s.push_str("\n  const outputs = {};\n");
    for (out, r) in &g.outputs {
        s.push_str(&format!(
            "  outputs[{out:?}] = env.get({r:?});\n",
            out = out,
            r = r
        ));
    }
    s.push_str("  return await builder.build(outputs);\n");
    s.push_str("}\n");
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{new_graph_json, ConstDecl, ConstInit, DataType, Node, OperandDesc};

    #[test]
    fn test_dt_to_js() {
        assert_eq!(dt_to_js(&DataType::Float32), "float32");
        assert_eq!(dt_to_js(&DataType::Float16), "float16");
        assert_eq!(dt_to_js(&DataType::Int4), "int4");
        assert_eq!(dt_to_js(&DataType::Uint4), "uint4");
        assert_eq!(dt_to_js(&DataType::Int32), "int32");
        assert_eq!(dt_to_js(&DataType::Uint32), "uint32");
        assert_eq!(dt_to_js(&DataType::Int64), "int64");
        assert_eq!(dt_to_js(&DataType::Uint64), "uint64");
        assert_eq!(dt_to_js(&DataType::Int8), "int8");
        assert_eq!(dt_to_js(&DataType::Uint8), "uint8");
    }

    #[test]
    fn test_emit_simple_graph() {
        let mut g = new_graph_json();
        g.inputs.insert(
            "x".to_string(),
            OperandDesc {
                data_type: DataType::Float32,
                shape: vec![1, 10],
            },
        );
        g.nodes.push(Node {
            id: "result".to_string(),
            op: "relu".to_string(),
            inputs: vec!["x".to_string()],
            options: serde_json::Map::new(),
            outputs: None,
        });
        g.outputs.insert("result".to_string(), "result".to_string());

        let js = emit_builder_js(&g);
        assert!(js.contains("export async function buildGraph"));
        assert!(js.contains("MLGraphBuilder(context)"));
        assert!(js.contains("builder.input(\"x\""));
        assert!(js.contains("builder[\"relu\"]"));
        assert!(js.contains("env.get(\"x\")"));
        assert!(js.contains("outputs[\"result\"]"));
        assert!(js.contains("builder.build(outputs)"));
    }

    #[test]
    fn test_emit_with_weights() {
        let mut g = new_graph_json();
        g.inputs.insert(
            "x".to_string(),
            OperandDesc {
                data_type: DataType::Float32,
                shape: vec![1, 10],
            },
        );
        g.consts.insert(
            "W".to_string(),
            ConstDecl {
                data_type: DataType::Float32,
                shape: vec![10, 5],
                init: ConstInit::Weights {
                    r#ref: "W".to_string(),
                },
            },
        );
        g.nodes.push(Node {
            id: "result".to_string(),
            op: "matmul".to_string(),
            inputs: vec!["x".to_string(), "W".to_string()],
            options: serde_json::Map::new(),
            outputs: None,
        });
        g.outputs.insert("result".to_string(), "result".to_string());

        let js = emit_builder_js(&g);
        assert!(js.contains("weights.getSlice(\"W\")"));
        assert!(js.contains("weights.buffer.slice"));
        assert!(js.contains("builder.constant"));
        assert!(js.contains("dataType: \"float32\""));
        assert!(js.contains("shape: [10, 5]"));
    }

    #[test]
    fn test_emit_with_scalar() {
        let mut g = new_graph_json();
        g.inputs.insert(
            "x".to_string(),
            OperandDesc {
                data_type: DataType::Float32,
                shape: vec![1],
            },
        );
        g.consts.insert(
            "scale".to_string(),
            ConstDecl {
                data_type: DataType::Float32,
                shape: vec![1],
                init: ConstInit::Scalar {
                    value: serde_json::json!(2.5),
                },
            },
        );
        g.nodes.push(Node {
            id: "result".to_string(),
            op: "mul".to_string(),
            inputs: vec!["x".to_string(), "scale".to_string()],
            options: serde_json::Map::new(),
            outputs: None,
        });
        g.outputs.insert("result".to_string(), "result".to_string());

        let js = emit_builder_js(&g);
        assert!(js.contains("builder.constant(\"float32\", 2.5)"));
    }

    #[test]
    fn test_emit_with_inline_bytes() {
        let mut g = new_graph_json();
        g.consts.insert(
            "data".to_string(),
            ConstDecl {
                data_type: DataType::Uint8,
                shape: vec![4],
                init: ConstInit::InlineBytes {
                    bytes: vec![1, 2, 3, 4],
                },
            },
        );
        g.outputs.insert("data".to_string(), "data".to_string());

        let js = emit_builder_js(&g);
        assert!(js.contains("new Uint8Array([1, 2, 3, 4]).buffer"));
    }

    #[test]
    fn test_emit_with_options() {
        let mut g = new_graph_json();
        g.inputs.insert(
            "x".to_string(),
            OperandDesc {
                data_type: DataType::Float32,
                shape: vec![1, 10],
            },
        );

        let mut options = serde_json::Map::new();
        options.insert("axis".to_string(), serde_json::json!(1));

        g.nodes.push(Node {
            id: "result".to_string(),
            op: "softmax".to_string(),
            inputs: vec!["x".to_string()],
            options,
            outputs: None,
        });
        g.outputs.insert("result".to_string(), "result".to_string());

        let js = emit_builder_js(&g);
        assert!(js.contains("builder[\"softmax\"]"));
        assert!(js.contains("\"axis\":1"));
    }

    #[test]
    fn test_emit_with_multi_outputs() {
        let mut g = new_graph_json();
        g.inputs.insert(
            "x".to_string(),
            OperandDesc {
                data_type: DataType::Float32,
                shape: vec![10],
            },
        );
        g.nodes.push(Node {
            id: "a".to_string(),
            op: "split".to_string(),
            inputs: vec!["x".to_string()],
            options: serde_json::Map::new(),
            outputs: Some(vec!["a".to_string(), "b".to_string()]),
        });
        g.outputs.insert("a".to_string(), "a".to_string());
        g.outputs.insert("b".to_string(), "b".to_string());

        let js = emit_builder_js(&g);
        assert!(js.contains("const tmp = builder[\"split\"]"));
        assert!(js.contains("env.set(\"a\", tmp[0])"));
        assert!(js.contains("env.set(\"b\", tmp[1])"));
    }

    #[test]
    fn test_emit_multiple_inputs_outputs() {
        let mut g = new_graph_json();
        g.inputs.insert(
            "x".to_string(),
            OperandDesc {
                data_type: DataType::Float32,
                shape: vec![1],
            },
        );
        g.inputs.insert(
            "y".to_string(),
            OperandDesc {
                data_type: DataType::Float32,
                shape: vec![1],
            },
        );
        g.nodes.push(Node {
            id: "a".to_string(),
            op: "relu".to_string(),
            inputs: vec!["x".to_string()],
            options: serde_json::Map::new(),
            outputs: None,
        });
        g.nodes.push(Node {
            id: "b".to_string(),
            op: "sigmoid".to_string(),
            inputs: vec!["y".to_string()],
            options: serde_json::Map::new(),
            outputs: None,
        });
        g.outputs.insert("out1".to_string(), "a".to_string());
        g.outputs.insert("out2".to_string(), "b".to_string());

        let js = emit_builder_js(&g);
        assert!(js.contains("builder.input(\"x\""));
        assert!(js.contains("builder.input(\"y\""));
        assert!(js.contains("outputs[\"out1\"] = env.get(\"a\")"));
        assert!(js.contains("outputs[\"out2\"] = env.get(\"b\")"));
    }
}
