/**
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

/**
 * Build a WebNN MLGraph from the graph definition
 * @param {MLContext} context - WebNN context
 * @param {WeightsFile} weights - Loaded weights file
 * @returns {Promise<MLGraph>}
 */
export async function buildGraph(context, weights) {
  const builder = new MLGraphBuilder(context);
  const env = new Map();

  env.set("attention_mask", builder.input("attention_mask", { dataType: "int64", shape: [1, 128] }));
  env.set("input_ids", builder.input("input_ids", { dataType: "int64", shape: [1, 128] }));
  env.set("token_type_ids", builder.input("token_type_ids", { dataType: "int64", shape: [1, 128] }));

  {
    const sl = weights.getSlice("_embeddings_Constant_1_output_0");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_embeddings_Constant_1_output_0", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("_embeddings_Constant_2_output_0");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_embeddings_Constant_2_output_0", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("_embeddings_Constant_3_output_0");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_embeddings_Constant_3_output_0", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("_embeddings_Constant_output_0");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_embeddings_Constant_output_0", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  env.set("_embeddings_Gather_output_0", builder.constant({ dataType: "int64", shape: [] }, new Uint8Array([128, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_embeddings_Shape_output_0", builder.constant({ dataType: "int64", shape: [2] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_embeddings_Unsqueeze_output_0", builder.constant({ dataType: "int64", shape: [] }, new Uint8Array([128, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_0_attention_self_Concat_1_output_0", builder.constant({ dataType: "int64", shape: [4] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_0_attention_self_Concat_2_output_0", builder.constant({ dataType: "int64", shape: [4] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_0_attention_self_Concat_3_output_0", builder.constant({ dataType: "int64", shape: [3] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 128, 1, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_0_attention_self_Concat_output_0", builder.constant({ dataType: "int64", shape: [4] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0]).buffer));
  {
    const sl = weights.getSlice("_encoder_layer_0_attention_self_Constant_10_output_0");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_encoder_layer_0_attention_self_Constant_10_output_0", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("_encoder_layer_0_attention_self_Constant_11_output_0");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_encoder_layer_0_attention_self_Constant_11_output_0", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("_encoder_layer_0_attention_self_Constant_13_output_0");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_encoder_layer_0_attention_self_Constant_13_output_0", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("_encoder_layer_0_attention_self_Constant_14_output_0");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_encoder_layer_0_attention_self_Constant_14_output_0", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("_encoder_layer_0_attention_self_Constant_15_output_0");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_encoder_layer_0_attention_self_Constant_15_output_0", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("_encoder_layer_0_attention_self_Constant_1_output_0");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_encoder_layer_0_attention_self_Constant_1_output_0", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("_encoder_layer_0_attention_self_Constant_2_output_0");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_encoder_layer_0_attention_self_Constant_2_output_0", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("_encoder_layer_0_attention_self_Constant_3_output_0");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_encoder_layer_0_attention_self_Constant_3_output_0", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("_encoder_layer_0_attention_self_Constant_4_output_0");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_encoder_layer_0_attention_self_Constant_4_output_0", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("_encoder_layer_0_attention_self_Constant_5_output_0");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_encoder_layer_0_attention_self_Constant_5_output_0", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("_encoder_layer_0_attention_self_Constant_6_output_0");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_encoder_layer_0_attention_self_Constant_6_output_0", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("_encoder_layer_0_attention_self_Constant_7_output_0");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_encoder_layer_0_attention_self_Constant_7_output_0", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("_encoder_layer_0_attention_self_Constant_8_output_0");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_encoder_layer_0_attention_self_Constant_8_output_0", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("_encoder_layer_0_attention_self_Constant_9_output_0");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_encoder_layer_0_attention_self_Constant_9_output_0", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("_encoder_layer_0_attention_self_Constant_output_0");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_encoder_layer_0_attention_self_Constant_output_0", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  env.set("_encoder_layer_0_attention_self_Gather_1_output_0", builder.constant({ dataType: "int64", shape: [] }, new Uint8Array([128, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_0_attention_self_Gather_2_output_0", builder.constant({ dataType: "int64", shape: [] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_0_attention_self_Gather_3_output_0", builder.constant({ dataType: "int64", shape: [] }, new Uint8Array([128, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_0_attention_self_Gather_4_output_0", builder.constant({ dataType: "int64", shape: [] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_0_attention_self_Gather_5_output_0", builder.constant({ dataType: "int64", shape: [] }, new Uint8Array([128, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_0_attention_self_Gather_6_output_0", builder.constant({ dataType: "int64", shape: [] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_0_attention_self_Gather_7_output_0", builder.constant({ dataType: "int64", shape: [] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_0_attention_self_Gather_output_0", builder.constant({ dataType: "int64", shape: [] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_0_attention_self_Shape_1_output_0", builder.constant({ dataType: "int64", shape: [3] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0, 0, 128, 1, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_0_attention_self_Shape_2_output_0", builder.constant({ dataType: "int64", shape: [3] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0, 0, 128, 1, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_0_attention_self_Shape_3_output_0", builder.constant({ dataType: "int64", shape: [3] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0, 0, 128, 1, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_0_attention_self_Shape_4_output_0", builder.constant({ dataType: "int64", shape: [3] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0, 0, 128, 1, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_0_attention_self_Shape_5_output_0", builder.constant({ dataType: "int64", shape: [3] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0, 0, 128, 1, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_0_attention_self_Shape_6_output_0", builder.constant({ dataType: "int64", shape: [4] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_0_attention_self_Shape_7_output_0", builder.constant({ dataType: "int64", shape: [4] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_0_attention_self_Shape_output_0", builder.constant({ dataType: "int64", shape: [3] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0, 0, 128, 1, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_0_attention_self_Unsqueeze_1_output_0", builder.constant({ dataType: "int64", shape: [] }, new Uint8Array([128, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_0_attention_self_Unsqueeze_2_output_0", builder.constant({ dataType: "int64", shape: [] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_0_attention_self_Unsqueeze_3_output_0", builder.constant({ dataType: "int64", shape: [] }, new Uint8Array([128, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_0_attention_self_Unsqueeze_4_output_0", builder.constant({ dataType: "int64", shape: [] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_0_attention_self_Unsqueeze_5_output_0", builder.constant({ dataType: "int64", shape: [] }, new Uint8Array([128, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_0_attention_self_Unsqueeze_6_output_0", builder.constant({ dataType: "int64", shape: [] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_0_attention_self_Unsqueeze_7_output_0", builder.constant({ dataType: "int64", shape: [] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_0_attention_self_Unsqueeze_output_0", builder.constant({ dataType: "int64", shape: [] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_1_attention_self_Concat_1_output_0", builder.constant({ dataType: "int64", shape: [4] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_1_attention_self_Concat_2_output_0", builder.constant({ dataType: "int64", shape: [4] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_1_attention_self_Concat_3_output_0", builder.constant({ dataType: "int64", shape: [3] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 128, 1, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_1_attention_self_Concat_output_0", builder.constant({ dataType: "int64", shape: [4] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0]).buffer));
  {
    const sl = weights.getSlice("_encoder_layer_1_attention_self_Constant_10_output_0");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_encoder_layer_1_attention_self_Constant_10_output_0", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("_encoder_layer_1_attention_self_Constant_11_output_0");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_encoder_layer_1_attention_self_Constant_11_output_0", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("_encoder_layer_1_attention_self_Constant_13_output_0");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_encoder_layer_1_attention_self_Constant_13_output_0", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("_encoder_layer_1_attention_self_Constant_14_output_0");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_encoder_layer_1_attention_self_Constant_14_output_0", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("_encoder_layer_1_attention_self_Constant_15_output_0");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_encoder_layer_1_attention_self_Constant_15_output_0", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("_encoder_layer_1_attention_self_Constant_1_output_0");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_encoder_layer_1_attention_self_Constant_1_output_0", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("_encoder_layer_1_attention_self_Constant_2_output_0");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_encoder_layer_1_attention_self_Constant_2_output_0", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("_encoder_layer_1_attention_self_Constant_3_output_0");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_encoder_layer_1_attention_self_Constant_3_output_0", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("_encoder_layer_1_attention_self_Constant_4_output_0");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_encoder_layer_1_attention_self_Constant_4_output_0", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("_encoder_layer_1_attention_self_Constant_5_output_0");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_encoder_layer_1_attention_self_Constant_5_output_0", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("_encoder_layer_1_attention_self_Constant_6_output_0");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_encoder_layer_1_attention_self_Constant_6_output_0", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("_encoder_layer_1_attention_self_Constant_7_output_0");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_encoder_layer_1_attention_self_Constant_7_output_0", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("_encoder_layer_1_attention_self_Constant_8_output_0");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_encoder_layer_1_attention_self_Constant_8_output_0", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("_encoder_layer_1_attention_self_Constant_9_output_0");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_encoder_layer_1_attention_self_Constant_9_output_0", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("_encoder_layer_1_attention_self_Constant_output_0");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_encoder_layer_1_attention_self_Constant_output_0", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  env.set("_encoder_layer_1_attention_self_Gather_1_output_0", builder.constant({ dataType: "int64", shape: [] }, new Uint8Array([128, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_1_attention_self_Gather_2_output_0", builder.constant({ dataType: "int64", shape: [] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_1_attention_self_Gather_3_output_0", builder.constant({ dataType: "int64", shape: [] }, new Uint8Array([128, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_1_attention_self_Gather_4_output_0", builder.constant({ dataType: "int64", shape: [] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_1_attention_self_Gather_5_output_0", builder.constant({ dataType: "int64", shape: [] }, new Uint8Array([128, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_1_attention_self_Gather_6_output_0", builder.constant({ dataType: "int64", shape: [] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_1_attention_self_Gather_7_output_0", builder.constant({ dataType: "int64", shape: [] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_1_attention_self_Gather_output_0", builder.constant({ dataType: "int64", shape: [] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_1_attention_self_Shape_1_output_0", builder.constant({ dataType: "int64", shape: [3] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0, 0, 128, 1, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_1_attention_self_Shape_2_output_0", builder.constant({ dataType: "int64", shape: [3] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0, 0, 128, 1, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_1_attention_self_Shape_3_output_0", builder.constant({ dataType: "int64", shape: [3] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0, 0, 128, 1, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_1_attention_self_Shape_4_output_0", builder.constant({ dataType: "int64", shape: [3] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0, 0, 128, 1, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_1_attention_self_Shape_5_output_0", builder.constant({ dataType: "int64", shape: [3] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0, 0, 128, 1, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_1_attention_self_Shape_6_output_0", builder.constant({ dataType: "int64", shape: [4] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_1_attention_self_Shape_7_output_0", builder.constant({ dataType: "int64", shape: [4] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_1_attention_self_Shape_output_0", builder.constant({ dataType: "int64", shape: [3] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0, 0, 128, 1, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_1_attention_self_Unsqueeze_1_output_0", builder.constant({ dataType: "int64", shape: [] }, new Uint8Array([128, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_1_attention_self_Unsqueeze_2_output_0", builder.constant({ dataType: "int64", shape: [] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_1_attention_self_Unsqueeze_3_output_0", builder.constant({ dataType: "int64", shape: [] }, new Uint8Array([128, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_1_attention_self_Unsqueeze_4_output_0", builder.constant({ dataType: "int64", shape: [] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_1_attention_self_Unsqueeze_5_output_0", builder.constant({ dataType: "int64", shape: [] }, new Uint8Array([128, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_1_attention_self_Unsqueeze_6_output_0", builder.constant({ dataType: "int64", shape: [] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_1_attention_self_Unsqueeze_7_output_0", builder.constant({ dataType: "int64", shape: [] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_1_attention_self_Unsqueeze_output_0", builder.constant({ dataType: "int64", shape: [] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_2_attention_self_Concat_1_output_0", builder.constant({ dataType: "int64", shape: [4] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_2_attention_self_Concat_2_output_0", builder.constant({ dataType: "int64", shape: [4] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_2_attention_self_Concat_3_output_0", builder.constant({ dataType: "int64", shape: [3] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 128, 1, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_2_attention_self_Concat_output_0", builder.constant({ dataType: "int64", shape: [4] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0]).buffer));
  {
    const sl = weights.getSlice("_encoder_layer_2_attention_self_Constant_10_output_0");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_encoder_layer_2_attention_self_Constant_10_output_0", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("_encoder_layer_2_attention_self_Constant_11_output_0");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_encoder_layer_2_attention_self_Constant_11_output_0", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("_encoder_layer_2_attention_self_Constant_13_output_0");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_encoder_layer_2_attention_self_Constant_13_output_0", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("_encoder_layer_2_attention_self_Constant_14_output_0");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_encoder_layer_2_attention_self_Constant_14_output_0", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("_encoder_layer_2_attention_self_Constant_15_output_0");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_encoder_layer_2_attention_self_Constant_15_output_0", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("_encoder_layer_2_attention_self_Constant_1_output_0");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_encoder_layer_2_attention_self_Constant_1_output_0", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("_encoder_layer_2_attention_self_Constant_2_output_0");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_encoder_layer_2_attention_self_Constant_2_output_0", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("_encoder_layer_2_attention_self_Constant_3_output_0");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_encoder_layer_2_attention_self_Constant_3_output_0", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("_encoder_layer_2_attention_self_Constant_4_output_0");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_encoder_layer_2_attention_self_Constant_4_output_0", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("_encoder_layer_2_attention_self_Constant_5_output_0");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_encoder_layer_2_attention_self_Constant_5_output_0", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("_encoder_layer_2_attention_self_Constant_6_output_0");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_encoder_layer_2_attention_self_Constant_6_output_0", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("_encoder_layer_2_attention_self_Constant_7_output_0");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_encoder_layer_2_attention_self_Constant_7_output_0", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("_encoder_layer_2_attention_self_Constant_8_output_0");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_encoder_layer_2_attention_self_Constant_8_output_0", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("_encoder_layer_2_attention_self_Constant_9_output_0");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_encoder_layer_2_attention_self_Constant_9_output_0", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("_encoder_layer_2_attention_self_Constant_output_0");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_encoder_layer_2_attention_self_Constant_output_0", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  env.set("_encoder_layer_2_attention_self_Gather_1_output_0", builder.constant({ dataType: "int64", shape: [] }, new Uint8Array([128, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_2_attention_self_Gather_2_output_0", builder.constant({ dataType: "int64", shape: [] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_2_attention_self_Gather_3_output_0", builder.constant({ dataType: "int64", shape: [] }, new Uint8Array([128, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_2_attention_self_Gather_4_output_0", builder.constant({ dataType: "int64", shape: [] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_2_attention_self_Gather_5_output_0", builder.constant({ dataType: "int64", shape: [] }, new Uint8Array([128, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_2_attention_self_Gather_6_output_0", builder.constant({ dataType: "int64", shape: [] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_2_attention_self_Gather_7_output_0", builder.constant({ dataType: "int64", shape: [] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_2_attention_self_Gather_output_0", builder.constant({ dataType: "int64", shape: [] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_2_attention_self_Shape_1_output_0", builder.constant({ dataType: "int64", shape: [3] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0, 0, 128, 1, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_2_attention_self_Shape_2_output_0", builder.constant({ dataType: "int64", shape: [3] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0, 0, 128, 1, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_2_attention_self_Shape_3_output_0", builder.constant({ dataType: "int64", shape: [3] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0, 0, 128, 1, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_2_attention_self_Shape_4_output_0", builder.constant({ dataType: "int64", shape: [3] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0, 0, 128, 1, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_2_attention_self_Shape_5_output_0", builder.constant({ dataType: "int64", shape: [3] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0, 0, 128, 1, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_2_attention_self_Shape_6_output_0", builder.constant({ dataType: "int64", shape: [4] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_2_attention_self_Shape_7_output_0", builder.constant({ dataType: "int64", shape: [4] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_2_attention_self_Shape_output_0", builder.constant({ dataType: "int64", shape: [3] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0, 0, 128, 1, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_2_attention_self_Unsqueeze_1_output_0", builder.constant({ dataType: "int64", shape: [] }, new Uint8Array([128, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_2_attention_self_Unsqueeze_2_output_0", builder.constant({ dataType: "int64", shape: [] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_2_attention_self_Unsqueeze_3_output_0", builder.constant({ dataType: "int64", shape: [] }, new Uint8Array([128, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_2_attention_self_Unsqueeze_4_output_0", builder.constant({ dataType: "int64", shape: [] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_2_attention_self_Unsqueeze_5_output_0", builder.constant({ dataType: "int64", shape: [] }, new Uint8Array([128, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_2_attention_self_Unsqueeze_6_output_0", builder.constant({ dataType: "int64", shape: [] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_2_attention_self_Unsqueeze_7_output_0", builder.constant({ dataType: "int64", shape: [] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_2_attention_self_Unsqueeze_output_0", builder.constant({ dataType: "int64", shape: [] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_3_attention_self_Concat_1_output_0", builder.constant({ dataType: "int64", shape: [4] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_3_attention_self_Concat_2_output_0", builder.constant({ dataType: "int64", shape: [4] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_3_attention_self_Concat_3_output_0", builder.constant({ dataType: "int64", shape: [3] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 128, 1, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_3_attention_self_Concat_output_0", builder.constant({ dataType: "int64", shape: [4] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0]).buffer));
  {
    const sl = weights.getSlice("_encoder_layer_3_attention_self_Constant_10_output_0");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_encoder_layer_3_attention_self_Constant_10_output_0", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("_encoder_layer_3_attention_self_Constant_11_output_0");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_encoder_layer_3_attention_self_Constant_11_output_0", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("_encoder_layer_3_attention_self_Constant_13_output_0");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_encoder_layer_3_attention_self_Constant_13_output_0", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("_encoder_layer_3_attention_self_Constant_14_output_0");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_encoder_layer_3_attention_self_Constant_14_output_0", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("_encoder_layer_3_attention_self_Constant_15_output_0");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_encoder_layer_3_attention_self_Constant_15_output_0", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("_encoder_layer_3_attention_self_Constant_1_output_0");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_encoder_layer_3_attention_self_Constant_1_output_0", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("_encoder_layer_3_attention_self_Constant_2_output_0");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_encoder_layer_3_attention_self_Constant_2_output_0", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("_encoder_layer_3_attention_self_Constant_3_output_0");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_encoder_layer_3_attention_self_Constant_3_output_0", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("_encoder_layer_3_attention_self_Constant_4_output_0");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_encoder_layer_3_attention_self_Constant_4_output_0", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("_encoder_layer_3_attention_self_Constant_5_output_0");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_encoder_layer_3_attention_self_Constant_5_output_0", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("_encoder_layer_3_attention_self_Constant_6_output_0");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_encoder_layer_3_attention_self_Constant_6_output_0", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("_encoder_layer_3_attention_self_Constant_7_output_0");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_encoder_layer_3_attention_self_Constant_7_output_0", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("_encoder_layer_3_attention_self_Constant_8_output_0");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_encoder_layer_3_attention_self_Constant_8_output_0", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("_encoder_layer_3_attention_self_Constant_9_output_0");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_encoder_layer_3_attention_self_Constant_9_output_0", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("_encoder_layer_3_attention_self_Constant_output_0");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_encoder_layer_3_attention_self_Constant_output_0", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  env.set("_encoder_layer_3_attention_self_Gather_1_output_0", builder.constant({ dataType: "int64", shape: [] }, new Uint8Array([128, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_3_attention_self_Gather_2_output_0", builder.constant({ dataType: "int64", shape: [] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_3_attention_self_Gather_3_output_0", builder.constant({ dataType: "int64", shape: [] }, new Uint8Array([128, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_3_attention_self_Gather_4_output_0", builder.constant({ dataType: "int64", shape: [] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_3_attention_self_Gather_5_output_0", builder.constant({ dataType: "int64", shape: [] }, new Uint8Array([128, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_3_attention_self_Gather_6_output_0", builder.constant({ dataType: "int64", shape: [] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_3_attention_self_Gather_7_output_0", builder.constant({ dataType: "int64", shape: [] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_3_attention_self_Gather_output_0", builder.constant({ dataType: "int64", shape: [] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_3_attention_self_Shape_1_output_0", builder.constant({ dataType: "int64", shape: [3] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0, 0, 128, 1, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_3_attention_self_Shape_2_output_0", builder.constant({ dataType: "int64", shape: [3] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0, 0, 128, 1, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_3_attention_self_Shape_3_output_0", builder.constant({ dataType: "int64", shape: [3] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0, 0, 128, 1, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_3_attention_self_Shape_4_output_0", builder.constant({ dataType: "int64", shape: [3] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0, 0, 128, 1, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_3_attention_self_Shape_5_output_0", builder.constant({ dataType: "int64", shape: [3] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0, 0, 128, 1, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_3_attention_self_Shape_6_output_0", builder.constant({ dataType: "int64", shape: [4] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_3_attention_self_Shape_7_output_0", builder.constant({ dataType: "int64", shape: [4] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_3_attention_self_Shape_output_0", builder.constant({ dataType: "int64", shape: [3] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0, 0, 128, 1, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_3_attention_self_Unsqueeze_1_output_0", builder.constant({ dataType: "int64", shape: [] }, new Uint8Array([128, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_3_attention_self_Unsqueeze_2_output_0", builder.constant({ dataType: "int64", shape: [] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_3_attention_self_Unsqueeze_3_output_0", builder.constant({ dataType: "int64", shape: [] }, new Uint8Array([128, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_3_attention_self_Unsqueeze_4_output_0", builder.constant({ dataType: "int64", shape: [] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_3_attention_self_Unsqueeze_5_output_0", builder.constant({ dataType: "int64", shape: [] }, new Uint8Array([128, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_3_attention_self_Unsqueeze_6_output_0", builder.constant({ dataType: "int64", shape: [] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_3_attention_self_Unsqueeze_7_output_0", builder.constant({ dataType: "int64", shape: [] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_3_attention_self_Unsqueeze_output_0", builder.constant({ dataType: "int64", shape: [] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_4_attention_self_Concat_1_output_0", builder.constant({ dataType: "int64", shape: [4] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_4_attention_self_Concat_2_output_0", builder.constant({ dataType: "int64", shape: [4] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_4_attention_self_Concat_3_output_0", builder.constant({ dataType: "int64", shape: [3] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 128, 1, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_4_attention_self_Concat_output_0", builder.constant({ dataType: "int64", shape: [4] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0]).buffer));
  {
    const sl = weights.getSlice("_encoder_layer_4_attention_self_Constant_10_output_0");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_encoder_layer_4_attention_self_Constant_10_output_0", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("_encoder_layer_4_attention_self_Constant_11_output_0");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_encoder_layer_4_attention_self_Constant_11_output_0", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("_encoder_layer_4_attention_self_Constant_13_output_0");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_encoder_layer_4_attention_self_Constant_13_output_0", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("_encoder_layer_4_attention_self_Constant_14_output_0");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_encoder_layer_4_attention_self_Constant_14_output_0", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("_encoder_layer_4_attention_self_Constant_15_output_0");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_encoder_layer_4_attention_self_Constant_15_output_0", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("_encoder_layer_4_attention_self_Constant_1_output_0");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_encoder_layer_4_attention_self_Constant_1_output_0", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("_encoder_layer_4_attention_self_Constant_2_output_0");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_encoder_layer_4_attention_self_Constant_2_output_0", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("_encoder_layer_4_attention_self_Constant_3_output_0");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_encoder_layer_4_attention_self_Constant_3_output_0", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("_encoder_layer_4_attention_self_Constant_4_output_0");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_encoder_layer_4_attention_self_Constant_4_output_0", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("_encoder_layer_4_attention_self_Constant_5_output_0");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_encoder_layer_4_attention_self_Constant_5_output_0", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("_encoder_layer_4_attention_self_Constant_6_output_0");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_encoder_layer_4_attention_self_Constant_6_output_0", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("_encoder_layer_4_attention_self_Constant_7_output_0");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_encoder_layer_4_attention_self_Constant_7_output_0", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("_encoder_layer_4_attention_self_Constant_8_output_0");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_encoder_layer_4_attention_self_Constant_8_output_0", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("_encoder_layer_4_attention_self_Constant_9_output_0");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_encoder_layer_4_attention_self_Constant_9_output_0", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("_encoder_layer_4_attention_self_Constant_output_0");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_encoder_layer_4_attention_self_Constant_output_0", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  env.set("_encoder_layer_4_attention_self_Gather_1_output_0", builder.constant({ dataType: "int64", shape: [] }, new Uint8Array([128, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_4_attention_self_Gather_2_output_0", builder.constant({ dataType: "int64", shape: [] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_4_attention_self_Gather_3_output_0", builder.constant({ dataType: "int64", shape: [] }, new Uint8Array([128, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_4_attention_self_Gather_4_output_0", builder.constant({ dataType: "int64", shape: [] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_4_attention_self_Gather_5_output_0", builder.constant({ dataType: "int64", shape: [] }, new Uint8Array([128, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_4_attention_self_Gather_6_output_0", builder.constant({ dataType: "int64", shape: [] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_4_attention_self_Gather_7_output_0", builder.constant({ dataType: "int64", shape: [] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_4_attention_self_Gather_output_0", builder.constant({ dataType: "int64", shape: [] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_4_attention_self_Shape_1_output_0", builder.constant({ dataType: "int64", shape: [3] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0, 0, 128, 1, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_4_attention_self_Shape_2_output_0", builder.constant({ dataType: "int64", shape: [3] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0, 0, 128, 1, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_4_attention_self_Shape_3_output_0", builder.constant({ dataType: "int64", shape: [3] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0, 0, 128, 1, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_4_attention_self_Shape_4_output_0", builder.constant({ dataType: "int64", shape: [3] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0, 0, 128, 1, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_4_attention_self_Shape_5_output_0", builder.constant({ dataType: "int64", shape: [3] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0, 0, 128, 1, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_4_attention_self_Shape_6_output_0", builder.constant({ dataType: "int64", shape: [4] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_4_attention_self_Shape_7_output_0", builder.constant({ dataType: "int64", shape: [4] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_4_attention_self_Shape_output_0", builder.constant({ dataType: "int64", shape: [3] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0, 0, 128, 1, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_4_attention_self_Unsqueeze_1_output_0", builder.constant({ dataType: "int64", shape: [] }, new Uint8Array([128, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_4_attention_self_Unsqueeze_2_output_0", builder.constant({ dataType: "int64", shape: [] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_4_attention_self_Unsqueeze_3_output_0", builder.constant({ dataType: "int64", shape: [] }, new Uint8Array([128, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_4_attention_self_Unsqueeze_4_output_0", builder.constant({ dataType: "int64", shape: [] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_4_attention_self_Unsqueeze_5_output_0", builder.constant({ dataType: "int64", shape: [] }, new Uint8Array([128, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_4_attention_self_Unsqueeze_6_output_0", builder.constant({ dataType: "int64", shape: [] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_4_attention_self_Unsqueeze_7_output_0", builder.constant({ dataType: "int64", shape: [] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_4_attention_self_Unsqueeze_output_0", builder.constant({ dataType: "int64", shape: [] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_5_attention_self_Concat_1_output_0", builder.constant({ dataType: "int64", shape: [4] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_5_attention_self_Concat_2_output_0", builder.constant({ dataType: "int64", shape: [4] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_5_attention_self_Concat_3_output_0", builder.constant({ dataType: "int64", shape: [3] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 128, 1, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_5_attention_self_Concat_output_0", builder.constant({ dataType: "int64", shape: [4] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0]).buffer));
  {
    const sl = weights.getSlice("_encoder_layer_5_attention_self_Constant_10_output_0");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_encoder_layer_5_attention_self_Constant_10_output_0", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("_encoder_layer_5_attention_self_Constant_11_output_0");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_encoder_layer_5_attention_self_Constant_11_output_0", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("_encoder_layer_5_attention_self_Constant_13_output_0");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_encoder_layer_5_attention_self_Constant_13_output_0", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("_encoder_layer_5_attention_self_Constant_14_output_0");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_encoder_layer_5_attention_self_Constant_14_output_0", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("_encoder_layer_5_attention_self_Constant_15_output_0");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_encoder_layer_5_attention_self_Constant_15_output_0", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("_encoder_layer_5_attention_self_Constant_1_output_0");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_encoder_layer_5_attention_self_Constant_1_output_0", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("_encoder_layer_5_attention_self_Constant_2_output_0");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_encoder_layer_5_attention_self_Constant_2_output_0", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("_encoder_layer_5_attention_self_Constant_3_output_0");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_encoder_layer_5_attention_self_Constant_3_output_0", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("_encoder_layer_5_attention_self_Constant_4_output_0");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_encoder_layer_5_attention_self_Constant_4_output_0", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("_encoder_layer_5_attention_self_Constant_5_output_0");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_encoder_layer_5_attention_self_Constant_5_output_0", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("_encoder_layer_5_attention_self_Constant_6_output_0");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_encoder_layer_5_attention_self_Constant_6_output_0", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("_encoder_layer_5_attention_self_Constant_7_output_0");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_encoder_layer_5_attention_self_Constant_7_output_0", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("_encoder_layer_5_attention_self_Constant_8_output_0");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_encoder_layer_5_attention_self_Constant_8_output_0", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("_encoder_layer_5_attention_self_Constant_9_output_0");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_encoder_layer_5_attention_self_Constant_9_output_0", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("_encoder_layer_5_attention_self_Constant_output_0");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_encoder_layer_5_attention_self_Constant_output_0", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  env.set("_encoder_layer_5_attention_self_Gather_1_output_0", builder.constant({ dataType: "int64", shape: [] }, new Uint8Array([128, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_5_attention_self_Gather_2_output_0", builder.constant({ dataType: "int64", shape: [] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_5_attention_self_Gather_3_output_0", builder.constant({ dataType: "int64", shape: [] }, new Uint8Array([128, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_5_attention_self_Gather_4_output_0", builder.constant({ dataType: "int64", shape: [] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_5_attention_self_Gather_5_output_0", builder.constant({ dataType: "int64", shape: [] }, new Uint8Array([128, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_5_attention_self_Gather_6_output_0", builder.constant({ dataType: "int64", shape: [] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_5_attention_self_Gather_7_output_0", builder.constant({ dataType: "int64", shape: [] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_5_attention_self_Gather_output_0", builder.constant({ dataType: "int64", shape: [] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_5_attention_self_Shape_1_output_0", builder.constant({ dataType: "int64", shape: [3] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0, 0, 128, 1, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_5_attention_self_Shape_2_output_0", builder.constant({ dataType: "int64", shape: [3] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0, 0, 128, 1, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_5_attention_self_Shape_3_output_0", builder.constant({ dataType: "int64", shape: [3] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0, 0, 128, 1, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_5_attention_self_Shape_4_output_0", builder.constant({ dataType: "int64", shape: [3] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0, 0, 128, 1, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_5_attention_self_Shape_5_output_0", builder.constant({ dataType: "int64", shape: [3] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0, 0, 128, 1, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_5_attention_self_Shape_6_output_0", builder.constant({ dataType: "int64", shape: [4] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_5_attention_self_Shape_7_output_0", builder.constant({ dataType: "int64", shape: [4] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_5_attention_self_Shape_output_0", builder.constant({ dataType: "int64", shape: [3] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0, 0, 128, 1, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_5_attention_self_Unsqueeze_1_output_0", builder.constant({ dataType: "int64", shape: [] }, new Uint8Array([128, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_5_attention_self_Unsqueeze_2_output_0", builder.constant({ dataType: "int64", shape: [] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_5_attention_self_Unsqueeze_3_output_0", builder.constant({ dataType: "int64", shape: [] }, new Uint8Array([128, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_5_attention_self_Unsqueeze_4_output_0", builder.constant({ dataType: "int64", shape: [] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_5_attention_self_Unsqueeze_5_output_0", builder.constant({ dataType: "int64", shape: [] }, new Uint8Array([128, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_5_attention_self_Unsqueeze_6_output_0", builder.constant({ dataType: "int64", shape: [] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_5_attention_self_Unsqueeze_7_output_0", builder.constant({ dataType: "int64", shape: [] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0]).buffer));
  env.set("_encoder_layer_5_attention_self_Unsqueeze_output_0", builder.constant({ dataType: "int64", shape: [] }, new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0]).buffer));
  {
    const sl = weights.getSlice("embeddings_LayerNorm_bias");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("embeddings_LayerNorm_bias", builder.constant({ dataType: "float16", shape: [384] }, buf));
  }
  {
    const sl = weights.getSlice("embeddings_LayerNorm_weight");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("embeddings_LayerNorm_weight", builder.constant({ dataType: "float16", shape: [384] }, buf));
  }
  {
    const sl = weights.getSlice("embeddings_position_embeddings_weight");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("embeddings_position_embeddings_weight", builder.constant({ dataType: "float16", shape: [512, 384] }, buf));
  }
  {
    const sl = weights.getSlice("embeddings_position_ids");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("embeddings_position_ids", builder.constant({ dataType: "int64", shape: [1, 512] }, buf));
  }
  {
    const sl = weights.getSlice("embeddings_token_type_embeddings_weight");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("embeddings_token_type_embeddings_weight", builder.constant({ dataType: "float16", shape: [2, 384] }, buf));
  }
  {
    const sl = weights.getSlice("embeddings_word_embeddings_weight");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("embeddings_word_embeddings_weight", builder.constant({ dataType: "float16", shape: [30522, 384] }, buf));
  }
  {
    const sl = weights.getSlice("encoder_layer_0_attention_output_LayerNorm_bias");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("encoder_layer_0_attention_output_LayerNorm_bias", builder.constant({ dataType: "float16", shape: [384] }, buf));
  }
  {
    const sl = weights.getSlice("encoder_layer_0_attention_output_LayerNorm_weight");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("encoder_layer_0_attention_output_LayerNorm_weight", builder.constant({ dataType: "float16", shape: [384] }, buf));
  }
  {
    const sl = weights.getSlice("encoder_layer_0_attention_output_dense_bias");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("encoder_layer_0_attention_output_dense_bias", builder.constant({ dataType: "float16", shape: [384] }, buf));
  }
  {
    const sl = weights.getSlice("encoder_layer_0_attention_self_key_bias");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("encoder_layer_0_attention_self_key_bias", builder.constant({ dataType: "float16", shape: [384] }, buf));
  }
  {
    const sl = weights.getSlice("encoder_layer_0_attention_self_query_bias");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("encoder_layer_0_attention_self_query_bias", builder.constant({ dataType: "float16", shape: [384] }, buf));
  }
  {
    const sl = weights.getSlice("encoder_layer_0_attention_self_value_bias");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("encoder_layer_0_attention_self_value_bias", builder.constant({ dataType: "float16", shape: [384] }, buf));
  }
  {
    const sl = weights.getSlice("encoder_layer_0_intermediate_dense_bias");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("encoder_layer_0_intermediate_dense_bias", builder.constant({ dataType: "float16", shape: [1536] }, buf));
  }
  {
    const sl = weights.getSlice("encoder_layer_0_output_LayerNorm_bias");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("encoder_layer_0_output_LayerNorm_bias", builder.constant({ dataType: "float16", shape: [384] }, buf));
  }
  {
    const sl = weights.getSlice("encoder_layer_0_output_LayerNorm_weight");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("encoder_layer_0_output_LayerNorm_weight", builder.constant({ dataType: "float16", shape: [384] }, buf));
  }
  {
    const sl = weights.getSlice("encoder_layer_0_output_dense_bias");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("encoder_layer_0_output_dense_bias", builder.constant({ dataType: "float16", shape: [384] }, buf));
  }
  {
    const sl = weights.getSlice("encoder_layer_1_attention_output_LayerNorm_bias");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("encoder_layer_1_attention_output_LayerNorm_bias", builder.constant({ dataType: "float16", shape: [384] }, buf));
  }
  {
    const sl = weights.getSlice("encoder_layer_1_attention_output_LayerNorm_weight");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("encoder_layer_1_attention_output_LayerNorm_weight", builder.constant({ dataType: "float16", shape: [384] }, buf));
  }
  {
    const sl = weights.getSlice("encoder_layer_1_attention_output_dense_bias");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("encoder_layer_1_attention_output_dense_bias", builder.constant({ dataType: "float16", shape: [384] }, buf));
  }
  {
    const sl = weights.getSlice("encoder_layer_1_attention_self_key_bias");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("encoder_layer_1_attention_self_key_bias", builder.constant({ dataType: "float16", shape: [384] }, buf));
  }
  {
    const sl = weights.getSlice("encoder_layer_1_attention_self_query_bias");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("encoder_layer_1_attention_self_query_bias", builder.constant({ dataType: "float16", shape: [384] }, buf));
  }
  {
    const sl = weights.getSlice("encoder_layer_1_attention_self_value_bias");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("encoder_layer_1_attention_self_value_bias", builder.constant({ dataType: "float16", shape: [384] }, buf));
  }
  {
    const sl = weights.getSlice("encoder_layer_1_intermediate_dense_bias");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("encoder_layer_1_intermediate_dense_bias", builder.constant({ dataType: "float16", shape: [1536] }, buf));
  }
  {
    const sl = weights.getSlice("encoder_layer_1_output_LayerNorm_bias");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("encoder_layer_1_output_LayerNorm_bias", builder.constant({ dataType: "float16", shape: [384] }, buf));
  }
  {
    const sl = weights.getSlice("encoder_layer_1_output_LayerNorm_weight");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("encoder_layer_1_output_LayerNorm_weight", builder.constant({ dataType: "float16", shape: [384] }, buf));
  }
  {
    const sl = weights.getSlice("encoder_layer_1_output_dense_bias");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("encoder_layer_1_output_dense_bias", builder.constant({ dataType: "float16", shape: [384] }, buf));
  }
  {
    const sl = weights.getSlice("encoder_layer_2_attention_output_LayerNorm_bias");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("encoder_layer_2_attention_output_LayerNorm_bias", builder.constant({ dataType: "float16", shape: [384] }, buf));
  }
  {
    const sl = weights.getSlice("encoder_layer_2_attention_output_LayerNorm_weight");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("encoder_layer_2_attention_output_LayerNorm_weight", builder.constant({ dataType: "float16", shape: [384] }, buf));
  }
  {
    const sl = weights.getSlice("encoder_layer_2_attention_output_dense_bias");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("encoder_layer_2_attention_output_dense_bias", builder.constant({ dataType: "float16", shape: [384] }, buf));
  }
  {
    const sl = weights.getSlice("encoder_layer_2_attention_self_key_bias");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("encoder_layer_2_attention_self_key_bias", builder.constant({ dataType: "float16", shape: [384] }, buf));
  }
  {
    const sl = weights.getSlice("encoder_layer_2_attention_self_query_bias");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("encoder_layer_2_attention_self_query_bias", builder.constant({ dataType: "float16", shape: [384] }, buf));
  }
  {
    const sl = weights.getSlice("encoder_layer_2_attention_self_value_bias");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("encoder_layer_2_attention_self_value_bias", builder.constant({ dataType: "float16", shape: [384] }, buf));
  }
  {
    const sl = weights.getSlice("encoder_layer_2_intermediate_dense_bias");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("encoder_layer_2_intermediate_dense_bias", builder.constant({ dataType: "float16", shape: [1536] }, buf));
  }
  {
    const sl = weights.getSlice("encoder_layer_2_output_LayerNorm_bias");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("encoder_layer_2_output_LayerNorm_bias", builder.constant({ dataType: "float16", shape: [384] }, buf));
  }
  {
    const sl = weights.getSlice("encoder_layer_2_output_LayerNorm_weight");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("encoder_layer_2_output_LayerNorm_weight", builder.constant({ dataType: "float16", shape: [384] }, buf));
  }
  {
    const sl = weights.getSlice("encoder_layer_2_output_dense_bias");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("encoder_layer_2_output_dense_bias", builder.constant({ dataType: "float16", shape: [384] }, buf));
  }
  {
    const sl = weights.getSlice("encoder_layer_3_attention_output_LayerNorm_bias");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("encoder_layer_3_attention_output_LayerNorm_bias", builder.constant({ dataType: "float16", shape: [384] }, buf));
  }
  {
    const sl = weights.getSlice("encoder_layer_3_attention_output_LayerNorm_weight");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("encoder_layer_3_attention_output_LayerNorm_weight", builder.constant({ dataType: "float16", shape: [384] }, buf));
  }
  {
    const sl = weights.getSlice("encoder_layer_3_attention_output_dense_bias");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("encoder_layer_3_attention_output_dense_bias", builder.constant({ dataType: "float16", shape: [384] }, buf));
  }
  {
    const sl = weights.getSlice("encoder_layer_3_attention_self_key_bias");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("encoder_layer_3_attention_self_key_bias", builder.constant({ dataType: "float16", shape: [384] }, buf));
  }
  {
    const sl = weights.getSlice("encoder_layer_3_attention_self_query_bias");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("encoder_layer_3_attention_self_query_bias", builder.constant({ dataType: "float16", shape: [384] }, buf));
  }
  {
    const sl = weights.getSlice("encoder_layer_3_attention_self_value_bias");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("encoder_layer_3_attention_self_value_bias", builder.constant({ dataType: "float16", shape: [384] }, buf));
  }
  {
    const sl = weights.getSlice("encoder_layer_3_intermediate_dense_bias");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("encoder_layer_3_intermediate_dense_bias", builder.constant({ dataType: "float16", shape: [1536] }, buf));
  }
  {
    const sl = weights.getSlice("encoder_layer_3_output_LayerNorm_bias");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("encoder_layer_3_output_LayerNorm_bias", builder.constant({ dataType: "float16", shape: [384] }, buf));
  }
  {
    const sl = weights.getSlice("encoder_layer_3_output_LayerNorm_weight");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("encoder_layer_3_output_LayerNorm_weight", builder.constant({ dataType: "float16", shape: [384] }, buf));
  }
  {
    const sl = weights.getSlice("encoder_layer_3_output_dense_bias");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("encoder_layer_3_output_dense_bias", builder.constant({ dataType: "float16", shape: [384] }, buf));
  }
  {
    const sl = weights.getSlice("encoder_layer_4_attention_output_LayerNorm_bias");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("encoder_layer_4_attention_output_LayerNorm_bias", builder.constant({ dataType: "float16", shape: [384] }, buf));
  }
  {
    const sl = weights.getSlice("encoder_layer_4_attention_output_LayerNorm_weight");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("encoder_layer_4_attention_output_LayerNorm_weight", builder.constant({ dataType: "float16", shape: [384] }, buf));
  }
  {
    const sl = weights.getSlice("encoder_layer_4_attention_output_dense_bias");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("encoder_layer_4_attention_output_dense_bias", builder.constant({ dataType: "float16", shape: [384] }, buf));
  }
  {
    const sl = weights.getSlice("encoder_layer_4_attention_self_key_bias");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("encoder_layer_4_attention_self_key_bias", builder.constant({ dataType: "float16", shape: [384] }, buf));
  }
  {
    const sl = weights.getSlice("encoder_layer_4_attention_self_query_bias");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("encoder_layer_4_attention_self_query_bias", builder.constant({ dataType: "float16", shape: [384] }, buf));
  }
  {
    const sl = weights.getSlice("encoder_layer_4_attention_self_value_bias");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("encoder_layer_4_attention_self_value_bias", builder.constant({ dataType: "float16", shape: [384] }, buf));
  }
  {
    const sl = weights.getSlice("encoder_layer_4_intermediate_dense_bias");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("encoder_layer_4_intermediate_dense_bias", builder.constant({ dataType: "float16", shape: [1536] }, buf));
  }
  {
    const sl = weights.getSlice("encoder_layer_4_output_LayerNorm_bias");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("encoder_layer_4_output_LayerNorm_bias", builder.constant({ dataType: "float16", shape: [384] }, buf));
  }
  {
    const sl = weights.getSlice("encoder_layer_4_output_LayerNorm_weight");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("encoder_layer_4_output_LayerNorm_weight", builder.constant({ dataType: "float16", shape: [384] }, buf));
  }
  {
    const sl = weights.getSlice("encoder_layer_4_output_dense_bias");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("encoder_layer_4_output_dense_bias", builder.constant({ dataType: "float16", shape: [384] }, buf));
  }
  {
    const sl = weights.getSlice("encoder_layer_5_attention_output_LayerNorm_bias");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("encoder_layer_5_attention_output_LayerNorm_bias", builder.constant({ dataType: "float16", shape: [384] }, buf));
  }
  {
    const sl = weights.getSlice("encoder_layer_5_attention_output_LayerNorm_weight");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("encoder_layer_5_attention_output_LayerNorm_weight", builder.constant({ dataType: "float16", shape: [384] }, buf));
  }
  {
    const sl = weights.getSlice("encoder_layer_5_attention_output_dense_bias");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("encoder_layer_5_attention_output_dense_bias", builder.constant({ dataType: "float16", shape: [384] }, buf));
  }
  {
    const sl = weights.getSlice("encoder_layer_5_attention_self_key_bias");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("encoder_layer_5_attention_self_key_bias", builder.constant({ dataType: "float16", shape: [384] }, buf));
  }
  {
    const sl = weights.getSlice("encoder_layer_5_attention_self_query_bias");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("encoder_layer_5_attention_self_query_bias", builder.constant({ dataType: "float16", shape: [384] }, buf));
  }
  {
    const sl = weights.getSlice("encoder_layer_5_attention_self_value_bias");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("encoder_layer_5_attention_self_value_bias", builder.constant({ dataType: "float16", shape: [384] }, buf));
  }
  {
    const sl = weights.getSlice("encoder_layer_5_intermediate_dense_bias");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("encoder_layer_5_intermediate_dense_bias", builder.constant({ dataType: "float16", shape: [1536] }, buf));
  }
  {
    const sl = weights.getSlice("encoder_layer_5_output_LayerNorm_bias");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("encoder_layer_5_output_LayerNorm_bias", builder.constant({ dataType: "float16", shape: [384] }, buf));
  }
  {
    const sl = weights.getSlice("encoder_layer_5_output_LayerNorm_weight");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("encoder_layer_5_output_LayerNorm_weight", builder.constant({ dataType: "float16", shape: [384] }, buf));
  }
  {
    const sl = weights.getSlice("encoder_layer_5_output_dense_bias");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("encoder_layer_5_output_dense_bias", builder.constant({ dataType: "float16", shape: [384] }, buf));
  }
  {
    const sl = weights.getSlice("onnx__MatMul_787");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("onnx__MatMul_787", builder.constant({ dataType: "float16", shape: [384, 384] }, buf));
  }
  {
    const sl = weights.getSlice("onnx__MatMul_788");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("onnx__MatMul_788", builder.constant({ dataType: "float16", shape: [384, 384] }, buf));
  }
  {
    const sl = weights.getSlice("onnx__MatMul_791");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("onnx__MatMul_791", builder.constant({ dataType: "float16", shape: [384, 384] }, buf));
  }
  {
    const sl = weights.getSlice("onnx__MatMul_797");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("onnx__MatMul_797", builder.constant({ dataType: "float16", shape: [384, 384] }, buf));
  }
  {
    const sl = weights.getSlice("onnx__MatMul_798");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("onnx__MatMul_798", builder.constant({ dataType: "float16", shape: [384, 1536] }, buf));
  }
  {
    const sl = weights.getSlice("onnx__MatMul_799");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("onnx__MatMul_799", builder.constant({ dataType: "float16", shape: [1536, 384] }, buf));
  }
  {
    const sl = weights.getSlice("onnx__MatMul_800");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("onnx__MatMul_800", builder.constant({ dataType: "float16", shape: [384, 384] }, buf));
  }
  {
    const sl = weights.getSlice("onnx__MatMul_801");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("onnx__MatMul_801", builder.constant({ dataType: "float16", shape: [384, 384] }, buf));
  }
  {
    const sl = weights.getSlice("onnx__MatMul_804");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("onnx__MatMul_804", builder.constant({ dataType: "float16", shape: [384, 384] }, buf));
  }
  {
    const sl = weights.getSlice("onnx__MatMul_810");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("onnx__MatMul_810", builder.constant({ dataType: "float16", shape: [384, 384] }, buf));
  }
  {
    const sl = weights.getSlice("onnx__MatMul_811");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("onnx__MatMul_811", builder.constant({ dataType: "float16", shape: [384, 1536] }, buf));
  }
  {
    const sl = weights.getSlice("onnx__MatMul_812");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("onnx__MatMul_812", builder.constant({ dataType: "float16", shape: [1536, 384] }, buf));
  }
  {
    const sl = weights.getSlice("onnx__MatMul_813");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("onnx__MatMul_813", builder.constant({ dataType: "float16", shape: [384, 384] }, buf));
  }
  {
    const sl = weights.getSlice("onnx__MatMul_814");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("onnx__MatMul_814", builder.constant({ dataType: "float16", shape: [384, 384] }, buf));
  }
  {
    const sl = weights.getSlice("onnx__MatMul_817");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("onnx__MatMul_817", builder.constant({ dataType: "float16", shape: [384, 384] }, buf));
  }
  {
    const sl = weights.getSlice("onnx__MatMul_823");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("onnx__MatMul_823", builder.constant({ dataType: "float16", shape: [384, 384] }, buf));
  }
  {
    const sl = weights.getSlice("onnx__MatMul_824");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("onnx__MatMul_824", builder.constant({ dataType: "float16", shape: [384, 1536] }, buf));
  }
  {
    const sl = weights.getSlice("onnx__MatMul_825");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("onnx__MatMul_825", builder.constant({ dataType: "float16", shape: [1536, 384] }, buf));
  }
  {
    const sl = weights.getSlice("onnx__MatMul_826");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("onnx__MatMul_826", builder.constant({ dataType: "float16", shape: [384, 384] }, buf));
  }
  {
    const sl = weights.getSlice("onnx__MatMul_827");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("onnx__MatMul_827", builder.constant({ dataType: "float16", shape: [384, 384] }, buf));
  }
  {
    const sl = weights.getSlice("onnx__MatMul_830");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("onnx__MatMul_830", builder.constant({ dataType: "float16", shape: [384, 384] }, buf));
  }
  {
    const sl = weights.getSlice("onnx__MatMul_836");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("onnx__MatMul_836", builder.constant({ dataType: "float16", shape: [384, 384] }, buf));
  }
  {
    const sl = weights.getSlice("onnx__MatMul_837");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("onnx__MatMul_837", builder.constant({ dataType: "float16", shape: [384, 1536] }, buf));
  }
  {
    const sl = weights.getSlice("onnx__MatMul_838");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("onnx__MatMul_838", builder.constant({ dataType: "float16", shape: [1536, 384] }, buf));
  }
  {
    const sl = weights.getSlice("onnx__MatMul_839");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("onnx__MatMul_839", builder.constant({ dataType: "float16", shape: [384, 384] }, buf));
  }
  {
    const sl = weights.getSlice("onnx__MatMul_840");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("onnx__MatMul_840", builder.constant({ dataType: "float16", shape: [384, 384] }, buf));
  }
  {
    const sl = weights.getSlice("onnx__MatMul_843");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("onnx__MatMul_843", builder.constant({ dataType: "float16", shape: [384, 384] }, buf));
  }
  {
    const sl = weights.getSlice("onnx__MatMul_849");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("onnx__MatMul_849", builder.constant({ dataType: "float16", shape: [384, 384] }, buf));
  }
  {
    const sl = weights.getSlice("onnx__MatMul_850");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("onnx__MatMul_850", builder.constant({ dataType: "float16", shape: [384, 1536] }, buf));
  }
  {
    const sl = weights.getSlice("onnx__MatMul_851");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("onnx__MatMul_851", builder.constant({ dataType: "float16", shape: [1536, 384] }, buf));
  }
  {
    const sl = weights.getSlice("onnx__MatMul_852");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("onnx__MatMul_852", builder.constant({ dataType: "float16", shape: [384, 384] }, buf));
  }
  {
    const sl = weights.getSlice("onnx__MatMul_853");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("onnx__MatMul_853", builder.constant({ dataType: "float16", shape: [384, 384] }, buf));
  }
  {
    const sl = weights.getSlice("onnx__MatMul_856");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("onnx__MatMul_856", builder.constant({ dataType: "float16", shape: [384, 384] }, buf));
  }
  {
    const sl = weights.getSlice("onnx__MatMul_862");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("onnx__MatMul_862", builder.constant({ dataType: "float16", shape: [384, 384] }, buf));
  }
  {
    const sl = weights.getSlice("onnx__MatMul_863");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("onnx__MatMul_863", builder.constant({ dataType: "float16", shape: [384, 1536] }, buf));
  }
  {
    const sl = weights.getSlice("onnx__MatMul_864");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("onnx__MatMul_864", builder.constant({ dataType: "float16", shape: [1536, 384] }, buf));
  }

  env.set("_Unsqueeze_output_0", builder.reshape(env.get("attention_mask"), [1,1,128]));
  env.set("_Unsqueeze_1_output_0", builder.reshape(env.get("_Unsqueeze_output_0"), [1,1,1,128]));
  env.set("_Cast_output_0", builder.cast(env.get("_Unsqueeze_1_output_0"), "float16"));
  env.set("_Constant_output_0", builder.constant({ dataType: "float16", shape: [] }, Uint8Array.from(atob("ADw="), c => c.charCodeAt(0)).buffer));
  env.set("_Sub_output_0", builder["sub"](env.get("_Constant_output_0"), env.get("_Cast_output_0"), {}));
  env.set("_Constant_1_output_0", builder.constant({ dataType: "float16", shape: [] }, Uint8Array.from(atob("4vA="), c => c.charCodeAt(0)).buffer));
  env.set("_Mul_output_0", builder["mul"](env.get("_Sub_output_0"), env.get("_Constant_1_output_0"), {}));
  env.set("_embeddings_Slice_output_0", builder.slice(env.get("embeddings_position_ids"), [0,0], [1, 128], {"strides":[1,1]}));
  env.set("_embeddings_word_embeddings_Gather_output_0", builder["gather"](env.get("embeddings_word_embeddings_weight"), env.get("input_ids"), {"axis":0,"shape":[1,128,384]}));
  env.set("_embeddings_token_type_embeddings_Gather_output_0", builder["gather"](env.get("embeddings_token_type_embeddings_weight"), env.get("token_type_ids"), {"axis":0,"shape":[1,128,384]}));
  env.set("_embeddings_Add_output_0", builder["add"](env.get("_embeddings_word_embeddings_Gather_output_0"), env.get("_embeddings_token_type_embeddings_Gather_output_0"), {}));
  env.set("_embeddings_position_embeddings_Gather_output_0", builder["gather"](env.get("embeddings_position_embeddings_weight"), env.get("_embeddings_Slice_output_0"), {"axis":0,"shape":[1,128,384]}));
  env.set("_embeddings_Add_1_output_0", builder["add"](env.get("_embeddings_Add_output_0"), env.get("_embeddings_position_embeddings_Gather_output_0"), {}));
  env.set("_embeddings_LayerNorm_ReduceMean_output_0", builder["reduceMean"](env.get("_embeddings_Add_1_output_0"), {"axes":[2],"keepDimensions":true}));
  env.set("_embeddings_LayerNorm_Sub_output_0", builder["sub"](env.get("_embeddings_Add_1_output_0"), env.get("_embeddings_LayerNorm_ReduceMean_output_0"), {}));
  env.set("_embeddings_LayerNorm_Constant_output_0", builder.constant({ dataType: "float16", shape: [] }, Uint8Array.from(atob("AEA="), c => c.charCodeAt(0)).buffer));
  env.set("_embeddings_LayerNorm_Pow_output_0", builder["pow"](env.get("_embeddings_LayerNorm_Sub_output_0"), env.get("_embeddings_LayerNorm_Constant_output_0"), {}));
  env.set("_embeddings_LayerNorm_ReduceMean_1_output_0", builder["reduceMean"](env.get("_embeddings_LayerNorm_Pow_output_0"), {"axes":[2],"keepDimensions":true}));
  env.set("_embeddings_LayerNorm_Constant_1_output_0", builder.constant({ dataType: "float16", shape: [] }, Uint8Array.from(atob("AgA="), c => c.charCodeAt(0)).buffer));
  env.set("_embeddings_LayerNorm_Add_output_0", builder["add"](env.get("_embeddings_LayerNorm_ReduceMean_1_output_0"), env.get("_embeddings_LayerNorm_Constant_1_output_0"), {}));
  env.set("_embeddings_LayerNorm_Sqrt_output_0", builder["sqrt"](env.get("_embeddings_LayerNorm_Add_output_0"), {}));
  env.set("_embeddings_LayerNorm_Div_output_0", builder["div"](env.get("_embeddings_LayerNorm_Sub_output_0"), env.get("_embeddings_LayerNorm_Sqrt_output_0"), {}));
  env.set("_embeddings_LayerNorm_Mul_output_0", builder["mul"](env.get("_embeddings_LayerNorm_Div_output_0"), env.get("embeddings_LayerNorm_weight"), {}));
  env.set("_embeddings_LayerNorm_Add_1_output_0", builder["add"](env.get("_embeddings_LayerNorm_Mul_output_0"), env.get("embeddings_LayerNorm_bias"), {}));
  env.set("_encoder_layer_0_attention_self_query_MatMul_output_0", builder["matmul"](env.get("_embeddings_LayerNorm_Add_1_output_0"), env.get("onnx__MatMul_787"), {}));
  env.set("_encoder_layer_0_attention_self_query_Add_output_0", builder["add"](env.get("encoder_layer_0_attention_self_query_bias"), env.get("_encoder_layer_0_attention_self_query_MatMul_output_0"), {}));
  env.set("_encoder_layer_0_attention_self_key_MatMul_output_0", builder["matmul"](env.get("_embeddings_LayerNorm_Add_1_output_0"), env.get("onnx__MatMul_788"), {}));
  env.set("_encoder_layer_0_attention_self_key_Add_output_0", builder["add"](env.get("encoder_layer_0_attention_self_key_bias"), env.get("_encoder_layer_0_attention_self_key_MatMul_output_0"), {}));
  env.set("_encoder_layer_0_attention_self_Reshape_output_0", builder.reshape(env.get("_encoder_layer_0_attention_self_key_Add_output_0"), [1,128,12,32]));
  env.set("_encoder_layer_0_attention_self_value_MatMul_output_0", builder["matmul"](env.get("_embeddings_LayerNorm_Add_1_output_0"), env.get("onnx__MatMul_791"), {}));
  env.set("_encoder_layer_0_attention_self_value_Add_output_0", builder["add"](env.get("encoder_layer_0_attention_self_value_bias"), env.get("_encoder_layer_0_attention_self_value_MatMul_output_0"), {}));
  env.set("_encoder_layer_0_attention_self_Reshape_1_output_0", builder.reshape(env.get("_encoder_layer_0_attention_self_value_Add_output_0"), [1,128,12,32]));
  env.set("_encoder_layer_0_attention_self_Transpose_output_0", builder["transpose"](env.get("_encoder_layer_0_attention_self_Reshape_1_output_0"), {"permutation":[0,2,1,3]}));
  env.set("_encoder_layer_0_attention_self_Reshape_2_output_0", builder.reshape(env.get("_encoder_layer_0_attention_self_query_Add_output_0"), [1,128,12,32]));
  env.set("_encoder_layer_0_attention_self_Transpose_1_output_0", builder["transpose"](env.get("_encoder_layer_0_attention_self_Reshape_2_output_0"), {"permutation":[0,2,1,3]}));
  env.set("_encoder_layer_0_attention_self_Transpose_2_output_0", builder["transpose"](env.get("_encoder_layer_0_attention_self_Reshape_output_0"), {"permutation":[0,2,3,1]}));
  env.set("_encoder_layer_0_attention_self_MatMul_output_0", builder["matmul"](env.get("_encoder_layer_0_attention_self_Transpose_1_output_0"), env.get("_encoder_layer_0_attention_self_Transpose_2_output_0"), {}));
  env.set("_encoder_layer_0_attention_self_Constant_12_output_0", builder.constant({ dataType: "float16", shape: [] }, Uint8Array.from(atob("qEU="), c => c.charCodeAt(0)).buffer));
  env.set("_encoder_layer_0_attention_self_Div_output_0", builder["div"](env.get("_encoder_layer_0_attention_self_MatMul_output_0"), env.get("_encoder_layer_0_attention_self_Constant_12_output_0"), {}));
  env.set("_encoder_layer_0_attention_self_Add_output_0", builder["add"](env.get("_encoder_layer_0_attention_self_Div_output_0"), env.get("_Mul_output_0"), {}));
  env.set("_encoder_layer_0_attention_self_Softmax_output_0", builder["softmax"](env.get("_encoder_layer_0_attention_self_Add_output_0"), 3));
  env.set("_encoder_layer_0_attention_self_MatMul_1_output_0", builder["matmul"](env.get("_encoder_layer_0_attention_self_Softmax_output_0"), env.get("_encoder_layer_0_attention_self_Transpose_output_0"), {}));
  env.set("_encoder_layer_0_attention_self_Transpose_3_output_0", builder["transpose"](env.get("_encoder_layer_0_attention_self_MatMul_1_output_0"), {"permutation":[0,2,1,3]}));
  env.set("_encoder_layer_0_attention_self_Reshape_3_output_0", builder.reshape(env.get("_encoder_layer_0_attention_self_Transpose_3_output_0"), [1,128,384]));
  env.set("_encoder_layer_0_attention_output_dense_MatMul_output_0", builder["matmul"](env.get("_encoder_layer_0_attention_self_Reshape_3_output_0"), env.get("onnx__MatMul_797"), {}));
  env.set("_encoder_layer_0_attention_output_dense_Add_output_0", builder["add"](env.get("encoder_layer_0_attention_output_dense_bias"), env.get("_encoder_layer_0_attention_output_dense_MatMul_output_0"), {}));
  env.set("_encoder_layer_0_attention_output_Add_output_0", builder["add"](env.get("_encoder_layer_0_attention_output_dense_Add_output_0"), env.get("_embeddings_LayerNorm_Add_1_output_0"), {}));
  env.set("_encoder_layer_0_attention_output_LayerNorm_ReduceMean_output_0", builder["reduceMean"](env.get("_encoder_layer_0_attention_output_Add_output_0"), {"axes":[2],"keepDimensions":true}));
  env.set("_encoder_layer_0_attention_output_LayerNorm_Sub_output_0", builder["sub"](env.get("_encoder_layer_0_attention_output_Add_output_0"), env.get("_encoder_layer_0_attention_output_LayerNorm_ReduceMean_output_0"), {}));
  env.set("_encoder_layer_0_attention_output_LayerNorm_Constant_output_0", builder.constant({ dataType: "float16", shape: [] }, Uint8Array.from(atob("AEA="), c => c.charCodeAt(0)).buffer));
  env.set("_encoder_layer_0_attention_output_LayerNorm_Pow_output_0", builder["pow"](env.get("_encoder_layer_0_attention_output_LayerNorm_Sub_output_0"), env.get("_encoder_layer_0_attention_output_LayerNorm_Constant_output_0"), {}));
  env.set("_encoder_layer_0_attention_output_LayerNorm_ReduceMean_1_output_0", builder["reduceMean"](env.get("_encoder_layer_0_attention_output_LayerNorm_Pow_output_0"), {"axes":[2],"keepDimensions":true}));
  env.set("_encoder_layer_0_attention_output_LayerNorm_Constant_1_output_0", builder.constant({ dataType: "float16", shape: [] }, Uint8Array.from(atob("AgA="), c => c.charCodeAt(0)).buffer));
  env.set("_encoder_layer_0_attention_output_LayerNorm_Add_output_0", builder["add"](env.get("_encoder_layer_0_attention_output_LayerNorm_ReduceMean_1_output_0"), env.get("_encoder_layer_0_attention_output_LayerNorm_Constant_1_output_0"), {}));
  env.set("_encoder_layer_0_attention_output_LayerNorm_Sqrt_output_0", builder["sqrt"](env.get("_encoder_layer_0_attention_output_LayerNorm_Add_output_0"), {}));
  env.set("_encoder_layer_0_attention_output_LayerNorm_Div_output_0", builder["div"](env.get("_encoder_layer_0_attention_output_LayerNorm_Sub_output_0"), env.get("_encoder_layer_0_attention_output_LayerNorm_Sqrt_output_0"), {}));
  env.set("_encoder_layer_0_attention_output_LayerNorm_Mul_output_0", builder["mul"](env.get("_encoder_layer_0_attention_output_LayerNorm_Div_output_0"), env.get("encoder_layer_0_attention_output_LayerNorm_weight"), {}));
  env.set("_encoder_layer_0_attention_output_LayerNorm_Add_1_output_0", builder["add"](env.get("_encoder_layer_0_attention_output_LayerNorm_Mul_output_0"), env.get("encoder_layer_0_attention_output_LayerNorm_bias"), {}));
  env.set("_encoder_layer_0_intermediate_dense_MatMul_output_0", builder["matmul"](env.get("_encoder_layer_0_attention_output_LayerNorm_Add_1_output_0"), env.get("onnx__MatMul_798"), {}));
  env.set("_encoder_layer_0_intermediate_dense_Add_output_0", builder["add"](env.get("encoder_layer_0_intermediate_dense_bias"), env.get("_encoder_layer_0_intermediate_dense_MatMul_output_0"), {}));
  env.set("_encoder_layer_0_intermediate_intermediate_act_fn_Constant_output_0", builder.constant({ dataType: "float16", shape: [] }, Uint8Array.from(atob("qD0="), c => c.charCodeAt(0)).buffer));
  env.set("_encoder_layer_0_intermediate_intermediate_act_fn_Div_output_0", builder["div"](env.get("_encoder_layer_0_intermediate_dense_Add_output_0"), env.get("_encoder_layer_0_intermediate_intermediate_act_fn_Constant_output_0"), {}));
  env.set("_encoder_layer_0_intermediate_intermediate_act_fn_Erf_output_0", builder["erf"](env.get("_encoder_layer_0_intermediate_intermediate_act_fn_Div_output_0"), {}));
  env.set("_encoder_layer_0_intermediate_intermediate_act_fn_Constant_1_output_0", builder.constant({ dataType: "float16", shape: [] }, Uint8Array.from(atob("ADw="), c => c.charCodeAt(0)).buffer));
  env.set("_encoder_layer_0_intermediate_intermediate_act_fn_Add_output_0", builder["add"](env.get("_encoder_layer_0_intermediate_intermediate_act_fn_Erf_output_0"), env.get("_encoder_layer_0_intermediate_intermediate_act_fn_Constant_1_output_0"), {}));
  env.set("_encoder_layer_0_intermediate_intermediate_act_fn_Mul_output_0", builder["mul"](env.get("_encoder_layer_0_intermediate_dense_Add_output_0"), env.get("_encoder_layer_0_intermediate_intermediate_act_fn_Add_output_0"), {}));
  env.set("_encoder_layer_0_intermediate_intermediate_act_fn_Constant_2_output_0", builder.constant({ dataType: "float16", shape: [] }, Uint8Array.from(atob("ADg="), c => c.charCodeAt(0)).buffer));
  env.set("_encoder_layer_0_intermediate_intermediate_act_fn_Mul_1_output_0", builder["mul"](env.get("_encoder_layer_0_intermediate_intermediate_act_fn_Mul_output_0"), env.get("_encoder_layer_0_intermediate_intermediate_act_fn_Constant_2_output_0"), {}));
  env.set("_encoder_layer_0_output_dense_MatMul_output_0", builder["matmul"](env.get("_encoder_layer_0_intermediate_intermediate_act_fn_Mul_1_output_0"), env.get("onnx__MatMul_799"), {}));
  env.set("_encoder_layer_0_output_dense_Add_output_0", builder["add"](env.get("encoder_layer_0_output_dense_bias"), env.get("_encoder_layer_0_output_dense_MatMul_output_0"), {}));
  env.set("_encoder_layer_0_output_Add_output_0", builder["add"](env.get("_encoder_layer_0_output_dense_Add_output_0"), env.get("_encoder_layer_0_attention_output_LayerNorm_Add_1_output_0"), {}));
  env.set("_encoder_layer_0_output_LayerNorm_ReduceMean_output_0", builder["reduceMean"](env.get("_encoder_layer_0_output_Add_output_0"), {"axes":[2],"keepDimensions":true}));
  env.set("_encoder_layer_0_output_LayerNorm_Sub_output_0", builder["sub"](env.get("_encoder_layer_0_output_Add_output_0"), env.get("_encoder_layer_0_output_LayerNorm_ReduceMean_output_0"), {}));
  env.set("_encoder_layer_0_output_LayerNorm_Constant_output_0", builder.constant({ dataType: "float16", shape: [] }, Uint8Array.from(atob("AEA="), c => c.charCodeAt(0)).buffer));
  env.set("_encoder_layer_0_output_LayerNorm_Pow_output_0", builder["pow"](env.get("_encoder_layer_0_output_LayerNorm_Sub_output_0"), env.get("_encoder_layer_0_output_LayerNorm_Constant_output_0"), {}));
  env.set("_encoder_layer_0_output_LayerNorm_ReduceMean_1_output_0", builder["reduceMean"](env.get("_encoder_layer_0_output_LayerNorm_Pow_output_0"), {"axes":[2],"keepDimensions":true}));
  env.set("_encoder_layer_0_output_LayerNorm_Constant_1_output_0", builder.constant({ dataType: "float16", shape: [] }, Uint8Array.from(atob("AgA="), c => c.charCodeAt(0)).buffer));
  env.set("_encoder_layer_0_output_LayerNorm_Add_output_0", builder["add"](env.get("_encoder_layer_0_output_LayerNorm_ReduceMean_1_output_0"), env.get("_encoder_layer_0_output_LayerNorm_Constant_1_output_0"), {}));
  env.set("_encoder_layer_0_output_LayerNorm_Sqrt_output_0", builder["sqrt"](env.get("_encoder_layer_0_output_LayerNorm_Add_output_0"), {}));
  env.set("_encoder_layer_0_output_LayerNorm_Div_output_0", builder["div"](env.get("_encoder_layer_0_output_LayerNorm_Sub_output_0"), env.get("_encoder_layer_0_output_LayerNorm_Sqrt_output_0"), {}));
  env.set("_encoder_layer_0_output_LayerNorm_Mul_output_0", builder["mul"](env.get("_encoder_layer_0_output_LayerNorm_Div_output_0"), env.get("encoder_layer_0_output_LayerNorm_weight"), {}));
  env.set("_encoder_layer_0_output_LayerNorm_Add_1_output_0", builder["add"](env.get("_encoder_layer_0_output_LayerNorm_Mul_output_0"), env.get("encoder_layer_0_output_LayerNorm_bias"), {}));
  env.set("_encoder_layer_1_attention_self_query_MatMul_output_0", builder["matmul"](env.get("_encoder_layer_0_output_LayerNorm_Add_1_output_0"), env.get("onnx__MatMul_800"), {}));
  env.set("_encoder_layer_1_attention_self_query_Add_output_0", builder["add"](env.get("encoder_layer_1_attention_self_query_bias"), env.get("_encoder_layer_1_attention_self_query_MatMul_output_0"), {}));
  env.set("_encoder_layer_1_attention_self_key_MatMul_output_0", builder["matmul"](env.get("_encoder_layer_0_output_LayerNorm_Add_1_output_0"), env.get("onnx__MatMul_801"), {}));
  env.set("_encoder_layer_1_attention_self_key_Add_output_0", builder["add"](env.get("encoder_layer_1_attention_self_key_bias"), env.get("_encoder_layer_1_attention_self_key_MatMul_output_0"), {}));
  env.set("_encoder_layer_1_attention_self_Reshape_output_0", builder.reshape(env.get("_encoder_layer_1_attention_self_key_Add_output_0"), [1,128,12,32]));
  env.set("_encoder_layer_1_attention_self_value_MatMul_output_0", builder["matmul"](env.get("_encoder_layer_0_output_LayerNorm_Add_1_output_0"), env.get("onnx__MatMul_804"), {}));
  env.set("_encoder_layer_1_attention_self_value_Add_output_0", builder["add"](env.get("encoder_layer_1_attention_self_value_bias"), env.get("_encoder_layer_1_attention_self_value_MatMul_output_0"), {}));
  env.set("_encoder_layer_1_attention_self_Reshape_1_output_0", builder.reshape(env.get("_encoder_layer_1_attention_self_value_Add_output_0"), [1,128,12,32]));
  env.set("_encoder_layer_1_attention_self_Transpose_output_0", builder["transpose"](env.get("_encoder_layer_1_attention_self_Reshape_1_output_0"), {"permutation":[0,2,1,3]}));
  env.set("_encoder_layer_1_attention_self_Reshape_2_output_0", builder.reshape(env.get("_encoder_layer_1_attention_self_query_Add_output_0"), [1,128,12,32]));
  env.set("_encoder_layer_1_attention_self_Transpose_1_output_0", builder["transpose"](env.get("_encoder_layer_1_attention_self_Reshape_2_output_0"), {"permutation":[0,2,1,3]}));
  env.set("_encoder_layer_1_attention_self_Transpose_2_output_0", builder["transpose"](env.get("_encoder_layer_1_attention_self_Reshape_output_0"), {"permutation":[0,2,3,1]}));
  env.set("_encoder_layer_1_attention_self_MatMul_output_0", builder["matmul"](env.get("_encoder_layer_1_attention_self_Transpose_1_output_0"), env.get("_encoder_layer_1_attention_self_Transpose_2_output_0"), {}));
  env.set("_encoder_layer_1_attention_self_Constant_12_output_0", builder.constant({ dataType: "float16", shape: [] }, Uint8Array.from(atob("qEU="), c => c.charCodeAt(0)).buffer));
  env.set("_encoder_layer_1_attention_self_Div_output_0", builder["div"](env.get("_encoder_layer_1_attention_self_MatMul_output_0"), env.get("_encoder_layer_1_attention_self_Constant_12_output_0"), {}));
  env.set("_encoder_layer_1_attention_self_Add_output_0", builder["add"](env.get("_encoder_layer_1_attention_self_Div_output_0"), env.get("_Mul_output_0"), {}));
  env.set("_encoder_layer_1_attention_self_Softmax_output_0", builder["softmax"](env.get("_encoder_layer_1_attention_self_Add_output_0"), 3));
  env.set("_encoder_layer_1_attention_self_MatMul_1_output_0", builder["matmul"](env.get("_encoder_layer_1_attention_self_Softmax_output_0"), env.get("_encoder_layer_1_attention_self_Transpose_output_0"), {}));
  env.set("_encoder_layer_1_attention_self_Transpose_3_output_0", builder["transpose"](env.get("_encoder_layer_1_attention_self_MatMul_1_output_0"), {"permutation":[0,2,1,3]}));
  env.set("_encoder_layer_1_attention_self_Reshape_3_output_0", builder.reshape(env.get("_encoder_layer_1_attention_self_Transpose_3_output_0"), [1,128,384]));
  env.set("_encoder_layer_1_attention_output_dense_MatMul_output_0", builder["matmul"](env.get("_encoder_layer_1_attention_self_Reshape_3_output_0"), env.get("onnx__MatMul_810"), {}));
  env.set("_encoder_layer_1_attention_output_dense_Add_output_0", builder["add"](env.get("encoder_layer_1_attention_output_dense_bias"), env.get("_encoder_layer_1_attention_output_dense_MatMul_output_0"), {}));
  env.set("_encoder_layer_1_attention_output_Add_output_0", builder["add"](env.get("_encoder_layer_1_attention_output_dense_Add_output_0"), env.get("_encoder_layer_0_output_LayerNorm_Add_1_output_0"), {}));
  env.set("_encoder_layer_1_attention_output_LayerNorm_ReduceMean_output_0", builder["reduceMean"](env.get("_encoder_layer_1_attention_output_Add_output_0"), {"axes":[2],"keepDimensions":true}));
  env.set("_encoder_layer_1_attention_output_LayerNorm_Sub_output_0", builder["sub"](env.get("_encoder_layer_1_attention_output_Add_output_0"), env.get("_encoder_layer_1_attention_output_LayerNorm_ReduceMean_output_0"), {}));
  env.set("_encoder_layer_1_attention_output_LayerNorm_Constant_output_0", builder.constant({ dataType: "float16", shape: [] }, Uint8Array.from(atob("AEA="), c => c.charCodeAt(0)).buffer));
  env.set("_encoder_layer_1_attention_output_LayerNorm_Pow_output_0", builder["pow"](env.get("_encoder_layer_1_attention_output_LayerNorm_Sub_output_0"), env.get("_encoder_layer_1_attention_output_LayerNorm_Constant_output_0"), {}));
  env.set("_encoder_layer_1_attention_output_LayerNorm_ReduceMean_1_output_0", builder["reduceMean"](env.get("_encoder_layer_1_attention_output_LayerNorm_Pow_output_0"), {"axes":[2],"keepDimensions":true}));
  env.set("_encoder_layer_1_attention_output_LayerNorm_Constant_1_output_0", builder.constant({ dataType: "float16", shape: [] }, Uint8Array.from(atob("AgA="), c => c.charCodeAt(0)).buffer));
  env.set("_encoder_layer_1_attention_output_LayerNorm_Add_output_0", builder["add"](env.get("_encoder_layer_1_attention_output_LayerNorm_ReduceMean_1_output_0"), env.get("_encoder_layer_1_attention_output_LayerNorm_Constant_1_output_0"), {}));
  env.set("_encoder_layer_1_attention_output_LayerNorm_Sqrt_output_0", builder["sqrt"](env.get("_encoder_layer_1_attention_output_LayerNorm_Add_output_0"), {}));
  env.set("_encoder_layer_1_attention_output_LayerNorm_Div_output_0", builder["div"](env.get("_encoder_layer_1_attention_output_LayerNorm_Sub_output_0"), env.get("_encoder_layer_1_attention_output_LayerNorm_Sqrt_output_0"), {}));
  env.set("_encoder_layer_1_attention_output_LayerNorm_Mul_output_0", builder["mul"](env.get("_encoder_layer_1_attention_output_LayerNorm_Div_output_0"), env.get("encoder_layer_1_attention_output_LayerNorm_weight"), {}));
  env.set("_encoder_layer_1_attention_output_LayerNorm_Add_1_output_0", builder["add"](env.get("_encoder_layer_1_attention_output_LayerNorm_Mul_output_0"), env.get("encoder_layer_1_attention_output_LayerNorm_bias"), {}));
  env.set("_encoder_layer_1_intermediate_dense_MatMul_output_0", builder["matmul"](env.get("_encoder_layer_1_attention_output_LayerNorm_Add_1_output_0"), env.get("onnx__MatMul_811"), {}));
  env.set("_encoder_layer_1_intermediate_dense_Add_output_0", builder["add"](env.get("encoder_layer_1_intermediate_dense_bias"), env.get("_encoder_layer_1_intermediate_dense_MatMul_output_0"), {}));
  env.set("_encoder_layer_1_intermediate_intermediate_act_fn_Constant_output_0", builder.constant({ dataType: "float16", shape: [] }, Uint8Array.from(atob("qD0="), c => c.charCodeAt(0)).buffer));
  env.set("_encoder_layer_1_intermediate_intermediate_act_fn_Div_output_0", builder["div"](env.get("_encoder_layer_1_intermediate_dense_Add_output_0"), env.get("_encoder_layer_1_intermediate_intermediate_act_fn_Constant_output_0"), {}));
  env.set("_encoder_layer_1_intermediate_intermediate_act_fn_Erf_output_0", builder["erf"](env.get("_encoder_layer_1_intermediate_intermediate_act_fn_Div_output_0"), {}));
  env.set("_encoder_layer_1_intermediate_intermediate_act_fn_Constant_1_output_0", builder.constant({ dataType: "float16", shape: [] }, Uint8Array.from(atob("ADw="), c => c.charCodeAt(0)).buffer));
  env.set("_encoder_layer_1_intermediate_intermediate_act_fn_Add_output_0", builder["add"](env.get("_encoder_layer_1_intermediate_intermediate_act_fn_Erf_output_0"), env.get("_encoder_layer_1_intermediate_intermediate_act_fn_Constant_1_output_0"), {}));
  env.set("_encoder_layer_1_intermediate_intermediate_act_fn_Mul_output_0", builder["mul"](env.get("_encoder_layer_1_intermediate_dense_Add_output_0"), env.get("_encoder_layer_1_intermediate_intermediate_act_fn_Add_output_0"), {}));
  env.set("_encoder_layer_1_intermediate_intermediate_act_fn_Constant_2_output_0", builder.constant({ dataType: "float16", shape: [] }, Uint8Array.from(atob("ADg="), c => c.charCodeAt(0)).buffer));
  env.set("_encoder_layer_1_intermediate_intermediate_act_fn_Mul_1_output_0", builder["mul"](env.get("_encoder_layer_1_intermediate_intermediate_act_fn_Mul_output_0"), env.get("_encoder_layer_1_intermediate_intermediate_act_fn_Constant_2_output_0"), {}));
  env.set("_encoder_layer_1_output_dense_MatMul_output_0", builder["matmul"](env.get("_encoder_layer_1_intermediate_intermediate_act_fn_Mul_1_output_0"), env.get("onnx__MatMul_812"), {}));
  env.set("_encoder_layer_1_output_dense_Add_output_0", builder["add"](env.get("encoder_layer_1_output_dense_bias"), env.get("_encoder_layer_1_output_dense_MatMul_output_0"), {}));
  env.set("_encoder_layer_1_output_Add_output_0", builder["add"](env.get("_encoder_layer_1_output_dense_Add_output_0"), env.get("_encoder_layer_1_attention_output_LayerNorm_Add_1_output_0"), {}));
  env.set("_encoder_layer_1_output_LayerNorm_ReduceMean_output_0", builder["reduceMean"](env.get("_encoder_layer_1_output_Add_output_0"), {"axes":[2],"keepDimensions":true}));
  env.set("_encoder_layer_1_output_LayerNorm_Sub_output_0", builder["sub"](env.get("_encoder_layer_1_output_Add_output_0"), env.get("_encoder_layer_1_output_LayerNorm_ReduceMean_output_0"), {}));
  env.set("_encoder_layer_1_output_LayerNorm_Constant_output_0", builder.constant({ dataType: "float16", shape: [] }, Uint8Array.from(atob("AEA="), c => c.charCodeAt(0)).buffer));
  env.set("_encoder_layer_1_output_LayerNorm_Pow_output_0", builder["pow"](env.get("_encoder_layer_1_output_LayerNorm_Sub_output_0"), env.get("_encoder_layer_1_output_LayerNorm_Constant_output_0"), {}));
  env.set("_encoder_layer_1_output_LayerNorm_ReduceMean_1_output_0", builder["reduceMean"](env.get("_encoder_layer_1_output_LayerNorm_Pow_output_0"), {"axes":[2],"keepDimensions":true}));
  env.set("_encoder_layer_1_output_LayerNorm_Constant_1_output_0", builder.constant({ dataType: "float16", shape: [] }, Uint8Array.from(atob("AgA="), c => c.charCodeAt(0)).buffer));
  env.set("_encoder_layer_1_output_LayerNorm_Add_output_0", builder["add"](env.get("_encoder_layer_1_output_LayerNorm_ReduceMean_1_output_0"), env.get("_encoder_layer_1_output_LayerNorm_Constant_1_output_0"), {}));
  env.set("_encoder_layer_1_output_LayerNorm_Sqrt_output_0", builder["sqrt"](env.get("_encoder_layer_1_output_LayerNorm_Add_output_0"), {}));
  env.set("_encoder_layer_1_output_LayerNorm_Div_output_0", builder["div"](env.get("_encoder_layer_1_output_LayerNorm_Sub_output_0"), env.get("_encoder_layer_1_output_LayerNorm_Sqrt_output_0"), {}));
  env.set("_encoder_layer_1_output_LayerNorm_Mul_output_0", builder["mul"](env.get("_encoder_layer_1_output_LayerNorm_Div_output_0"), env.get("encoder_layer_1_output_LayerNorm_weight"), {}));
  env.set("_encoder_layer_1_output_LayerNorm_Add_1_output_0", builder["add"](env.get("_encoder_layer_1_output_LayerNorm_Mul_output_0"), env.get("encoder_layer_1_output_LayerNorm_bias"), {}));
  env.set("_encoder_layer_2_attention_self_query_MatMul_output_0", builder["matmul"](env.get("_encoder_layer_1_output_LayerNorm_Add_1_output_0"), env.get("onnx__MatMul_813"), {}));
  env.set("_encoder_layer_2_attention_self_query_Add_output_0", builder["add"](env.get("encoder_layer_2_attention_self_query_bias"), env.get("_encoder_layer_2_attention_self_query_MatMul_output_0"), {}));
  env.set("_encoder_layer_2_attention_self_key_MatMul_output_0", builder["matmul"](env.get("_encoder_layer_1_output_LayerNorm_Add_1_output_0"), env.get("onnx__MatMul_814"), {}));
  env.set("_encoder_layer_2_attention_self_key_Add_output_0", builder["add"](env.get("encoder_layer_2_attention_self_key_bias"), env.get("_encoder_layer_2_attention_self_key_MatMul_output_0"), {}));
  env.set("_encoder_layer_2_attention_self_Reshape_output_0", builder.reshape(env.get("_encoder_layer_2_attention_self_key_Add_output_0"), [1,128,12,32]));
  env.set("_encoder_layer_2_attention_self_value_MatMul_output_0", builder["matmul"](env.get("_encoder_layer_1_output_LayerNorm_Add_1_output_0"), env.get("onnx__MatMul_817"), {}));
  env.set("_encoder_layer_2_attention_self_value_Add_output_0", builder["add"](env.get("encoder_layer_2_attention_self_value_bias"), env.get("_encoder_layer_2_attention_self_value_MatMul_output_0"), {}));
  env.set("_encoder_layer_2_attention_self_Reshape_1_output_0", builder.reshape(env.get("_encoder_layer_2_attention_self_value_Add_output_0"), [1,128,12,32]));
  env.set("_encoder_layer_2_attention_self_Transpose_output_0", builder["transpose"](env.get("_encoder_layer_2_attention_self_Reshape_1_output_0"), {"permutation":[0,2,1,3]}));
  env.set("_encoder_layer_2_attention_self_Reshape_2_output_0", builder.reshape(env.get("_encoder_layer_2_attention_self_query_Add_output_0"), [1,128,12,32]));
  env.set("_encoder_layer_2_attention_self_Transpose_1_output_0", builder["transpose"](env.get("_encoder_layer_2_attention_self_Reshape_2_output_0"), {"permutation":[0,2,1,3]}));
  env.set("_encoder_layer_2_attention_self_Transpose_2_output_0", builder["transpose"](env.get("_encoder_layer_2_attention_self_Reshape_output_0"), {"permutation":[0,2,3,1]}));
  env.set("_encoder_layer_2_attention_self_MatMul_output_0", builder["matmul"](env.get("_encoder_layer_2_attention_self_Transpose_1_output_0"), env.get("_encoder_layer_2_attention_self_Transpose_2_output_0"), {}));
  env.set("_encoder_layer_2_attention_self_Constant_12_output_0", builder.constant({ dataType: "float16", shape: [] }, Uint8Array.from(atob("qEU="), c => c.charCodeAt(0)).buffer));
  env.set("_encoder_layer_2_attention_self_Div_output_0", builder["div"](env.get("_encoder_layer_2_attention_self_MatMul_output_0"), env.get("_encoder_layer_2_attention_self_Constant_12_output_0"), {}));
  env.set("_encoder_layer_2_attention_self_Add_output_0", builder["add"](env.get("_encoder_layer_2_attention_self_Div_output_0"), env.get("_Mul_output_0"), {}));
  env.set("_encoder_layer_2_attention_self_Softmax_output_0", builder["softmax"](env.get("_encoder_layer_2_attention_self_Add_output_0"), 3));
  env.set("_encoder_layer_2_attention_self_MatMul_1_output_0", builder["matmul"](env.get("_encoder_layer_2_attention_self_Softmax_output_0"), env.get("_encoder_layer_2_attention_self_Transpose_output_0"), {}));
  env.set("_encoder_layer_2_attention_self_Transpose_3_output_0", builder["transpose"](env.get("_encoder_layer_2_attention_self_MatMul_1_output_0"), {"permutation":[0,2,1,3]}));
  env.set("_encoder_layer_2_attention_self_Reshape_3_output_0", builder.reshape(env.get("_encoder_layer_2_attention_self_Transpose_3_output_0"), [1,128,384]));
  env.set("_encoder_layer_2_attention_output_dense_MatMul_output_0", builder["matmul"](env.get("_encoder_layer_2_attention_self_Reshape_3_output_0"), env.get("onnx__MatMul_823"), {}));
  env.set("_encoder_layer_2_attention_output_dense_Add_output_0", builder["add"](env.get("encoder_layer_2_attention_output_dense_bias"), env.get("_encoder_layer_2_attention_output_dense_MatMul_output_0"), {}));
  env.set("_encoder_layer_2_attention_output_Add_output_0", builder["add"](env.get("_encoder_layer_2_attention_output_dense_Add_output_0"), env.get("_encoder_layer_1_output_LayerNorm_Add_1_output_0"), {}));
  env.set("_encoder_layer_2_attention_output_LayerNorm_ReduceMean_output_0", builder["reduceMean"](env.get("_encoder_layer_2_attention_output_Add_output_0"), {"axes":[2],"keepDimensions":true}));
  env.set("_encoder_layer_2_attention_output_LayerNorm_Sub_output_0", builder["sub"](env.get("_encoder_layer_2_attention_output_Add_output_0"), env.get("_encoder_layer_2_attention_output_LayerNorm_ReduceMean_output_0"), {}));
  env.set("_encoder_layer_2_attention_output_LayerNorm_Constant_output_0", builder.constant({ dataType: "float16", shape: [] }, Uint8Array.from(atob("AEA="), c => c.charCodeAt(0)).buffer));
  env.set("_encoder_layer_2_attention_output_LayerNorm_Pow_output_0", builder["pow"](env.get("_encoder_layer_2_attention_output_LayerNorm_Sub_output_0"), env.get("_encoder_layer_2_attention_output_LayerNorm_Constant_output_0"), {}));
  env.set("_encoder_layer_2_attention_output_LayerNorm_ReduceMean_1_output_0", builder["reduceMean"](env.get("_encoder_layer_2_attention_output_LayerNorm_Pow_output_0"), {"axes":[2],"keepDimensions":true}));
  env.set("_encoder_layer_2_attention_output_LayerNorm_Constant_1_output_0", builder.constant({ dataType: "float16", shape: [] }, Uint8Array.from(atob("AgA="), c => c.charCodeAt(0)).buffer));
  env.set("_encoder_layer_2_attention_output_LayerNorm_Add_output_0", builder["add"](env.get("_encoder_layer_2_attention_output_LayerNorm_ReduceMean_1_output_0"), env.get("_encoder_layer_2_attention_output_LayerNorm_Constant_1_output_0"), {}));
  env.set("_encoder_layer_2_attention_output_LayerNorm_Sqrt_output_0", builder["sqrt"](env.get("_encoder_layer_2_attention_output_LayerNorm_Add_output_0"), {}));
  env.set("_encoder_layer_2_attention_output_LayerNorm_Div_output_0", builder["div"](env.get("_encoder_layer_2_attention_output_LayerNorm_Sub_output_0"), env.get("_encoder_layer_2_attention_output_LayerNorm_Sqrt_output_0"), {}));
  env.set("_encoder_layer_2_attention_output_LayerNorm_Mul_output_0", builder["mul"](env.get("_encoder_layer_2_attention_output_LayerNorm_Div_output_0"), env.get("encoder_layer_2_attention_output_LayerNorm_weight"), {}));
  env.set("_encoder_layer_2_attention_output_LayerNorm_Add_1_output_0", builder["add"](env.get("_encoder_layer_2_attention_output_LayerNorm_Mul_output_0"), env.get("encoder_layer_2_attention_output_LayerNorm_bias"), {}));
  env.set("_encoder_layer_2_intermediate_dense_MatMul_output_0", builder["matmul"](env.get("_encoder_layer_2_attention_output_LayerNorm_Add_1_output_0"), env.get("onnx__MatMul_824"), {}));
  env.set("_encoder_layer_2_intermediate_dense_Add_output_0", builder["add"](env.get("encoder_layer_2_intermediate_dense_bias"), env.get("_encoder_layer_2_intermediate_dense_MatMul_output_0"), {}));
  env.set("_encoder_layer_2_intermediate_intermediate_act_fn_Constant_output_0", builder.constant({ dataType: "float16", shape: [] }, Uint8Array.from(atob("qD0="), c => c.charCodeAt(0)).buffer));
  env.set("_encoder_layer_2_intermediate_intermediate_act_fn_Div_output_0", builder["div"](env.get("_encoder_layer_2_intermediate_dense_Add_output_0"), env.get("_encoder_layer_2_intermediate_intermediate_act_fn_Constant_output_0"), {}));
  env.set("_encoder_layer_2_intermediate_intermediate_act_fn_Erf_output_0", builder["erf"](env.get("_encoder_layer_2_intermediate_intermediate_act_fn_Div_output_0"), {}));
  env.set("_encoder_layer_2_intermediate_intermediate_act_fn_Constant_1_output_0", builder.constant({ dataType: "float16", shape: [] }, Uint8Array.from(atob("ADw="), c => c.charCodeAt(0)).buffer));
  env.set("_encoder_layer_2_intermediate_intermediate_act_fn_Add_output_0", builder["add"](env.get("_encoder_layer_2_intermediate_intermediate_act_fn_Erf_output_0"), env.get("_encoder_layer_2_intermediate_intermediate_act_fn_Constant_1_output_0"), {}));
  env.set("_encoder_layer_2_intermediate_intermediate_act_fn_Mul_output_0", builder["mul"](env.get("_encoder_layer_2_intermediate_dense_Add_output_0"), env.get("_encoder_layer_2_intermediate_intermediate_act_fn_Add_output_0"), {}));
  env.set("_encoder_layer_2_intermediate_intermediate_act_fn_Constant_2_output_0", builder.constant({ dataType: "float16", shape: [] }, Uint8Array.from(atob("ADg="), c => c.charCodeAt(0)).buffer));
  env.set("_encoder_layer_2_intermediate_intermediate_act_fn_Mul_1_output_0", builder["mul"](env.get("_encoder_layer_2_intermediate_intermediate_act_fn_Mul_output_0"), env.get("_encoder_layer_2_intermediate_intermediate_act_fn_Constant_2_output_0"), {}));
  env.set("_encoder_layer_2_output_dense_MatMul_output_0", builder["matmul"](env.get("_encoder_layer_2_intermediate_intermediate_act_fn_Mul_1_output_0"), env.get("onnx__MatMul_825"), {}));
  env.set("_encoder_layer_2_output_dense_Add_output_0", builder["add"](env.get("encoder_layer_2_output_dense_bias"), env.get("_encoder_layer_2_output_dense_MatMul_output_0"), {}));
  env.set("_encoder_layer_2_output_Add_output_0", builder["add"](env.get("_encoder_layer_2_output_dense_Add_output_0"), env.get("_encoder_layer_2_attention_output_LayerNorm_Add_1_output_0"), {}));
  env.set("_encoder_layer_2_output_LayerNorm_ReduceMean_output_0", builder["reduceMean"](env.get("_encoder_layer_2_output_Add_output_0"), {"axes":[2],"keepDimensions":true}));
  env.set("_encoder_layer_2_output_LayerNorm_Sub_output_0", builder["sub"](env.get("_encoder_layer_2_output_Add_output_0"), env.get("_encoder_layer_2_output_LayerNorm_ReduceMean_output_0"), {}));
  env.set("_encoder_layer_2_output_LayerNorm_Constant_output_0", builder.constant({ dataType: "float16", shape: [] }, Uint8Array.from(atob("AEA="), c => c.charCodeAt(0)).buffer));
  env.set("_encoder_layer_2_output_LayerNorm_Pow_output_0", builder["pow"](env.get("_encoder_layer_2_output_LayerNorm_Sub_output_0"), env.get("_encoder_layer_2_output_LayerNorm_Constant_output_0"), {}));
  env.set("_encoder_layer_2_output_LayerNorm_ReduceMean_1_output_0", builder["reduceMean"](env.get("_encoder_layer_2_output_LayerNorm_Pow_output_0"), {"axes":[2],"keepDimensions":true}));
  env.set("_encoder_layer_2_output_LayerNorm_Constant_1_output_0", builder.constant({ dataType: "float16", shape: [] }, Uint8Array.from(atob("AgA="), c => c.charCodeAt(0)).buffer));
  env.set("_encoder_layer_2_output_LayerNorm_Add_output_0", builder["add"](env.get("_encoder_layer_2_output_LayerNorm_ReduceMean_1_output_0"), env.get("_encoder_layer_2_output_LayerNorm_Constant_1_output_0"), {}));
  env.set("_encoder_layer_2_output_LayerNorm_Sqrt_output_0", builder["sqrt"](env.get("_encoder_layer_2_output_LayerNorm_Add_output_0"), {}));
  env.set("_encoder_layer_2_output_LayerNorm_Div_output_0", builder["div"](env.get("_encoder_layer_2_output_LayerNorm_Sub_output_0"), env.get("_encoder_layer_2_output_LayerNorm_Sqrt_output_0"), {}));
  env.set("_encoder_layer_2_output_LayerNorm_Mul_output_0", builder["mul"](env.get("_encoder_layer_2_output_LayerNorm_Div_output_0"), env.get("encoder_layer_2_output_LayerNorm_weight"), {}));
  env.set("_encoder_layer_2_output_LayerNorm_Add_1_output_0", builder["add"](env.get("_encoder_layer_2_output_LayerNorm_Mul_output_0"), env.get("encoder_layer_2_output_LayerNorm_bias"), {}));
  env.set("_encoder_layer_3_attention_self_query_MatMul_output_0", builder["matmul"](env.get("_encoder_layer_2_output_LayerNorm_Add_1_output_0"), env.get("onnx__MatMul_826"), {}));
  env.set("_encoder_layer_3_attention_self_query_Add_output_0", builder["add"](env.get("encoder_layer_3_attention_self_query_bias"), env.get("_encoder_layer_3_attention_self_query_MatMul_output_0"), {}));
  env.set("_encoder_layer_3_attention_self_key_MatMul_output_0", builder["matmul"](env.get("_encoder_layer_2_output_LayerNorm_Add_1_output_0"), env.get("onnx__MatMul_827"), {}));
  env.set("_encoder_layer_3_attention_self_key_Add_output_0", builder["add"](env.get("encoder_layer_3_attention_self_key_bias"), env.get("_encoder_layer_3_attention_self_key_MatMul_output_0"), {}));
  env.set("_encoder_layer_3_attention_self_Reshape_output_0", builder.reshape(env.get("_encoder_layer_3_attention_self_key_Add_output_0"), [1,128,12,32]));
  env.set("_encoder_layer_3_attention_self_value_MatMul_output_0", builder["matmul"](env.get("_encoder_layer_2_output_LayerNorm_Add_1_output_0"), env.get("onnx__MatMul_830"), {}));
  env.set("_encoder_layer_3_attention_self_value_Add_output_0", builder["add"](env.get("encoder_layer_3_attention_self_value_bias"), env.get("_encoder_layer_3_attention_self_value_MatMul_output_0"), {}));
  env.set("_encoder_layer_3_attention_self_Reshape_1_output_0", builder.reshape(env.get("_encoder_layer_3_attention_self_value_Add_output_0"), [1,128,12,32]));
  env.set("_encoder_layer_3_attention_self_Transpose_output_0", builder["transpose"](env.get("_encoder_layer_3_attention_self_Reshape_1_output_0"), {"permutation":[0,2,1,3]}));
  env.set("_encoder_layer_3_attention_self_Reshape_2_output_0", builder.reshape(env.get("_encoder_layer_3_attention_self_query_Add_output_0"), [1,128,12,32]));
  env.set("_encoder_layer_3_attention_self_Transpose_1_output_0", builder["transpose"](env.get("_encoder_layer_3_attention_self_Reshape_2_output_0"), {"permutation":[0,2,1,3]}));
  env.set("_encoder_layer_3_attention_self_Transpose_2_output_0", builder["transpose"](env.get("_encoder_layer_3_attention_self_Reshape_output_0"), {"permutation":[0,2,3,1]}));
  env.set("_encoder_layer_3_attention_self_MatMul_output_0", builder["matmul"](env.get("_encoder_layer_3_attention_self_Transpose_1_output_0"), env.get("_encoder_layer_3_attention_self_Transpose_2_output_0"), {}));
  env.set("_encoder_layer_3_attention_self_Constant_12_output_0", builder.constant({ dataType: "float16", shape: [] }, Uint8Array.from(atob("qEU="), c => c.charCodeAt(0)).buffer));
  env.set("_encoder_layer_3_attention_self_Div_output_0", builder["div"](env.get("_encoder_layer_3_attention_self_MatMul_output_0"), env.get("_encoder_layer_3_attention_self_Constant_12_output_0"), {}));
  env.set("_encoder_layer_3_attention_self_Add_output_0", builder["add"](env.get("_encoder_layer_3_attention_self_Div_output_0"), env.get("_Mul_output_0"), {}));
  env.set("_encoder_layer_3_attention_self_Softmax_output_0", builder["softmax"](env.get("_encoder_layer_3_attention_self_Add_output_0"), 3));
  env.set("_encoder_layer_3_attention_self_MatMul_1_output_0", builder["matmul"](env.get("_encoder_layer_3_attention_self_Softmax_output_0"), env.get("_encoder_layer_3_attention_self_Transpose_output_0"), {}));
  env.set("_encoder_layer_3_attention_self_Transpose_3_output_0", builder["transpose"](env.get("_encoder_layer_3_attention_self_MatMul_1_output_0"), {"permutation":[0,2,1,3]}));
  env.set("_encoder_layer_3_attention_self_Reshape_3_output_0", builder.reshape(env.get("_encoder_layer_3_attention_self_Transpose_3_output_0"), [1,128,384]));
  env.set("_encoder_layer_3_attention_output_dense_MatMul_output_0", builder["matmul"](env.get("_encoder_layer_3_attention_self_Reshape_3_output_0"), env.get("onnx__MatMul_836"), {}));
  env.set("_encoder_layer_3_attention_output_dense_Add_output_0", builder["add"](env.get("encoder_layer_3_attention_output_dense_bias"), env.get("_encoder_layer_3_attention_output_dense_MatMul_output_0"), {}));
  env.set("_encoder_layer_3_attention_output_Add_output_0", builder["add"](env.get("_encoder_layer_3_attention_output_dense_Add_output_0"), env.get("_encoder_layer_2_output_LayerNorm_Add_1_output_0"), {}));
  env.set("_encoder_layer_3_attention_output_LayerNorm_ReduceMean_output_0", builder["reduceMean"](env.get("_encoder_layer_3_attention_output_Add_output_0"), {"axes":[2],"keepDimensions":true}));
  env.set("_encoder_layer_3_attention_output_LayerNorm_Sub_output_0", builder["sub"](env.get("_encoder_layer_3_attention_output_Add_output_0"), env.get("_encoder_layer_3_attention_output_LayerNorm_ReduceMean_output_0"), {}));
  env.set("_encoder_layer_3_attention_output_LayerNorm_Constant_output_0", builder.constant({ dataType: "float16", shape: [] }, Uint8Array.from(atob("AEA="), c => c.charCodeAt(0)).buffer));
  env.set("_encoder_layer_3_attention_output_LayerNorm_Pow_output_0", builder["pow"](env.get("_encoder_layer_3_attention_output_LayerNorm_Sub_output_0"), env.get("_encoder_layer_3_attention_output_LayerNorm_Constant_output_0"), {}));
  env.set("_encoder_layer_3_attention_output_LayerNorm_ReduceMean_1_output_0", builder["reduceMean"](env.get("_encoder_layer_3_attention_output_LayerNorm_Pow_output_0"), {"axes":[2],"keepDimensions":true}));
  env.set("_encoder_layer_3_attention_output_LayerNorm_Constant_1_output_0", builder.constant({ dataType: "float16", shape: [] }, Uint8Array.from(atob("AgA="), c => c.charCodeAt(0)).buffer));
  env.set("_encoder_layer_3_attention_output_LayerNorm_Add_output_0", builder["add"](env.get("_encoder_layer_3_attention_output_LayerNorm_ReduceMean_1_output_0"), env.get("_encoder_layer_3_attention_output_LayerNorm_Constant_1_output_0"), {}));
  env.set("_encoder_layer_3_attention_output_LayerNorm_Sqrt_output_0", builder["sqrt"](env.get("_encoder_layer_3_attention_output_LayerNorm_Add_output_0"), {}));
  env.set("_encoder_layer_3_attention_output_LayerNorm_Div_output_0", builder["div"](env.get("_encoder_layer_3_attention_output_LayerNorm_Sub_output_0"), env.get("_encoder_layer_3_attention_output_LayerNorm_Sqrt_output_0"), {}));
  env.set("_encoder_layer_3_attention_output_LayerNorm_Mul_output_0", builder["mul"](env.get("_encoder_layer_3_attention_output_LayerNorm_Div_output_0"), env.get("encoder_layer_3_attention_output_LayerNorm_weight"), {}));
  env.set("_encoder_layer_3_attention_output_LayerNorm_Add_1_output_0", builder["add"](env.get("_encoder_layer_3_attention_output_LayerNorm_Mul_output_0"), env.get("encoder_layer_3_attention_output_LayerNorm_bias"), {}));
  env.set("_encoder_layer_3_intermediate_dense_MatMul_output_0", builder["matmul"](env.get("_encoder_layer_3_attention_output_LayerNorm_Add_1_output_0"), env.get("onnx__MatMul_837"), {}));
  env.set("_encoder_layer_3_intermediate_dense_Add_output_0", builder["add"](env.get("encoder_layer_3_intermediate_dense_bias"), env.get("_encoder_layer_3_intermediate_dense_MatMul_output_0"), {}));
  env.set("_encoder_layer_3_intermediate_intermediate_act_fn_Constant_output_0", builder.constant({ dataType: "float16", shape: [] }, Uint8Array.from(atob("qD0="), c => c.charCodeAt(0)).buffer));
  env.set("_encoder_layer_3_intermediate_intermediate_act_fn_Div_output_0", builder["div"](env.get("_encoder_layer_3_intermediate_dense_Add_output_0"), env.get("_encoder_layer_3_intermediate_intermediate_act_fn_Constant_output_0"), {}));
  env.set("_encoder_layer_3_intermediate_intermediate_act_fn_Erf_output_0", builder["erf"](env.get("_encoder_layer_3_intermediate_intermediate_act_fn_Div_output_0"), {}));
  env.set("_encoder_layer_3_intermediate_intermediate_act_fn_Constant_1_output_0", builder.constant({ dataType: "float16", shape: [] }, Uint8Array.from(atob("ADw="), c => c.charCodeAt(0)).buffer));
  env.set("_encoder_layer_3_intermediate_intermediate_act_fn_Add_output_0", builder["add"](env.get("_encoder_layer_3_intermediate_intermediate_act_fn_Erf_output_0"), env.get("_encoder_layer_3_intermediate_intermediate_act_fn_Constant_1_output_0"), {}));
  env.set("_encoder_layer_3_intermediate_intermediate_act_fn_Mul_output_0", builder["mul"](env.get("_encoder_layer_3_intermediate_dense_Add_output_0"), env.get("_encoder_layer_3_intermediate_intermediate_act_fn_Add_output_0"), {}));
  env.set("_encoder_layer_3_intermediate_intermediate_act_fn_Constant_2_output_0", builder.constant({ dataType: "float16", shape: [] }, Uint8Array.from(atob("ADg="), c => c.charCodeAt(0)).buffer));
  env.set("_encoder_layer_3_intermediate_intermediate_act_fn_Mul_1_output_0", builder["mul"](env.get("_encoder_layer_3_intermediate_intermediate_act_fn_Mul_output_0"), env.get("_encoder_layer_3_intermediate_intermediate_act_fn_Constant_2_output_0"), {}));
  env.set("_encoder_layer_3_output_dense_MatMul_output_0", builder["matmul"](env.get("_encoder_layer_3_intermediate_intermediate_act_fn_Mul_1_output_0"), env.get("onnx__MatMul_838"), {}));
  env.set("_encoder_layer_3_output_dense_Add_output_0", builder["add"](env.get("encoder_layer_3_output_dense_bias"), env.get("_encoder_layer_3_output_dense_MatMul_output_0"), {}));
  env.set("_encoder_layer_3_output_Add_output_0", builder["add"](env.get("_encoder_layer_3_output_dense_Add_output_0"), env.get("_encoder_layer_3_attention_output_LayerNorm_Add_1_output_0"), {}));
  env.set("_encoder_layer_3_output_LayerNorm_ReduceMean_output_0", builder["reduceMean"](env.get("_encoder_layer_3_output_Add_output_0"), {"axes":[2],"keepDimensions":true}));
  env.set("_encoder_layer_3_output_LayerNorm_Sub_output_0", builder["sub"](env.get("_encoder_layer_3_output_Add_output_0"), env.get("_encoder_layer_3_output_LayerNorm_ReduceMean_output_0"), {}));
  env.set("_encoder_layer_3_output_LayerNorm_Constant_output_0", builder.constant({ dataType: "float16", shape: [] }, Uint8Array.from(atob("AEA="), c => c.charCodeAt(0)).buffer));
  env.set("_encoder_layer_3_output_LayerNorm_Pow_output_0", builder["pow"](env.get("_encoder_layer_3_output_LayerNorm_Sub_output_0"), env.get("_encoder_layer_3_output_LayerNorm_Constant_output_0"), {}));
  env.set("_encoder_layer_3_output_LayerNorm_ReduceMean_1_output_0", builder["reduceMean"](env.get("_encoder_layer_3_output_LayerNorm_Pow_output_0"), {"axes":[2],"keepDimensions":true}));
  env.set("_encoder_layer_3_output_LayerNorm_Constant_1_output_0", builder.constant({ dataType: "float16", shape: [] }, Uint8Array.from(atob("AgA="), c => c.charCodeAt(0)).buffer));
  env.set("_encoder_layer_3_output_LayerNorm_Add_output_0", builder["add"](env.get("_encoder_layer_3_output_LayerNorm_ReduceMean_1_output_0"), env.get("_encoder_layer_3_output_LayerNorm_Constant_1_output_0"), {}));
  env.set("_encoder_layer_3_output_LayerNorm_Sqrt_output_0", builder["sqrt"](env.get("_encoder_layer_3_output_LayerNorm_Add_output_0"), {}));
  env.set("_encoder_layer_3_output_LayerNorm_Div_output_0", builder["div"](env.get("_encoder_layer_3_output_LayerNorm_Sub_output_0"), env.get("_encoder_layer_3_output_LayerNorm_Sqrt_output_0"), {}));
  env.set("_encoder_layer_3_output_LayerNorm_Mul_output_0", builder["mul"](env.get("_encoder_layer_3_output_LayerNorm_Div_output_0"), env.get("encoder_layer_3_output_LayerNorm_weight"), {}));
  env.set("_encoder_layer_3_output_LayerNorm_Add_1_output_0", builder["add"](env.get("_encoder_layer_3_output_LayerNorm_Mul_output_0"), env.get("encoder_layer_3_output_LayerNorm_bias"), {}));
  env.set("_encoder_layer_4_attention_self_query_MatMul_output_0", builder["matmul"](env.get("_encoder_layer_3_output_LayerNorm_Add_1_output_0"), env.get("onnx__MatMul_839"), {}));
  env.set("_encoder_layer_4_attention_self_query_Add_output_0", builder["add"](env.get("encoder_layer_4_attention_self_query_bias"), env.get("_encoder_layer_4_attention_self_query_MatMul_output_0"), {}));
  env.set("_encoder_layer_4_attention_self_key_MatMul_output_0", builder["matmul"](env.get("_encoder_layer_3_output_LayerNorm_Add_1_output_0"), env.get("onnx__MatMul_840"), {}));
  env.set("_encoder_layer_4_attention_self_key_Add_output_0", builder["add"](env.get("encoder_layer_4_attention_self_key_bias"), env.get("_encoder_layer_4_attention_self_key_MatMul_output_0"), {}));
  env.set("_encoder_layer_4_attention_self_Reshape_output_0", builder.reshape(env.get("_encoder_layer_4_attention_self_key_Add_output_0"), [1,128,12,32]));
  env.set("_encoder_layer_4_attention_self_value_MatMul_output_0", builder["matmul"](env.get("_encoder_layer_3_output_LayerNorm_Add_1_output_0"), env.get("onnx__MatMul_843"), {}));
  env.set("_encoder_layer_4_attention_self_value_Add_output_0", builder["add"](env.get("encoder_layer_4_attention_self_value_bias"), env.get("_encoder_layer_4_attention_self_value_MatMul_output_0"), {}));
  env.set("_encoder_layer_4_attention_self_Reshape_1_output_0", builder.reshape(env.get("_encoder_layer_4_attention_self_value_Add_output_0"), [1,128,12,32]));
  env.set("_encoder_layer_4_attention_self_Transpose_output_0", builder["transpose"](env.get("_encoder_layer_4_attention_self_Reshape_1_output_0"), {"permutation":[0,2,1,3]}));
  env.set("_encoder_layer_4_attention_self_Reshape_2_output_0", builder.reshape(env.get("_encoder_layer_4_attention_self_query_Add_output_0"), [1,128,12,32]));
  env.set("_encoder_layer_4_attention_self_Transpose_1_output_0", builder["transpose"](env.get("_encoder_layer_4_attention_self_Reshape_2_output_0"), {"permutation":[0,2,1,3]}));
  env.set("_encoder_layer_4_attention_self_Transpose_2_output_0", builder["transpose"](env.get("_encoder_layer_4_attention_self_Reshape_output_0"), {"permutation":[0,2,3,1]}));
  env.set("_encoder_layer_4_attention_self_MatMul_output_0", builder["matmul"](env.get("_encoder_layer_4_attention_self_Transpose_1_output_0"), env.get("_encoder_layer_4_attention_self_Transpose_2_output_0"), {}));
  env.set("_encoder_layer_4_attention_self_Constant_12_output_0", builder.constant({ dataType: "float16", shape: [] }, Uint8Array.from(atob("qEU="), c => c.charCodeAt(0)).buffer));
  env.set("_encoder_layer_4_attention_self_Div_output_0", builder["div"](env.get("_encoder_layer_4_attention_self_MatMul_output_0"), env.get("_encoder_layer_4_attention_self_Constant_12_output_0"), {}));
  env.set("_encoder_layer_4_attention_self_Add_output_0", builder["add"](env.get("_encoder_layer_4_attention_self_Div_output_0"), env.get("_Mul_output_0"), {}));
  env.set("_encoder_layer_4_attention_self_Softmax_output_0", builder["softmax"](env.get("_encoder_layer_4_attention_self_Add_output_0"), 3));
  env.set("_encoder_layer_4_attention_self_MatMul_1_output_0", builder["matmul"](env.get("_encoder_layer_4_attention_self_Softmax_output_0"), env.get("_encoder_layer_4_attention_self_Transpose_output_0"), {}));
  env.set("_encoder_layer_4_attention_self_Transpose_3_output_0", builder["transpose"](env.get("_encoder_layer_4_attention_self_MatMul_1_output_0"), {"permutation":[0,2,1,3]}));
  env.set("_encoder_layer_4_attention_self_Reshape_3_output_0", builder.reshape(env.get("_encoder_layer_4_attention_self_Transpose_3_output_0"), [1,128,384]));
  env.set("_encoder_layer_4_attention_output_dense_MatMul_output_0", builder["matmul"](env.get("_encoder_layer_4_attention_self_Reshape_3_output_0"), env.get("onnx__MatMul_849"), {}));
  env.set("_encoder_layer_4_attention_output_dense_Add_output_0", builder["add"](env.get("encoder_layer_4_attention_output_dense_bias"), env.get("_encoder_layer_4_attention_output_dense_MatMul_output_0"), {}));
  env.set("_encoder_layer_4_attention_output_Add_output_0", builder["add"](env.get("_encoder_layer_4_attention_output_dense_Add_output_0"), env.get("_encoder_layer_3_output_LayerNorm_Add_1_output_0"), {}));
  env.set("_encoder_layer_4_attention_output_LayerNorm_ReduceMean_output_0", builder["reduceMean"](env.get("_encoder_layer_4_attention_output_Add_output_0"), {"axes":[2],"keepDimensions":true}));
  env.set("_encoder_layer_4_attention_output_LayerNorm_Sub_output_0", builder["sub"](env.get("_encoder_layer_4_attention_output_Add_output_0"), env.get("_encoder_layer_4_attention_output_LayerNorm_ReduceMean_output_0"), {}));
  env.set("_encoder_layer_4_attention_output_LayerNorm_Constant_output_0", builder.constant({ dataType: "float16", shape: [] }, Uint8Array.from(atob("AEA="), c => c.charCodeAt(0)).buffer));
  env.set("_encoder_layer_4_attention_output_LayerNorm_Pow_output_0", builder["pow"](env.get("_encoder_layer_4_attention_output_LayerNorm_Sub_output_0"), env.get("_encoder_layer_4_attention_output_LayerNorm_Constant_output_0"), {}));
  env.set("_encoder_layer_4_attention_output_LayerNorm_ReduceMean_1_output_0", builder["reduceMean"](env.get("_encoder_layer_4_attention_output_LayerNorm_Pow_output_0"), {"axes":[2],"keepDimensions":true}));
  env.set("_encoder_layer_4_attention_output_LayerNorm_Constant_1_output_0", builder.constant({ dataType: "float16", shape: [] }, Uint8Array.from(atob("AgA="), c => c.charCodeAt(0)).buffer));
  env.set("_encoder_layer_4_attention_output_LayerNorm_Add_output_0", builder["add"](env.get("_encoder_layer_4_attention_output_LayerNorm_ReduceMean_1_output_0"), env.get("_encoder_layer_4_attention_output_LayerNorm_Constant_1_output_0"), {}));
  env.set("_encoder_layer_4_attention_output_LayerNorm_Sqrt_output_0", builder["sqrt"](env.get("_encoder_layer_4_attention_output_LayerNorm_Add_output_0"), {}));
  env.set("_encoder_layer_4_attention_output_LayerNorm_Div_output_0", builder["div"](env.get("_encoder_layer_4_attention_output_LayerNorm_Sub_output_0"), env.get("_encoder_layer_4_attention_output_LayerNorm_Sqrt_output_0"), {}));
  env.set("_encoder_layer_4_attention_output_LayerNorm_Mul_output_0", builder["mul"](env.get("_encoder_layer_4_attention_output_LayerNorm_Div_output_0"), env.get("encoder_layer_4_attention_output_LayerNorm_weight"), {}));
  env.set("_encoder_layer_4_attention_output_LayerNorm_Add_1_output_0", builder["add"](env.get("_encoder_layer_4_attention_output_LayerNorm_Mul_output_0"), env.get("encoder_layer_4_attention_output_LayerNorm_bias"), {}));
  env.set("_encoder_layer_4_intermediate_dense_MatMul_output_0", builder["matmul"](env.get("_encoder_layer_4_attention_output_LayerNorm_Add_1_output_0"), env.get("onnx__MatMul_850"), {}));
  env.set("_encoder_layer_4_intermediate_dense_Add_output_0", builder["add"](env.get("encoder_layer_4_intermediate_dense_bias"), env.get("_encoder_layer_4_intermediate_dense_MatMul_output_0"), {}));
  env.set("_encoder_layer_4_intermediate_intermediate_act_fn_Constant_output_0", builder.constant({ dataType: "float16", shape: [] }, Uint8Array.from(atob("qD0="), c => c.charCodeAt(0)).buffer));
  env.set("_encoder_layer_4_intermediate_intermediate_act_fn_Div_output_0", builder["div"](env.get("_encoder_layer_4_intermediate_dense_Add_output_0"), env.get("_encoder_layer_4_intermediate_intermediate_act_fn_Constant_output_0"), {}));
  env.set("_encoder_layer_4_intermediate_intermediate_act_fn_Erf_output_0", builder["erf"](env.get("_encoder_layer_4_intermediate_intermediate_act_fn_Div_output_0"), {}));
  env.set("_encoder_layer_4_intermediate_intermediate_act_fn_Constant_1_output_0", builder.constant({ dataType: "float16", shape: [] }, Uint8Array.from(atob("ADw="), c => c.charCodeAt(0)).buffer));
  env.set("_encoder_layer_4_intermediate_intermediate_act_fn_Add_output_0", builder["add"](env.get("_encoder_layer_4_intermediate_intermediate_act_fn_Erf_output_0"), env.get("_encoder_layer_4_intermediate_intermediate_act_fn_Constant_1_output_0"), {}));
  env.set("_encoder_layer_4_intermediate_intermediate_act_fn_Mul_output_0", builder["mul"](env.get("_encoder_layer_4_intermediate_dense_Add_output_0"), env.get("_encoder_layer_4_intermediate_intermediate_act_fn_Add_output_0"), {}));
  env.set("_encoder_layer_4_intermediate_intermediate_act_fn_Constant_2_output_0", builder.constant({ dataType: "float16", shape: [] }, Uint8Array.from(atob("ADg="), c => c.charCodeAt(0)).buffer));
  env.set("_encoder_layer_4_intermediate_intermediate_act_fn_Mul_1_output_0", builder["mul"](env.get("_encoder_layer_4_intermediate_intermediate_act_fn_Mul_output_0"), env.get("_encoder_layer_4_intermediate_intermediate_act_fn_Constant_2_output_0"), {}));
  env.set("_encoder_layer_4_output_dense_MatMul_output_0", builder["matmul"](env.get("_encoder_layer_4_intermediate_intermediate_act_fn_Mul_1_output_0"), env.get("onnx__MatMul_851"), {}));
  env.set("_encoder_layer_4_output_dense_Add_output_0", builder["add"](env.get("encoder_layer_4_output_dense_bias"), env.get("_encoder_layer_4_output_dense_MatMul_output_0"), {}));
  env.set("_encoder_layer_4_output_Add_output_0", builder["add"](env.get("_encoder_layer_4_output_dense_Add_output_0"), env.get("_encoder_layer_4_attention_output_LayerNorm_Add_1_output_0"), {}));
  env.set("_encoder_layer_4_output_LayerNorm_ReduceMean_output_0", builder["reduceMean"](env.get("_encoder_layer_4_output_Add_output_0"), {"axes":[2],"keepDimensions":true}));
  env.set("_encoder_layer_4_output_LayerNorm_Sub_output_0", builder["sub"](env.get("_encoder_layer_4_output_Add_output_0"), env.get("_encoder_layer_4_output_LayerNorm_ReduceMean_output_0"), {}));
  env.set("_encoder_layer_4_output_LayerNorm_Constant_output_0", builder.constant({ dataType: "float16", shape: [] }, Uint8Array.from(atob("AEA="), c => c.charCodeAt(0)).buffer));
  env.set("_encoder_layer_4_output_LayerNorm_Pow_output_0", builder["pow"](env.get("_encoder_layer_4_output_LayerNorm_Sub_output_0"), env.get("_encoder_layer_4_output_LayerNorm_Constant_output_0"), {}));
  env.set("_encoder_layer_4_output_LayerNorm_ReduceMean_1_output_0", builder["reduceMean"](env.get("_encoder_layer_4_output_LayerNorm_Pow_output_0"), {"axes":[2],"keepDimensions":true}));
  env.set("_encoder_layer_4_output_LayerNorm_Constant_1_output_0", builder.constant({ dataType: "float16", shape: [] }, Uint8Array.from(atob("AgA="), c => c.charCodeAt(0)).buffer));
  env.set("_encoder_layer_4_output_LayerNorm_Add_output_0", builder["add"](env.get("_encoder_layer_4_output_LayerNorm_ReduceMean_1_output_0"), env.get("_encoder_layer_4_output_LayerNorm_Constant_1_output_0"), {}));
  env.set("_encoder_layer_4_output_LayerNorm_Sqrt_output_0", builder["sqrt"](env.get("_encoder_layer_4_output_LayerNorm_Add_output_0"), {}));
  env.set("_encoder_layer_4_output_LayerNorm_Div_output_0", builder["div"](env.get("_encoder_layer_4_output_LayerNorm_Sub_output_0"), env.get("_encoder_layer_4_output_LayerNorm_Sqrt_output_0"), {}));
  env.set("_encoder_layer_4_output_LayerNorm_Mul_output_0", builder["mul"](env.get("_encoder_layer_4_output_LayerNorm_Div_output_0"), env.get("encoder_layer_4_output_LayerNorm_weight"), {}));
  env.set("_encoder_layer_4_output_LayerNorm_Add_1_output_0", builder["add"](env.get("_encoder_layer_4_output_LayerNorm_Mul_output_0"), env.get("encoder_layer_4_output_LayerNorm_bias"), {}));
  env.set("_encoder_layer_5_attention_self_query_MatMul_output_0", builder["matmul"](env.get("_encoder_layer_4_output_LayerNorm_Add_1_output_0"), env.get("onnx__MatMul_852"), {}));
  env.set("_encoder_layer_5_attention_self_query_Add_output_0", builder["add"](env.get("encoder_layer_5_attention_self_query_bias"), env.get("_encoder_layer_5_attention_self_query_MatMul_output_0"), {}));
  env.set("_encoder_layer_5_attention_self_key_MatMul_output_0", builder["matmul"](env.get("_encoder_layer_4_output_LayerNorm_Add_1_output_0"), env.get("onnx__MatMul_853"), {}));
  env.set("_encoder_layer_5_attention_self_key_Add_output_0", builder["add"](env.get("encoder_layer_5_attention_self_key_bias"), env.get("_encoder_layer_5_attention_self_key_MatMul_output_0"), {}));
  env.set("_encoder_layer_5_attention_self_Reshape_output_0", builder.reshape(env.get("_encoder_layer_5_attention_self_key_Add_output_0"), [1,128,12,32]));
  env.set("_encoder_layer_5_attention_self_value_MatMul_output_0", builder["matmul"](env.get("_encoder_layer_4_output_LayerNorm_Add_1_output_0"), env.get("onnx__MatMul_856"), {}));
  env.set("_encoder_layer_5_attention_self_value_Add_output_0", builder["add"](env.get("encoder_layer_5_attention_self_value_bias"), env.get("_encoder_layer_5_attention_self_value_MatMul_output_0"), {}));
  env.set("_encoder_layer_5_attention_self_Reshape_1_output_0", builder.reshape(env.get("_encoder_layer_5_attention_self_value_Add_output_0"), [1,128,12,32]));
  env.set("_encoder_layer_5_attention_self_Transpose_output_0", builder["transpose"](env.get("_encoder_layer_5_attention_self_Reshape_1_output_0"), {"permutation":[0,2,1,3]}));
  env.set("_encoder_layer_5_attention_self_Reshape_2_output_0", builder.reshape(env.get("_encoder_layer_5_attention_self_query_Add_output_0"), [1,128,12,32]));
  env.set("_encoder_layer_5_attention_self_Transpose_1_output_0", builder["transpose"](env.get("_encoder_layer_5_attention_self_Reshape_2_output_0"), {"permutation":[0,2,1,3]}));
  env.set("_encoder_layer_5_attention_self_Transpose_2_output_0", builder["transpose"](env.get("_encoder_layer_5_attention_self_Reshape_output_0"), {"permutation":[0,2,3,1]}));
  env.set("_encoder_layer_5_attention_self_MatMul_output_0", builder["matmul"](env.get("_encoder_layer_5_attention_self_Transpose_1_output_0"), env.get("_encoder_layer_5_attention_self_Transpose_2_output_0"), {}));
  env.set("_encoder_layer_5_attention_self_Constant_12_output_0", builder.constant({ dataType: "float16", shape: [] }, Uint8Array.from(atob("qEU="), c => c.charCodeAt(0)).buffer));
  env.set("_encoder_layer_5_attention_self_Div_output_0", builder["div"](env.get("_encoder_layer_5_attention_self_MatMul_output_0"), env.get("_encoder_layer_5_attention_self_Constant_12_output_0"), {}));
  env.set("_encoder_layer_5_attention_self_Add_output_0", builder["add"](env.get("_encoder_layer_5_attention_self_Div_output_0"), env.get("_Mul_output_0"), {}));
  env.set("_encoder_layer_5_attention_self_Softmax_output_0", builder["softmax"](env.get("_encoder_layer_5_attention_self_Add_output_0"), 3));
  env.set("_encoder_layer_5_attention_self_MatMul_1_output_0", builder["matmul"](env.get("_encoder_layer_5_attention_self_Softmax_output_0"), env.get("_encoder_layer_5_attention_self_Transpose_output_0"), {}));
  env.set("_encoder_layer_5_attention_self_Transpose_3_output_0", builder["transpose"](env.get("_encoder_layer_5_attention_self_MatMul_1_output_0"), {"permutation":[0,2,1,3]}));
  env.set("_encoder_layer_5_attention_self_Reshape_3_output_0", builder.reshape(env.get("_encoder_layer_5_attention_self_Transpose_3_output_0"), [1,128,384]));
  env.set("_encoder_layer_5_attention_output_dense_MatMul_output_0", builder["matmul"](env.get("_encoder_layer_5_attention_self_Reshape_3_output_0"), env.get("onnx__MatMul_862"), {}));
  env.set("_encoder_layer_5_attention_output_dense_Add_output_0", builder["add"](env.get("encoder_layer_5_attention_output_dense_bias"), env.get("_encoder_layer_5_attention_output_dense_MatMul_output_0"), {}));
  env.set("_encoder_layer_5_attention_output_Add_output_0", builder["add"](env.get("_encoder_layer_5_attention_output_dense_Add_output_0"), env.get("_encoder_layer_4_output_LayerNorm_Add_1_output_0"), {}));
  env.set("_encoder_layer_5_attention_output_LayerNorm_ReduceMean_output_0", builder["reduceMean"](env.get("_encoder_layer_5_attention_output_Add_output_0"), {"axes":[2],"keepDimensions":true}));
  env.set("_encoder_layer_5_attention_output_LayerNorm_Sub_output_0", builder["sub"](env.get("_encoder_layer_5_attention_output_Add_output_0"), env.get("_encoder_layer_5_attention_output_LayerNorm_ReduceMean_output_0"), {}));
  env.set("_encoder_layer_5_attention_output_LayerNorm_Constant_output_0", builder.constant({ dataType: "float16", shape: [] }, Uint8Array.from(atob("AEA="), c => c.charCodeAt(0)).buffer));
  env.set("_encoder_layer_5_attention_output_LayerNorm_Pow_output_0", builder["pow"](env.get("_encoder_layer_5_attention_output_LayerNorm_Sub_output_0"), env.get("_encoder_layer_5_attention_output_LayerNorm_Constant_output_0"), {}));
  env.set("_encoder_layer_5_attention_output_LayerNorm_ReduceMean_1_output_0", builder["reduceMean"](env.get("_encoder_layer_5_attention_output_LayerNorm_Pow_output_0"), {"axes":[2],"keepDimensions":true}));
  env.set("_encoder_layer_5_attention_output_LayerNorm_Constant_1_output_0", builder.constant({ dataType: "float16", shape: [] }, Uint8Array.from(atob("AgA="), c => c.charCodeAt(0)).buffer));
  env.set("_encoder_layer_5_attention_output_LayerNorm_Add_output_0", builder["add"](env.get("_encoder_layer_5_attention_output_LayerNorm_ReduceMean_1_output_0"), env.get("_encoder_layer_5_attention_output_LayerNorm_Constant_1_output_0"), {}));
  env.set("_encoder_layer_5_attention_output_LayerNorm_Sqrt_output_0", builder["sqrt"](env.get("_encoder_layer_5_attention_output_LayerNorm_Add_output_0"), {}));
  env.set("_encoder_layer_5_attention_output_LayerNorm_Div_output_0", builder["div"](env.get("_encoder_layer_5_attention_output_LayerNorm_Sub_output_0"), env.get("_encoder_layer_5_attention_output_LayerNorm_Sqrt_output_0"), {}));
  env.set("_encoder_layer_5_attention_output_LayerNorm_Mul_output_0", builder["mul"](env.get("_encoder_layer_5_attention_output_LayerNorm_Div_output_0"), env.get("encoder_layer_5_attention_output_LayerNorm_weight"), {}));
  env.set("_encoder_layer_5_attention_output_LayerNorm_Add_1_output_0", builder["add"](env.get("_encoder_layer_5_attention_output_LayerNorm_Mul_output_0"), env.get("encoder_layer_5_attention_output_LayerNorm_bias"), {}));
  env.set("_encoder_layer_5_intermediate_dense_MatMul_output_0", builder["matmul"](env.get("_encoder_layer_5_attention_output_LayerNorm_Add_1_output_0"), env.get("onnx__MatMul_863"), {}));
  env.set("_encoder_layer_5_intermediate_dense_Add_output_0", builder["add"](env.get("encoder_layer_5_intermediate_dense_bias"), env.get("_encoder_layer_5_intermediate_dense_MatMul_output_0"), {}));
  env.set("_encoder_layer_5_intermediate_intermediate_act_fn_Constant_output_0", builder.constant({ dataType: "float16", shape: [] }, Uint8Array.from(atob("qD0="), c => c.charCodeAt(0)).buffer));
  env.set("_encoder_layer_5_intermediate_intermediate_act_fn_Div_output_0", builder["div"](env.get("_encoder_layer_5_intermediate_dense_Add_output_0"), env.get("_encoder_layer_5_intermediate_intermediate_act_fn_Constant_output_0"), {}));
  env.set("_encoder_layer_5_intermediate_intermediate_act_fn_Erf_output_0", builder["erf"](env.get("_encoder_layer_5_intermediate_intermediate_act_fn_Div_output_0"), {}));
  env.set("_encoder_layer_5_intermediate_intermediate_act_fn_Constant_1_output_0", builder.constant({ dataType: "float16", shape: [] }, Uint8Array.from(atob("ADw="), c => c.charCodeAt(0)).buffer));
  env.set("_encoder_layer_5_intermediate_intermediate_act_fn_Add_output_0", builder["add"](env.get("_encoder_layer_5_intermediate_intermediate_act_fn_Erf_output_0"), env.get("_encoder_layer_5_intermediate_intermediate_act_fn_Constant_1_output_0"), {}));
  env.set("_encoder_layer_5_intermediate_intermediate_act_fn_Mul_output_0", builder["mul"](env.get("_encoder_layer_5_intermediate_dense_Add_output_0"), env.get("_encoder_layer_5_intermediate_intermediate_act_fn_Add_output_0"), {}));
  env.set("_encoder_layer_5_intermediate_intermediate_act_fn_Constant_2_output_0", builder.constant({ dataType: "float16", shape: [] }, Uint8Array.from(atob("ADg="), c => c.charCodeAt(0)).buffer));
  env.set("_encoder_layer_5_intermediate_intermediate_act_fn_Mul_1_output_0", builder["mul"](env.get("_encoder_layer_5_intermediate_intermediate_act_fn_Mul_output_0"), env.get("_encoder_layer_5_intermediate_intermediate_act_fn_Constant_2_output_0"), {}));
  env.set("_encoder_layer_5_output_dense_MatMul_output_0", builder["matmul"](env.get("_encoder_layer_5_intermediate_intermediate_act_fn_Mul_1_output_0"), env.get("onnx__MatMul_864"), {}));
  env.set("_encoder_layer_5_output_dense_Add_output_0", builder["add"](env.get("encoder_layer_5_output_dense_bias"), env.get("_encoder_layer_5_output_dense_MatMul_output_0"), {}));
  env.set("_encoder_layer_5_output_Add_output_0", builder["add"](env.get("_encoder_layer_5_output_dense_Add_output_0"), env.get("_encoder_layer_5_attention_output_LayerNorm_Add_1_output_0"), {}));
  env.set("_encoder_layer_5_output_LayerNorm_ReduceMean_output_0", builder["reduceMean"](env.get("_encoder_layer_5_output_Add_output_0"), {"axes":[2],"keepDimensions":true}));
  env.set("_encoder_layer_5_output_LayerNorm_Sub_output_0", builder["sub"](env.get("_encoder_layer_5_output_Add_output_0"), env.get("_encoder_layer_5_output_LayerNorm_ReduceMean_output_0"), {}));
  env.set("_encoder_layer_5_output_LayerNorm_Constant_output_0", builder.constant({ dataType: "float16", shape: [] }, Uint8Array.from(atob("AEA="), c => c.charCodeAt(0)).buffer));
  env.set("_encoder_layer_5_output_LayerNorm_Pow_output_0", builder["pow"](env.get("_encoder_layer_5_output_LayerNorm_Sub_output_0"), env.get("_encoder_layer_5_output_LayerNorm_Constant_output_0"), {}));
  env.set("_encoder_layer_5_output_LayerNorm_ReduceMean_1_output_0", builder["reduceMean"](env.get("_encoder_layer_5_output_LayerNorm_Pow_output_0"), {"axes":[2],"keepDimensions":true}));
  env.set("_encoder_layer_5_output_LayerNorm_Constant_1_output_0", builder.constant({ dataType: "float16", shape: [] }, Uint8Array.from(atob("AgA="), c => c.charCodeAt(0)).buffer));
  env.set("_encoder_layer_5_output_LayerNorm_Add_output_0", builder["add"](env.get("_encoder_layer_5_output_LayerNorm_ReduceMean_1_output_0"), env.get("_encoder_layer_5_output_LayerNorm_Constant_1_output_0"), {}));
  env.set("_encoder_layer_5_output_LayerNorm_Sqrt_output_0", builder["sqrt"](env.get("_encoder_layer_5_output_LayerNorm_Add_output_0"), {}));
  env.set("_encoder_layer_5_output_LayerNorm_Div_output_0", builder["div"](env.get("_encoder_layer_5_output_LayerNorm_Sub_output_0"), env.get("_encoder_layer_5_output_LayerNorm_Sqrt_output_0"), {}));
  env.set("_encoder_layer_5_output_LayerNorm_Mul_output_0", builder["mul"](env.get("_encoder_layer_5_output_LayerNorm_Div_output_0"), env.get("encoder_layer_5_output_LayerNorm_weight"), {}));
  env.set("graph_output_cast_0", builder["add"](env.get("_encoder_layer_5_output_LayerNorm_Mul_output_0"), env.get("encoder_layer_5_output_LayerNorm_bias"), {}));
  env.set("last_hidden_state", builder.cast(env.get("graph_output_cast_0"), "float32"));

  const outputs = {};
  outputs["last_hidden_state"] = env.get("last_hidden_state");
  return await builder.build(outputs);
}
