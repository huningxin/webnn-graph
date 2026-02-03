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

  env.set("attention_mask_596", builder.input("attention_mask_596", { dataType: "int64", shape: [1, 128] }));
  env.set("input_ids_580", builder.input("input_ids_580", { dataType: "int64", shape: [1, 128] }));
  env.set("past_key_values_0_key_652", builder.input("past_key_values_0_key_652", { dataType: "float16", shape: [1, 4, 512, 64] }));
  env.set("past_key_values_0_value_609", builder.input("past_key_values_0_value_609", { dataType: "float16", shape: [1, 4, 512, 64] }));
  env.set("past_key_values_10_key_1842", builder.input("past_key_values_10_key_1842", { dataType: "float16", shape: [1, 4, 512, 64] }));
  env.set("past_key_values_10_value_1806", builder.input("past_key_values_10_value_1806", { dataType: "float16", shape: [1, 4, 512, 64] }));
  env.set("past_key_values_11_key_1961", builder.input("past_key_values_11_key_1961", { dataType: "float16", shape: [1, 4, 512, 64] }));
  env.set("past_key_values_11_value_1925", builder.input("past_key_values_11_value_1925", { dataType: "float16", shape: [1, 4, 512, 64] }));
  env.set("past_key_values_12_key_2080", builder.input("past_key_values_12_key_2080", { dataType: "float16", shape: [1, 4, 512, 64] }));
  env.set("past_key_values_12_value_2044", builder.input("past_key_values_12_value_2044", { dataType: "float16", shape: [1, 4, 512, 64] }));
  env.set("past_key_values_13_key_2199", builder.input("past_key_values_13_key_2199", { dataType: "float16", shape: [1, 4, 512, 64] }));
  env.set("past_key_values_13_value_2163", builder.input("past_key_values_13_value_2163", { dataType: "float16", shape: [1, 4, 512, 64] }));
  env.set("past_key_values_14_key_2318", builder.input("past_key_values_14_key_2318", { dataType: "float16", shape: [1, 4, 512, 64] }));
  env.set("past_key_values_14_value_2282", builder.input("past_key_values_14_value_2282", { dataType: "float16", shape: [1, 4, 512, 64] }));
  env.set("past_key_values_15_key_2437", builder.input("past_key_values_15_key_2437", { dataType: "float16", shape: [1, 4, 512, 64] }));
  env.set("past_key_values_15_value_2401", builder.input("past_key_values_15_value_2401", { dataType: "float16", shape: [1, 4, 512, 64] }));
  env.set("past_key_values_16_key_2556", builder.input("past_key_values_16_key_2556", { dataType: "float16", shape: [1, 4, 512, 64] }));
  env.set("past_key_values_16_value_2520", builder.input("past_key_values_16_value_2520", { dataType: "float16", shape: [1, 4, 512, 64] }));
  env.set("past_key_values_17_key_2675", builder.input("past_key_values_17_key_2675", { dataType: "float16", shape: [1, 4, 512, 64] }));
  env.set("past_key_values_17_value_2639", builder.input("past_key_values_17_value_2639", { dataType: "float16", shape: [1, 4, 512, 64] }));
  env.set("past_key_values_18_key_2794", builder.input("past_key_values_18_key_2794", { dataType: "float16", shape: [1, 4, 512, 64] }));
  env.set("past_key_values_18_value_2758", builder.input("past_key_values_18_value_2758", { dataType: "float16", shape: [1, 4, 512, 64] }));
  env.set("past_key_values_19_key_2913", builder.input("past_key_values_19_key_2913", { dataType: "float16", shape: [1, 4, 512, 64] }));
  env.set("past_key_values_19_value_2877", builder.input("past_key_values_19_value_2877", { dataType: "float16", shape: [1, 4, 512, 64] }));
  env.set("past_key_values_1_key_771", builder.input("past_key_values_1_key_771", { dataType: "float16", shape: [1, 4, 512, 64] }));
  env.set("past_key_values_1_value_735", builder.input("past_key_values_1_value_735", { dataType: "float16", shape: [1, 4, 512, 64] }));
  env.set("past_key_values_20_key_3032", builder.input("past_key_values_20_key_3032", { dataType: "float16", shape: [1, 4, 512, 64] }));
  env.set("past_key_values_20_value_2996", builder.input("past_key_values_20_value_2996", { dataType: "float16", shape: [1, 4, 512, 64] }));
  env.set("past_key_values_21_key_3151", builder.input("past_key_values_21_key_3151", { dataType: "float16", shape: [1, 4, 512, 64] }));
  env.set("past_key_values_21_value_3115", builder.input("past_key_values_21_value_3115", { dataType: "float16", shape: [1, 4, 512, 64] }));
  env.set("past_key_values_2_key_890", builder.input("past_key_values_2_key_890", { dataType: "float16", shape: [1, 4, 512, 64] }));
  env.set("past_key_values_2_value_854", builder.input("past_key_values_2_value_854", { dataType: "float16", shape: [1, 4, 512, 64] }));
  env.set("past_key_values_3_key_1009", builder.input("past_key_values_3_key_1009", { dataType: "float16", shape: [1, 4, 512, 64] }));
  env.set("past_key_values_3_value_973", builder.input("past_key_values_3_value_973", { dataType: "float16", shape: [1, 4, 512, 64] }));
  env.set("past_key_values_4_key_1128", builder.input("past_key_values_4_key_1128", { dataType: "float16", shape: [1, 4, 512, 64] }));
  env.set("past_key_values_4_value_1092", builder.input("past_key_values_4_value_1092", { dataType: "float16", shape: [1, 4, 512, 64] }));
  env.set("past_key_values_5_key_1247", builder.input("past_key_values_5_key_1247", { dataType: "float16", shape: [1, 4, 512, 64] }));
  env.set("past_key_values_5_value_1211", builder.input("past_key_values_5_value_1211", { dataType: "float16", shape: [1, 4, 512, 64] }));
  env.set("past_key_values_6_key_1366", builder.input("past_key_values_6_key_1366", { dataType: "float16", shape: [1, 4, 512, 64] }));
  env.set("past_key_values_6_value_1330", builder.input("past_key_values_6_value_1330", { dataType: "float16", shape: [1, 4, 512, 64] }));
  env.set("past_key_values_7_key_1485", builder.input("past_key_values_7_key_1485", { dataType: "float16", shape: [1, 4, 512, 64] }));
  env.set("past_key_values_7_value_1449", builder.input("past_key_values_7_value_1449", { dataType: "float16", shape: [1, 4, 512, 64] }));
  env.set("past_key_values_8_key_1604", builder.input("past_key_values_8_key_1604", { dataType: "float16", shape: [1, 4, 512, 64] }));
  env.set("past_key_values_8_value_1568", builder.input("past_key_values_8_value_1568", { dataType: "float16", shape: [1, 4, 512, 64] }));
  env.set("past_key_values_9_key_1723", builder.input("past_key_values_9_key_1723", { dataType: "float16", shape: [1, 4, 512, 64] }));
  env.set("past_key_values_9_value_1687", builder.input("past_key_values_9_value_1687", { dataType: "float16", shape: [1, 4, 512, 64] }));
  env.set("position_ids_625", builder.input("position_ids_625", { dataType: "int64", shape: [1, 128] }));

  {
    const sl = weights.getSlice("Inserted_1");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_10");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_10", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1001");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1001", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1003");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1003", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1004");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1004", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1005");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1005", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1006");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1006", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1007");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1007", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1008");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1008", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_101");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_101", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1012");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1012", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1013");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1013", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1014");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1014", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1015");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1015", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1016");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1016", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1017");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1017", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1018");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1018", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1019");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1019", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1021");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1021", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1022");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1022", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1023");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1023", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1024");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1024", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1025");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1025", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1026");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1026", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1028");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1028", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_103");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_103", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1030");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1030", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1032");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1032", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1033");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1033", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1034");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1034", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1036");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1036", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1037");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1037", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1038");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1038", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1039");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1039", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_104");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_104", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1040");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1040", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1041");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1041", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1042");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1042", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1043");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1043", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1045");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1045", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1046");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1046", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1047");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1047", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1048");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1048", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1049");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1049", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_105");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_105", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1050");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1050", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1051");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1051", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1052");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1052", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1053");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1053", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1054");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1054", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1055");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1055", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1057");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1057", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1059");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1059", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_106");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_106", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1061");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1061", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1063");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1063", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1064");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1064", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1065");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1065", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1066");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1066", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1067");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1067", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1068");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1068", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_107");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_107", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1072");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1072", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1073");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1073", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1074");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1074", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1075");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1075", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1076");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1076", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1077");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1077", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1078");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1078", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1079");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1079", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_108");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_108", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1081");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1081", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1082");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1082", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1083");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1083", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1084");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1084", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1085");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1085", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1086");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1086", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1088");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1088", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1090");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1090", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1092");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1092", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1093");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1093", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1094");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1094", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1096");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1096", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1097");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1097", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1098");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1098", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1099");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1099", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_11");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_11", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1100");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1100", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1101");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1101", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1102");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1102", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1103");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1103", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1105");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1105", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1106");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1106", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1107");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1107", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1108");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1108", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1109");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1109", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1110");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1110", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1111");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1111", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1112");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1112", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1113");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1113", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1114");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1114", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1115");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1115", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1117");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1117", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1119");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1119", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_112");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_112", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1121");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1121", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1123");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1123", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1124");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1124", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1125");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1125", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1126");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1126", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1127");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1127", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1128");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1128", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_113");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_113", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1132");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1132", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1133");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1133", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1134");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1134", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1135");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1135", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1136");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1136", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1137");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1137", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1138");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1138", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1139");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1139", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_114");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_114", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1141");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1141", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1142");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1142", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1143");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1143", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1144");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1144", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1145");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1145", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1146");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1146", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1148");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1148", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_115");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_115", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1150");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1150", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1152");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1152", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1153");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1153", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1154");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1154", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1156");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1156", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1157");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1157", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1158");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1158", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1159");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1159", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_116");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_116", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1160");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1160", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1161");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1161", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1162");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1162", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1163");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1163", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1165");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1165", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1166");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1166", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1167");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1167", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1168");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1168", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1169");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1169", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_117");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_117", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1170");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1170", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1171");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1171", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1172");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1172", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1173");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1173", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1174");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1174", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1175");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1175", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1177");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1177", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1179");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1179", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_118");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_118", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1181");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1181", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1183");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1183", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1184");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1184", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1185");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1185", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1186");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1186", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1187");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1187", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1188");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1188", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_119");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_119", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1192");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1192", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1193");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1193", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1194");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1194", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1195");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1195", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1196");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1196", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1197");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1197", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1198");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1198", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1199");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1199", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_12");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_12", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1201");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1201", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1202");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1202", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1203");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1203", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1204");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1204", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1205");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1205", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1206");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1206", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1208");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1208", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_121");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_121", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1210");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1210", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1212");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1212", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1213");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1213", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1214");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1214", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1216");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1216", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1217");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1217", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1218");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1218", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1219");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1219", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_122");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_122", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1220");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1220", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1221");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1221", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1222");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1222", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1223");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1223", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1225");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1225", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1226");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1226", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1227");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1227", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1228");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1228", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1229");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1229", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_123");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_123", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1230");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1230", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1231");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1231", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1232");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1232", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1233");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1233", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1234");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1234", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1235");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1235", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1237");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1237", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1239");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1239", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_124");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_124", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1241");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1241", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1243");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1243", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1244");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1244", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1245");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1245", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1246");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1246", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1247");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1247", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1248");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1248", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_125");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_125", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1252");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1252", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1253");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1253", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1254");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1254", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1255");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1255", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1256");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1256", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1257");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1257", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1258");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1258", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1259");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1259", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_126");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_126", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1261");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1261", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1262");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1262", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1263");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1263", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1264");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1264", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1265");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1265", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1266");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1266", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1268");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1268", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1270");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1270", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1272");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1272", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1273");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1273", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1274");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1274", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1276");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1276", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1277");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1277", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1278");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1278", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1279");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1279", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_128");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_128", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1280");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1280", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1281");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1281", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1282");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1282", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1283");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1283", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1285");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1285", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1286");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1286", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1287");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1287", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1288");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1288", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1289");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1289", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1290");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1290", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1291");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1291", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1292");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1292", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1293");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1293", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1294");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1294", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1295");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1295", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1297");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1297", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1299");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1299", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_13");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_13", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_130");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_130", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1301");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1301", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1303");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1303", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1304");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1304", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1305");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1305", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1306");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1306", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1307");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1307", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1308");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1308", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1312");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1312", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1313");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1313", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1314");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1314", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1315");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1315", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1316");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1316", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1317");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1317", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1318");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1318", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1319");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1319", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_132");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_132", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1321");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1321", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1322");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1322", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1323");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1323", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1324");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1324", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1325");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1325", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1326");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1326", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1328");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1328", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_133");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_133", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1330");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1330", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1332");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1332", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1333");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1333", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1334");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1334", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1336");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1336", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1337");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1337", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1338");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1338", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1339");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1339", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_134");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_134", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1340");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1340", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1341");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1341", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1342");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1342", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1343");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1343", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1345");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1345", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1346");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1346", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1347");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1347", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1348");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1348", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1349");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1349", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1350");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1350", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1351");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1351", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1352");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1352", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1353");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1353", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1354");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1354", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1355");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1355", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1357");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1357", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1359");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1359", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_136");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_136", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1361");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1361", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1363");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1363", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1364");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1364", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1365");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1365", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1366");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1366", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1367");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1367", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1368");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1368", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_137");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_137", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1372");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1372", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1373");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1373", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1374");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1374", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1375");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1375", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1376");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1376", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1377");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1377", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1378");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1378", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1379");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1379", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_138");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_138", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1381");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1381", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1382");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1382", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1383");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1383", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1384");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1384", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1385");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1385", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1386");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1386", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1388");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1388", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_139");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_139", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1390");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1390", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1392");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1392", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1393");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1393", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1394");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1394", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1396");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1396", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1397");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1397", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1398");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1398", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1399");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1399", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_14");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_14", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_140");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_140", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1400");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1400", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1401");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1401", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1402");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1402", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1403");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1403", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1405");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1405", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1406");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1406", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1407");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1407", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1408");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1408", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1409");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1409", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_141");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_141", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1410");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1410", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1411");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1411", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1412");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1412", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1413");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1413", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_1414");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_1414", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_142");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_142", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_143");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_143", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_145");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_145", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_146");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_146", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_147");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_147", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_148");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_148", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_149");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_149", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_15");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_15", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_150");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_150", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_151");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_151", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_152");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_152", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_153");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_153", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_154");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_154", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_155");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_155", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_157");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_157", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_159");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_159", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_16");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_16", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_161");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_161", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_163");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_163", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_164");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_164", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_165");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_165", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_166");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_166", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_167");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_167", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_168");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_168", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_17");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_17", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_172");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_172", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_173");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_173", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_174");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_174", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_175");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_175", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_176");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_176", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_177");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_177", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_178");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_178", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_179");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_179", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_18");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_18", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_181");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_181", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_182");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_182", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_183");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_183", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_184");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_184", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_185");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_185", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_186");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_186", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_188");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_188", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_19");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_19", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_190");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_190", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_192");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_192", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_193");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_193", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_194");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_194", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_196");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_196", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_197");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_197", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_198");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_198", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_199");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_199", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_2");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_2", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_20");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_20", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_200");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_200", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_201");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_201", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_202");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_202", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_203");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_203", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_205");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_205", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_206");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_206", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_207");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_207", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_208");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_208", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_209");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_209", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_21");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_21", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_210");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_210", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_211");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_211", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_212");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_212", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_213");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_213", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_214");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_214", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_215");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_215", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_217");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_217", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_219");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_219", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_22");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_22", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_221");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_221", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_223");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_223", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_224");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_224", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_225");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_225", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_226");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_226", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_227");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_227", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_228");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_228", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_23");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_23", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_232");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_232", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_233");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_233", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_234");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_234", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_235");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_235", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_236");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_236", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_237");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_237", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_238");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_238", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_239");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_239", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_24");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_24", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_241");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_241", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_242");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_242", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_243");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_243", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_244");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_244", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_245");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_245", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_246");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_246", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_248");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_248", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_25");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_25", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_250");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_250", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_252");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_252", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_253");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_253", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_254");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_254", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_256");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_256", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_257");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_257", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_258");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_258", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_259");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_259", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_26");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_26", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_260");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_260", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_261");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_261", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_262");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_262", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_263");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_263", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_265");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_265", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_266");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_266", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_267");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_267", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_268");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_268", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_269");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_269", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_27");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_27", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_270");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_270", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_271");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_271", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_272");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_272", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_273");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_273", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_274");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_274", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_275");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_275", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_277");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_277", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_279");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_279", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_28");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_28", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_281");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_281", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_283");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_283", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_284");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_284", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_285");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_285", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_286");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_286", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_287");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_287", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_288");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_288", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_29");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_29", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_292");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_292", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_293");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_293", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_294");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_294", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_295");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_295", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_296");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_296", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_297");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_297", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_298");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_298", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_299");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_299", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_3");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_3", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_30");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_30", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_301");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_301", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_302");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_302", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_303");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_303", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_304");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_304", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_305");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_305", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_306");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_306", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_308");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_308", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_31");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_31", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_310");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_310", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_312");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_312", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_313");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_313", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_314");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_314", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_316");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_316", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_317");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_317", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_318");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_318", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_319");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_319", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_32");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_32", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_320");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_320", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_321");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_321", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_322");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_322", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_323");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_323", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_325");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_325", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_326");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_326", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_327");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_327", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_328");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_328", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_329");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_329", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_33");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_33", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_330");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_330", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_331");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_331", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_332");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_332", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_333");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_333", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_334");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_334", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_335");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_335", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_337");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_337", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_339");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_339", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_34");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_34", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_341");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_341", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_343");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_343", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_344");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_344", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_345");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_345", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_346");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_346", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_347");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_347", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_348");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_348", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_35");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_35", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_352");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_352", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_353");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_353", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_354");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_354", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_355");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_355", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_356");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_356", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_357");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_357", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_358");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_358", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_359");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_359", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_36");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_36", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_361");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_361", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_362");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_362", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_363");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_363", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_364");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_364", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_365");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_365", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_366");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_366", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_368");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_368", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_37");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_37", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_370");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_370", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_372");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_372", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_373");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_373", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_374");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_374", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_376");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_376", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_377");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_377", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_378");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_378", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_379");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_379", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_38");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_38", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_380");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_380", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_381");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_381", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_382");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_382", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_383");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_383", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_385");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_385", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_386");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_386", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_387");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_387", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_388");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_388", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_389");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_389", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_39");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_39", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_390");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_390", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_391");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_391", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_392");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_392", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_393");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_393", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_394");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_394", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_395");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_395", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_397");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_397", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_399");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_399", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_4");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_4", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_40");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_40", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_401");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_401", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_403");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_403", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_404");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_404", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_405");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_405", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_406");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_406", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_407");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_407", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_408");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_408", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_41");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_41", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_412");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_412", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_413");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_413", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_414");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_414", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_415");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_415", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_416");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_416", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_417");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_417", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_418");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_418", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_419");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_419", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_42");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_42", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_421");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_421", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_422");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_422", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_423");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_423", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_424");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_424", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_425");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_425", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_426");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_426", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_428");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_428", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_43");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_43", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_430");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_430", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_432");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_432", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_433");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_433", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_434");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_434", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_436");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_436", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_437");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_437", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_438");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_438", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_439");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_439", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_44");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_44", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_440");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_440", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_441");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_441", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_442");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_442", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_443");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_443", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_445");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_445", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_446");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_446", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_447");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_447", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_448");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_448", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_449");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_449", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_45");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_45", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_450");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_450", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_451");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_451", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_452");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_452", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_453");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_453", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_454");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_454", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_455");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_455", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_457");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_457", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_459");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_459", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_46");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_46", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_461");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_461", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_463");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_463", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_464");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_464", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_465");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_465", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_466");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_466", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_467");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_467", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_468");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_468", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_47");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_47", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_472");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_472", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_473");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_473", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_474");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_474", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_475");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_475", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_476");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_476", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_477");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_477", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_478");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_478", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_479");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_479", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_48");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_48", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_481");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_481", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_482");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_482", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_483");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_483", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_484");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_484", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_485");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_485", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_486");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_486", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_488");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_488", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_49");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_49", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_490");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_490", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_492");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_492", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_493");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_493", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_494");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_494", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_496");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_496", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_497");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_497", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_498");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_498", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_499");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_499", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_5");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_5", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_50");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_50", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_500");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_500", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_501");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_501", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_502");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_502", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_503");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_503", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_505");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_505", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_506");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_506", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_507");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_507", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_508");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_508", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_509");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_509", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_51");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_51", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_510");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_510", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_511");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_511", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_512");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_512", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_513");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_513", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_514");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_514", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_515");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_515", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_517");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_517", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_519");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_519", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_52");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_52", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_521");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_521", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_523");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_523", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_524");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_524", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_525");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_525", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_526");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_526", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_527");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_527", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_528");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_528", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_53");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_53", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_532");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_532", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_533");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_533", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_534");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_534", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_535");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_535", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_536");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_536", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_537");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_537", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_538");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_538", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_539");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_539", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_54");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_54", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_541");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_541", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_542");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_542", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_543");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_543", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_544");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_544", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_545");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_545", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_546");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_546", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_548");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_548", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_55");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_55", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_550");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_550", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_552");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_552", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_553");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_553", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_554");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_554", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_556");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_556", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_557");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_557", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_558");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_558", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_559");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_559", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_56");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_56", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_560");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_560", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_561");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_561", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_562");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_562", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_563");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_563", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_565");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_565", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_566");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_566", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_567");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_567", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_568");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_568", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_569");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_569", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_57");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_57", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_570");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_570", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_571");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_571", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_572");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_572", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_573");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_573", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_574");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_574", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_575");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_575", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_577");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_577", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_579");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_579", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_58");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_58", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_581");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_581", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_583");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_583", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_584");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_584", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_585");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_585", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_586");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_586", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_587");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_587", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_588");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_588", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_59");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_59", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_592");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_592", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_593");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_593", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_594");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_594", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_595");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_595", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_596");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_596", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_597");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_597", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_598");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_598", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_599");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_599", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_6");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_6", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_60");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_60", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_601");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_601", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_602");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_602", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_603");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_603", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_604");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_604", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_605");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_605", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_606");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_606", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_608");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_608", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_61");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_61", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_610");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_610", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_612");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_612", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_613");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_613", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_614");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_614", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_616");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_616", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_617");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_617", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_618");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_618", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_619");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_619", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_62");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_62", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_620");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_620", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_621");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_621", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_622");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_622", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_623");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_623", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_625");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_625", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_626");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_626", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_627");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_627", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_628");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_628", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_629");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_629", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_63");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_63", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_630");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_630", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_631");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_631", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_632");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_632", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_633");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_633", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_634");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_634", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_635");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_635", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_637");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_637", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_639");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_639", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_64");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_64", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_641");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_641", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_643");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_643", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_644");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_644", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_645");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_645", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_646");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_646", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_647");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_647", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_648");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_648", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_65");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_65", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_652");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_652", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_653");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_653", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_654");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_654", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_655");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_655", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_656");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_656", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_657");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_657", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_658");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_658", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_659");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_659", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_66");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_66", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_661");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_661", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_662");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_662", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_663");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_663", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_664");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_664", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_665");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_665", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_666");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_666", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_668");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_668", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_67");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_67", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_670");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_670", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_672");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_672", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_673");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_673", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_674");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_674", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_676");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_676", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_677");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_677", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_678");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_678", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_679");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_679", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_68");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_68", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_680");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_680", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_681");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_681", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_682");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_682", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_683");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_683", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_685");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_685", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_686");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_686", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_687");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_687", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_688");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_688", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_689");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_689", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_69");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_69", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_690");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_690", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_691");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_691", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_692");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_692", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_693");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_693", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_694");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_694", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_695");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_695", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_697");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_697", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_699");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_699", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_7");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_7", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_70");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_70", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_701");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_701", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_703");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_703", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_704");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_704", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_705");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_705", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_706");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_706", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_707");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_707", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_708");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_708", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_71");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_71", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_712");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_712", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_713");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_713", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_714");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_714", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_715");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_715", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_716");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_716", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_717");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_717", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_718");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_718", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_719");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_719", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_72");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_72", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_721");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_721", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_722");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_722", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_723");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_723", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_724");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_724", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_725");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_725", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_726");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_726", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_728");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_728", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_73");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_73", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_730");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_730", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_732");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_732", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_733");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_733", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_734");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_734", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_736");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_736", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_737");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_737", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_738");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_738", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_739");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_739", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_74");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_74", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_740");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_740", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_741");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_741", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_742");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_742", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_743");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_743", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_745");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_745", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_746");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_746", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_747");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_747", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_748");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_748", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_749");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_749", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_75");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_75", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_750");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_750", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_751");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_751", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_752");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_752", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_753");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_753", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_754");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_754", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_755");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_755", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_757");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_757", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_759");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_759", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_76");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_76", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_761");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_761", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_763");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_763", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_764");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_764", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_765");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_765", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_766");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_766", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_767");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_767", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_768");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_768", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_77");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_77", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_772");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_772", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_773");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_773", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_774");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_774", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_775");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_775", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_776");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_776", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_777");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_777", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_778");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_778", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_779");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_779", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_78");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_78", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_781");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_781", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_782");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_782", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_783");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_783", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_784");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_784", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_785");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_785", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_786");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_786", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_788");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_788", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_79");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_79", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_790");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_790", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_792");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_792", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_793");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_793", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_794");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_794", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_796");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_796", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_797");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_797", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_798");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_798", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_799");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_799", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_8");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_8", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_80");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_80", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_800");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_800", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_801");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_801", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_802");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_802", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_803");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_803", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_805");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_805", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_806");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_806", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_807");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_807", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_808");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_808", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_809");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_809", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_81");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_81", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_810");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_810", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_811");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_811", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_812");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_812", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_813");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_813", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_814");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_814", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_815");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_815", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_817");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_817", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_819");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_819", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_82");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_82", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_821");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_821", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_823");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_823", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_824");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_824", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_825");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_825", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_826");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_826", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_827");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_827", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_828");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_828", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_83");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_83", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_832");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_832", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_833");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_833", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_834");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_834", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_835");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_835", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_836");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_836", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_837");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_837", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_838");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_838", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_839");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_839", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_84");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_84", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_841");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_841", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_842");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_842", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_843");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_843", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_844");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_844", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_845");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_845", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_846");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_846", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_848");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_848", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_85");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_85", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_850");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_850", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_852");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_852", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_853");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_853", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_854");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_854", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_856");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_856", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_857");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_857", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_858");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_858", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_859");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_859", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_86");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_86", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_860");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_860", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_861");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_861", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_862");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_862", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_863");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_863", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_865");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_865", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_866");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_866", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_867");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_867", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_868");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_868", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_869");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_869", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_87");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_87", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_870");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_870", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_871");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_871", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_872");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_872", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_873");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_873", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_874");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_874", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_875");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_875", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_877");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_877", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_879");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_879", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_88");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_88", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_881");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_881", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_883");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_883", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_884");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_884", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_885");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_885", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_886");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_886", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_887");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_887", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_888");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_888", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_89");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_89", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_892");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_892", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_893");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_893", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_894");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_894", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_895");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_895", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_896");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_896", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_897");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_897", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_898");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_898", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_899");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_899", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_9");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_9", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_901");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_901", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_902");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_902", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_903");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_903", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_904");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_904", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_905");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_905", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_906");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_906", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_908");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_908", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_91");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_91", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_910");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_910", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_912");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_912", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_913");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_913", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_914");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_914", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_916");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_916", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_917");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_917", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_918");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_918", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_919");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_919", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_92");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_92", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_920");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_920", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_921");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_921", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_922");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_922", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_923");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_923", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_925");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_925", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_926");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_926", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_927");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_927", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_928");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_928", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_929");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_929", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_93");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_93", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_930");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_930", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_931");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_931", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_932");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_932", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_933");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_933", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_934");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_934", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_935");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_935", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_937");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_937", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_939");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_939", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_94");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_94", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_941");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_941", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_943");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_943", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_944");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_944", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_945");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_945", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_946");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_946", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_947");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_947", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_948");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_948", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_95");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_95", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_952");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_952", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_953");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_953", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_954");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_954", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_955");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_955", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_956");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_956", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_957");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_957", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_958");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_958", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_959");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_959", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_961");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_961", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_962");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_962", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_963");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_963", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_964");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_964", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_965");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_965", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_966");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_966", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_968");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_968", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_97");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_97", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_970");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_970", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_972");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_972", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_973");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_973", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_974");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_974", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_976");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_976", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_977");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_977", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_978");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_978", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_979");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_979", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_980");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_980", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_981");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_981", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_982");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_982", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_983");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_983", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_985");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_985", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_986");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_986", builder.constant({ dataType: "int64", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_987");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_987", builder.constant({ dataType: "int64", shape: [5] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_988");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_988", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_989");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_989", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_99");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_99", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_990");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_990", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_991");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_991", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_992");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_992", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_993");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_993", builder.constant({ dataType: "int64", shape: [2] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_994");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_994", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_995");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_995", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_997");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_997", builder.constant({ dataType: "int64", shape: [4] }, buf));
  }
  {
    const sl = weights.getSlice("Inserted_999");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("Inserted_999", builder.constant({ dataType: "int64", shape: [3] }, buf));
  }
  {
    const sl = weights.getSlice("_100");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_100", builder.constant({ dataType: "float16", shape: [2048, 176, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_101");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_101", builder.constant({ dataType: "uint4", shape: [2048, 176, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1016");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1016", builder.constant({ dataType: "uint4", shape: [2048, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_1017");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1017", builder.constant({ dataType: "float16", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1018");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1018", builder.constant({ dataType: "uint4", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1029");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1029", builder.constant({ dataType: "float16", shape: [1, 1, 2, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_105");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_105", builder.constant({ dataType: "uint4", shape: [5632, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_1057");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1057", builder.constant({ dataType: "float16", shape: [2048] }, buf));
  }
  {
    const sl = weights.getSlice("_106");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_106", builder.constant({ dataType: "float16", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1062");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1062", builder.constant({ dataType: "uint4", shape: [5632, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_1063");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1063", builder.constant({ dataType: "float16", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1064");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1064", builder.constant({ dataType: "uint4", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_107");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_107", builder.constant({ dataType: "uint4", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1082");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1082", builder.constant({ dataType: "float16", shape: [2048] }, buf));
  }
  {
    const sl = weights.getSlice("_1105");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1105", builder.constant({ dataType: "uint4", shape: [256, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_1106");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1106", builder.constant({ dataType: "float16", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1107");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1107", builder.constant({ dataType: "uint4", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_111");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_111", builder.constant({ dataType: "uint4", shape: [2048, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_1118");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1118", builder.constant({ dataType: "float16", shape: [1, 1, 2, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_112");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_112", builder.constant({ dataType: "float16", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_113");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_113", builder.constant({ dataType: "uint4", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1135");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1135", builder.constant({ dataType: "uint4", shape: [2048, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_1136");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1136", builder.constant({ dataType: "float16", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1137");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1137", builder.constant({ dataType: "uint4", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1148");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1148", builder.constant({ dataType: "float16", shape: [1, 1, 2, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_117");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_117", builder.constant({ dataType: "uint4", shape: [256, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_1176");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1176", builder.constant({ dataType: "float16", shape: [2048] }, buf));
  }
  {
    const sl = weights.getSlice("_118");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_118", builder.constant({ dataType: "float16", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1181");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1181", builder.constant({ dataType: "uint4", shape: [5632, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_1182");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1182", builder.constant({ dataType: "float16", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1183");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1183", builder.constant({ dataType: "uint4", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_119");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_119", builder.constant({ dataType: "uint4", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1201");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1201", builder.constant({ dataType: "float16", shape: [2048] }, buf));
  }
  {
    const sl = weights.getSlice("_1224");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1224", builder.constant({ dataType: "uint4", shape: [256, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_1225");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1225", builder.constant({ dataType: "float16", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1226");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1226", builder.constant({ dataType: "uint4", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_123");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_123", builder.constant({ dataType: "uint4", shape: [2048, 176, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_1237");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1237", builder.constant({ dataType: "float16", shape: [1, 1, 2, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_124");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_124", builder.constant({ dataType: "float16", shape: [2048, 176, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_125");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_125", builder.constant({ dataType: "uint4", shape: [2048, 176, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1254");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1254", builder.constant({ dataType: "uint4", shape: [2048, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_1255");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1255", builder.constant({ dataType: "float16", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1256");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1256", builder.constant({ dataType: "uint4", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1267");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1267", builder.constant({ dataType: "float16", shape: [1, 1, 2, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_129");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_129", builder.constant({ dataType: "uint4", shape: [5632, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_1295");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1295", builder.constant({ dataType: "float16", shape: [2048] }, buf));
  }
  {
    const sl = weights.getSlice("_130");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_130", builder.constant({ dataType: "float16", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1300");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1300", builder.constant({ dataType: "uint4", shape: [5632, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_1301");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1301", builder.constant({ dataType: "float16", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1302");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1302", builder.constant({ dataType: "uint4", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_131");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_131", builder.constant({ dataType: "uint4", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1320");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1320", builder.constant({ dataType: "float16", shape: [2048] }, buf));
  }
  {
    const sl = weights.getSlice("_1343");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1343", builder.constant({ dataType: "uint4", shape: [256, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_1344");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1344", builder.constant({ dataType: "float16", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1345");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1345", builder.constant({ dataType: "uint4", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_135");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_135", builder.constant({ dataType: "uint4", shape: [2048, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_1356");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1356", builder.constant({ dataType: "float16", shape: [1, 1, 2, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_136");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_136", builder.constant({ dataType: "float16", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_137");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_137", builder.constant({ dataType: "uint4", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1373");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1373", builder.constant({ dataType: "uint4", shape: [2048, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_1374");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1374", builder.constant({ dataType: "float16", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1375");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1375", builder.constant({ dataType: "uint4", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1386");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1386", builder.constant({ dataType: "float16", shape: [1, 1, 2, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_141");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_141", builder.constant({ dataType: "uint4", shape: [256, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_1414");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1414", builder.constant({ dataType: "float16", shape: [2048] }, buf));
  }
  {
    const sl = weights.getSlice("_1419");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1419", builder.constant({ dataType: "uint4", shape: [5632, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_142");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_142", builder.constant({ dataType: "float16", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1420");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1420", builder.constant({ dataType: "float16", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1421");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1421", builder.constant({ dataType: "uint4", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_143");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_143", builder.constant({ dataType: "uint4", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1439");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1439", builder.constant({ dataType: "float16", shape: [2048] }, buf));
  }
  {
    const sl = weights.getSlice("_1462");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1462", builder.constant({ dataType: "uint4", shape: [256, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_1463");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1463", builder.constant({ dataType: "float16", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1464");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1464", builder.constant({ dataType: "uint4", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_147");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_147", builder.constant({ dataType: "uint4", shape: [2048, 176, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_1475");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1475", builder.constant({ dataType: "float16", shape: [1, 1, 2, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_148");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_148", builder.constant({ dataType: "float16", shape: [2048, 176, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_149");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_149", builder.constant({ dataType: "uint4", shape: [2048, 176, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1492");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1492", builder.constant({ dataType: "uint4", shape: [2048, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_1493");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1493", builder.constant({ dataType: "float16", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1494");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1494", builder.constant({ dataType: "uint4", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1505");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1505", builder.constant({ dataType: "float16", shape: [1, 1, 2, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_153");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_153", builder.constant({ dataType: "uint4", shape: [5632, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_1533");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1533", builder.constant({ dataType: "float16", shape: [2048] }, buf));
  }
  {
    const sl = weights.getSlice("_1538");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1538", builder.constant({ dataType: "uint4", shape: [5632, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_1539");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1539", builder.constant({ dataType: "float16", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_154");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_154", builder.constant({ dataType: "float16", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1540");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1540", builder.constant({ dataType: "uint4", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_155");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_155", builder.constant({ dataType: "uint4", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1558");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1558", builder.constant({ dataType: "float16", shape: [2048] }, buf));
  }
  {
    const sl = weights.getSlice("_1581");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1581", builder.constant({ dataType: "uint4", shape: [256, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_1582");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1582", builder.constant({ dataType: "float16", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1583");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1583", builder.constant({ dataType: "uint4", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_159");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_159", builder.constant({ dataType: "uint4", shape: [2048, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_1594");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1594", builder.constant({ dataType: "float16", shape: [1, 1, 2, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_160");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_160", builder.constant({ dataType: "float16", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_161");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_161", builder.constant({ dataType: "uint4", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1611");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1611", builder.constant({ dataType: "uint4", shape: [2048, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_1612");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1612", builder.constant({ dataType: "float16", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1613");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1613", builder.constant({ dataType: "uint4", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1624");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1624", builder.constant({ dataType: "float16", shape: [1, 1, 2, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_165");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_165", builder.constant({ dataType: "uint4", shape: [256, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_1652");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1652", builder.constant({ dataType: "float16", shape: [2048] }, buf));
  }
  {
    const sl = weights.getSlice("_1657");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1657", builder.constant({ dataType: "uint4", shape: [5632, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_1658");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1658", builder.constant({ dataType: "float16", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1659");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1659", builder.constant({ dataType: "uint4", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_166");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_166", builder.constant({ dataType: "float16", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_167");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_167", builder.constant({ dataType: "uint4", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1677");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1677", builder.constant({ dataType: "float16", shape: [2048] }, buf));
  }
  {
    const sl = weights.getSlice("_1700");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1700", builder.constant({ dataType: "uint4", shape: [256, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_1701");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1701", builder.constant({ dataType: "float16", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1702");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1702", builder.constant({ dataType: "uint4", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_171");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_171", builder.constant({ dataType: "uint4", shape: [2048, 176, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_1713");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1713", builder.constant({ dataType: "float16", shape: [1, 1, 2, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_172");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_172", builder.constant({ dataType: "float16", shape: [2048, 176, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_173");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_173", builder.constant({ dataType: "uint4", shape: [2048, 176, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1730");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1730", builder.constant({ dataType: "uint4", shape: [2048, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_1731");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1731", builder.constant({ dataType: "float16", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1732");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1732", builder.constant({ dataType: "uint4", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1743");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1743", builder.constant({ dataType: "float16", shape: [1, 1, 2, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_177");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_177", builder.constant({ dataType: "uint4", shape: [5632, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_1771");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1771", builder.constant({ dataType: "float16", shape: [2048] }, buf));
  }
  {
    const sl = weights.getSlice("_1776");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1776", builder.constant({ dataType: "uint4", shape: [5632, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_1777");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1777", builder.constant({ dataType: "float16", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1778");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1778", builder.constant({ dataType: "uint4", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_178");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_178", builder.constant({ dataType: "float16", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_179");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_179", builder.constant({ dataType: "uint4", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1796");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1796", builder.constant({ dataType: "float16", shape: [2048] }, buf));
  }
  {
    const sl = weights.getSlice("_1819");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1819", builder.constant({ dataType: "uint4", shape: [256, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_1820");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1820", builder.constant({ dataType: "float16", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1821");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1821", builder.constant({ dataType: "uint4", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_183");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_183", builder.constant({ dataType: "uint4", shape: [2048, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_1832");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1832", builder.constant({ dataType: "float16", shape: [1, 1, 2, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_184");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_184", builder.constant({ dataType: "float16", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1849");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1849", builder.constant({ dataType: "uint4", shape: [2048, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_185");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_185", builder.constant({ dataType: "uint4", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1850");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1850", builder.constant({ dataType: "float16", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1851");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1851", builder.constant({ dataType: "uint4", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1862");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1862", builder.constant({ dataType: "float16", shape: [1, 1, 2, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_189");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_189", builder.constant({ dataType: "uint4", shape: [256, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_1890");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1890", builder.constant({ dataType: "float16", shape: [2048] }, buf));
  }
  {
    const sl = weights.getSlice("_1895");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1895", builder.constant({ dataType: "uint4", shape: [5632, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_1896");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1896", builder.constant({ dataType: "float16", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1897");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1897", builder.constant({ dataType: "uint4", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_190");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_190", builder.constant({ dataType: "float16", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_191");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_191", builder.constant({ dataType: "uint4", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1915");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1915", builder.constant({ dataType: "float16", shape: [2048] }, buf));
  }
  {
    const sl = weights.getSlice("_1938");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1938", builder.constant({ dataType: "uint4", shape: [256, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_1939");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1939", builder.constant({ dataType: "float16", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1940");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1940", builder.constant({ dataType: "uint4", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_195");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_195", builder.constant({ dataType: "uint4", shape: [2048, 176, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_1951");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1951", builder.constant({ dataType: "float16", shape: [1, 1, 2, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_196");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_196", builder.constant({ dataType: "float16", shape: [2048, 176, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1968");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1968", builder.constant({ dataType: "uint4", shape: [2048, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_1969");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1969", builder.constant({ dataType: "float16", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_197");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_197", builder.constant({ dataType: "uint4", shape: [2048, 176, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1970");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1970", builder.constant({ dataType: "uint4", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_1981");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_1981", builder.constant({ dataType: "float16", shape: [1, 1, 2, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2009");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2009", builder.constant({ dataType: "float16", shape: [2048] }, buf));
  }
  {
    const sl = weights.getSlice("_201");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_201", builder.constant({ dataType: "uint4", shape: [5632, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_2014");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2014", builder.constant({ dataType: "uint4", shape: [5632, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_2015");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2015", builder.constant({ dataType: "float16", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2016");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2016", builder.constant({ dataType: "uint4", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_202");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_202", builder.constant({ dataType: "float16", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_203");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_203", builder.constant({ dataType: "uint4", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2034");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2034", builder.constant({ dataType: "float16", shape: [2048] }, buf));
  }
  {
    const sl = weights.getSlice("_2057");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2057", builder.constant({ dataType: "uint4", shape: [256, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_2058");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2058", builder.constant({ dataType: "float16", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2059");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2059", builder.constant({ dataType: "uint4", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_207");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_207", builder.constant({ dataType: "uint4", shape: [2048, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_2070");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2070", builder.constant({ dataType: "float16", shape: [1, 1, 2, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_208");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_208", builder.constant({ dataType: "float16", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2087");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2087", builder.constant({ dataType: "uint4", shape: [2048, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_2088");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2088", builder.constant({ dataType: "float16", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2089");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2089", builder.constant({ dataType: "uint4", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_209");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_209", builder.constant({ dataType: "uint4", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2100");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2100", builder.constant({ dataType: "float16", shape: [1, 1, 2, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2128");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2128", builder.constant({ dataType: "float16", shape: [2048] }, buf));
  }
  {
    const sl = weights.getSlice("_213");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_213", builder.constant({ dataType: "uint4", shape: [256, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_2133");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2133", builder.constant({ dataType: "uint4", shape: [5632, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_2134");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2134", builder.constant({ dataType: "float16", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2135");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2135", builder.constant({ dataType: "uint4", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_214");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_214", builder.constant({ dataType: "float16", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_215");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_215", builder.constant({ dataType: "uint4", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2153");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2153", builder.constant({ dataType: "float16", shape: [2048] }, buf));
  }
  {
    const sl = weights.getSlice("_2176");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2176", builder.constant({ dataType: "uint4", shape: [256, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_2177");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2177", builder.constant({ dataType: "float16", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2178");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2178", builder.constant({ dataType: "uint4", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2189");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2189", builder.constant({ dataType: "float16", shape: [1, 1, 2, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_219");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_219", builder.constant({ dataType: "uint4", shape: [2048, 176, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_220");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_220", builder.constant({ dataType: "float16", shape: [2048, 176, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2206");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2206", builder.constant({ dataType: "uint4", shape: [2048, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_2207");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2207", builder.constant({ dataType: "float16", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2208");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2208", builder.constant({ dataType: "uint4", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_221");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_221", builder.constant({ dataType: "uint4", shape: [2048, 176, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2219");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2219", builder.constant({ dataType: "float16", shape: [1, 1, 2, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2247");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2247", builder.constant({ dataType: "float16", shape: [2048] }, buf));
  }
  {
    const sl = weights.getSlice("_225");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_225", builder.constant({ dataType: "uint4", shape: [5632, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_2252");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2252", builder.constant({ dataType: "uint4", shape: [5632, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_2253");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2253", builder.constant({ dataType: "float16", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2254");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2254", builder.constant({ dataType: "uint4", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_226");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_226", builder.constant({ dataType: "float16", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_227");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_227", builder.constant({ dataType: "uint4", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2272");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2272", builder.constant({ dataType: "float16", shape: [2048] }, buf));
  }
  {
    const sl = weights.getSlice("_2295");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2295", builder.constant({ dataType: "uint4", shape: [256, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_2296");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2296", builder.constant({ dataType: "float16", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2297");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2297", builder.constant({ dataType: "uint4", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2308");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2308", builder.constant({ dataType: "float16", shape: [1, 1, 2, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_231");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_231", builder.constant({ dataType: "uint4", shape: [2048, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_232");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_232", builder.constant({ dataType: "float16", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2325");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2325", builder.constant({ dataType: "uint4", shape: [2048, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_2326");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2326", builder.constant({ dataType: "float16", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2327");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2327", builder.constant({ dataType: "uint4", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_233");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_233", builder.constant({ dataType: "uint4", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2338");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2338", builder.constant({ dataType: "float16", shape: [1, 1, 2, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2366");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2366", builder.constant({ dataType: "float16", shape: [2048] }, buf));
  }
  {
    const sl = weights.getSlice("_237");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_237", builder.constant({ dataType: "uint4", shape: [256, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_2371");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2371", builder.constant({ dataType: "uint4", shape: [5632, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_2372");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2372", builder.constant({ dataType: "float16", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2373");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2373", builder.constant({ dataType: "uint4", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_238");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_238", builder.constant({ dataType: "float16", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_239");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_239", builder.constant({ dataType: "uint4", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2391");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2391", builder.constant({ dataType: "float16", shape: [2048] }, buf));
  }
  {
    const sl = weights.getSlice("_2414");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2414", builder.constant({ dataType: "uint4", shape: [256, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_2415");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2415", builder.constant({ dataType: "float16", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2416");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2416", builder.constant({ dataType: "uint4", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2427");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2427", builder.constant({ dataType: "float16", shape: [1, 1, 2, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_243");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_243", builder.constant({ dataType: "uint4", shape: [2048, 176, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_244");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_244", builder.constant({ dataType: "float16", shape: [2048, 176, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2444");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2444", builder.constant({ dataType: "uint4", shape: [2048, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_2445");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2445", builder.constant({ dataType: "float16", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2446");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2446", builder.constant({ dataType: "uint4", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_245");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_245", builder.constant({ dataType: "uint4", shape: [2048, 176, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2457");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2457", builder.constant({ dataType: "float16", shape: [1, 1, 2, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2485");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2485", builder.constant({ dataType: "float16", shape: [2048] }, buf));
  }
  {
    const sl = weights.getSlice("_249");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_249", builder.constant({ dataType: "uint4", shape: [5632, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_2490");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2490", builder.constant({ dataType: "uint4", shape: [5632, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_2491");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2491", builder.constant({ dataType: "float16", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2492");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2492", builder.constant({ dataType: "uint4", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_250");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_250", builder.constant({ dataType: "float16", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_251");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_251", builder.constant({ dataType: "uint4", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2510");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2510", builder.constant({ dataType: "float16", shape: [2048] }, buf));
  }
  {
    const sl = weights.getSlice("_2533");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2533", builder.constant({ dataType: "uint4", shape: [256, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_2534");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2534", builder.constant({ dataType: "float16", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2535");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2535", builder.constant({ dataType: "uint4", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2546");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2546", builder.constant({ dataType: "float16", shape: [1, 1, 2, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_255");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_255", builder.constant({ dataType: "uint4", shape: [2048, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_256");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_256", builder.constant({ dataType: "float16", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2563");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2563", builder.constant({ dataType: "uint4", shape: [2048, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_2564");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2564", builder.constant({ dataType: "float16", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2565");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2565", builder.constant({ dataType: "uint4", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_257");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_257", builder.constant({ dataType: "uint4", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2576");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2576", builder.constant({ dataType: "float16", shape: [1, 1, 2, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2604");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2604", builder.constant({ dataType: "float16", shape: [2048] }, buf));
  }
  {
    const sl = weights.getSlice("_2609");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2609", builder.constant({ dataType: "uint4", shape: [5632, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_261");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_261", builder.constant({ dataType: "uint4", shape: [256, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_2610");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2610", builder.constant({ dataType: "float16", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2611");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2611", builder.constant({ dataType: "uint4", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_262");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_262", builder.constant({ dataType: "float16", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2629");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2629", builder.constant({ dataType: "float16", shape: [2048] }, buf));
  }
  {
    const sl = weights.getSlice("_263");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_263", builder.constant({ dataType: "uint4", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2652");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2652", builder.constant({ dataType: "uint4", shape: [256, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_2653");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2653", builder.constant({ dataType: "float16", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2654");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2654", builder.constant({ dataType: "uint4", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2665");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2665", builder.constant({ dataType: "float16", shape: [1, 1, 2, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_267");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_267", builder.constant({ dataType: "uint4", shape: [2048, 176, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_268");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_268", builder.constant({ dataType: "float16", shape: [2048, 176, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2682");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2682", builder.constant({ dataType: "uint4", shape: [2048, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_2683");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2683", builder.constant({ dataType: "float16", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2684");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2684", builder.constant({ dataType: "uint4", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_269");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_269", builder.constant({ dataType: "uint4", shape: [2048, 176, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2695");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2695", builder.constant({ dataType: "float16", shape: [1, 1, 2, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2723");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2723", builder.constant({ dataType: "float16", shape: [2048] }, buf));
  }
  {
    const sl = weights.getSlice("_2728");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2728", builder.constant({ dataType: "uint4", shape: [5632, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_2729");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2729", builder.constant({ dataType: "float16", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_273");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_273", builder.constant({ dataType: "uint4", shape: [5632, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_2730");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2730", builder.constant({ dataType: "uint4", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_274");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_274", builder.constant({ dataType: "float16", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2748");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2748", builder.constant({ dataType: "float16", shape: [2048] }, buf));
  }
  {
    const sl = weights.getSlice("_275");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_275", builder.constant({ dataType: "uint4", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2771");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2771", builder.constant({ dataType: "uint4", shape: [256, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_2772");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2772", builder.constant({ dataType: "float16", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2773");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2773", builder.constant({ dataType: "uint4", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2784");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2784", builder.constant({ dataType: "float16", shape: [1, 1, 2, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_279");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_279", builder.constant({ dataType: "uint4", shape: [2048, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_280");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_280", builder.constant({ dataType: "float16", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2801");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2801", builder.constant({ dataType: "uint4", shape: [2048, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_2802");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2802", builder.constant({ dataType: "float16", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2803");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2803", builder.constant({ dataType: "uint4", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_281");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_281", builder.constant({ dataType: "uint4", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2814");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2814", builder.constant({ dataType: "float16", shape: [1, 1, 2, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2842");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2842", builder.constant({ dataType: "float16", shape: [2048] }, buf));
  }
  {
    const sl = weights.getSlice("_2847");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2847", builder.constant({ dataType: "uint4", shape: [5632, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_2848");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2848", builder.constant({ dataType: "float16", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2849");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2849", builder.constant({ dataType: "uint4", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_285");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_285", builder.constant({ dataType: "uint4", shape: [256, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_286");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_286", builder.constant({ dataType: "float16", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2867");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2867", builder.constant({ dataType: "float16", shape: [2048] }, buf));
  }
  {
    const sl = weights.getSlice("_287");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_287", builder.constant({ dataType: "uint4", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2890");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2890", builder.constant({ dataType: "uint4", shape: [256, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_2891");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2891", builder.constant({ dataType: "float16", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2892");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2892", builder.constant({ dataType: "uint4", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2903");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2903", builder.constant({ dataType: "float16", shape: [1, 1, 2, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_291");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_291", builder.constant({ dataType: "uint4", shape: [2048, 176, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_292");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_292", builder.constant({ dataType: "float16", shape: [2048, 176, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2920");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2920", builder.constant({ dataType: "uint4", shape: [2048, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_2921");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2921", builder.constant({ dataType: "float16", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2922");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2922", builder.constant({ dataType: "uint4", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_293");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_293", builder.constant({ dataType: "uint4", shape: [2048, 176, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2933");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2933", builder.constant({ dataType: "float16", shape: [1, 1, 2, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2961");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2961", builder.constant({ dataType: "float16", shape: [2048] }, buf));
  }
  {
    const sl = weights.getSlice("_2966");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2966", builder.constant({ dataType: "uint4", shape: [5632, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_2967");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2967", builder.constant({ dataType: "float16", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2968");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2968", builder.constant({ dataType: "uint4", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_297");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_297", builder.constant({ dataType: "uint4", shape: [5632, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_298");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_298", builder.constant({ dataType: "float16", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_2986");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_2986", builder.constant({ dataType: "float16", shape: [2048] }, buf));
  }
  {
    const sl = weights.getSlice("_299");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_299", builder.constant({ dataType: "uint4", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_3009");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_3009", builder.constant({ dataType: "uint4", shape: [256, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_3010");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_3010", builder.constant({ dataType: "float16", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_3011");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_3011", builder.constant({ dataType: "uint4", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_3022");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_3022", builder.constant({ dataType: "float16", shape: [1, 1, 2, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_303");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_303", builder.constant({ dataType: "uint4", shape: [2048, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_3039");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_3039", builder.constant({ dataType: "uint4", shape: [2048, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_304");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_304", builder.constant({ dataType: "float16", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_3040");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_3040", builder.constant({ dataType: "float16", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_3041");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_3041", builder.constant({ dataType: "uint4", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_305");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_305", builder.constant({ dataType: "uint4", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_3052");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_3052", builder.constant({ dataType: "float16", shape: [1, 1, 2, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_3080");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_3080", builder.constant({ dataType: "float16", shape: [2048] }, buf));
  }
  {
    const sl = weights.getSlice("_3085");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_3085", builder.constant({ dataType: "uint4", shape: [5632, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_3086");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_3086", builder.constant({ dataType: "float16", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_3087");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_3087", builder.constant({ dataType: "uint4", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_309");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_309", builder.constant({ dataType: "uint4", shape: [256, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_310");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_310", builder.constant({ dataType: "float16", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_3105");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_3105", builder.constant({ dataType: "float16", shape: [2048] }, buf));
  }
  {
    const sl = weights.getSlice("_311");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_311", builder.constant({ dataType: "uint4", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_3128");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_3128", builder.constant({ dataType: "uint4", shape: [256, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_3129");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_3129", builder.constant({ dataType: "float16", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_3130");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_3130", builder.constant({ dataType: "uint4", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_3141");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_3141", builder.constant({ dataType: "float16", shape: [1, 1, 2, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_315");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_315", builder.constant({ dataType: "uint4", shape: [2048, 176, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_3158");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_3158", builder.constant({ dataType: "uint4", shape: [2048, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_3159");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_3159", builder.constant({ dataType: "float16", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_316");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_316", builder.constant({ dataType: "float16", shape: [2048, 176, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_3160");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_3160", builder.constant({ dataType: "uint4", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_317");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_317", builder.constant({ dataType: "uint4", shape: [2048, 176, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_3171");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_3171", builder.constant({ dataType: "float16", shape: [1, 1, 2, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_3199");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_3199", builder.constant({ dataType: "float16", shape: [2048] }, buf));
  }
  {
    const sl = weights.getSlice("_3204");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_3204", builder.constant({ dataType: "uint4", shape: [5632, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_3205");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_3205", builder.constant({ dataType: "float16", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_3206");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_3206", builder.constant({ dataType: "uint4", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_321");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_321", builder.constant({ dataType: "uint4", shape: [5632, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_322");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_322", builder.constant({ dataType: "float16", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_3224");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_3224", builder.constant({ dataType: "float16", shape: [2048] }, buf));
  }
  {
    const sl = weights.getSlice("_323");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_323", builder.constant({ dataType: "uint4", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_327");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_327", builder.constant({ dataType: "uint4", shape: [2048, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_328");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_328", builder.constant({ dataType: "float16", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_329");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_329", builder.constant({ dataType: "uint4", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_333");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_333", builder.constant({ dataType: "uint4", shape: [256, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_334");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_334", builder.constant({ dataType: "float16", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_335");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_335", builder.constant({ dataType: "uint4", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_339");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_339", builder.constant({ dataType: "uint4", shape: [2048, 176, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_340");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_340", builder.constant({ dataType: "float16", shape: [2048, 176, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_341");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_341", builder.constant({ dataType: "uint4", shape: [2048, 176, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_345");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_345", builder.constant({ dataType: "uint4", shape: [5632, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_346");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_346", builder.constant({ dataType: "float16", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_347");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_347", builder.constant({ dataType: "uint4", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_351");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_351", builder.constant({ dataType: "uint4", shape: [2048, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_352");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_352", builder.constant({ dataType: "float16", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_353");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_353", builder.constant({ dataType: "uint4", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_357");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_357", builder.constant({ dataType: "uint4", shape: [256, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_358");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_358", builder.constant({ dataType: "float16", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_359");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_359", builder.constant({ dataType: "uint4", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_363");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_363", builder.constant({ dataType: "uint4", shape: [2048, 176, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_364");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_364", builder.constant({ dataType: "float16", shape: [2048, 176, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_365");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_365", builder.constant({ dataType: "uint4", shape: [2048, 176, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_369");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_369", builder.constant({ dataType: "uint4", shape: [5632, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_370");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_370", builder.constant({ dataType: "float16", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_371");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_371", builder.constant({ dataType: "uint4", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_375");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_375", builder.constant({ dataType: "uint4", shape: [2048, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_376");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_376", builder.constant({ dataType: "float16", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_377");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_377", builder.constant({ dataType: "uint4", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_381");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_381", builder.constant({ dataType: "uint4", shape: [256, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_382");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_382", builder.constant({ dataType: "float16", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_383");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_383", builder.constant({ dataType: "uint4", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_387");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_387", builder.constant({ dataType: "uint4", shape: [2048, 176, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_388");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_388", builder.constant({ dataType: "float16", shape: [2048, 176, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_389");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_389", builder.constant({ dataType: "uint4", shape: [2048, 176, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_393");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_393", builder.constant({ dataType: "uint4", shape: [5632, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_394");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_394", builder.constant({ dataType: "float16", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_395");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_395", builder.constant({ dataType: "uint4", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_399");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_399", builder.constant({ dataType: "uint4", shape: [2048, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_400");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_400", builder.constant({ dataType: "float16", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_401");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_401", builder.constant({ dataType: "uint4", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_405");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_405", builder.constant({ dataType: "uint4", shape: [256, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_406");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_406", builder.constant({ dataType: "float16", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_407");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_407", builder.constant({ dataType: "uint4", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_411");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_411", builder.constant({ dataType: "uint4", shape: [2048, 176, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_412");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_412", builder.constant({ dataType: "float16", shape: [2048, 176, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_413");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_413", builder.constant({ dataType: "uint4", shape: [2048, 176, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_417");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_417", builder.constant({ dataType: "uint4", shape: [5632, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_418");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_418", builder.constant({ dataType: "float16", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_419");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_419", builder.constant({ dataType: "uint4", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_423");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_423", builder.constant({ dataType: "uint4", shape: [2048, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_424");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_424", builder.constant({ dataType: "float16", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_425");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_425", builder.constant({ dataType: "uint4", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_429");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_429", builder.constant({ dataType: "uint4", shape: [256, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_430");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_430", builder.constant({ dataType: "float16", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_431");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_431", builder.constant({ dataType: "uint4", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_435");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_435", builder.constant({ dataType: "uint4", shape: [2048, 176, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_436");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_436", builder.constant({ dataType: "float16", shape: [2048, 176, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_437");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_437", builder.constant({ dataType: "uint4", shape: [2048, 176, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_441");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_441", builder.constant({ dataType: "uint4", shape: [5632, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_442");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_442", builder.constant({ dataType: "float16", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_443");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_443", builder.constant({ dataType: "uint4", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_447");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_447", builder.constant({ dataType: "uint4", shape: [2048, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_448");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_448", builder.constant({ dataType: "float16", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_449");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_449", builder.constant({ dataType: "uint4", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_45");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_45", builder.constant({ dataType: "uint4", shape: [32000, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_453");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_453", builder.constant({ dataType: "uint4", shape: [256, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_454");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_454", builder.constant({ dataType: "float16", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_455");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_455", builder.constant({ dataType: "uint4", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_459");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_459", builder.constant({ dataType: "uint4", shape: [2048, 176, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_46");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_46", builder.constant({ dataType: "float16", shape: [32000, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_460");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_460", builder.constant({ dataType: "float16", shape: [2048, 176, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_461");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_461", builder.constant({ dataType: "uint4", shape: [2048, 176, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_465");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_465", builder.constant({ dataType: "uint4", shape: [5632, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_466");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_466", builder.constant({ dataType: "float16", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_467");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_467", builder.constant({ dataType: "uint4", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_47");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_47", builder.constant({ dataType: "uint4", shape: [32000, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_471");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_471", builder.constant({ dataType: "uint4", shape: [2048, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_472");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_472", builder.constant({ dataType: "float16", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_473");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_473", builder.constant({ dataType: "uint4", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_477");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_477", builder.constant({ dataType: "uint4", shape: [256, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_478");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_478", builder.constant({ dataType: "float16", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_479");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_479", builder.constant({ dataType: "uint4", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_483");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_483", builder.constant({ dataType: "uint4", shape: [2048, 176, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_484");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_484", builder.constant({ dataType: "float16", shape: [2048, 176, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_485");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_485", builder.constant({ dataType: "uint4", shape: [2048, 176, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_489");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_489", builder.constant({ dataType: "uint4", shape: [5632, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_490");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_490", builder.constant({ dataType: "float16", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_491");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_491", builder.constant({ dataType: "uint4", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_495");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_495", builder.constant({ dataType: "uint4", shape: [2048, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_496");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_496", builder.constant({ dataType: "float16", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_497");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_497", builder.constant({ dataType: "uint4", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_501");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_501", builder.constant({ dataType: "uint4", shape: [256, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_502");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_502", builder.constant({ dataType: "float16", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_503");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_503", builder.constant({ dataType: "uint4", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_507");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_507", builder.constant({ dataType: "uint4", shape: [2048, 176, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_508");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_508", builder.constant({ dataType: "float16", shape: [2048, 176, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_509");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_509", builder.constant({ dataType: "uint4", shape: [2048, 176, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_51");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_51", builder.constant({ dataType: "uint4", shape: [2048, 176, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_513");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_513", builder.constant({ dataType: "uint4", shape: [5632, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_514");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_514", builder.constant({ dataType: "float16", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_515");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_515", builder.constant({ dataType: "uint4", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_519");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_519", builder.constant({ dataType: "uint4", shape: [2048, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_52");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_52", builder.constant({ dataType: "float16", shape: [2048, 176, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_520");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_520", builder.constant({ dataType: "float16", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_521");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_521", builder.constant({ dataType: "uint4", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_525");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_525", builder.constant({ dataType: "uint4", shape: [256, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_526");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_526", builder.constant({ dataType: "float16", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_527");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_527", builder.constant({ dataType: "uint4", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_53");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_53", builder.constant({ dataType: "uint4", shape: [2048, 176, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_531");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_531", builder.constant({ dataType: "uint4", shape: [2048, 176, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_532");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_532", builder.constant({ dataType: "float16", shape: [2048, 176, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_533");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_533", builder.constant({ dataType: "uint4", shape: [2048, 176, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_537");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_537", builder.constant({ dataType: "uint4", shape: [5632, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_538");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_538", builder.constant({ dataType: "float16", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_539");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_539", builder.constant({ dataType: "uint4", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_543");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_543", builder.constant({ dataType: "uint4", shape: [2048, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_544");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_544", builder.constant({ dataType: "float16", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_545");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_545", builder.constant({ dataType: "uint4", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_549");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_549", builder.constant({ dataType: "uint4", shape: [256, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_550");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_550", builder.constant({ dataType: "float16", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_551");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_551", builder.constant({ dataType: "uint4", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_555");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_555", builder.constant({ dataType: "uint4", shape: [2048, 176, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_556");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_556", builder.constant({ dataType: "float16", shape: [2048, 176, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_557");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_557", builder.constant({ dataType: "uint4", shape: [2048, 176, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_561");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_561", builder.constant({ dataType: "uint4", shape: [5632, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_562");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_562", builder.constant({ dataType: "float16", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_563");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_563", builder.constant({ dataType: "uint4", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_567");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_567", builder.constant({ dataType: "uint4", shape: [2048, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_568");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_568", builder.constant({ dataType: "float16", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_569");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_569", builder.constant({ dataType: "uint4", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_57");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_57", builder.constant({ dataType: "uint4", shape: [5632, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_573");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_573", builder.constant({ dataType: "uint4", shape: [256, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_574");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_574", builder.constant({ dataType: "float16", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_575");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_575", builder.constant({ dataType: "uint4", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_579");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_579", builder.constant({ dataType: "float16", shape: [32000, 2048] }, buf));
  }
  {
    const sl = weights.getSlice("_58");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_58", builder.constant({ dataType: "float16", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_583");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_583", builder.constant({ dataType: "float32", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("_586");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_586", builder.constant({ dataType: "float32", shape: [] }, buf));
  }
  {
    const sl = weights.getSlice("_59");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_59", builder.constant({ dataType: "uint4", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_590");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_590", builder.constant({ dataType: "float16", shape: [2048] }, buf));
  }
  {
    const sl = weights.getSlice("_598");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_598", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("_601");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_601", builder.constant({ dataType: "uint8", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("_602");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_602", builder.constant({ dataType: "int32", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("_604");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_604", builder.constant({ dataType: "int32", shape: [512, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_606");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_606", builder.constant({ dataType: "int32", shape: [512, 2] }, buf));
  }
  {
    const sl = weights.getSlice("_613");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_613", builder.constant({ dataType: "int32", shape: [128] }, buf));
  }
  {
    const sl = weights.getSlice("_617");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_617", builder.constant({ dataType: "int32", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("_621");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_621", builder.constant({ dataType: "float16", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("_622");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_622", builder.constant({ dataType: "float16", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("_624");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_624", builder.constant({ dataType: "float16", shape: [2048, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_628");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_628", builder.constant({ dataType: "uint4", shape: [256, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_629");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_629", builder.constant({ dataType: "float16", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_63");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_63", builder.constant({ dataType: "uint4", shape: [2048, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_630");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_630", builder.constant({ dataType: "uint4", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_64");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_64", builder.constant({ dataType: "float16", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_641");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_641", builder.constant({ dataType: "float16", shape: [1, 1, 2, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_644");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_644", builder.constant({ dataType: "float16", shape: [2048, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_65");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_65", builder.constant({ dataType: "uint4", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_659");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_659", builder.constant({ dataType: "uint4", shape: [2048, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_660");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_660", builder.constant({ dataType: "float16", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_661");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_661", builder.constant({ dataType: "uint4", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_672");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_672", builder.constant({ dataType: "float16", shape: [1, 1, 2, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_684");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_684", builder.constant({ dataType: "float16", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("_69");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_69", builder.constant({ dataType: "uint4", shape: [256, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_70");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_70", builder.constant({ dataType: "float16", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_700");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_700", builder.constant({ dataType: "float16", shape: [2048] }, buf));
  }
  {
    const sl = weights.getSlice("_705");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_705", builder.constant({ dataType: "uint4", shape: [5632, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_706");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_706", builder.constant({ dataType: "float16", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_707");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_707", builder.constant({ dataType: "uint4", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_71");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_71", builder.constant({ dataType: "uint4", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_725");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_725", builder.constant({ dataType: "float16", shape: [2048] }, buf));
  }
  {
    const sl = weights.getSlice("_748");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_748", builder.constant({ dataType: "uint4", shape: [256, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_749");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_749", builder.constant({ dataType: "float16", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_75");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_75", builder.constant({ dataType: "uint4", shape: [2048, 176, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_750");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_750", builder.constant({ dataType: "uint4", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_76");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_76", builder.constant({ dataType: "float16", shape: [2048, 176, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_761");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_761", builder.constant({ dataType: "float16", shape: [1, 1, 2, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_77");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_77", builder.constant({ dataType: "uint4", shape: [2048, 176, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_778");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_778", builder.constant({ dataType: "uint4", shape: [2048, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_779");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_779", builder.constant({ dataType: "float16", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_780");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_780", builder.constant({ dataType: "uint4", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_791");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_791", builder.constant({ dataType: "float16", shape: [1, 1, 2, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_81");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_81", builder.constant({ dataType: "uint4", shape: [5632, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_819");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_819", builder.constant({ dataType: "float16", shape: [2048] }, buf));
  }
  {
    const sl = weights.getSlice("_82");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_82", builder.constant({ dataType: "float16", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_824");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_824", builder.constant({ dataType: "uint4", shape: [5632, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_825");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_825", builder.constant({ dataType: "float16", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_826");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_826", builder.constant({ dataType: "uint4", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_83");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_83", builder.constant({ dataType: "uint4", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_844");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_844", builder.constant({ dataType: "float16", shape: [2048] }, buf));
  }
  {
    const sl = weights.getSlice("_867");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_867", builder.constant({ dataType: "uint4", shape: [256, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_868");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_868", builder.constant({ dataType: "float16", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_869");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_869", builder.constant({ dataType: "uint4", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_87");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_87", builder.constant({ dataType: "uint4", shape: [2048, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_88");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_88", builder.constant({ dataType: "float16", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_880");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_880", builder.constant({ dataType: "float16", shape: [1, 1, 2, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_89");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_89", builder.constant({ dataType: "uint4", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_897");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_897", builder.constant({ dataType: "uint4", shape: [2048, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_898");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_898", builder.constant({ dataType: "float16", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_899");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_899", builder.constant({ dataType: "uint4", shape: [2048, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_910");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_910", builder.constant({ dataType: "float16", shape: [1, 1, 2, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_93");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_93", builder.constant({ dataType: "uint4", shape: [256, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_938");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_938", builder.constant({ dataType: "float16", shape: [2048] }, buf));
  }
  {
    const sl = weights.getSlice("_94");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_94", builder.constant({ dataType: "float16", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_943");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_943", builder.constant({ dataType: "uint4", shape: [5632, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_944");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_944", builder.constant({ dataType: "float16", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_945");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_945", builder.constant({ dataType: "uint4", shape: [5632, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_95");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_95", builder.constant({ dataType: "uint4", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_963");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_963", builder.constant({ dataType: "float16", shape: [2048] }, buf));
  }
  {
    const sl = weights.getSlice("_986");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_986", builder.constant({ dataType: "uint4", shape: [256, 64, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_987");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_987", builder.constant({ dataType: "float16", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_988");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_988", builder.constant({ dataType: "uint4", shape: [256, 64, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_99");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_99", builder.constant({ dataType: "uint4", shape: [2048, 176, 32] }, buf));
  }
  {
    const sl = weights.getSlice("_999");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_999", builder.constant({ dataType: "float16", shape: [1, 1, 2, 1] }, buf));
  }
  {
    const sl = weights.getSlice("_inlfunc_Softmax_axes");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_inlfunc_Softmax_axes", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("_inlfunc_Softmax_token_0_axes");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_inlfunc_Softmax_token_0_axes", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("_inlfunc_Softmax_token_10_axes");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_inlfunc_Softmax_token_10_axes", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("_inlfunc_Softmax_token_11_axes");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_inlfunc_Softmax_token_11_axes", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("_inlfunc_Softmax_token_12_axes");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_inlfunc_Softmax_token_12_axes", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("_inlfunc_Softmax_token_13_axes");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_inlfunc_Softmax_token_13_axes", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("_inlfunc_Softmax_token_14_axes");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_inlfunc_Softmax_token_14_axes", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("_inlfunc_Softmax_token_15_axes");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_inlfunc_Softmax_token_15_axes", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("_inlfunc_Softmax_token_16_axes");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_inlfunc_Softmax_token_16_axes", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("_inlfunc_Softmax_token_17_axes");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_inlfunc_Softmax_token_17_axes", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("_inlfunc_Softmax_token_18_axes");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_inlfunc_Softmax_token_18_axes", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("_inlfunc_Softmax_token_19_axes");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_inlfunc_Softmax_token_19_axes", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("_inlfunc_Softmax_token_1_axes");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_inlfunc_Softmax_token_1_axes", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("_inlfunc_Softmax_token_20_axes");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_inlfunc_Softmax_token_20_axes", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("_inlfunc_Softmax_token_2_axes");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_inlfunc_Softmax_token_2_axes", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("_inlfunc_Softmax_token_3_axes");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_inlfunc_Softmax_token_3_axes", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("_inlfunc_Softmax_token_4_axes");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_inlfunc_Softmax_token_4_axes", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("_inlfunc_Softmax_token_5_axes");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_inlfunc_Softmax_token_5_axes", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("_inlfunc_Softmax_token_6_axes");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_inlfunc_Softmax_token_6_axes", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("_inlfunc_Softmax_token_7_axes");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_inlfunc_Softmax_token_7_axes", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("_inlfunc_Softmax_token_8_axes");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_inlfunc_Softmax_token_8_axes", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }
  {
    const sl = weights.getSlice("_inlfunc_Softmax_token_9_axes");
    const buf = weights.buffer.slice(sl.byteOffset, sl.byteOffset + sl.byteLength);
    env.set("_inlfunc_Softmax_token_9_axes", builder.constant({ dataType: "int64", shape: [1] }, buf));
  }

  env.set("_48", builder.dequantizeLinear(env.get("_45"), env.get("_46"), env.get("_47"), {"axis":2,"blockSize":32,"label":"/lm_head/MatMul_Q4_dequantizeLinear_0"}));
  env.set("_49", builder.reshape(env.get("_48"), [32000,2048]));
  env.set("_50", builder["transpose"](env.get("_49"), {"label":"/lm_head/MatMul_Q4_transpose_dequantizeLinear_2","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__50", builder.cast(env.get("_50"), "float32"));
  env.set("_3225", builder.cast(env.get("_3224"), "float32"));
  env.set("Inserted_90", builder["clamp"](env.get("input_ids_580"), {"label":"Inserted_Clip_268","maxValue":31999,"minValue":-32000}));
  env.set("_581", builder["gather"](env.get("_579"), env.get("Inserted_90"), {"axis":0,"label":"/model/embed_tokens/Gather_267"}));
  env.set("_693", builder.cast(env.get("_581"), "float32"));
  env.set("_570", builder.dequantizeLinear(env.get("_567"), env.get("_568"), env.get("_569"), {"axis":2,"blockSize":32,"label":"/model/layers.0/attn/o_proj/MatMul_Q4_dequantizeLinear_261"}));
  env.set("_571", builder.reshape(env.get("_570"), [2048,2048]));
  env.set("_572", builder["transpose"](env.get("_571"), {"label":"/model/layers.0/attn/o_proj/MatMul_Q4_transpose_dequantizeLinear_263","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__572", builder.cast(env.get("_572"), "float32"));
  env.set("Inserted_96", builder.cast(env.get("_601"), "uint8"));
  env.set("_597", builder["reduceSum"](env.get("attention_mask_596"), {"axes":[1],"keepDimensions":true,"label":"/model/attn_mask_reformat/attn_mask_subgraph/ReduceSum_280"}));
  env.set("_599", builder["sub"](env.get("_597"), env.get("_598"), {"label":"/model/attn_mask_reformat/attn_mask_subgraph/Sub_281"}));
  env.set("_600", builder.cast(env.get("_599"), "int32"));
  env.set("_603", builder["where"](env.get("Inserted_96"), env.get("_602"), env.get("_600"), {"label":"/model/layers.0/attn/GroupQueryAttention_/GQA/scatter/where_283"}));
  env.set("_605", builder["add"](env.get("_604"), env.get("_603"), {"label":"/model/layers.0/attn/GroupQueryAttention_/GQA/right_constant/add_285"}));
  env.set("_607", builder.concat([env.get("_606"), env.get("_605")], 1, {"label":"/model/layers.0/attn/GroupQueryAttention_/GQA/concat_for_pre_scatter_indices_286"}));
  env.set("_608", builder.reshape(env.get("_607"), [1,128,4,3]));
  env.set("Inserted_98", builder.cast(env.get("_608"), "int64"));
  env.set("Inserted_100", builder["max"](env.get("Inserted_98"), env.get("Inserted_99"), {"label":"Inserted_Max_290"}));
  env.set("Inserted_102", builder["min"](env.get("Inserted_100"), env.get("Inserted_101"), {"label":"Inserted_Min_291"}));
  env.set("_576", builder.dequantizeLinear(env.get("_573"), env.get("_574"), env.get("_575"), {"axis":2,"blockSize":32,"label":"/model/layers.0/attn/v_proj/MatMul_Q4_dequantizeLinear_264"}));
  env.set("_577", builder.reshape(env.get("_576"), [256,2048]));
  env.set("_578", builder["transpose"](env.get("_577"), {"label":"/model/layers.0/attn/v_proj/MatMul_Q4_transpose_dequantizeLinear_266","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__578", builder.cast(env.get("_578"), "float32"));
  env.set("_591", builder.cast(env.get("_590"), "float32"));
  env.set("_582", builder.cast(env.get("_581"), "float32"));
  env.set("_584", builder["pow"](env.get("_582"), env.get("_583"), {"label":"/model/layers.0/input_layernorm/LayerNorm_pow_270"}));
  env.set("_585", builder["reduceMean"](env.get("_584"), {"axes":[2],"keepDimensions":true,"label":"/model/layers.0/input_layernorm/LayerNorm_reduceMean_271"}));
  env.set("_587", builder["add"](env.get("_585"), env.get("_586"), {"label":"/model/layers.0/input_layernorm/LayerNorm_add_272"}));
  env.set("_588", builder["sqrt"](env.get("_587"), {"label":"/model/layers.0/input_layernorm/LayerNorm_sqrt_273"}));
  env.set("_589", builder["div"](env.get("_582"), env.get("_588"), {"label":"/model/layers.0/input_layernorm/LayerNorm_div_274"}));
  env.set("_592", builder["mul"](env.get("_591"), env.get("_589"), {"label":"/model/layers.0/input_layernorm/LayerNorm_mul_276"}));
  env.set("InsertedPrecisionFreeCast__594", builder["matmul"](env.get("_592"), env.get("InsertedPrecisionFreeCast__578"), {"label":"/model/layers.0/attn/v_proj/MatMul_Q4_matmul_278"}));
  env.set("_594", builder.cast(env.get("InsertedPrecisionFreeCast__594"), "float16"));
  env.set("_595", builder.reshape(env.get("_594"), [1,128,4,64]));
  env.set("present_0_value_1", builder["scatterND"](env.get("past_key_values_0_value_609"), env.get("Inserted_102"), env.get("_595"), {"label":"/model/layers.0/attn/GroupQueryAttention_/GQA/present_value/ScatterND_288"}));
  env.set("_610", builder.reshape(env.get("present_0_value_1"), [1,4,1,512,64]));
  env.set("_611", builder.expand(env.get("_610"), [1,4,8,512,64], {"label":"/model/layers.0/attn/GroupQueryAttention_/GQA/true_present_value/expand_293"}));
  env.set("_612", builder.reshape(env.get("_611"), [1,32,512,64]));
  env.set("InsertedPrecisionFreeCast__612", builder.cast(env.get("_612"), "float32"));
  env.set("InsertedPrecisionFreeCast__684", builder.cast(env.get("_684"), "float32"));
  env.set("Inserted_127", builder.cast(env.get("_608"), "int64"));
  env.set("Inserted_129", builder["max"](env.get("Inserted_127"), env.get("Inserted_128"), {"label":"Inserted_Max_328"}));
  env.set("Inserted_131", builder["min"](env.get("Inserted_129"), env.get("Inserted_130"), {"label":"Inserted_Min_329"}));
  env.set("Inserted_120", builder["clamp"](env.get("position_ids_625"), {"label":"Inserted_Clip_319","maxValue":2047,"minValue":-2048}));
  env.set("_645", builder["gather"](env.get("_644"), env.get("Inserted_120"), {"axis":0,"label":"/model/layers.0/attn/k_rotary/RotaryEmbedding_gather_cos_318"}));
  env.set("_646", builder.reshape(env.get("_645"), [1,128,1,1,32]));
  env.set("InsertedPrecisionFreeCast__646", builder.cast(env.get("_646"), "float32"));
  env.set("_631", builder.dequantizeLinear(env.get("_628"), env.get("_629"), env.get("_630"), {"axis":2,"blockSize":32,"label":"/model/layers.0/attn/k_proj/MatMul_Q4_dequantizeLinear_307"}));
  env.set("_632", builder.reshape(env.get("_631"), [256,2048]));
  env.set("_633", builder["transpose"](env.get("_632"), {"label":"/model/layers.0/attn/k_proj/MatMul_Q4_transpose_dequantizeLinear_309","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__633", builder.cast(env.get("_633"), "float32"));
  env.set("InsertedPrecisionFreeCast__634", builder["matmul"](env.get("_592"), env.get("InsertedPrecisionFreeCast__633"), {"label":"/model/layers.0/attn/k_proj/MatMul_Q4_matmul_310"}));
  env.set("_634", builder.cast(env.get("InsertedPrecisionFreeCast__634"), "float16"));
  env.set("_635", builder.reshape(env.get("_634"), [1,128,4,64]));
  env.set("_636", builder.reshape(env.get("_635"), [1,128,4,2,32]));
  env.set("InsertedPrecisionFreeCast__636", builder.cast(env.get("_636"), "float32"));
  env.set("InsertedPrecisionFreeCast__647", builder["mul"](env.get("InsertedPrecisionFreeCast__636"), env.get("InsertedPrecisionFreeCast__646"), {"label":"/model/layers.0/attn/k_rotary/RotaryEmbedding_mul_cos_321"}));
  env.set("InsertedPrecisionFreeCast__648", builder.reshape(env.get("InsertedPrecisionFreeCast__647"), [1,128,4,64]));
  env.set("InsertedPrecisionFreeCast__641", builder.cast(env.get("_641"), "float32"));
  env.set("Inserted_111", builder["clamp"](env.get("position_ids_625"), {"label":"Inserted_Clip_305","maxValue":2047,"minValue":-2048}));
  env.set("_626", builder["gather"](env.get("_624"), env.get("Inserted_111"), {"axis":0,"label":"/model/layers.0/attn/k_rotary/RotaryEmbedding_gather_sin_304"}));
  env.set("_627", builder.reshape(env.get("_626"), [1,128,1,1,32]));
  env.set("InsertedPrecisionFreeCast__627", builder.cast(env.get("_627"), "float32"));
  {
    const tmp = builder.split(env.get("_636"), 2, {"axis":3,"label":"/model/layers.0/attn/k_rotary/RotaryEmbedding_split_partial_input0_313"});
    env.set("_637", tmp[0]);
    env.set("_638", tmp[1]);
  }
  env.set("_639", builder.concat([env.get("_638"), env.get("_637")], 3, {"label":"/model/layers.0/attn/k_rotary/RotaryEmbedding_concat_partial_input0_314"}));
  env.set("InsertedPrecisionFreeCast__639", builder.cast(env.get("_639"), "float32"));
  env.set("InsertedPrecisionFreeCast__640", builder["mul"](env.get("InsertedPrecisionFreeCast__639"), env.get("InsertedPrecisionFreeCast__627"), {"label":"/model/layers.0/attn/k_rotary/RotaryEmbedding_mul_sin_315"}));
  env.set("InsertedPrecisionFreeCast__642", builder["mul"](env.get("InsertedPrecisionFreeCast__640"), env.get("InsertedPrecisionFreeCast__641"), {"label":"/model/layers.0/attn/k_rotary/RotaryEmbedding_mul_sign_316"}));
  env.set("InsertedPrecisionFreeCast__643", builder.reshape(env.get("InsertedPrecisionFreeCast__642"), [1,128,4,64]));
  env.set("InsertedPrecisionFreeCast__649", builder["add"](env.get("InsertedPrecisionFreeCast__648"), env.get("InsertedPrecisionFreeCast__643"), {"label":"/model/layers.0/attn/k_rotary/RotaryEmbedding_add_mul_cos_sin_323"}));
  env.set("_649", builder.cast(env.get("InsertedPrecisionFreeCast__649"), "float16"));
  env.set("_650", builder.reshape(env.get("_649"), [1,128,256]));
  env.set("_651", builder.reshape(env.get("_650"), [1,128,4,64]));
  env.set("present_0_key_0", builder["scatterND"](env.get("past_key_values_0_key_652"), env.get("Inserted_131"), env.get("_651"), {"label":"/model/layers.0/attn/GroupQueryAttention_/GQA/present_key/ScatterND_326"}));
  env.set("_653", builder.reshape(env.get("present_0_key_0"), [1,4,1,512,64]));
  env.set("_654", builder.expand(env.get("_653"), [1,4,8,512,64], {"label":"/model/layers.0/attn/GroupQueryAttention_/GQA/true_present_key/expand_331"}));
  env.set("_655", builder.reshape(env.get("_654"), [1,32,512,64]));
  env.set("_656", builder["transpose"](env.get("_655"), {"label":"/model/layers.0/attn/GroupQueryAttention_/GQA/present_key/transpose_333","permutation":[0,1,3,2]}));
  env.set("InsertedPrecisionFreeCast__656", builder.cast(env.get("_656"), "float32"));
  env.set("Inserted_144", builder["clamp"](env.get("position_ids_625"), {"label":"Inserted_Clip_349","maxValue":2047,"minValue":-2048}));
  env.set("_675", builder["gather"](env.get("_644"), env.get("Inserted_144"), {"axis":0,"label":"/model/layers.0/attn/q_rotary/RotaryEmbedding_gather_cos_348"}));
  env.set("_676", builder.reshape(env.get("_675"), [1,128,1,1,32]));
  env.set("InsertedPrecisionFreeCast__676", builder.cast(env.get("_676"), "float32"));
  env.set("_662", builder.dequantizeLinear(env.get("_659"), env.get("_660"), env.get("_661"), {"axis":2,"blockSize":32,"label":"/model/layers.0/attn/q_proj/MatMul_Q4_dequantizeLinear_337"}));
  env.set("_663", builder.reshape(env.get("_662"), [2048,2048]));
  env.set("_664", builder["transpose"](env.get("_663"), {"label":"/model/layers.0/attn/q_proj/MatMul_Q4_transpose_dequantizeLinear_339","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__664", builder.cast(env.get("_664"), "float32"));
  env.set("InsertedPrecisionFreeCast__665", builder["matmul"](env.get("_592"), env.get("InsertedPrecisionFreeCast__664"), {"label":"/model/layers.0/attn/q_proj/MatMul_Q4_matmul_340"}));
  env.set("_665", builder.cast(env.get("InsertedPrecisionFreeCast__665"), "float16"));
  env.set("_666", builder.reshape(env.get("_665"), [1,128,32,64]));
  env.set("_667", builder.reshape(env.get("_666"), [1,128,32,2,32]));
  env.set("InsertedPrecisionFreeCast__667", builder.cast(env.get("_667"), "float32"));
  env.set("InsertedPrecisionFreeCast__677", builder["mul"](env.get("InsertedPrecisionFreeCast__667"), env.get("InsertedPrecisionFreeCast__676"), {"label":"/model/layers.0/attn/q_rotary/RotaryEmbedding_mul_cos_351"}));
  env.set("InsertedPrecisionFreeCast__678", builder.reshape(env.get("InsertedPrecisionFreeCast__677"), [1,128,32,64]));
  env.set("InsertedPrecisionFreeCast__672", builder.cast(env.get("_672"), "float32"));
  env.set("Inserted_135", builder["clamp"](env.get("position_ids_625"), {"label":"Inserted_Clip_335","maxValue":2047,"minValue":-2048}));
  env.set("_657", builder["gather"](env.get("_624"), env.get("Inserted_135"), {"axis":0,"label":"/model/layers.0/attn/q_rotary/RotaryEmbedding_gather_sin_334"}));
  env.set("_658", builder.reshape(env.get("_657"), [1,128,1,1,32]));
  env.set("InsertedPrecisionFreeCast__658", builder.cast(env.get("_658"), "float32"));
  {
    const tmp = builder.split(env.get("_667"), 2, {"axis":3,"label":"/model/layers.0/attn/q_rotary/RotaryEmbedding_split_partial_input0_343"});
    env.set("_668", tmp[0]);
    env.set("_669", tmp[1]);
  }
  env.set("_670", builder.concat([env.get("_669"), env.get("_668")], 3, {"label":"/model/layers.0/attn/q_rotary/RotaryEmbedding_concat_partial_input0_344"}));
  env.set("InsertedPrecisionFreeCast__670", builder.cast(env.get("_670"), "float32"));
  env.set("InsertedPrecisionFreeCast__671", builder["mul"](env.get("InsertedPrecisionFreeCast__670"), env.get("InsertedPrecisionFreeCast__658"), {"label":"/model/layers.0/attn/q_rotary/RotaryEmbedding_mul_sin_345"}));
  env.set("InsertedPrecisionFreeCast__673", builder["mul"](env.get("InsertedPrecisionFreeCast__671"), env.get("InsertedPrecisionFreeCast__672"), {"label":"/model/layers.0/attn/q_rotary/RotaryEmbedding_mul_sign_346"}));
  env.set("InsertedPrecisionFreeCast__674", builder.reshape(env.get("InsertedPrecisionFreeCast__673"), [1,128,32,64]));
  env.set("InsertedPrecisionFreeCast__679", builder["add"](env.get("InsertedPrecisionFreeCast__678"), env.get("InsertedPrecisionFreeCast__674"), {"label":"/model/layers.0/attn/q_rotary/RotaryEmbedding_add_mul_cos_sin_353"}));
  env.set("_679", builder.cast(env.get("InsertedPrecisionFreeCast__679"), "float16"));
  env.set("_680", builder.reshape(env.get("_679"), [1,128,2048]));
  env.set("_681", builder.reshape(env.get("_680"), [1,128,32,64]));
  env.set("_682", builder["transpose"](env.get("_681"), {"label":"/model/layers.0/attn/GroupQueryAttention_/GQA/query/transpose_356","permutation":[0,2,1,3]}));
  env.set("InsertedPrecisionFreeCast__682", builder.cast(env.get("_682"), "float32"));
  env.set("InsertedPrecisionFreeCast__683", builder["matmul"](env.get("InsertedPrecisionFreeCast__682"), env.get("InsertedPrecisionFreeCast__656"), {"label":"/model/layers.0/attn/GroupQueryAttention_/Attention/qkv/matmul_1_357"}));
  env.set("InsertedPrecisionFreeCast__685", builder["mul"](env.get("InsertedPrecisionFreeCast__683"), env.get("InsertedPrecisionFreeCast__684"), {"label":"/model/layers.0/attn/GroupQueryAttention_/Attention/qkv/div_358"}));
  env.set("InsertedPrecisionFreeCast__622", builder.cast(env.get("_622"), "float32"));
  env.set("InsertedPrecisionFreeCast__621", builder.cast(env.get("_621"), "float32"));
  env.set("_618", builder.expand(env.get("_617"), [1,32,128,512], {"label":"/model/layers.0/attn/GroupQueryAttention_/GQA/GQA_mask_shape_ones/expand_298"}));
  env.set("_619", builder.cumulativeSum(env.get("_618"), 3, {"exclusive":true,"label":"/model/layers.0/attn/GroupQueryAttention_range_of_mask_shape_299"}));
  env.set("_614", builder["add"](env.get("_613"), env.get("_603"), {"label":"/model/layers.0/attn/GroupQueryAttention_/GQA/attn_mask/add_295"}));
  env.set("_615", builder.expand(env.get("_614"), [512,128], {"label":"/model/layers.0/attn/GroupQueryAttention_/GQA/expand_neq_right_296"}));
  env.set("_616", builder["transpose"](env.get("_615"), {"label":"/model/layers.0/attn/GroupQueryAttention_/GQA/neq_right/transpose_297","permutation":[1,0]}));
  env.set("Inserted_109", builder["lesser"](env.get("_619"), env.get("_616"), {"label":"/model/layers.0/attn/GroupQueryAttention_/GQA/attn_mask/condition_1_300"}));
  env.set("InsertedPrecisionFreeCast__623", builder["where"](env.get("Inserted_109"), env.get("InsertedPrecisionFreeCast__621"), env.get("InsertedPrecisionFreeCast__622"), {"label":"/model/layers.0/attn/GroupQueryAttention_/GQA/attn_mask/where_302"}));
  env.set("InsertedPrecisionFreeCast__686", builder["add"](env.get("InsertedPrecisionFreeCast__685"), env.get("InsertedPrecisionFreeCast__623"), {"label":"/model/layers.0/attn/GroupQueryAttention_/Attention/attn_mask/softmax_input_359"}));
  env.set("InsertedPrecisionFreeCast__inlfunc_Softmax_X_ReduceMax", builder["reduceMax"](env.get("InsertedPrecisionFreeCast__686"), {"axes":[3],"keepDimensions":true}));
  env.set("InsertedPrecisionFreeCast__inlfunc_Softmax_X_Sub", builder["sub"](env.get("InsertedPrecisionFreeCast__686"), env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_X_ReduceMax"), {}));
  env.set("InsertedPrecisionFreeCast__inlfunc_Softmax_X_Exp", builder["exp"](env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_X_Sub"), {}));
  env.set("InsertedPrecisionFreeCast__inlfunc_Softmax_X_ReduceSum", builder["reduceSum"](env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_X_Exp"), {"axes":[3],"keepDimensions":true}));
  env.set("InsertedPrecisionFreeCast__687", builder["div"](env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_X_Exp"), env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_X_ReduceSum"), {}));
  env.set("InsertedPrecisionFreeCast__688", builder["matmul"](env.get("InsertedPrecisionFreeCast__687"), env.get("InsertedPrecisionFreeCast__612"), {"label":"/model/layers.0/attn/GroupQueryAttention_/Attention/qkv/matmul_2_361"}));
  env.set("_688", builder.cast(env.get("InsertedPrecisionFreeCast__688"), "float16"));
  env.set("_689", builder["transpose"](env.get("_688"), {"label":"/model/layers.0/attn/GroupQueryAttention_/Attention/qkv/transpose_362","permutation":[0,2,1,3]}));
  env.set("_690", builder.reshape(env.get("_689"), [1,128,2048]));
  env.set("InsertedPrecisionFreeCast__690", builder.cast(env.get("_690"), "float32"));
  env.set("InsertedPrecisionFreeCast__691", builder["matmul"](env.get("InsertedPrecisionFreeCast__690"), env.get("InsertedPrecisionFreeCast__572"), {"label":"/model/layers.0/attn/o_proj/MatMul_Q4_matmul_364"}));
  env.set("_694", builder["add"](env.get("_693"), env.get("InsertedPrecisionFreeCast__691"), {"label":"/model/layers.0/post_attention_layernorm/SkipLayerNorm_add_skip_367"}));
  env.set("_717", builder.cast(env.get("_694"), "float16"));
  env.set("_718", builder.cast(env.get("_717"), "float32"));
  env.set("_558", builder.dequantizeLinear(env.get("_555"), env.get("_556"), env.get("_557"), {"axis":2,"blockSize":32,"label":"/model/layers.0/mlp/down_proj/MatMul_Q4_dequantizeLinear_255"}));
  env.set("_559", builder.reshape(env.get("_558"), [2048,5632]));
  env.set("_560", builder["transpose"](env.get("_559"), {"label":"/model/layers.0/mlp/down_proj/MatMul_Q4_transpose_dequantizeLinear_257","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__560", builder.cast(env.get("_560"), "float32"));
  env.set("_708", builder.dequantizeLinear(env.get("_705"), env.get("_706"), env.get("_707"), {"axis":2,"blockSize":32,"label":"/model/layers.0/mlp/gate_proj/MatMul_Q4_dequantizeLinear_377"}));
  env.set("_709", builder.reshape(env.get("_708"), [5632,2048]));
  env.set("_710", builder["transpose"](env.get("_709"), {"label":"/model/layers.0/mlp/gate_proj/MatMul_Q4_transpose_dequantizeLinear_379","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__710", builder.cast(env.get("_710"), "float32"));
  env.set("_701", builder.cast(env.get("_700"), "float32"));
  env.set("_695", builder["pow"](env.get("_694"), env.get("_583"), {"label":"/model/layers.0/post_attention_layernorm/SkipLayerNorm_pow_368"}));
  env.set("_696", builder["reduceMean"](env.get("_695"), {"axes":[2],"keepDimensions":true,"label":"/model/layers.0/post_attention_layernorm/SkipLayerNorm_reduceMean_369"}));
  env.set("_697", builder["add"](env.get("_696"), env.get("_586"), {"label":"/model/layers.0/post_attention_layernorm/SkipLayerNorm_add_370"}));
  env.set("_698", builder["sqrt"](env.get("_697"), {"label":"/model/layers.0/post_attention_layernorm/SkipLayerNorm_sqrt_371"}));
  env.set("_699", builder["div"](env.get("_694"), env.get("_698"), {"label":"/model/layers.0/post_attention_layernorm/SkipLayerNorm_div_372"}));
  env.set("_702", builder["mul"](env.get("_701"), env.get("_699"), {"label":"/model/layers.0/post_attention_layernorm/SkipLayerNorm_mul_374"}));
  env.set("InsertedPrecisionFreeCast__711", builder["matmul"](env.get("_702"), env.get("InsertedPrecisionFreeCast__710"), {"label":"/model/layers.0/mlp/gate_proj/MatMul_Q4_matmul_380"}));
  env.set("InsertedPrecisionFreeCast__712", builder["sigmoid"](env.get("InsertedPrecisionFreeCast__711"), {"label":"/model/layers.0/mlp/act_fn/Sigmoid_381"}));
  env.set("InsertedPrecisionFreeCast__713", builder["mul"](env.get("InsertedPrecisionFreeCast__711"), env.get("InsertedPrecisionFreeCast__712"), {"label":"/model/layers.0/mlp/act_fn/Mul_382"}));
  env.set("_564", builder.dequantizeLinear(env.get("_561"), env.get("_562"), env.get("_563"), {"axis":2,"blockSize":32,"label":"/model/layers.0/mlp/up_proj/MatMul_Q4_dequantizeLinear_258"}));
  env.set("_565", builder.reshape(env.get("_564"), [5632,2048]));
  env.set("_566", builder["transpose"](env.get("_565"), {"label":"/model/layers.0/mlp/up_proj/MatMul_Q4_transpose_dequantizeLinear_260","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__566", builder.cast(env.get("_566"), "float32"));
  env.set("InsertedPrecisionFreeCast__704", builder["matmul"](env.get("_702"), env.get("InsertedPrecisionFreeCast__566"), {"label":"/model/layers.0/mlp/up_proj/MatMul_Q4_matmul_376"}));
  env.set("InsertedPrecisionFreeCast__714", builder["mul"](env.get("InsertedPrecisionFreeCast__713"), env.get("InsertedPrecisionFreeCast__704"), {"label":"/model/layers.0/mlp/Mul_383"}));
  env.set("InsertedPrecisionFreeCast__715", builder["matmul"](env.get("InsertedPrecisionFreeCast__714"), env.get("InsertedPrecisionFreeCast__560"), {"label":"/model/layers.0/mlp/down_proj/MatMul_Q4_matmul_384"}));
  env.set("_719", builder["add"](env.get("_718"), env.get("InsertedPrecisionFreeCast__715"), {"label":"/model/layers.1/input_layernorm/SkipLayerNorm_add_skip_388"}));
  env.set("_811", builder.cast(env.get("_719"), "float16"));
  env.set("_812", builder.cast(env.get("_811"), "float32"));
  env.set("_546", builder.dequantizeLinear(env.get("_543"), env.get("_544"), env.get("_545"), {"axis":2,"blockSize":32,"label":"/model/layers.1/attn/o_proj/MatMul_Q4_dequantizeLinear_249"}));
  env.set("_547", builder.reshape(env.get("_546"), [2048,2048]));
  env.set("_548", builder["transpose"](env.get("_547"), {"label":"/model/layers.1/attn/o_proj/MatMul_Q4_transpose_dequantizeLinear_251","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__548", builder.cast(env.get("_548"), "float32"));
  env.set("Inserted_156", builder.cast(env.get("_601"), "uint8"));
  env.set("_731", builder["where"](env.get("Inserted_156"), env.get("_602"), env.get("_600"), {"label":"/model/layers.1/attn/GroupQueryAttention_/GQA/scatter/where_399"}));
  env.set("_732", builder["add"](env.get("_604"), env.get("_731"), {"label":"/model/layers.1/attn/GroupQueryAttention_/GQA/right_constant/add_401"}));
  env.set("_733", builder.concat([env.get("_606"), env.get("_732")], 1, {"label":"/model/layers.1/attn/GroupQueryAttention_/GQA/concat_for_pre_scatter_indices_402"}));
  env.set("_734", builder.reshape(env.get("_733"), [1,128,4,3]));
  env.set("Inserted_158", builder.cast(env.get("_734"), "int64"));
  env.set("Inserted_160", builder["max"](env.get("Inserted_158"), env.get("Inserted_159"), {"label":"Inserted_Max_406"}));
  env.set("Inserted_162", builder["min"](env.get("Inserted_160"), env.get("Inserted_161"), {"label":"Inserted_Min_407"}));
  env.set("_552", builder.dequantizeLinear(env.get("_549"), env.get("_550"), env.get("_551"), {"axis":2,"blockSize":32,"label":"/model/layers.1/attn/v_proj/MatMul_Q4_dequantizeLinear_252"}));
  env.set("_553", builder.reshape(env.get("_552"), [256,2048]));
  env.set("_554", builder["transpose"](env.get("_553"), {"label":"/model/layers.1/attn/v_proj/MatMul_Q4_transpose_dequantizeLinear_254","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__554", builder.cast(env.get("_554"), "float32"));
  env.set("_726", builder.cast(env.get("_725"), "float32"));
  env.set("_720", builder["pow"](env.get("_719"), env.get("_583"), {"label":"/model/layers.1/input_layernorm/SkipLayerNorm_pow_389"}));
  env.set("_721", builder["reduceMean"](env.get("_720"), {"axes":[2],"keepDimensions":true,"label":"/model/layers.1/input_layernorm/SkipLayerNorm_reduceMean_390"}));
  env.set("_722", builder["add"](env.get("_721"), env.get("_586"), {"label":"/model/layers.1/input_layernorm/SkipLayerNorm_add_391"}));
  env.set("_723", builder["sqrt"](env.get("_722"), {"label":"/model/layers.1/input_layernorm/SkipLayerNorm_sqrt_392"}));
  env.set("_724", builder["div"](env.get("_719"), env.get("_723"), {"label":"/model/layers.1/input_layernorm/SkipLayerNorm_div_393"}));
  env.set("_727", builder["mul"](env.get("_726"), env.get("_724"), {"label":"/model/layers.1/input_layernorm/SkipLayerNorm_mul_395"}));
  env.set("InsertedPrecisionFreeCast__729", builder["matmul"](env.get("_727"), env.get("InsertedPrecisionFreeCast__554"), {"label":"/model/layers.1/attn/v_proj/MatMul_Q4_matmul_397"}));
  env.set("_729", builder.cast(env.get("InsertedPrecisionFreeCast__729"), "float16"));
  env.set("_730", builder.reshape(env.get("_729"), [1,128,4,64]));
  env.set("present_1_value_3", builder["scatterND"](env.get("past_key_values_1_value_735"), env.get("Inserted_162"), env.get("_730"), {"label":"/model/layers.1/attn/GroupQueryAttention_/GQA/present_value/ScatterND_404"}));
  env.set("_736", builder.reshape(env.get("present_1_value_3"), [1,4,1,512,64]));
  env.set("_737", builder.expand(env.get("_736"), [1,4,8,512,64], {"label":"/model/layers.1/attn/GroupQueryAttention_/GQA/true_present_value/expand_409"}));
  env.set("_738", builder.reshape(env.get("_737"), [1,32,512,64]));
  env.set("InsertedPrecisionFreeCast__738", builder.cast(env.get("_738"), "float32"));
  env.set("Inserted_187", builder.cast(env.get("_734"), "int64"));
  env.set("Inserted_189", builder["max"](env.get("Inserted_187"), env.get("Inserted_188"), {"label":"Inserted_Max_444"}));
  env.set("Inserted_191", builder["min"](env.get("Inserted_189"), env.get("Inserted_190"), {"label":"Inserted_Min_445"}));
  env.set("Inserted_180", builder["clamp"](env.get("position_ids_625"), {"label":"Inserted_Clip_435","maxValue":2047,"minValue":-2048}));
  env.set("_764", builder["gather"](env.get("_644"), env.get("Inserted_180"), {"axis":0,"label":"/model/layers.1/attn/k_rotary/RotaryEmbedding_gather_cos_434"}));
  env.set("_765", builder.reshape(env.get("_764"), [1,128,1,1,32]));
  env.set("InsertedPrecisionFreeCast__765", builder.cast(env.get("_765"), "float32"));
  env.set("_751", builder.dequantizeLinear(env.get("_748"), env.get("_749"), env.get("_750"), {"axis":2,"blockSize":32,"label":"/model/layers.1/attn/k_proj/MatMul_Q4_dequantizeLinear_423"}));
  env.set("_752", builder.reshape(env.get("_751"), [256,2048]));
  env.set("_753", builder["transpose"](env.get("_752"), {"label":"/model/layers.1/attn/k_proj/MatMul_Q4_transpose_dequantizeLinear_425","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__753", builder.cast(env.get("_753"), "float32"));
  env.set("InsertedPrecisionFreeCast__754", builder["matmul"](env.get("_727"), env.get("InsertedPrecisionFreeCast__753"), {"label":"/model/layers.1/attn/k_proj/MatMul_Q4_matmul_426"}));
  env.set("_754", builder.cast(env.get("InsertedPrecisionFreeCast__754"), "float16"));
  env.set("_755", builder.reshape(env.get("_754"), [1,128,4,64]));
  env.set("_756", builder.reshape(env.get("_755"), [1,128,4,2,32]));
  env.set("InsertedPrecisionFreeCast__756", builder.cast(env.get("_756"), "float32"));
  env.set("InsertedPrecisionFreeCast__766", builder["mul"](env.get("InsertedPrecisionFreeCast__756"), env.get("InsertedPrecisionFreeCast__765"), {"label":"/model/layers.1/attn/k_rotary/RotaryEmbedding_mul_cos_437"}));
  env.set("InsertedPrecisionFreeCast__767", builder.reshape(env.get("InsertedPrecisionFreeCast__766"), [1,128,4,64]));
  env.set("InsertedPrecisionFreeCast__761", builder.cast(env.get("_761"), "float32"));
  env.set("Inserted_171", builder["clamp"](env.get("position_ids_625"), {"label":"Inserted_Clip_421","maxValue":2047,"minValue":-2048}));
  env.set("_746", builder["gather"](env.get("_624"), env.get("Inserted_171"), {"axis":0,"label":"/model/layers.1/attn/k_rotary/RotaryEmbedding_gather_sin_420"}));
  env.set("_747", builder.reshape(env.get("_746"), [1,128,1,1,32]));
  env.set("InsertedPrecisionFreeCast__747", builder.cast(env.get("_747"), "float32"));
  {
    const tmp = builder.split(env.get("_756"), 2, {"axis":3,"label":"/model/layers.1/attn/k_rotary/RotaryEmbedding_split_partial_input0_429"});
    env.set("_757", tmp[0]);
    env.set("_758", tmp[1]);
  }
  env.set("_759", builder.concat([env.get("_758"), env.get("_757")], 3, {"label":"/model/layers.1/attn/k_rotary/RotaryEmbedding_concat_partial_input0_430"}));
  env.set("InsertedPrecisionFreeCast__759", builder.cast(env.get("_759"), "float32"));
  env.set("InsertedPrecisionFreeCast__760", builder["mul"](env.get("InsertedPrecisionFreeCast__759"), env.get("InsertedPrecisionFreeCast__747"), {"label":"/model/layers.1/attn/k_rotary/RotaryEmbedding_mul_sin_431"}));
  env.set("InsertedPrecisionFreeCast__762", builder["mul"](env.get("InsertedPrecisionFreeCast__760"), env.get("InsertedPrecisionFreeCast__761"), {"label":"/model/layers.1/attn/k_rotary/RotaryEmbedding_mul_sign_432"}));
  env.set("InsertedPrecisionFreeCast__763", builder.reshape(env.get("InsertedPrecisionFreeCast__762"), [1,128,4,64]));
  env.set("InsertedPrecisionFreeCast__768", builder["add"](env.get("InsertedPrecisionFreeCast__767"), env.get("InsertedPrecisionFreeCast__763"), {"label":"/model/layers.1/attn/k_rotary/RotaryEmbedding_add_mul_cos_sin_439"}));
  env.set("_768", builder.cast(env.get("InsertedPrecisionFreeCast__768"), "float16"));
  env.set("_769", builder.reshape(env.get("_768"), [1,128,256]));
  env.set("_770", builder.reshape(env.get("_769"), [1,128,4,64]));
  env.set("present_1_key_2", builder["scatterND"](env.get("past_key_values_1_key_771"), env.get("Inserted_191"), env.get("_770"), {"label":"/model/layers.1/attn/GroupQueryAttention_/GQA/present_key/ScatterND_442"}));
  env.set("_772", builder.reshape(env.get("present_1_key_2"), [1,4,1,512,64]));
  env.set("_773", builder.expand(env.get("_772"), [1,4,8,512,64], {"label":"/model/layers.1/attn/GroupQueryAttention_/GQA/true_present_key/expand_447"}));
  env.set("_774", builder.reshape(env.get("_773"), [1,32,512,64]));
  env.set("_775", builder["transpose"](env.get("_774"), {"label":"/model/layers.1/attn/GroupQueryAttention_/GQA/present_key/transpose_449","permutation":[0,1,3,2]}));
  env.set("InsertedPrecisionFreeCast__775", builder.cast(env.get("_775"), "float32"));
  env.set("Inserted_204", builder["clamp"](env.get("position_ids_625"), {"label":"Inserted_Clip_465","maxValue":2047,"minValue":-2048}));
  env.set("_794", builder["gather"](env.get("_644"), env.get("Inserted_204"), {"axis":0,"label":"/model/layers.1/attn/q_rotary/RotaryEmbedding_gather_cos_464"}));
  env.set("_795", builder.reshape(env.get("_794"), [1,128,1,1,32]));
  env.set("InsertedPrecisionFreeCast__795", builder.cast(env.get("_795"), "float32"));
  env.set("_781", builder.dequantizeLinear(env.get("_778"), env.get("_779"), env.get("_780"), {"axis":2,"blockSize":32,"label":"/model/layers.1/attn/q_proj/MatMul_Q4_dequantizeLinear_453"}));
  env.set("_782", builder.reshape(env.get("_781"), [2048,2048]));
  env.set("_783", builder["transpose"](env.get("_782"), {"label":"/model/layers.1/attn/q_proj/MatMul_Q4_transpose_dequantizeLinear_455","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__783", builder.cast(env.get("_783"), "float32"));
  env.set("InsertedPrecisionFreeCast__784", builder["matmul"](env.get("_727"), env.get("InsertedPrecisionFreeCast__783"), {"label":"/model/layers.1/attn/q_proj/MatMul_Q4_matmul_456"}));
  env.set("_784", builder.cast(env.get("InsertedPrecisionFreeCast__784"), "float16"));
  env.set("_785", builder.reshape(env.get("_784"), [1,128,32,64]));
  env.set("_786", builder.reshape(env.get("_785"), [1,128,32,2,32]));
  env.set("InsertedPrecisionFreeCast__786", builder.cast(env.get("_786"), "float32"));
  env.set("InsertedPrecisionFreeCast__796", builder["mul"](env.get("InsertedPrecisionFreeCast__786"), env.get("InsertedPrecisionFreeCast__795"), {"label":"/model/layers.1/attn/q_rotary/RotaryEmbedding_mul_cos_467"}));
  env.set("InsertedPrecisionFreeCast__797", builder.reshape(env.get("InsertedPrecisionFreeCast__796"), [1,128,32,64]));
  env.set("InsertedPrecisionFreeCast__791", builder.cast(env.get("_791"), "float32"));
  env.set("Inserted_195", builder["clamp"](env.get("position_ids_625"), {"label":"Inserted_Clip_451","maxValue":2047,"minValue":-2048}));
  env.set("_776", builder["gather"](env.get("_624"), env.get("Inserted_195"), {"axis":0,"label":"/model/layers.1/attn/q_rotary/RotaryEmbedding_gather_sin_450"}));
  env.set("_777", builder.reshape(env.get("_776"), [1,128,1,1,32]));
  env.set("InsertedPrecisionFreeCast__777", builder.cast(env.get("_777"), "float32"));
  {
    const tmp = builder.split(env.get("_786"), 2, {"axis":3,"label":"/model/layers.1/attn/q_rotary/RotaryEmbedding_split_partial_input0_459"});
    env.set("_787", tmp[0]);
    env.set("_788", tmp[1]);
  }
  env.set("_789", builder.concat([env.get("_788"), env.get("_787")], 3, {"label":"/model/layers.1/attn/q_rotary/RotaryEmbedding_concat_partial_input0_460"}));
  env.set("InsertedPrecisionFreeCast__789", builder.cast(env.get("_789"), "float32"));
  env.set("InsertedPrecisionFreeCast__790", builder["mul"](env.get("InsertedPrecisionFreeCast__789"), env.get("InsertedPrecisionFreeCast__777"), {"label":"/model/layers.1/attn/q_rotary/RotaryEmbedding_mul_sin_461"}));
  env.set("InsertedPrecisionFreeCast__792", builder["mul"](env.get("InsertedPrecisionFreeCast__790"), env.get("InsertedPrecisionFreeCast__791"), {"label":"/model/layers.1/attn/q_rotary/RotaryEmbedding_mul_sign_462"}));
  env.set("InsertedPrecisionFreeCast__793", builder.reshape(env.get("InsertedPrecisionFreeCast__792"), [1,128,32,64]));
  env.set("InsertedPrecisionFreeCast__798", builder["add"](env.get("InsertedPrecisionFreeCast__797"), env.get("InsertedPrecisionFreeCast__793"), {"label":"/model/layers.1/attn/q_rotary/RotaryEmbedding_add_mul_cos_sin_469"}));
  env.set("_798", builder.cast(env.get("InsertedPrecisionFreeCast__798"), "float16"));
  env.set("_799", builder.reshape(env.get("_798"), [1,128,2048]));
  env.set("_800", builder.reshape(env.get("_799"), [1,128,32,64]));
  env.set("_801", builder["transpose"](env.get("_800"), {"label":"/model/layers.1/attn/GroupQueryAttention_/GQA/query/transpose_472","permutation":[0,2,1,3]}));
  env.set("InsertedPrecisionFreeCast__801", builder.cast(env.get("_801"), "float32"));
  env.set("InsertedPrecisionFreeCast__802", builder["matmul"](env.get("InsertedPrecisionFreeCast__801"), env.get("InsertedPrecisionFreeCast__775"), {"label":"/model/layers.1/attn/GroupQueryAttention_/Attention/qkv/matmul_1_473"}));
  env.set("InsertedPrecisionFreeCast__803", builder["mul"](env.get("InsertedPrecisionFreeCast__802"), env.get("InsertedPrecisionFreeCast__684"), {"label":"/model/layers.1/attn/GroupQueryAttention_/Attention/qkv/div_474"}));
  env.set("_742", builder.expand(env.get("_617"), [1,32,128,512], {"label":"/model/layers.1/attn/GroupQueryAttention_/GQA/GQA_mask_shape_ones/expand_414"}));
  env.set("_743", builder.cumulativeSum(env.get("_742"), 3, {"exclusive":true,"label":"/model/layers.1/attn/GroupQueryAttention_range_of_mask_shape_415"}));
  env.set("_739", builder["add"](env.get("_613"), env.get("_731"), {"label":"/model/layers.1/attn/GroupQueryAttention_/GQA/attn_mask/add_411"}));
  env.set("_740", builder.expand(env.get("_739"), [512,128], {"label":"/model/layers.1/attn/GroupQueryAttention_/GQA/expand_neq_right_412"}));
  env.set("_741", builder["transpose"](env.get("_740"), {"label":"/model/layers.1/attn/GroupQueryAttention_/GQA/neq_right/transpose_413","permutation":[1,0]}));
  env.set("Inserted_169", builder["lesser"](env.get("_743"), env.get("_741"), {"label":"/model/layers.1/attn/GroupQueryAttention_/GQA/attn_mask/condition_1_416"}));
  env.set("InsertedPrecisionFreeCast__745", builder["where"](env.get("Inserted_169"), env.get("InsertedPrecisionFreeCast__621"), env.get("InsertedPrecisionFreeCast__622"), {"label":"/model/layers.1/attn/GroupQueryAttention_/GQA/attn_mask/where_418"}));
  env.set("InsertedPrecisionFreeCast__804", builder["add"](env.get("InsertedPrecisionFreeCast__803"), env.get("InsertedPrecisionFreeCast__745"), {"label":"/model/layers.1/attn/GroupQueryAttention_/Attention/attn_mask/softmax_input_475"}));
  env.set("InsertedPrecisionFreeCast__inlfunc_Softmax_token_0_X_ReduceMax", builder["reduceMax"](env.get("InsertedPrecisionFreeCast__804"), {"axes":[3],"keepDimensions":true}));
  env.set("InsertedPrecisionFreeCast__inlfunc_Softmax_token_0_X_Sub", builder["sub"](env.get("InsertedPrecisionFreeCast__804"), env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_0_X_ReduceMax"), {}));
  env.set("InsertedPrecisionFreeCast__inlfunc_Softmax_token_0_X_Exp", builder["exp"](env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_0_X_Sub"), {}));
  env.set("InsertedPrecisionFreeCast__inlfunc_Softmax_token_0_X_ReduceSum", builder["reduceSum"](env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_0_X_Exp"), {"axes":[3],"keepDimensions":true}));
  env.set("InsertedPrecisionFreeCast__805", builder["div"](env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_0_X_Exp"), env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_0_X_ReduceSum"), {}));
  env.set("InsertedPrecisionFreeCast__806", builder["matmul"](env.get("InsertedPrecisionFreeCast__805"), env.get("InsertedPrecisionFreeCast__738"), {"label":"/model/layers.1/attn/GroupQueryAttention_/Attention/qkv/matmul_2_477"}));
  env.set("_806", builder.cast(env.get("InsertedPrecisionFreeCast__806"), "float16"));
  env.set("_807", builder["transpose"](env.get("_806"), {"label":"/model/layers.1/attn/GroupQueryAttention_/Attention/qkv/transpose_478","permutation":[0,2,1,3]}));
  env.set("_808", builder.reshape(env.get("_807"), [1,128,2048]));
  env.set("InsertedPrecisionFreeCast__808", builder.cast(env.get("_808"), "float32"));
  env.set("InsertedPrecisionFreeCast__809", builder["matmul"](env.get("InsertedPrecisionFreeCast__808"), env.get("InsertedPrecisionFreeCast__548"), {"label":"/model/layers.1/attn/o_proj/MatMul_Q4_matmul_480"}));
  env.set("_813", builder["add"](env.get("_812"), env.get("InsertedPrecisionFreeCast__809"), {"label":"/model/layers.1/post_attention_layernorm/SkipLayerNorm_add_skip_484"}));
  env.set("_836", builder.cast(env.get("_813"), "float16"));
  env.set("_837", builder.cast(env.get("_836"), "float32"));
  env.set("_534", builder.dequantizeLinear(env.get("_531"), env.get("_532"), env.get("_533"), {"axis":2,"blockSize":32,"label":"/model/layers.1/mlp/down_proj/MatMul_Q4_dequantizeLinear_243"}));
  env.set("_535", builder.reshape(env.get("_534"), [2048,5632]));
  env.set("_536", builder["transpose"](env.get("_535"), {"label":"/model/layers.1/mlp/down_proj/MatMul_Q4_transpose_dequantizeLinear_245","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__536", builder.cast(env.get("_536"), "float32"));
  env.set("_827", builder.dequantizeLinear(env.get("_824"), env.get("_825"), env.get("_826"), {"axis":2,"blockSize":32,"label":"/model/layers.1/mlp/gate_proj/MatMul_Q4_dequantizeLinear_494"}));
  env.set("_828", builder.reshape(env.get("_827"), [5632,2048]));
  env.set("_829", builder["transpose"](env.get("_828"), {"label":"/model/layers.1/mlp/gate_proj/MatMul_Q4_transpose_dequantizeLinear_496","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__829", builder.cast(env.get("_829"), "float32"));
  env.set("_820", builder.cast(env.get("_819"), "float32"));
  env.set("_814", builder["pow"](env.get("_813"), env.get("_583"), {"label":"/model/layers.1/post_attention_layernorm/SkipLayerNorm_pow_485"}));
  env.set("_815", builder["reduceMean"](env.get("_814"), {"axes":[2],"keepDimensions":true,"label":"/model/layers.1/post_attention_layernorm/SkipLayerNorm_reduceMean_486"}));
  env.set("_816", builder["add"](env.get("_815"), env.get("_586"), {"label":"/model/layers.1/post_attention_layernorm/SkipLayerNorm_add_487"}));
  env.set("_817", builder["sqrt"](env.get("_816"), {"label":"/model/layers.1/post_attention_layernorm/SkipLayerNorm_sqrt_488"}));
  env.set("_818", builder["div"](env.get("_813"), env.get("_817"), {"label":"/model/layers.1/post_attention_layernorm/SkipLayerNorm_div_489"}));
  env.set("_821", builder["mul"](env.get("_820"), env.get("_818"), {"label":"/model/layers.1/post_attention_layernorm/SkipLayerNorm_mul_491"}));
  env.set("InsertedPrecisionFreeCast__830", builder["matmul"](env.get("_821"), env.get("InsertedPrecisionFreeCast__829"), {"label":"/model/layers.1/mlp/gate_proj/MatMul_Q4_matmul_497"}));
  env.set("InsertedPrecisionFreeCast__831", builder["sigmoid"](env.get("InsertedPrecisionFreeCast__830"), {"label":"/model/layers.1/mlp/act_fn/Sigmoid_498"}));
  env.set("InsertedPrecisionFreeCast__832", builder["mul"](env.get("InsertedPrecisionFreeCast__830"), env.get("InsertedPrecisionFreeCast__831"), {"label":"/model/layers.1/mlp/act_fn/Mul_499"}));
  env.set("_540", builder.dequantizeLinear(env.get("_537"), env.get("_538"), env.get("_539"), {"axis":2,"blockSize":32,"label":"/model/layers.1/mlp/up_proj/MatMul_Q4_dequantizeLinear_246"}));
  env.set("_541", builder.reshape(env.get("_540"), [5632,2048]));
  env.set("_542", builder["transpose"](env.get("_541"), {"label":"/model/layers.1/mlp/up_proj/MatMul_Q4_transpose_dequantizeLinear_248","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__542", builder.cast(env.get("_542"), "float32"));
  env.set("InsertedPrecisionFreeCast__823", builder["matmul"](env.get("_821"), env.get("InsertedPrecisionFreeCast__542"), {"label":"/model/layers.1/mlp/up_proj/MatMul_Q4_matmul_493"}));
  env.set("InsertedPrecisionFreeCast__833", builder["mul"](env.get("InsertedPrecisionFreeCast__832"), env.get("InsertedPrecisionFreeCast__823"), {"label":"/model/layers.1/mlp/Mul_500"}));
  env.set("InsertedPrecisionFreeCast__834", builder["matmul"](env.get("InsertedPrecisionFreeCast__833"), env.get("InsertedPrecisionFreeCast__536"), {"label":"/model/layers.1/mlp/down_proj/MatMul_Q4_matmul_501"}));
  env.set("_838", builder["add"](env.get("_837"), env.get("InsertedPrecisionFreeCast__834"), {"label":"/model/layers.2/input_layernorm/SkipLayerNorm_add_skip_505"}));
  env.set("_930", builder.cast(env.get("_838"), "float16"));
  env.set("_931", builder.cast(env.get("_930"), "float32"));
  env.set("_522", builder.dequantizeLinear(env.get("_519"), env.get("_520"), env.get("_521"), {"axis":2,"blockSize":32,"label":"/model/layers.2/attn/o_proj/MatMul_Q4_dequantizeLinear_237"}));
  env.set("_523", builder.reshape(env.get("_522"), [2048,2048]));
  env.set("_524", builder["transpose"](env.get("_523"), {"label":"/model/layers.2/attn/o_proj/MatMul_Q4_transpose_dequantizeLinear_239","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__524", builder.cast(env.get("_524"), "float32"));
  env.set("Inserted_216", builder.cast(env.get("_601"), "uint8"));
  env.set("_850", builder["where"](env.get("Inserted_216"), env.get("_602"), env.get("_600"), {"label":"/model/layers.2/attn/GroupQueryAttention_/GQA/scatter/where_516"}));
  env.set("_851", builder["add"](env.get("_604"), env.get("_850"), {"label":"/model/layers.2/attn/GroupQueryAttention_/GQA/right_constant/add_518"}));
  env.set("_852", builder.concat([env.get("_606"), env.get("_851")], 1, {"label":"/model/layers.2/attn/GroupQueryAttention_/GQA/concat_for_pre_scatter_indices_519"}));
  env.set("_853", builder.reshape(env.get("_852"), [1,128,4,3]));
  env.set("Inserted_218", builder.cast(env.get("_853"), "int64"));
  env.set("Inserted_220", builder["max"](env.get("Inserted_218"), env.get("Inserted_219"), {"label":"Inserted_Max_523"}));
  env.set("Inserted_222", builder["min"](env.get("Inserted_220"), env.get("Inserted_221"), {"label":"Inserted_Min_524"}));
  env.set("_528", builder.dequantizeLinear(env.get("_525"), env.get("_526"), env.get("_527"), {"axis":2,"blockSize":32,"label":"/model/layers.2/attn/v_proj/MatMul_Q4_dequantizeLinear_240"}));
  env.set("_529", builder.reshape(env.get("_528"), [256,2048]));
  env.set("_530", builder["transpose"](env.get("_529"), {"label":"/model/layers.2/attn/v_proj/MatMul_Q4_transpose_dequantizeLinear_242","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__530", builder.cast(env.get("_530"), "float32"));
  env.set("_845", builder.cast(env.get("_844"), "float32"));
  env.set("_839", builder["pow"](env.get("_838"), env.get("_583"), {"label":"/model/layers.2/input_layernorm/SkipLayerNorm_pow_506"}));
  env.set("_840", builder["reduceMean"](env.get("_839"), {"axes":[2],"keepDimensions":true,"label":"/model/layers.2/input_layernorm/SkipLayerNorm_reduceMean_507"}));
  env.set("_841", builder["add"](env.get("_840"), env.get("_586"), {"label":"/model/layers.2/input_layernorm/SkipLayerNorm_add_508"}));
  env.set("_842", builder["sqrt"](env.get("_841"), {"label":"/model/layers.2/input_layernorm/SkipLayerNorm_sqrt_509"}));
  env.set("_843", builder["div"](env.get("_838"), env.get("_842"), {"label":"/model/layers.2/input_layernorm/SkipLayerNorm_div_510"}));
  env.set("_846", builder["mul"](env.get("_845"), env.get("_843"), {"label":"/model/layers.2/input_layernorm/SkipLayerNorm_mul_512"}));
  env.set("InsertedPrecisionFreeCast__848", builder["matmul"](env.get("_846"), env.get("InsertedPrecisionFreeCast__530"), {"label":"/model/layers.2/attn/v_proj/MatMul_Q4_matmul_514"}));
  env.set("_848", builder.cast(env.get("InsertedPrecisionFreeCast__848"), "float16"));
  env.set("_849", builder.reshape(env.get("_848"), [1,128,4,64]));
  env.set("present_2_value_5", builder["scatterND"](env.get("past_key_values_2_value_854"), env.get("Inserted_222"), env.get("_849"), {"label":"/model/layers.2/attn/GroupQueryAttention_/GQA/present_value/ScatterND_521"}));
  env.set("_855", builder.reshape(env.get("present_2_value_5"), [1,4,1,512,64]));
  env.set("_856", builder.expand(env.get("_855"), [1,4,8,512,64], {"label":"/model/layers.2/attn/GroupQueryAttention_/GQA/true_present_value/expand_526"}));
  env.set("_857", builder.reshape(env.get("_856"), [1,32,512,64]));
  env.set("InsertedPrecisionFreeCast__857", builder.cast(env.get("_857"), "float32"));
  env.set("Inserted_247", builder.cast(env.get("_853"), "int64"));
  env.set("Inserted_249", builder["max"](env.get("Inserted_247"), env.get("Inserted_248"), {"label":"Inserted_Max_561"}));
  env.set("Inserted_251", builder["min"](env.get("Inserted_249"), env.get("Inserted_250"), {"label":"Inserted_Min_562"}));
  env.set("Inserted_240", builder["clamp"](env.get("position_ids_625"), {"label":"Inserted_Clip_552","maxValue":2047,"minValue":-2048}));
  env.set("_883", builder["gather"](env.get("_644"), env.get("Inserted_240"), {"axis":0,"label":"/model/layers.2/attn/k_rotary/RotaryEmbedding_gather_cos_551"}));
  env.set("_884", builder.reshape(env.get("_883"), [1,128,1,1,32]));
  env.set("InsertedPrecisionFreeCast__884", builder.cast(env.get("_884"), "float32"));
  env.set("_870", builder.dequantizeLinear(env.get("_867"), env.get("_868"), env.get("_869"), {"axis":2,"blockSize":32,"label":"/model/layers.2/attn/k_proj/MatMul_Q4_dequantizeLinear_540"}));
  env.set("_871", builder.reshape(env.get("_870"), [256,2048]));
  env.set("_872", builder["transpose"](env.get("_871"), {"label":"/model/layers.2/attn/k_proj/MatMul_Q4_transpose_dequantizeLinear_542","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__872", builder.cast(env.get("_872"), "float32"));
  env.set("InsertedPrecisionFreeCast__873", builder["matmul"](env.get("_846"), env.get("InsertedPrecisionFreeCast__872"), {"label":"/model/layers.2/attn/k_proj/MatMul_Q4_matmul_543"}));
  env.set("_873", builder.cast(env.get("InsertedPrecisionFreeCast__873"), "float16"));
  env.set("_874", builder.reshape(env.get("_873"), [1,128,4,64]));
  env.set("_875", builder.reshape(env.get("_874"), [1,128,4,2,32]));
  env.set("InsertedPrecisionFreeCast__875", builder.cast(env.get("_875"), "float32"));
  env.set("InsertedPrecisionFreeCast__885", builder["mul"](env.get("InsertedPrecisionFreeCast__875"), env.get("InsertedPrecisionFreeCast__884"), {"label":"/model/layers.2/attn/k_rotary/RotaryEmbedding_mul_cos_554"}));
  env.set("InsertedPrecisionFreeCast__886", builder.reshape(env.get("InsertedPrecisionFreeCast__885"), [1,128,4,64]));
  env.set("InsertedPrecisionFreeCast__880", builder.cast(env.get("_880"), "float32"));
  env.set("Inserted_231", builder["clamp"](env.get("position_ids_625"), {"label":"Inserted_Clip_538","maxValue":2047,"minValue":-2048}));
  env.set("_865", builder["gather"](env.get("_624"), env.get("Inserted_231"), {"axis":0,"label":"/model/layers.2/attn/k_rotary/RotaryEmbedding_gather_sin_537"}));
  env.set("_866", builder.reshape(env.get("_865"), [1,128,1,1,32]));
  env.set("InsertedPrecisionFreeCast__866", builder.cast(env.get("_866"), "float32"));
  {
    const tmp = builder.split(env.get("_875"), 2, {"axis":3,"label":"/model/layers.2/attn/k_rotary/RotaryEmbedding_split_partial_input0_546"});
    env.set("_876", tmp[0]);
    env.set("_877", tmp[1]);
  }
  env.set("_878", builder.concat([env.get("_877"), env.get("_876")], 3, {"label":"/model/layers.2/attn/k_rotary/RotaryEmbedding_concat_partial_input0_547"}));
  env.set("InsertedPrecisionFreeCast__878", builder.cast(env.get("_878"), "float32"));
  env.set("InsertedPrecisionFreeCast__879", builder["mul"](env.get("InsertedPrecisionFreeCast__878"), env.get("InsertedPrecisionFreeCast__866"), {"label":"/model/layers.2/attn/k_rotary/RotaryEmbedding_mul_sin_548"}));
  env.set("InsertedPrecisionFreeCast__881", builder["mul"](env.get("InsertedPrecisionFreeCast__879"), env.get("InsertedPrecisionFreeCast__880"), {"label":"/model/layers.2/attn/k_rotary/RotaryEmbedding_mul_sign_549"}));
  env.set("InsertedPrecisionFreeCast__882", builder.reshape(env.get("InsertedPrecisionFreeCast__881"), [1,128,4,64]));
  env.set("InsertedPrecisionFreeCast__887", builder["add"](env.get("InsertedPrecisionFreeCast__886"), env.get("InsertedPrecisionFreeCast__882"), {"label":"/model/layers.2/attn/k_rotary/RotaryEmbedding_add_mul_cos_sin_556"}));
  env.set("_887", builder.cast(env.get("InsertedPrecisionFreeCast__887"), "float16"));
  env.set("_888", builder.reshape(env.get("_887"), [1,128,256]));
  env.set("_889", builder.reshape(env.get("_888"), [1,128,4,64]));
  env.set("present_2_key_4", builder["scatterND"](env.get("past_key_values_2_key_890"), env.get("Inserted_251"), env.get("_889"), {"label":"/model/layers.2/attn/GroupQueryAttention_/GQA/present_key/ScatterND_559"}));
  env.set("_891", builder.reshape(env.get("present_2_key_4"), [1,4,1,512,64]));
  env.set("_892", builder.expand(env.get("_891"), [1,4,8,512,64], {"label":"/model/layers.2/attn/GroupQueryAttention_/GQA/true_present_key/expand_564"}));
  env.set("_893", builder.reshape(env.get("_892"), [1,32,512,64]));
  env.set("_894", builder["transpose"](env.get("_893"), {"label":"/model/layers.2/attn/GroupQueryAttention_/GQA/present_key/transpose_566","permutation":[0,1,3,2]}));
  env.set("InsertedPrecisionFreeCast__894", builder.cast(env.get("_894"), "float32"));
  env.set("Inserted_264", builder["clamp"](env.get("position_ids_625"), {"label":"Inserted_Clip_582","maxValue":2047,"minValue":-2048}));
  env.set("_913", builder["gather"](env.get("_644"), env.get("Inserted_264"), {"axis":0,"label":"/model/layers.2/attn/q_rotary/RotaryEmbedding_gather_cos_581"}));
  env.set("_914", builder.reshape(env.get("_913"), [1,128,1,1,32]));
  env.set("InsertedPrecisionFreeCast__914", builder.cast(env.get("_914"), "float32"));
  env.set("_900", builder.dequantizeLinear(env.get("_897"), env.get("_898"), env.get("_899"), {"axis":2,"blockSize":32,"label":"/model/layers.2/attn/q_proj/MatMul_Q4_dequantizeLinear_570"}));
  env.set("_901", builder.reshape(env.get("_900"), [2048,2048]));
  env.set("_902", builder["transpose"](env.get("_901"), {"label":"/model/layers.2/attn/q_proj/MatMul_Q4_transpose_dequantizeLinear_572","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__902", builder.cast(env.get("_902"), "float32"));
  env.set("InsertedPrecisionFreeCast__903", builder["matmul"](env.get("_846"), env.get("InsertedPrecisionFreeCast__902"), {"label":"/model/layers.2/attn/q_proj/MatMul_Q4_matmul_573"}));
  env.set("_903", builder.cast(env.get("InsertedPrecisionFreeCast__903"), "float16"));
  env.set("_904", builder.reshape(env.get("_903"), [1,128,32,64]));
  env.set("_905", builder.reshape(env.get("_904"), [1,128,32,2,32]));
  env.set("InsertedPrecisionFreeCast__905", builder.cast(env.get("_905"), "float32"));
  env.set("InsertedPrecisionFreeCast__915", builder["mul"](env.get("InsertedPrecisionFreeCast__905"), env.get("InsertedPrecisionFreeCast__914"), {"label":"/model/layers.2/attn/q_rotary/RotaryEmbedding_mul_cos_584"}));
  env.set("InsertedPrecisionFreeCast__916", builder.reshape(env.get("InsertedPrecisionFreeCast__915"), [1,128,32,64]));
  env.set("InsertedPrecisionFreeCast__910", builder.cast(env.get("_910"), "float32"));
  env.set("Inserted_255", builder["clamp"](env.get("position_ids_625"), {"label":"Inserted_Clip_568","maxValue":2047,"minValue":-2048}));
  env.set("_895", builder["gather"](env.get("_624"), env.get("Inserted_255"), {"axis":0,"label":"/model/layers.2/attn/q_rotary/RotaryEmbedding_gather_sin_567"}));
  env.set("_896", builder.reshape(env.get("_895"), [1,128,1,1,32]));
  env.set("InsertedPrecisionFreeCast__896", builder.cast(env.get("_896"), "float32"));
  {
    const tmp = builder.split(env.get("_905"), 2, {"axis":3,"label":"/model/layers.2/attn/q_rotary/RotaryEmbedding_split_partial_input0_576"});
    env.set("_906", tmp[0]);
    env.set("_907", tmp[1]);
  }
  env.set("_908", builder.concat([env.get("_907"), env.get("_906")], 3, {"label":"/model/layers.2/attn/q_rotary/RotaryEmbedding_concat_partial_input0_577"}));
  env.set("InsertedPrecisionFreeCast__908", builder.cast(env.get("_908"), "float32"));
  env.set("InsertedPrecisionFreeCast__909", builder["mul"](env.get("InsertedPrecisionFreeCast__908"), env.get("InsertedPrecisionFreeCast__896"), {"label":"/model/layers.2/attn/q_rotary/RotaryEmbedding_mul_sin_578"}));
  env.set("InsertedPrecisionFreeCast__911", builder["mul"](env.get("InsertedPrecisionFreeCast__909"), env.get("InsertedPrecisionFreeCast__910"), {"label":"/model/layers.2/attn/q_rotary/RotaryEmbedding_mul_sign_579"}));
  env.set("InsertedPrecisionFreeCast__912", builder.reshape(env.get("InsertedPrecisionFreeCast__911"), [1,128,32,64]));
  env.set("InsertedPrecisionFreeCast__917", builder["add"](env.get("InsertedPrecisionFreeCast__916"), env.get("InsertedPrecisionFreeCast__912"), {"label":"/model/layers.2/attn/q_rotary/RotaryEmbedding_add_mul_cos_sin_586"}));
  env.set("_917", builder.cast(env.get("InsertedPrecisionFreeCast__917"), "float16"));
  env.set("_918", builder.reshape(env.get("_917"), [1,128,2048]));
  env.set("_919", builder.reshape(env.get("_918"), [1,128,32,64]));
  env.set("_920", builder["transpose"](env.get("_919"), {"label":"/model/layers.2/attn/GroupQueryAttention_/GQA/query/transpose_589","permutation":[0,2,1,3]}));
  env.set("InsertedPrecisionFreeCast__920", builder.cast(env.get("_920"), "float32"));
  env.set("InsertedPrecisionFreeCast__921", builder["matmul"](env.get("InsertedPrecisionFreeCast__920"), env.get("InsertedPrecisionFreeCast__894"), {"label":"/model/layers.2/attn/GroupQueryAttention_/Attention/qkv/matmul_1_590"}));
  env.set("InsertedPrecisionFreeCast__922", builder["mul"](env.get("InsertedPrecisionFreeCast__921"), env.get("InsertedPrecisionFreeCast__684"), {"label":"/model/layers.2/attn/GroupQueryAttention_/Attention/qkv/div_591"}));
  env.set("_861", builder.expand(env.get("_617"), [1,32,128,512], {"label":"/model/layers.2/attn/GroupQueryAttention_/GQA/GQA_mask_shape_ones/expand_531"}));
  env.set("_862", builder.cumulativeSum(env.get("_861"), 3, {"exclusive":true,"label":"/model/layers.2/attn/GroupQueryAttention_range_of_mask_shape_532"}));
  env.set("_858", builder["add"](env.get("_613"), env.get("_850"), {"label":"/model/layers.2/attn/GroupQueryAttention_/GQA/attn_mask/add_528"}));
  env.set("_859", builder.expand(env.get("_858"), [512,128], {"label":"/model/layers.2/attn/GroupQueryAttention_/GQA/expand_neq_right_529"}));
  env.set("_860", builder["transpose"](env.get("_859"), {"label":"/model/layers.2/attn/GroupQueryAttention_/GQA/neq_right/transpose_530","permutation":[1,0]}));
  env.set("Inserted_229", builder["lesser"](env.get("_862"), env.get("_860"), {"label":"/model/layers.2/attn/GroupQueryAttention_/GQA/attn_mask/condition_1_533"}));
  env.set("InsertedPrecisionFreeCast__864", builder["where"](env.get("Inserted_229"), env.get("InsertedPrecisionFreeCast__621"), env.get("InsertedPrecisionFreeCast__622"), {"label":"/model/layers.2/attn/GroupQueryAttention_/GQA/attn_mask/where_535"}));
  env.set("InsertedPrecisionFreeCast__923", builder["add"](env.get("InsertedPrecisionFreeCast__922"), env.get("InsertedPrecisionFreeCast__864"), {"label":"/model/layers.2/attn/GroupQueryAttention_/Attention/attn_mask/softmax_input_592"}));
  env.set("InsertedPrecisionFreeCast__inlfunc_Softmax_token_1_X_ReduceMax", builder["reduceMax"](env.get("InsertedPrecisionFreeCast__923"), {"axes":[3],"keepDimensions":true}));
  env.set("InsertedPrecisionFreeCast__inlfunc_Softmax_token_1_X_Sub", builder["sub"](env.get("InsertedPrecisionFreeCast__923"), env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_1_X_ReduceMax"), {}));
  env.set("InsertedPrecisionFreeCast__inlfunc_Softmax_token_1_X_Exp", builder["exp"](env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_1_X_Sub"), {}));
  env.set("InsertedPrecisionFreeCast__inlfunc_Softmax_token_1_X_ReduceSum", builder["reduceSum"](env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_1_X_Exp"), {"axes":[3],"keepDimensions":true}));
  env.set("InsertedPrecisionFreeCast__924", builder["div"](env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_1_X_Exp"), env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_1_X_ReduceSum"), {}));
  env.set("InsertedPrecisionFreeCast__925", builder["matmul"](env.get("InsertedPrecisionFreeCast__924"), env.get("InsertedPrecisionFreeCast__857"), {"label":"/model/layers.2/attn/GroupQueryAttention_/Attention/qkv/matmul_2_594"}));
  env.set("_925", builder.cast(env.get("InsertedPrecisionFreeCast__925"), "float16"));
  env.set("_926", builder["transpose"](env.get("_925"), {"label":"/model/layers.2/attn/GroupQueryAttention_/Attention/qkv/transpose_595","permutation":[0,2,1,3]}));
  env.set("_927", builder.reshape(env.get("_926"), [1,128,2048]));
  env.set("InsertedPrecisionFreeCast__927", builder.cast(env.get("_927"), "float32"));
  env.set("InsertedPrecisionFreeCast__928", builder["matmul"](env.get("InsertedPrecisionFreeCast__927"), env.get("InsertedPrecisionFreeCast__524"), {"label":"/model/layers.2/attn/o_proj/MatMul_Q4_matmul_597"}));
  env.set("_932", builder["add"](env.get("_931"), env.get("InsertedPrecisionFreeCast__928"), {"label":"/model/layers.2/post_attention_layernorm/SkipLayerNorm_add_skip_601"}));
  env.set("_955", builder.cast(env.get("_932"), "float16"));
  env.set("_956", builder.cast(env.get("_955"), "float32"));
  env.set("_510", builder.dequantizeLinear(env.get("_507"), env.get("_508"), env.get("_509"), {"axis":2,"blockSize":32,"label":"/model/layers.2/mlp/down_proj/MatMul_Q4_dequantizeLinear_231"}));
  env.set("_511", builder.reshape(env.get("_510"), [2048,5632]));
  env.set("_512", builder["transpose"](env.get("_511"), {"label":"/model/layers.2/mlp/down_proj/MatMul_Q4_transpose_dequantizeLinear_233","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__512", builder.cast(env.get("_512"), "float32"));
  env.set("_946", builder.dequantizeLinear(env.get("_943"), env.get("_944"), env.get("_945"), {"axis":2,"blockSize":32,"label":"/model/layers.2/mlp/gate_proj/MatMul_Q4_dequantizeLinear_611"}));
  env.set("_947", builder.reshape(env.get("_946"), [5632,2048]));
  env.set("_948", builder["transpose"](env.get("_947"), {"label":"/model/layers.2/mlp/gate_proj/MatMul_Q4_transpose_dequantizeLinear_613","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__948", builder.cast(env.get("_948"), "float32"));
  env.set("_939", builder.cast(env.get("_938"), "float32"));
  env.set("_933", builder["pow"](env.get("_932"), env.get("_583"), {"label":"/model/layers.2/post_attention_layernorm/SkipLayerNorm_pow_602"}));
  env.set("_934", builder["reduceMean"](env.get("_933"), {"axes":[2],"keepDimensions":true,"label":"/model/layers.2/post_attention_layernorm/SkipLayerNorm_reduceMean_603"}));
  env.set("_935", builder["add"](env.get("_934"), env.get("_586"), {"label":"/model/layers.2/post_attention_layernorm/SkipLayerNorm_add_604"}));
  env.set("_936", builder["sqrt"](env.get("_935"), {"label":"/model/layers.2/post_attention_layernorm/SkipLayerNorm_sqrt_605"}));
  env.set("_937", builder["div"](env.get("_932"), env.get("_936"), {"label":"/model/layers.2/post_attention_layernorm/SkipLayerNorm_div_606"}));
  env.set("_940", builder["mul"](env.get("_939"), env.get("_937"), {"label":"/model/layers.2/post_attention_layernorm/SkipLayerNorm_mul_608"}));
  env.set("InsertedPrecisionFreeCast__949", builder["matmul"](env.get("_940"), env.get("InsertedPrecisionFreeCast__948"), {"label":"/model/layers.2/mlp/gate_proj/MatMul_Q4_matmul_614"}));
  env.set("InsertedPrecisionFreeCast__950", builder["sigmoid"](env.get("InsertedPrecisionFreeCast__949"), {"label":"/model/layers.2/mlp/act_fn/Sigmoid_615"}));
  env.set("InsertedPrecisionFreeCast__951", builder["mul"](env.get("InsertedPrecisionFreeCast__949"), env.get("InsertedPrecisionFreeCast__950"), {"label":"/model/layers.2/mlp/act_fn/Mul_616"}));
  env.set("_516", builder.dequantizeLinear(env.get("_513"), env.get("_514"), env.get("_515"), {"axis":2,"blockSize":32,"label":"/model/layers.2/mlp/up_proj/MatMul_Q4_dequantizeLinear_234"}));
  env.set("_517", builder.reshape(env.get("_516"), [5632,2048]));
  env.set("_518", builder["transpose"](env.get("_517"), {"label":"/model/layers.2/mlp/up_proj/MatMul_Q4_transpose_dequantizeLinear_236","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__518", builder.cast(env.get("_518"), "float32"));
  env.set("InsertedPrecisionFreeCast__942", builder["matmul"](env.get("_940"), env.get("InsertedPrecisionFreeCast__518"), {"label":"/model/layers.2/mlp/up_proj/MatMul_Q4_matmul_610"}));
  env.set("InsertedPrecisionFreeCast__952", builder["mul"](env.get("InsertedPrecisionFreeCast__951"), env.get("InsertedPrecisionFreeCast__942"), {"label":"/model/layers.2/mlp/Mul_617"}));
  env.set("InsertedPrecisionFreeCast__953", builder["matmul"](env.get("InsertedPrecisionFreeCast__952"), env.get("InsertedPrecisionFreeCast__512"), {"label":"/model/layers.2/mlp/down_proj/MatMul_Q4_matmul_618"}));
  env.set("_957", builder["add"](env.get("_956"), env.get("InsertedPrecisionFreeCast__953"), {"label":"/model/layers.3/input_layernorm/SkipLayerNorm_add_skip_622"}));
  env.set("_1049", builder.cast(env.get("_957"), "float16"));
  env.set("_1050", builder.cast(env.get("_1049"), "float32"));
  env.set("_498", builder.dequantizeLinear(env.get("_495"), env.get("_496"), env.get("_497"), {"axis":2,"blockSize":32,"label":"/model/layers.3/attn/o_proj/MatMul_Q4_dequantizeLinear_225"}));
  env.set("_499", builder.reshape(env.get("_498"), [2048,2048]));
  env.set("_500", builder["transpose"](env.get("_499"), {"label":"/model/layers.3/attn/o_proj/MatMul_Q4_transpose_dequantizeLinear_227","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__500", builder.cast(env.get("_500"), "float32"));
  env.set("Inserted_276", builder.cast(env.get("_601"), "uint8"));
  env.set("_969", builder["where"](env.get("Inserted_276"), env.get("_602"), env.get("_600"), {"label":"/model/layers.3/attn/GroupQueryAttention_/GQA/scatter/where_633"}));
  env.set("_970", builder["add"](env.get("_604"), env.get("_969"), {"label":"/model/layers.3/attn/GroupQueryAttention_/GQA/right_constant/add_635"}));
  env.set("_971", builder.concat([env.get("_606"), env.get("_970")], 1, {"label":"/model/layers.3/attn/GroupQueryAttention_/GQA/concat_for_pre_scatter_indices_636"}));
  env.set("_972", builder.reshape(env.get("_971"), [1,128,4,3]));
  env.set("Inserted_278", builder.cast(env.get("_972"), "int64"));
  env.set("Inserted_280", builder["max"](env.get("Inserted_278"), env.get("Inserted_279"), {"label":"Inserted_Max_640"}));
  env.set("Inserted_282", builder["min"](env.get("Inserted_280"), env.get("Inserted_281"), {"label":"Inserted_Min_641"}));
  env.set("_504", builder.dequantizeLinear(env.get("_501"), env.get("_502"), env.get("_503"), {"axis":2,"blockSize":32,"label":"/model/layers.3/attn/v_proj/MatMul_Q4_dequantizeLinear_228"}));
  env.set("_505", builder.reshape(env.get("_504"), [256,2048]));
  env.set("_506", builder["transpose"](env.get("_505"), {"label":"/model/layers.3/attn/v_proj/MatMul_Q4_transpose_dequantizeLinear_230","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__506", builder.cast(env.get("_506"), "float32"));
  env.set("_964", builder.cast(env.get("_963"), "float32"));
  env.set("_958", builder["pow"](env.get("_957"), env.get("_583"), {"label":"/model/layers.3/input_layernorm/SkipLayerNorm_pow_623"}));
  env.set("_959", builder["reduceMean"](env.get("_958"), {"axes":[2],"keepDimensions":true,"label":"/model/layers.3/input_layernorm/SkipLayerNorm_reduceMean_624"}));
  env.set("_960", builder["add"](env.get("_959"), env.get("_586"), {"label":"/model/layers.3/input_layernorm/SkipLayerNorm_add_625"}));
  env.set("_961", builder["sqrt"](env.get("_960"), {"label":"/model/layers.3/input_layernorm/SkipLayerNorm_sqrt_626"}));
  env.set("_962", builder["div"](env.get("_957"), env.get("_961"), {"label":"/model/layers.3/input_layernorm/SkipLayerNorm_div_627"}));
  env.set("_965", builder["mul"](env.get("_964"), env.get("_962"), {"label":"/model/layers.3/input_layernorm/SkipLayerNorm_mul_629"}));
  env.set("InsertedPrecisionFreeCast__967", builder["matmul"](env.get("_965"), env.get("InsertedPrecisionFreeCast__506"), {"label":"/model/layers.3/attn/v_proj/MatMul_Q4_matmul_631"}));
  env.set("_967", builder.cast(env.get("InsertedPrecisionFreeCast__967"), "float16"));
  env.set("_968", builder.reshape(env.get("_967"), [1,128,4,64]));
  env.set("present_3_value_7", builder["scatterND"](env.get("past_key_values_3_value_973"), env.get("Inserted_282"), env.get("_968"), {"label":"/model/layers.3/attn/GroupQueryAttention_/GQA/present_value/ScatterND_638"}));
  env.set("_974", builder.reshape(env.get("present_3_value_7"), [1,4,1,512,64]));
  env.set("_975", builder.expand(env.get("_974"), [1,4,8,512,64], {"label":"/model/layers.3/attn/GroupQueryAttention_/GQA/true_present_value/expand_643"}));
  env.set("_976", builder.reshape(env.get("_975"), [1,32,512,64]));
  env.set("InsertedPrecisionFreeCast__976", builder.cast(env.get("_976"), "float32"));
  env.set("Inserted_307", builder.cast(env.get("_972"), "int64"));
  env.set("Inserted_309", builder["max"](env.get("Inserted_307"), env.get("Inserted_308"), {"label":"Inserted_Max_678"}));
  env.set("Inserted_311", builder["min"](env.get("Inserted_309"), env.get("Inserted_310"), {"label":"Inserted_Min_679"}));
  env.set("Inserted_300", builder["clamp"](env.get("position_ids_625"), {"label":"Inserted_Clip_669","maxValue":2047,"minValue":-2048}));
  env.set("_1002", builder["gather"](env.get("_644"), env.get("Inserted_300"), {"axis":0,"label":"/model/layers.3/attn/k_rotary/RotaryEmbedding_gather_cos_668"}));
  env.set("_1003", builder.reshape(env.get("_1002"), [1,128,1,1,32]));
  env.set("InsertedPrecisionFreeCast__1003", builder.cast(env.get("_1003"), "float32"));
  env.set("_989", builder.dequantizeLinear(env.get("_986"), env.get("_987"), env.get("_988"), {"axis":2,"blockSize":32,"label":"/model/layers.3/attn/k_proj/MatMul_Q4_dequantizeLinear_657"}));
  env.set("_990", builder.reshape(env.get("_989"), [256,2048]));
  env.set("_991", builder["transpose"](env.get("_990"), {"label":"/model/layers.3/attn/k_proj/MatMul_Q4_transpose_dequantizeLinear_659","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__991", builder.cast(env.get("_991"), "float32"));
  env.set("InsertedPrecisionFreeCast__992", builder["matmul"](env.get("_965"), env.get("InsertedPrecisionFreeCast__991"), {"label":"/model/layers.3/attn/k_proj/MatMul_Q4_matmul_660"}));
  env.set("_992", builder.cast(env.get("InsertedPrecisionFreeCast__992"), "float16"));
  env.set("_993", builder.reshape(env.get("_992"), [1,128,4,64]));
  env.set("_994", builder.reshape(env.get("_993"), [1,128,4,2,32]));
  env.set("InsertedPrecisionFreeCast__994", builder.cast(env.get("_994"), "float32"));
  env.set("InsertedPrecisionFreeCast__1004", builder["mul"](env.get("InsertedPrecisionFreeCast__994"), env.get("InsertedPrecisionFreeCast__1003"), {"label":"/model/layers.3/attn/k_rotary/RotaryEmbedding_mul_cos_671"}));
  env.set("InsertedPrecisionFreeCast__1005", builder.reshape(env.get("InsertedPrecisionFreeCast__1004"), [1,128,4,64]));
  env.set("InsertedPrecisionFreeCast__999", builder.cast(env.get("_999"), "float32"));
  env.set("Inserted_291", builder["clamp"](env.get("position_ids_625"), {"label":"Inserted_Clip_655","maxValue":2047,"minValue":-2048}));
  env.set("_984", builder["gather"](env.get("_624"), env.get("Inserted_291"), {"axis":0,"label":"/model/layers.3/attn/k_rotary/RotaryEmbedding_gather_sin_654"}));
  env.set("_985", builder.reshape(env.get("_984"), [1,128,1,1,32]));
  env.set("InsertedPrecisionFreeCast__985", builder.cast(env.get("_985"), "float32"));
  {
    const tmp = builder.split(env.get("_994"), 2, {"axis":3,"label":"/model/layers.3/attn/k_rotary/RotaryEmbedding_split_partial_input0_663"});
    env.set("_995", tmp[0]);
    env.set("_996", tmp[1]);
  }
  env.set("_997", builder.concat([env.get("_996"), env.get("_995")], 3, {"label":"/model/layers.3/attn/k_rotary/RotaryEmbedding_concat_partial_input0_664"}));
  env.set("InsertedPrecisionFreeCast__997", builder.cast(env.get("_997"), "float32"));
  env.set("InsertedPrecisionFreeCast__998", builder["mul"](env.get("InsertedPrecisionFreeCast__997"), env.get("InsertedPrecisionFreeCast__985"), {"label":"/model/layers.3/attn/k_rotary/RotaryEmbedding_mul_sin_665"}));
  env.set("InsertedPrecisionFreeCast__1000", builder["mul"](env.get("InsertedPrecisionFreeCast__998"), env.get("InsertedPrecisionFreeCast__999"), {"label":"/model/layers.3/attn/k_rotary/RotaryEmbedding_mul_sign_666"}));
  env.set("InsertedPrecisionFreeCast__1001", builder.reshape(env.get("InsertedPrecisionFreeCast__1000"), [1,128,4,64]));
  env.set("InsertedPrecisionFreeCast__1006", builder["add"](env.get("InsertedPrecisionFreeCast__1005"), env.get("InsertedPrecisionFreeCast__1001"), {"label":"/model/layers.3/attn/k_rotary/RotaryEmbedding_add_mul_cos_sin_673"}));
  env.set("_1006", builder.cast(env.get("InsertedPrecisionFreeCast__1006"), "float16"));
  env.set("_1007", builder.reshape(env.get("_1006"), [1,128,256]));
  env.set("_1008", builder.reshape(env.get("_1007"), [1,128,4,64]));
  env.set("present_3_key_6", builder["scatterND"](env.get("past_key_values_3_key_1009"), env.get("Inserted_311"), env.get("_1008"), {"label":"/model/layers.3/attn/GroupQueryAttention_/GQA/present_key/ScatterND_676"}));
  env.set("_1010", builder.reshape(env.get("present_3_key_6"), [1,4,1,512,64]));
  env.set("_1011", builder.expand(env.get("_1010"), [1,4,8,512,64], {"label":"/model/layers.3/attn/GroupQueryAttention_/GQA/true_present_key/expand_681"}));
  env.set("_1012", builder.reshape(env.get("_1011"), [1,32,512,64]));
  env.set("_1013", builder["transpose"](env.get("_1012"), {"label":"/model/layers.3/attn/GroupQueryAttention_/GQA/present_key/transpose_683","permutation":[0,1,3,2]}));
  env.set("InsertedPrecisionFreeCast__1013", builder.cast(env.get("_1013"), "float32"));
  env.set("Inserted_324", builder["clamp"](env.get("position_ids_625"), {"label":"Inserted_Clip_699","maxValue":2047,"minValue":-2048}));
  env.set("_1032", builder["gather"](env.get("_644"), env.get("Inserted_324"), {"axis":0,"label":"/model/layers.3/attn/q_rotary/RotaryEmbedding_gather_cos_698"}));
  env.set("_1033", builder.reshape(env.get("_1032"), [1,128,1,1,32]));
  env.set("InsertedPrecisionFreeCast__1033", builder.cast(env.get("_1033"), "float32"));
  env.set("_1019", builder.dequantizeLinear(env.get("_1016"), env.get("_1017"), env.get("_1018"), {"axis":2,"blockSize":32,"label":"/model/layers.3/attn/q_proj/MatMul_Q4_dequantizeLinear_687"}));
  env.set("_1020", builder.reshape(env.get("_1019"), [2048,2048]));
  env.set("_1021", builder["transpose"](env.get("_1020"), {"label":"/model/layers.3/attn/q_proj/MatMul_Q4_transpose_dequantizeLinear_689","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__1021", builder.cast(env.get("_1021"), "float32"));
  env.set("InsertedPrecisionFreeCast__1022", builder["matmul"](env.get("_965"), env.get("InsertedPrecisionFreeCast__1021"), {"label":"/model/layers.3/attn/q_proj/MatMul_Q4_matmul_690"}));
  env.set("_1022", builder.cast(env.get("InsertedPrecisionFreeCast__1022"), "float16"));
  env.set("_1023", builder.reshape(env.get("_1022"), [1,128,32,64]));
  env.set("_1024", builder.reshape(env.get("_1023"), [1,128,32,2,32]));
  env.set("InsertedPrecisionFreeCast__1024", builder.cast(env.get("_1024"), "float32"));
  env.set("InsertedPrecisionFreeCast__1034", builder["mul"](env.get("InsertedPrecisionFreeCast__1024"), env.get("InsertedPrecisionFreeCast__1033"), {"label":"/model/layers.3/attn/q_rotary/RotaryEmbedding_mul_cos_701"}));
  env.set("InsertedPrecisionFreeCast__1035", builder.reshape(env.get("InsertedPrecisionFreeCast__1034"), [1,128,32,64]));
  env.set("InsertedPrecisionFreeCast__1029", builder.cast(env.get("_1029"), "float32"));
  env.set("Inserted_315", builder["clamp"](env.get("position_ids_625"), {"label":"Inserted_Clip_685","maxValue":2047,"minValue":-2048}));
  env.set("_1014", builder["gather"](env.get("_624"), env.get("Inserted_315"), {"axis":0,"label":"/model/layers.3/attn/q_rotary/RotaryEmbedding_gather_sin_684"}));
  env.set("_1015", builder.reshape(env.get("_1014"), [1,128,1,1,32]));
  env.set("InsertedPrecisionFreeCast__1015", builder.cast(env.get("_1015"), "float32"));
  {
    const tmp = builder.split(env.get("_1024"), 2, {"axis":3,"label":"/model/layers.3/attn/q_rotary/RotaryEmbedding_split_partial_input0_693"});
    env.set("_1025", tmp[0]);
    env.set("_1026", tmp[1]);
  }
  env.set("_1027", builder.concat([env.get("_1026"), env.get("_1025")], 3, {"label":"/model/layers.3/attn/q_rotary/RotaryEmbedding_concat_partial_input0_694"}));
  env.set("InsertedPrecisionFreeCast__1027", builder.cast(env.get("_1027"), "float32"));
  env.set("InsertedPrecisionFreeCast__1028", builder["mul"](env.get("InsertedPrecisionFreeCast__1027"), env.get("InsertedPrecisionFreeCast__1015"), {"label":"/model/layers.3/attn/q_rotary/RotaryEmbedding_mul_sin_695"}));
  env.set("InsertedPrecisionFreeCast__1030", builder["mul"](env.get("InsertedPrecisionFreeCast__1028"), env.get("InsertedPrecisionFreeCast__1029"), {"label":"/model/layers.3/attn/q_rotary/RotaryEmbedding_mul_sign_696"}));
  env.set("InsertedPrecisionFreeCast__1031", builder.reshape(env.get("InsertedPrecisionFreeCast__1030"), [1,128,32,64]));
  env.set("InsertedPrecisionFreeCast__1036", builder["add"](env.get("InsertedPrecisionFreeCast__1035"), env.get("InsertedPrecisionFreeCast__1031"), {"label":"/model/layers.3/attn/q_rotary/RotaryEmbedding_add_mul_cos_sin_703"}));
  env.set("_1036", builder.cast(env.get("InsertedPrecisionFreeCast__1036"), "float16"));
  env.set("_1037", builder.reshape(env.get("_1036"), [1,128,2048]));
  env.set("_1038", builder.reshape(env.get("_1037"), [1,128,32,64]));
  env.set("_1039", builder["transpose"](env.get("_1038"), {"label":"/model/layers.3/attn/GroupQueryAttention_/GQA/query/transpose_706","permutation":[0,2,1,3]}));
  env.set("InsertedPrecisionFreeCast__1039", builder.cast(env.get("_1039"), "float32"));
  env.set("InsertedPrecisionFreeCast__1040", builder["matmul"](env.get("InsertedPrecisionFreeCast__1039"), env.get("InsertedPrecisionFreeCast__1013"), {"label":"/model/layers.3/attn/GroupQueryAttention_/Attention/qkv/matmul_1_707"}));
  env.set("InsertedPrecisionFreeCast__1041", builder["mul"](env.get("InsertedPrecisionFreeCast__1040"), env.get("InsertedPrecisionFreeCast__684"), {"label":"/model/layers.3/attn/GroupQueryAttention_/Attention/qkv/div_708"}));
  env.set("_980", builder.expand(env.get("_617"), [1,32,128,512], {"label":"/model/layers.3/attn/GroupQueryAttention_/GQA/GQA_mask_shape_ones/expand_648"}));
  env.set("_981", builder.cumulativeSum(env.get("_980"), 3, {"exclusive":true,"label":"/model/layers.3/attn/GroupQueryAttention_range_of_mask_shape_649"}));
  env.set("_977", builder["add"](env.get("_613"), env.get("_969"), {"label":"/model/layers.3/attn/GroupQueryAttention_/GQA/attn_mask/add_645"}));
  env.set("_978", builder.expand(env.get("_977"), [512,128], {"label":"/model/layers.3/attn/GroupQueryAttention_/GQA/expand_neq_right_646"}));
  env.set("_979", builder["transpose"](env.get("_978"), {"label":"/model/layers.3/attn/GroupQueryAttention_/GQA/neq_right/transpose_647","permutation":[1,0]}));
  env.set("Inserted_289", builder["lesser"](env.get("_981"), env.get("_979"), {"label":"/model/layers.3/attn/GroupQueryAttention_/GQA/attn_mask/condition_1_650"}));
  env.set("InsertedPrecisionFreeCast__983", builder["where"](env.get("Inserted_289"), env.get("InsertedPrecisionFreeCast__621"), env.get("InsertedPrecisionFreeCast__622"), {"label":"/model/layers.3/attn/GroupQueryAttention_/GQA/attn_mask/where_652"}));
  env.set("InsertedPrecisionFreeCast__1042", builder["add"](env.get("InsertedPrecisionFreeCast__1041"), env.get("InsertedPrecisionFreeCast__983"), {"label":"/model/layers.3/attn/GroupQueryAttention_/Attention/attn_mask/softmax_input_709"}));
  env.set("InsertedPrecisionFreeCast__inlfunc_Softmax_token_2_X_ReduceMax", builder["reduceMax"](env.get("InsertedPrecisionFreeCast__1042"), {"axes":[3],"keepDimensions":true}));
  env.set("InsertedPrecisionFreeCast__inlfunc_Softmax_token_2_X_Sub", builder["sub"](env.get("InsertedPrecisionFreeCast__1042"), env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_2_X_ReduceMax"), {}));
  env.set("InsertedPrecisionFreeCast__inlfunc_Softmax_token_2_X_Exp", builder["exp"](env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_2_X_Sub"), {}));
  env.set("InsertedPrecisionFreeCast__inlfunc_Softmax_token_2_X_ReduceSum", builder["reduceSum"](env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_2_X_Exp"), {"axes":[3],"keepDimensions":true}));
  env.set("InsertedPrecisionFreeCast__1043", builder["div"](env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_2_X_Exp"), env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_2_X_ReduceSum"), {}));
  env.set("InsertedPrecisionFreeCast__1044", builder["matmul"](env.get("InsertedPrecisionFreeCast__1043"), env.get("InsertedPrecisionFreeCast__976"), {"label":"/model/layers.3/attn/GroupQueryAttention_/Attention/qkv/matmul_2_711"}));
  env.set("_1044", builder.cast(env.get("InsertedPrecisionFreeCast__1044"), "float16"));
  env.set("_1045", builder["transpose"](env.get("_1044"), {"label":"/model/layers.3/attn/GroupQueryAttention_/Attention/qkv/transpose_712","permutation":[0,2,1,3]}));
  env.set("_1046", builder.reshape(env.get("_1045"), [1,128,2048]));
  env.set("InsertedPrecisionFreeCast__1046", builder.cast(env.get("_1046"), "float32"));
  env.set("InsertedPrecisionFreeCast__1047", builder["matmul"](env.get("InsertedPrecisionFreeCast__1046"), env.get("InsertedPrecisionFreeCast__500"), {"label":"/model/layers.3/attn/o_proj/MatMul_Q4_matmul_714"}));
  env.set("_1051", builder["add"](env.get("_1050"), env.get("InsertedPrecisionFreeCast__1047"), {"label":"/model/layers.3/post_attention_layernorm/SkipLayerNorm_add_skip_718"}));
  env.set("_1074", builder.cast(env.get("_1051"), "float16"));
  env.set("_1075", builder.cast(env.get("_1074"), "float32"));
  env.set("_486", builder.dequantizeLinear(env.get("_483"), env.get("_484"), env.get("_485"), {"axis":2,"blockSize":32,"label":"/model/layers.3/mlp/down_proj/MatMul_Q4_dequantizeLinear_219"}));
  env.set("_487", builder.reshape(env.get("_486"), [2048,5632]));
  env.set("_488", builder["transpose"](env.get("_487"), {"label":"/model/layers.3/mlp/down_proj/MatMul_Q4_transpose_dequantizeLinear_221","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__488", builder.cast(env.get("_488"), "float32"));
  env.set("_1065", builder.dequantizeLinear(env.get("_1062"), env.get("_1063"), env.get("_1064"), {"axis":2,"blockSize":32,"label":"/model/layers.3/mlp/gate_proj/MatMul_Q4_dequantizeLinear_728"}));
  env.set("_1066", builder.reshape(env.get("_1065"), [5632,2048]));
  env.set("_1067", builder["transpose"](env.get("_1066"), {"label":"/model/layers.3/mlp/gate_proj/MatMul_Q4_transpose_dequantizeLinear_730","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__1067", builder.cast(env.get("_1067"), "float32"));
  env.set("_1058", builder.cast(env.get("_1057"), "float32"));
  env.set("_1052", builder["pow"](env.get("_1051"), env.get("_583"), {"label":"/model/layers.3/post_attention_layernorm/SkipLayerNorm_pow_719"}));
  env.set("_1053", builder["reduceMean"](env.get("_1052"), {"axes":[2],"keepDimensions":true,"label":"/model/layers.3/post_attention_layernorm/SkipLayerNorm_reduceMean_720"}));
  env.set("_1054", builder["add"](env.get("_1053"), env.get("_586"), {"label":"/model/layers.3/post_attention_layernorm/SkipLayerNorm_add_721"}));
  env.set("_1055", builder["sqrt"](env.get("_1054"), {"label":"/model/layers.3/post_attention_layernorm/SkipLayerNorm_sqrt_722"}));
  env.set("_1056", builder["div"](env.get("_1051"), env.get("_1055"), {"label":"/model/layers.3/post_attention_layernorm/SkipLayerNorm_div_723"}));
  env.set("_1059", builder["mul"](env.get("_1058"), env.get("_1056"), {"label":"/model/layers.3/post_attention_layernorm/SkipLayerNorm_mul_725"}));
  env.set("InsertedPrecisionFreeCast__1068", builder["matmul"](env.get("_1059"), env.get("InsertedPrecisionFreeCast__1067"), {"label":"/model/layers.3/mlp/gate_proj/MatMul_Q4_matmul_731"}));
  env.set("InsertedPrecisionFreeCast__1069", builder["sigmoid"](env.get("InsertedPrecisionFreeCast__1068"), {"label":"/model/layers.3/mlp/act_fn/Sigmoid_732"}));
  env.set("InsertedPrecisionFreeCast__1070", builder["mul"](env.get("InsertedPrecisionFreeCast__1068"), env.get("InsertedPrecisionFreeCast__1069"), {"label":"/model/layers.3/mlp/act_fn/Mul_733"}));
  env.set("_492", builder.dequantizeLinear(env.get("_489"), env.get("_490"), env.get("_491"), {"axis":2,"blockSize":32,"label":"/model/layers.3/mlp/up_proj/MatMul_Q4_dequantizeLinear_222"}));
  env.set("_493", builder.reshape(env.get("_492"), [5632,2048]));
  env.set("_494", builder["transpose"](env.get("_493"), {"label":"/model/layers.3/mlp/up_proj/MatMul_Q4_transpose_dequantizeLinear_224","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__494", builder.cast(env.get("_494"), "float32"));
  env.set("InsertedPrecisionFreeCast__1061", builder["matmul"](env.get("_1059"), env.get("InsertedPrecisionFreeCast__494"), {"label":"/model/layers.3/mlp/up_proj/MatMul_Q4_matmul_727"}));
  env.set("InsertedPrecisionFreeCast__1071", builder["mul"](env.get("InsertedPrecisionFreeCast__1070"), env.get("InsertedPrecisionFreeCast__1061"), {"label":"/model/layers.3/mlp/Mul_734"}));
  env.set("InsertedPrecisionFreeCast__1072", builder["matmul"](env.get("InsertedPrecisionFreeCast__1071"), env.get("InsertedPrecisionFreeCast__488"), {"label":"/model/layers.3/mlp/down_proj/MatMul_Q4_matmul_735"}));
  env.set("_1076", builder["add"](env.get("_1075"), env.get("InsertedPrecisionFreeCast__1072"), {"label":"/model/layers.4/input_layernorm/SkipLayerNorm_add_skip_739"}));
  env.set("_1168", builder.cast(env.get("_1076"), "float16"));
  env.set("_1169", builder.cast(env.get("_1168"), "float32"));
  env.set("_474", builder.dequantizeLinear(env.get("_471"), env.get("_472"), env.get("_473"), {"axis":2,"blockSize":32,"label":"/model/layers.4/attn/o_proj/MatMul_Q4_dequantizeLinear_213"}));
  env.set("_475", builder.reshape(env.get("_474"), [2048,2048]));
  env.set("_476", builder["transpose"](env.get("_475"), {"label":"/model/layers.4/attn/o_proj/MatMul_Q4_transpose_dequantizeLinear_215","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__476", builder.cast(env.get("_476"), "float32"));
  env.set("Inserted_336", builder.cast(env.get("_601"), "uint8"));
  env.set("_1088", builder["where"](env.get("Inserted_336"), env.get("_602"), env.get("_600"), {"label":"/model/layers.4/attn/GroupQueryAttention_/GQA/scatter/where_750"}));
  env.set("_1089", builder["add"](env.get("_604"), env.get("_1088"), {"label":"/model/layers.4/attn/GroupQueryAttention_/GQA/right_constant/add_752"}));
  env.set("_1090", builder.concat([env.get("_606"), env.get("_1089")], 1, {"label":"/model/layers.4/attn/GroupQueryAttention_/GQA/concat_for_pre_scatter_indices_753"}));
  env.set("_1091", builder.reshape(env.get("_1090"), [1,128,4,3]));
  env.set("Inserted_338", builder.cast(env.get("_1091"), "int64"));
  env.set("Inserted_340", builder["max"](env.get("Inserted_338"), env.get("Inserted_339"), {"label":"Inserted_Max_757"}));
  env.set("Inserted_342", builder["min"](env.get("Inserted_340"), env.get("Inserted_341"), {"label":"Inserted_Min_758"}));
  env.set("_480", builder.dequantizeLinear(env.get("_477"), env.get("_478"), env.get("_479"), {"axis":2,"blockSize":32,"label":"/model/layers.4/attn/v_proj/MatMul_Q4_dequantizeLinear_216"}));
  env.set("_481", builder.reshape(env.get("_480"), [256,2048]));
  env.set("_482", builder["transpose"](env.get("_481"), {"label":"/model/layers.4/attn/v_proj/MatMul_Q4_transpose_dequantizeLinear_218","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__482", builder.cast(env.get("_482"), "float32"));
  env.set("_1083", builder.cast(env.get("_1082"), "float32"));
  env.set("_1077", builder["pow"](env.get("_1076"), env.get("_583"), {"label":"/model/layers.4/input_layernorm/SkipLayerNorm_pow_740"}));
  env.set("_1078", builder["reduceMean"](env.get("_1077"), {"axes":[2],"keepDimensions":true,"label":"/model/layers.4/input_layernorm/SkipLayerNorm_reduceMean_741"}));
  env.set("_1079", builder["add"](env.get("_1078"), env.get("_586"), {"label":"/model/layers.4/input_layernorm/SkipLayerNorm_add_742"}));
  env.set("_1080", builder["sqrt"](env.get("_1079"), {"label":"/model/layers.4/input_layernorm/SkipLayerNorm_sqrt_743"}));
  env.set("_1081", builder["div"](env.get("_1076"), env.get("_1080"), {"label":"/model/layers.4/input_layernorm/SkipLayerNorm_div_744"}));
  env.set("_1084", builder["mul"](env.get("_1083"), env.get("_1081"), {"label":"/model/layers.4/input_layernorm/SkipLayerNorm_mul_746"}));
  env.set("InsertedPrecisionFreeCast__1086", builder["matmul"](env.get("_1084"), env.get("InsertedPrecisionFreeCast__482"), {"label":"/model/layers.4/attn/v_proj/MatMul_Q4_matmul_748"}));
  env.set("_1086", builder.cast(env.get("InsertedPrecisionFreeCast__1086"), "float16"));
  env.set("_1087", builder.reshape(env.get("_1086"), [1,128,4,64]));
  env.set("present_4_value_9", builder["scatterND"](env.get("past_key_values_4_value_1092"), env.get("Inserted_342"), env.get("_1087"), {"label":"/model/layers.4/attn/GroupQueryAttention_/GQA/present_value/ScatterND_755"}));
  env.set("_1093", builder.reshape(env.get("present_4_value_9"), [1,4,1,512,64]));
  env.set("_1094", builder.expand(env.get("_1093"), [1,4,8,512,64], {"label":"/model/layers.4/attn/GroupQueryAttention_/GQA/true_present_value/expand_760"}));
  env.set("_1095", builder.reshape(env.get("_1094"), [1,32,512,64]));
  env.set("InsertedPrecisionFreeCast__1095", builder.cast(env.get("_1095"), "float32"));
  env.set("Inserted_367", builder.cast(env.get("_1091"), "int64"));
  env.set("Inserted_369", builder["max"](env.get("Inserted_367"), env.get("Inserted_368"), {"label":"Inserted_Max_795"}));
  env.set("Inserted_371", builder["min"](env.get("Inserted_369"), env.get("Inserted_370"), {"label":"Inserted_Min_796"}));
  env.set("Inserted_360", builder["clamp"](env.get("position_ids_625"), {"label":"Inserted_Clip_786","maxValue":2047,"minValue":-2048}));
  env.set("_1121", builder["gather"](env.get("_644"), env.get("Inserted_360"), {"axis":0,"label":"/model/layers.4/attn/k_rotary/RotaryEmbedding_gather_cos_785"}));
  env.set("_1122", builder.reshape(env.get("_1121"), [1,128,1,1,32]));
  env.set("InsertedPrecisionFreeCast__1122", builder.cast(env.get("_1122"), "float32"));
  env.set("_1108", builder.dequantizeLinear(env.get("_1105"), env.get("_1106"), env.get("_1107"), {"axis":2,"blockSize":32,"label":"/model/layers.4/attn/k_proj/MatMul_Q4_dequantizeLinear_774"}));
  env.set("_1109", builder.reshape(env.get("_1108"), [256,2048]));
  env.set("_1110", builder["transpose"](env.get("_1109"), {"label":"/model/layers.4/attn/k_proj/MatMul_Q4_transpose_dequantizeLinear_776","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__1110", builder.cast(env.get("_1110"), "float32"));
  env.set("InsertedPrecisionFreeCast__1111", builder["matmul"](env.get("_1084"), env.get("InsertedPrecisionFreeCast__1110"), {"label":"/model/layers.4/attn/k_proj/MatMul_Q4_matmul_777"}));
  env.set("_1111", builder.cast(env.get("InsertedPrecisionFreeCast__1111"), "float16"));
  env.set("_1112", builder.reshape(env.get("_1111"), [1,128,4,64]));
  env.set("_1113", builder.reshape(env.get("_1112"), [1,128,4,2,32]));
  env.set("InsertedPrecisionFreeCast__1113", builder.cast(env.get("_1113"), "float32"));
  env.set("InsertedPrecisionFreeCast__1123", builder["mul"](env.get("InsertedPrecisionFreeCast__1113"), env.get("InsertedPrecisionFreeCast__1122"), {"label":"/model/layers.4/attn/k_rotary/RotaryEmbedding_mul_cos_788"}));
  env.set("InsertedPrecisionFreeCast__1124", builder.reshape(env.get("InsertedPrecisionFreeCast__1123"), [1,128,4,64]));
  env.set("InsertedPrecisionFreeCast__1118", builder.cast(env.get("_1118"), "float32"));
  env.set("Inserted_351", builder["clamp"](env.get("position_ids_625"), {"label":"Inserted_Clip_772","maxValue":2047,"minValue":-2048}));
  env.set("_1103", builder["gather"](env.get("_624"), env.get("Inserted_351"), {"axis":0,"label":"/model/layers.4/attn/k_rotary/RotaryEmbedding_gather_sin_771"}));
  env.set("_1104", builder.reshape(env.get("_1103"), [1,128,1,1,32]));
  env.set("InsertedPrecisionFreeCast__1104", builder.cast(env.get("_1104"), "float32"));
  {
    const tmp = builder.split(env.get("_1113"), 2, {"axis":3,"label":"/model/layers.4/attn/k_rotary/RotaryEmbedding_split_partial_input0_780"});
    env.set("_1114", tmp[0]);
    env.set("_1115", tmp[1]);
  }
  env.set("_1116", builder.concat([env.get("_1115"), env.get("_1114")], 3, {"label":"/model/layers.4/attn/k_rotary/RotaryEmbedding_concat_partial_input0_781"}));
  env.set("InsertedPrecisionFreeCast__1116", builder.cast(env.get("_1116"), "float32"));
  env.set("InsertedPrecisionFreeCast__1117", builder["mul"](env.get("InsertedPrecisionFreeCast__1116"), env.get("InsertedPrecisionFreeCast__1104"), {"label":"/model/layers.4/attn/k_rotary/RotaryEmbedding_mul_sin_782"}));
  env.set("InsertedPrecisionFreeCast__1119", builder["mul"](env.get("InsertedPrecisionFreeCast__1117"), env.get("InsertedPrecisionFreeCast__1118"), {"label":"/model/layers.4/attn/k_rotary/RotaryEmbedding_mul_sign_783"}));
  env.set("InsertedPrecisionFreeCast__1120", builder.reshape(env.get("InsertedPrecisionFreeCast__1119"), [1,128,4,64]));
  env.set("InsertedPrecisionFreeCast__1125", builder["add"](env.get("InsertedPrecisionFreeCast__1124"), env.get("InsertedPrecisionFreeCast__1120"), {"label":"/model/layers.4/attn/k_rotary/RotaryEmbedding_add_mul_cos_sin_790"}));
  env.set("_1125", builder.cast(env.get("InsertedPrecisionFreeCast__1125"), "float16"));
  env.set("_1126", builder.reshape(env.get("_1125"), [1,128,256]));
  env.set("_1127", builder.reshape(env.get("_1126"), [1,128,4,64]));
  env.set("present_4_key_8", builder["scatterND"](env.get("past_key_values_4_key_1128"), env.get("Inserted_371"), env.get("_1127"), {"label":"/model/layers.4/attn/GroupQueryAttention_/GQA/present_key/ScatterND_793"}));
  env.set("_1129", builder.reshape(env.get("present_4_key_8"), [1,4,1,512,64]));
  env.set("_1130", builder.expand(env.get("_1129"), [1,4,8,512,64], {"label":"/model/layers.4/attn/GroupQueryAttention_/GQA/true_present_key/expand_798"}));
  env.set("_1131", builder.reshape(env.get("_1130"), [1,32,512,64]));
  env.set("_1132", builder["transpose"](env.get("_1131"), {"label":"/model/layers.4/attn/GroupQueryAttention_/GQA/present_key/transpose_800","permutation":[0,1,3,2]}));
  env.set("InsertedPrecisionFreeCast__1132", builder.cast(env.get("_1132"), "float32"));
  env.set("Inserted_384", builder["clamp"](env.get("position_ids_625"), {"label":"Inserted_Clip_816","maxValue":2047,"minValue":-2048}));
  env.set("_1151", builder["gather"](env.get("_644"), env.get("Inserted_384"), {"axis":0,"label":"/model/layers.4/attn/q_rotary/RotaryEmbedding_gather_cos_815"}));
  env.set("_1152", builder.reshape(env.get("_1151"), [1,128,1,1,32]));
  env.set("InsertedPrecisionFreeCast__1152", builder.cast(env.get("_1152"), "float32"));
  env.set("_1138", builder.dequantizeLinear(env.get("_1135"), env.get("_1136"), env.get("_1137"), {"axis":2,"blockSize":32,"label":"/model/layers.4/attn/q_proj/MatMul_Q4_dequantizeLinear_804"}));
  env.set("_1139", builder.reshape(env.get("_1138"), [2048,2048]));
  env.set("_1140", builder["transpose"](env.get("_1139"), {"label":"/model/layers.4/attn/q_proj/MatMul_Q4_transpose_dequantizeLinear_806","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__1140", builder.cast(env.get("_1140"), "float32"));
  env.set("InsertedPrecisionFreeCast__1141", builder["matmul"](env.get("_1084"), env.get("InsertedPrecisionFreeCast__1140"), {"label":"/model/layers.4/attn/q_proj/MatMul_Q4_matmul_807"}));
  env.set("_1141", builder.cast(env.get("InsertedPrecisionFreeCast__1141"), "float16"));
  env.set("_1142", builder.reshape(env.get("_1141"), [1,128,32,64]));
  env.set("_1143", builder.reshape(env.get("_1142"), [1,128,32,2,32]));
  env.set("InsertedPrecisionFreeCast__1143", builder.cast(env.get("_1143"), "float32"));
  env.set("InsertedPrecisionFreeCast__1153", builder["mul"](env.get("InsertedPrecisionFreeCast__1143"), env.get("InsertedPrecisionFreeCast__1152"), {"label":"/model/layers.4/attn/q_rotary/RotaryEmbedding_mul_cos_818"}));
  env.set("InsertedPrecisionFreeCast__1154", builder.reshape(env.get("InsertedPrecisionFreeCast__1153"), [1,128,32,64]));
  env.set("InsertedPrecisionFreeCast__1148", builder.cast(env.get("_1148"), "float32"));
  env.set("Inserted_375", builder["clamp"](env.get("position_ids_625"), {"label":"Inserted_Clip_802","maxValue":2047,"minValue":-2048}));
  env.set("_1133", builder["gather"](env.get("_624"), env.get("Inserted_375"), {"axis":0,"label":"/model/layers.4/attn/q_rotary/RotaryEmbedding_gather_sin_801"}));
  env.set("_1134", builder.reshape(env.get("_1133"), [1,128,1,1,32]));
  env.set("InsertedPrecisionFreeCast__1134", builder.cast(env.get("_1134"), "float32"));
  {
    const tmp = builder.split(env.get("_1143"), 2, {"axis":3,"label":"/model/layers.4/attn/q_rotary/RotaryEmbedding_split_partial_input0_810"});
    env.set("_1144", tmp[0]);
    env.set("_1145", tmp[1]);
  }
  env.set("_1146", builder.concat([env.get("_1145"), env.get("_1144")], 3, {"label":"/model/layers.4/attn/q_rotary/RotaryEmbedding_concat_partial_input0_811"}));
  env.set("InsertedPrecisionFreeCast__1146", builder.cast(env.get("_1146"), "float32"));
  env.set("InsertedPrecisionFreeCast__1147", builder["mul"](env.get("InsertedPrecisionFreeCast__1146"), env.get("InsertedPrecisionFreeCast__1134"), {"label":"/model/layers.4/attn/q_rotary/RotaryEmbedding_mul_sin_812"}));
  env.set("InsertedPrecisionFreeCast__1149", builder["mul"](env.get("InsertedPrecisionFreeCast__1147"), env.get("InsertedPrecisionFreeCast__1148"), {"label":"/model/layers.4/attn/q_rotary/RotaryEmbedding_mul_sign_813"}));
  env.set("InsertedPrecisionFreeCast__1150", builder.reshape(env.get("InsertedPrecisionFreeCast__1149"), [1,128,32,64]));
  env.set("InsertedPrecisionFreeCast__1155", builder["add"](env.get("InsertedPrecisionFreeCast__1154"), env.get("InsertedPrecisionFreeCast__1150"), {"label":"/model/layers.4/attn/q_rotary/RotaryEmbedding_add_mul_cos_sin_820"}));
  env.set("_1155", builder.cast(env.get("InsertedPrecisionFreeCast__1155"), "float16"));
  env.set("_1156", builder.reshape(env.get("_1155"), [1,128,2048]));
  env.set("_1157", builder.reshape(env.get("_1156"), [1,128,32,64]));
  env.set("_1158", builder["transpose"](env.get("_1157"), {"label":"/model/layers.4/attn/GroupQueryAttention_/GQA/query/transpose_823","permutation":[0,2,1,3]}));
  env.set("InsertedPrecisionFreeCast__1158", builder.cast(env.get("_1158"), "float32"));
  env.set("InsertedPrecisionFreeCast__1159", builder["matmul"](env.get("InsertedPrecisionFreeCast__1158"), env.get("InsertedPrecisionFreeCast__1132"), {"label":"/model/layers.4/attn/GroupQueryAttention_/Attention/qkv/matmul_1_824"}));
  env.set("InsertedPrecisionFreeCast__1160", builder["mul"](env.get("InsertedPrecisionFreeCast__1159"), env.get("InsertedPrecisionFreeCast__684"), {"label":"/model/layers.4/attn/GroupQueryAttention_/Attention/qkv/div_825"}));
  env.set("_1099", builder.expand(env.get("_617"), [1,32,128,512], {"label":"/model/layers.4/attn/GroupQueryAttention_/GQA/GQA_mask_shape_ones/expand_765"}));
  env.set("_1100", builder.cumulativeSum(env.get("_1099"), 3, {"exclusive":true,"label":"/model/layers.4/attn/GroupQueryAttention_range_of_mask_shape_766"}));
  env.set("_1096", builder["add"](env.get("_613"), env.get("_1088"), {"label":"/model/layers.4/attn/GroupQueryAttention_/GQA/attn_mask/add_762"}));
  env.set("_1097", builder.expand(env.get("_1096"), [512,128], {"label":"/model/layers.4/attn/GroupQueryAttention_/GQA/expand_neq_right_763"}));
  env.set("_1098", builder["transpose"](env.get("_1097"), {"label":"/model/layers.4/attn/GroupQueryAttention_/GQA/neq_right/transpose_764","permutation":[1,0]}));
  env.set("Inserted_349", builder["lesser"](env.get("_1100"), env.get("_1098"), {"label":"/model/layers.4/attn/GroupQueryAttention_/GQA/attn_mask/condition_1_767"}));
  env.set("InsertedPrecisionFreeCast__1102", builder["where"](env.get("Inserted_349"), env.get("InsertedPrecisionFreeCast__621"), env.get("InsertedPrecisionFreeCast__622"), {"label":"/model/layers.4/attn/GroupQueryAttention_/GQA/attn_mask/where_769"}));
  env.set("InsertedPrecisionFreeCast__1161", builder["add"](env.get("InsertedPrecisionFreeCast__1160"), env.get("InsertedPrecisionFreeCast__1102"), {"label":"/model/layers.4/attn/GroupQueryAttention_/Attention/attn_mask/softmax_input_826"}));
  env.set("InsertedPrecisionFreeCast__inlfunc_Softmax_token_3_X_ReduceMax", builder["reduceMax"](env.get("InsertedPrecisionFreeCast__1161"), {"axes":[3],"keepDimensions":true}));
  env.set("InsertedPrecisionFreeCast__inlfunc_Softmax_token_3_X_Sub", builder["sub"](env.get("InsertedPrecisionFreeCast__1161"), env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_3_X_ReduceMax"), {}));
  env.set("InsertedPrecisionFreeCast__inlfunc_Softmax_token_3_X_Exp", builder["exp"](env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_3_X_Sub"), {}));
  env.set("InsertedPrecisionFreeCast__inlfunc_Softmax_token_3_X_ReduceSum", builder["reduceSum"](env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_3_X_Exp"), {"axes":[3],"keepDimensions":true}));
  env.set("InsertedPrecisionFreeCast__1162", builder["div"](env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_3_X_Exp"), env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_3_X_ReduceSum"), {}));
  env.set("InsertedPrecisionFreeCast__1163", builder["matmul"](env.get("InsertedPrecisionFreeCast__1162"), env.get("InsertedPrecisionFreeCast__1095"), {"label":"/model/layers.4/attn/GroupQueryAttention_/Attention/qkv/matmul_2_828"}));
  env.set("_1163", builder.cast(env.get("InsertedPrecisionFreeCast__1163"), "float16"));
  env.set("_1164", builder["transpose"](env.get("_1163"), {"label":"/model/layers.4/attn/GroupQueryAttention_/Attention/qkv/transpose_829","permutation":[0,2,1,3]}));
  env.set("_1165", builder.reshape(env.get("_1164"), [1,128,2048]));
  env.set("InsertedPrecisionFreeCast__1165", builder.cast(env.get("_1165"), "float32"));
  env.set("InsertedPrecisionFreeCast__1166", builder["matmul"](env.get("InsertedPrecisionFreeCast__1165"), env.get("InsertedPrecisionFreeCast__476"), {"label":"/model/layers.4/attn/o_proj/MatMul_Q4_matmul_831"}));
  env.set("_1170", builder["add"](env.get("_1169"), env.get("InsertedPrecisionFreeCast__1166"), {"label":"/model/layers.4/post_attention_layernorm/SkipLayerNorm_add_skip_835"}));
  env.set("_1193", builder.cast(env.get("_1170"), "float16"));
  env.set("_1194", builder.cast(env.get("_1193"), "float32"));
  env.set("_462", builder.dequantizeLinear(env.get("_459"), env.get("_460"), env.get("_461"), {"axis":2,"blockSize":32,"label":"/model/layers.4/mlp/down_proj/MatMul_Q4_dequantizeLinear_207"}));
  env.set("_463", builder.reshape(env.get("_462"), [2048,5632]));
  env.set("_464", builder["transpose"](env.get("_463"), {"label":"/model/layers.4/mlp/down_proj/MatMul_Q4_transpose_dequantizeLinear_209","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__464", builder.cast(env.get("_464"), "float32"));
  env.set("_1184", builder.dequantizeLinear(env.get("_1181"), env.get("_1182"), env.get("_1183"), {"axis":2,"blockSize":32,"label":"/model/layers.4/mlp/gate_proj/MatMul_Q4_dequantizeLinear_845"}));
  env.set("_1185", builder.reshape(env.get("_1184"), [5632,2048]));
  env.set("_1186", builder["transpose"](env.get("_1185"), {"label":"/model/layers.4/mlp/gate_proj/MatMul_Q4_transpose_dequantizeLinear_847","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__1186", builder.cast(env.get("_1186"), "float32"));
  env.set("_1177", builder.cast(env.get("_1176"), "float32"));
  env.set("_1171", builder["pow"](env.get("_1170"), env.get("_583"), {"label":"/model/layers.4/post_attention_layernorm/SkipLayerNorm_pow_836"}));
  env.set("_1172", builder["reduceMean"](env.get("_1171"), {"axes":[2],"keepDimensions":true,"label":"/model/layers.4/post_attention_layernorm/SkipLayerNorm_reduceMean_837"}));
  env.set("_1173", builder["add"](env.get("_1172"), env.get("_586"), {"label":"/model/layers.4/post_attention_layernorm/SkipLayerNorm_add_838"}));
  env.set("_1174", builder["sqrt"](env.get("_1173"), {"label":"/model/layers.4/post_attention_layernorm/SkipLayerNorm_sqrt_839"}));
  env.set("_1175", builder["div"](env.get("_1170"), env.get("_1174"), {"label":"/model/layers.4/post_attention_layernorm/SkipLayerNorm_div_840"}));
  env.set("_1178", builder["mul"](env.get("_1177"), env.get("_1175"), {"label":"/model/layers.4/post_attention_layernorm/SkipLayerNorm_mul_842"}));
  env.set("InsertedPrecisionFreeCast__1187", builder["matmul"](env.get("_1178"), env.get("InsertedPrecisionFreeCast__1186"), {"label":"/model/layers.4/mlp/gate_proj/MatMul_Q4_matmul_848"}));
  env.set("InsertedPrecisionFreeCast__1188", builder["sigmoid"](env.get("InsertedPrecisionFreeCast__1187"), {"label":"/model/layers.4/mlp/act_fn/Sigmoid_849"}));
  env.set("InsertedPrecisionFreeCast__1189", builder["mul"](env.get("InsertedPrecisionFreeCast__1187"), env.get("InsertedPrecisionFreeCast__1188"), {"label":"/model/layers.4/mlp/act_fn/Mul_850"}));
  env.set("_468", builder.dequantizeLinear(env.get("_465"), env.get("_466"), env.get("_467"), {"axis":2,"blockSize":32,"label":"/model/layers.4/mlp/up_proj/MatMul_Q4_dequantizeLinear_210"}));
  env.set("_469", builder.reshape(env.get("_468"), [5632,2048]));
  env.set("_470", builder["transpose"](env.get("_469"), {"label":"/model/layers.4/mlp/up_proj/MatMul_Q4_transpose_dequantizeLinear_212","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__470", builder.cast(env.get("_470"), "float32"));
  env.set("InsertedPrecisionFreeCast__1180", builder["matmul"](env.get("_1178"), env.get("InsertedPrecisionFreeCast__470"), {"label":"/model/layers.4/mlp/up_proj/MatMul_Q4_matmul_844"}));
  env.set("InsertedPrecisionFreeCast__1190", builder["mul"](env.get("InsertedPrecisionFreeCast__1189"), env.get("InsertedPrecisionFreeCast__1180"), {"label":"/model/layers.4/mlp/Mul_851"}));
  env.set("InsertedPrecisionFreeCast__1191", builder["matmul"](env.get("InsertedPrecisionFreeCast__1190"), env.get("InsertedPrecisionFreeCast__464"), {"label":"/model/layers.4/mlp/down_proj/MatMul_Q4_matmul_852"}));
  env.set("_1195", builder["add"](env.get("_1194"), env.get("InsertedPrecisionFreeCast__1191"), {"label":"/model/layers.5/input_layernorm/SkipLayerNorm_add_skip_856"}));
  env.set("_1287", builder.cast(env.get("_1195"), "float16"));
  env.set("_1288", builder.cast(env.get("_1287"), "float32"));
  env.set("_450", builder.dequantizeLinear(env.get("_447"), env.get("_448"), env.get("_449"), {"axis":2,"blockSize":32,"label":"/model/layers.5/attn/o_proj/MatMul_Q4_dequantizeLinear_201"}));
  env.set("_451", builder.reshape(env.get("_450"), [2048,2048]));
  env.set("_452", builder["transpose"](env.get("_451"), {"label":"/model/layers.5/attn/o_proj/MatMul_Q4_transpose_dequantizeLinear_203","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__452", builder.cast(env.get("_452"), "float32"));
  env.set("Inserted_396", builder.cast(env.get("_601"), "uint8"));
  env.set("_1207", builder["where"](env.get("Inserted_396"), env.get("_602"), env.get("_600"), {"label":"/model/layers.5/attn/GroupQueryAttention_/GQA/scatter/where_867"}));
  env.set("_1208", builder["add"](env.get("_604"), env.get("_1207"), {"label":"/model/layers.5/attn/GroupQueryAttention_/GQA/right_constant/add_869"}));
  env.set("_1209", builder.concat([env.get("_606"), env.get("_1208")], 1, {"label":"/model/layers.5/attn/GroupQueryAttention_/GQA/concat_for_pre_scatter_indices_870"}));
  env.set("_1210", builder.reshape(env.get("_1209"), [1,128,4,3]));
  env.set("Inserted_398", builder.cast(env.get("_1210"), "int64"));
  env.set("Inserted_400", builder["max"](env.get("Inserted_398"), env.get("Inserted_399"), {"label":"Inserted_Max_874"}));
  env.set("Inserted_402", builder["min"](env.get("Inserted_400"), env.get("Inserted_401"), {"label":"Inserted_Min_875"}));
  env.set("_456", builder.dequantizeLinear(env.get("_453"), env.get("_454"), env.get("_455"), {"axis":2,"blockSize":32,"label":"/model/layers.5/attn/v_proj/MatMul_Q4_dequantizeLinear_204"}));
  env.set("_457", builder.reshape(env.get("_456"), [256,2048]));
  env.set("_458", builder["transpose"](env.get("_457"), {"label":"/model/layers.5/attn/v_proj/MatMul_Q4_transpose_dequantizeLinear_206","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__458", builder.cast(env.get("_458"), "float32"));
  env.set("_1202", builder.cast(env.get("_1201"), "float32"));
  env.set("_1196", builder["pow"](env.get("_1195"), env.get("_583"), {"label":"/model/layers.5/input_layernorm/SkipLayerNorm_pow_857"}));
  env.set("_1197", builder["reduceMean"](env.get("_1196"), {"axes":[2],"keepDimensions":true,"label":"/model/layers.5/input_layernorm/SkipLayerNorm_reduceMean_858"}));
  env.set("_1198", builder["add"](env.get("_1197"), env.get("_586"), {"label":"/model/layers.5/input_layernorm/SkipLayerNorm_add_859"}));
  env.set("_1199", builder["sqrt"](env.get("_1198"), {"label":"/model/layers.5/input_layernorm/SkipLayerNorm_sqrt_860"}));
  env.set("_1200", builder["div"](env.get("_1195"), env.get("_1199"), {"label":"/model/layers.5/input_layernorm/SkipLayerNorm_div_861"}));
  env.set("_1203", builder["mul"](env.get("_1202"), env.get("_1200"), {"label":"/model/layers.5/input_layernorm/SkipLayerNorm_mul_863"}));
  env.set("InsertedPrecisionFreeCast__1205", builder["matmul"](env.get("_1203"), env.get("InsertedPrecisionFreeCast__458"), {"label":"/model/layers.5/attn/v_proj/MatMul_Q4_matmul_865"}));
  env.set("_1205", builder.cast(env.get("InsertedPrecisionFreeCast__1205"), "float16"));
  env.set("_1206", builder.reshape(env.get("_1205"), [1,128,4,64]));
  env.set("present_5_value_11", builder["scatterND"](env.get("past_key_values_5_value_1211"), env.get("Inserted_402"), env.get("_1206"), {"label":"/model/layers.5/attn/GroupQueryAttention_/GQA/present_value/ScatterND_872"}));
  env.set("_1212", builder.reshape(env.get("present_5_value_11"), [1,4,1,512,64]));
  env.set("_1213", builder.expand(env.get("_1212"), [1,4,8,512,64], {"label":"/model/layers.5/attn/GroupQueryAttention_/GQA/true_present_value/expand_877"}));
  env.set("_1214", builder.reshape(env.get("_1213"), [1,32,512,64]));
  env.set("InsertedPrecisionFreeCast__1214", builder.cast(env.get("_1214"), "float32"));
  env.set("Inserted_427", builder.cast(env.get("_1210"), "int64"));
  env.set("Inserted_429", builder["max"](env.get("Inserted_427"), env.get("Inserted_428"), {"label":"Inserted_Max_912"}));
  env.set("Inserted_431", builder["min"](env.get("Inserted_429"), env.get("Inserted_430"), {"label":"Inserted_Min_913"}));
  env.set("Inserted_420", builder["clamp"](env.get("position_ids_625"), {"label":"Inserted_Clip_903","maxValue":2047,"minValue":-2048}));
  env.set("_1240", builder["gather"](env.get("_644"), env.get("Inserted_420"), {"axis":0,"label":"/model/layers.5/attn/k_rotary/RotaryEmbedding_gather_cos_902"}));
  env.set("_1241", builder.reshape(env.get("_1240"), [1,128,1,1,32]));
  env.set("InsertedPrecisionFreeCast__1241", builder.cast(env.get("_1241"), "float32"));
  env.set("_1227", builder.dequantizeLinear(env.get("_1224"), env.get("_1225"), env.get("_1226"), {"axis":2,"blockSize":32,"label":"/model/layers.5/attn/k_proj/MatMul_Q4_dequantizeLinear_891"}));
  env.set("_1228", builder.reshape(env.get("_1227"), [256,2048]));
  env.set("_1229", builder["transpose"](env.get("_1228"), {"label":"/model/layers.5/attn/k_proj/MatMul_Q4_transpose_dequantizeLinear_893","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__1229", builder.cast(env.get("_1229"), "float32"));
  env.set("InsertedPrecisionFreeCast__1230", builder["matmul"](env.get("_1203"), env.get("InsertedPrecisionFreeCast__1229"), {"label":"/model/layers.5/attn/k_proj/MatMul_Q4_matmul_894"}));
  env.set("_1230", builder.cast(env.get("InsertedPrecisionFreeCast__1230"), "float16"));
  env.set("_1231", builder.reshape(env.get("_1230"), [1,128,4,64]));
  env.set("_1232", builder.reshape(env.get("_1231"), [1,128,4,2,32]));
  env.set("InsertedPrecisionFreeCast__1232", builder.cast(env.get("_1232"), "float32"));
  env.set("InsertedPrecisionFreeCast__1242", builder["mul"](env.get("InsertedPrecisionFreeCast__1232"), env.get("InsertedPrecisionFreeCast__1241"), {"label":"/model/layers.5/attn/k_rotary/RotaryEmbedding_mul_cos_905"}));
  env.set("InsertedPrecisionFreeCast__1243", builder.reshape(env.get("InsertedPrecisionFreeCast__1242"), [1,128,4,64]));
  env.set("InsertedPrecisionFreeCast__1237", builder.cast(env.get("_1237"), "float32"));
  env.set("Inserted_411", builder["clamp"](env.get("position_ids_625"), {"label":"Inserted_Clip_889","maxValue":2047,"minValue":-2048}));
  env.set("_1222", builder["gather"](env.get("_624"), env.get("Inserted_411"), {"axis":0,"label":"/model/layers.5/attn/k_rotary/RotaryEmbedding_gather_sin_888"}));
  env.set("_1223", builder.reshape(env.get("_1222"), [1,128,1,1,32]));
  env.set("InsertedPrecisionFreeCast__1223", builder.cast(env.get("_1223"), "float32"));
  {
    const tmp = builder.split(env.get("_1232"), 2, {"axis":3,"label":"/model/layers.5/attn/k_rotary/RotaryEmbedding_split_partial_input0_897"});
    env.set("_1233", tmp[0]);
    env.set("_1234", tmp[1]);
  }
  env.set("_1235", builder.concat([env.get("_1234"), env.get("_1233")], 3, {"label":"/model/layers.5/attn/k_rotary/RotaryEmbedding_concat_partial_input0_898"}));
  env.set("InsertedPrecisionFreeCast__1235", builder.cast(env.get("_1235"), "float32"));
  env.set("InsertedPrecisionFreeCast__1236", builder["mul"](env.get("InsertedPrecisionFreeCast__1235"), env.get("InsertedPrecisionFreeCast__1223"), {"label":"/model/layers.5/attn/k_rotary/RotaryEmbedding_mul_sin_899"}));
  env.set("InsertedPrecisionFreeCast__1238", builder["mul"](env.get("InsertedPrecisionFreeCast__1236"), env.get("InsertedPrecisionFreeCast__1237"), {"label":"/model/layers.5/attn/k_rotary/RotaryEmbedding_mul_sign_900"}));
  env.set("InsertedPrecisionFreeCast__1239", builder.reshape(env.get("InsertedPrecisionFreeCast__1238"), [1,128,4,64]));
  env.set("InsertedPrecisionFreeCast__1244", builder["add"](env.get("InsertedPrecisionFreeCast__1243"), env.get("InsertedPrecisionFreeCast__1239"), {"label":"/model/layers.5/attn/k_rotary/RotaryEmbedding_add_mul_cos_sin_907"}));
  env.set("_1244", builder.cast(env.get("InsertedPrecisionFreeCast__1244"), "float16"));
  env.set("_1245", builder.reshape(env.get("_1244"), [1,128,256]));
  env.set("_1246", builder.reshape(env.get("_1245"), [1,128,4,64]));
  env.set("present_5_key_10", builder["scatterND"](env.get("past_key_values_5_key_1247"), env.get("Inserted_431"), env.get("_1246"), {"label":"/model/layers.5/attn/GroupQueryAttention_/GQA/present_key/ScatterND_910"}));
  env.set("_1248", builder.reshape(env.get("present_5_key_10"), [1,4,1,512,64]));
  env.set("_1249", builder.expand(env.get("_1248"), [1,4,8,512,64], {"label":"/model/layers.5/attn/GroupQueryAttention_/GQA/true_present_key/expand_915"}));
  env.set("_1250", builder.reshape(env.get("_1249"), [1,32,512,64]));
  env.set("_1251", builder["transpose"](env.get("_1250"), {"label":"/model/layers.5/attn/GroupQueryAttention_/GQA/present_key/transpose_917","permutation":[0,1,3,2]}));
  env.set("InsertedPrecisionFreeCast__1251", builder.cast(env.get("_1251"), "float32"));
  env.set("Inserted_444", builder["clamp"](env.get("position_ids_625"), {"label":"Inserted_Clip_933","maxValue":2047,"minValue":-2048}));
  env.set("_1270", builder["gather"](env.get("_644"), env.get("Inserted_444"), {"axis":0,"label":"/model/layers.5/attn/q_rotary/RotaryEmbedding_gather_cos_932"}));
  env.set("_1271", builder.reshape(env.get("_1270"), [1,128,1,1,32]));
  env.set("InsertedPrecisionFreeCast__1271", builder.cast(env.get("_1271"), "float32"));
  env.set("_1257", builder.dequantizeLinear(env.get("_1254"), env.get("_1255"), env.get("_1256"), {"axis":2,"blockSize":32,"label":"/model/layers.5/attn/q_proj/MatMul_Q4_dequantizeLinear_921"}));
  env.set("_1258", builder.reshape(env.get("_1257"), [2048,2048]));
  env.set("_1259", builder["transpose"](env.get("_1258"), {"label":"/model/layers.5/attn/q_proj/MatMul_Q4_transpose_dequantizeLinear_923","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__1259", builder.cast(env.get("_1259"), "float32"));
  env.set("InsertedPrecisionFreeCast__1260", builder["matmul"](env.get("_1203"), env.get("InsertedPrecisionFreeCast__1259"), {"label":"/model/layers.5/attn/q_proj/MatMul_Q4_matmul_924"}));
  env.set("_1260", builder.cast(env.get("InsertedPrecisionFreeCast__1260"), "float16"));
  env.set("_1261", builder.reshape(env.get("_1260"), [1,128,32,64]));
  env.set("_1262", builder.reshape(env.get("_1261"), [1,128,32,2,32]));
  env.set("InsertedPrecisionFreeCast__1262", builder.cast(env.get("_1262"), "float32"));
  env.set("InsertedPrecisionFreeCast__1272", builder["mul"](env.get("InsertedPrecisionFreeCast__1262"), env.get("InsertedPrecisionFreeCast__1271"), {"label":"/model/layers.5/attn/q_rotary/RotaryEmbedding_mul_cos_935"}));
  env.set("InsertedPrecisionFreeCast__1273", builder.reshape(env.get("InsertedPrecisionFreeCast__1272"), [1,128,32,64]));
  env.set("InsertedPrecisionFreeCast__1267", builder.cast(env.get("_1267"), "float32"));
  env.set("Inserted_435", builder["clamp"](env.get("position_ids_625"), {"label":"Inserted_Clip_919","maxValue":2047,"minValue":-2048}));
  env.set("_1252", builder["gather"](env.get("_624"), env.get("Inserted_435"), {"axis":0,"label":"/model/layers.5/attn/q_rotary/RotaryEmbedding_gather_sin_918"}));
  env.set("_1253", builder.reshape(env.get("_1252"), [1,128,1,1,32]));
  env.set("InsertedPrecisionFreeCast__1253", builder.cast(env.get("_1253"), "float32"));
  {
    const tmp = builder.split(env.get("_1262"), 2, {"axis":3,"label":"/model/layers.5/attn/q_rotary/RotaryEmbedding_split_partial_input0_927"});
    env.set("_1263", tmp[0]);
    env.set("_1264", tmp[1]);
  }
  env.set("_1265", builder.concat([env.get("_1264"), env.get("_1263")], 3, {"label":"/model/layers.5/attn/q_rotary/RotaryEmbedding_concat_partial_input0_928"}));
  env.set("InsertedPrecisionFreeCast__1265", builder.cast(env.get("_1265"), "float32"));
  env.set("InsertedPrecisionFreeCast__1266", builder["mul"](env.get("InsertedPrecisionFreeCast__1265"), env.get("InsertedPrecisionFreeCast__1253"), {"label":"/model/layers.5/attn/q_rotary/RotaryEmbedding_mul_sin_929"}));
  env.set("InsertedPrecisionFreeCast__1268", builder["mul"](env.get("InsertedPrecisionFreeCast__1266"), env.get("InsertedPrecisionFreeCast__1267"), {"label":"/model/layers.5/attn/q_rotary/RotaryEmbedding_mul_sign_930"}));
  env.set("InsertedPrecisionFreeCast__1269", builder.reshape(env.get("InsertedPrecisionFreeCast__1268"), [1,128,32,64]));
  env.set("InsertedPrecisionFreeCast__1274", builder["add"](env.get("InsertedPrecisionFreeCast__1273"), env.get("InsertedPrecisionFreeCast__1269"), {"label":"/model/layers.5/attn/q_rotary/RotaryEmbedding_add_mul_cos_sin_937"}));
  env.set("_1274", builder.cast(env.get("InsertedPrecisionFreeCast__1274"), "float16"));
  env.set("_1275", builder.reshape(env.get("_1274"), [1,128,2048]));
  env.set("_1276", builder.reshape(env.get("_1275"), [1,128,32,64]));
  env.set("_1277", builder["transpose"](env.get("_1276"), {"label":"/model/layers.5/attn/GroupQueryAttention_/GQA/query/transpose_940","permutation":[0,2,1,3]}));
  env.set("InsertedPrecisionFreeCast__1277", builder.cast(env.get("_1277"), "float32"));
  env.set("InsertedPrecisionFreeCast__1278", builder["matmul"](env.get("InsertedPrecisionFreeCast__1277"), env.get("InsertedPrecisionFreeCast__1251"), {"label":"/model/layers.5/attn/GroupQueryAttention_/Attention/qkv/matmul_1_941"}));
  env.set("InsertedPrecisionFreeCast__1279", builder["mul"](env.get("InsertedPrecisionFreeCast__1278"), env.get("InsertedPrecisionFreeCast__684"), {"label":"/model/layers.5/attn/GroupQueryAttention_/Attention/qkv/div_942"}));
  env.set("_1218", builder.expand(env.get("_617"), [1,32,128,512], {"label":"/model/layers.5/attn/GroupQueryAttention_/GQA/GQA_mask_shape_ones/expand_882"}));
  env.set("_1219", builder.cumulativeSum(env.get("_1218"), 3, {"exclusive":true,"label":"/model/layers.5/attn/GroupQueryAttention_range_of_mask_shape_883"}));
  env.set("_1215", builder["add"](env.get("_613"), env.get("_1207"), {"label":"/model/layers.5/attn/GroupQueryAttention_/GQA/attn_mask/add_879"}));
  env.set("_1216", builder.expand(env.get("_1215"), [512,128], {"label":"/model/layers.5/attn/GroupQueryAttention_/GQA/expand_neq_right_880"}));
  env.set("_1217", builder["transpose"](env.get("_1216"), {"label":"/model/layers.5/attn/GroupQueryAttention_/GQA/neq_right/transpose_881","permutation":[1,0]}));
  env.set("Inserted_409", builder["lesser"](env.get("_1219"), env.get("_1217"), {"label":"/model/layers.5/attn/GroupQueryAttention_/GQA/attn_mask/condition_1_884"}));
  env.set("InsertedPrecisionFreeCast__1221", builder["where"](env.get("Inserted_409"), env.get("InsertedPrecisionFreeCast__621"), env.get("InsertedPrecisionFreeCast__622"), {"label":"/model/layers.5/attn/GroupQueryAttention_/GQA/attn_mask/where_886"}));
  env.set("InsertedPrecisionFreeCast__1280", builder["add"](env.get("InsertedPrecisionFreeCast__1279"), env.get("InsertedPrecisionFreeCast__1221"), {"label":"/model/layers.5/attn/GroupQueryAttention_/Attention/attn_mask/softmax_input_943"}));
  env.set("InsertedPrecisionFreeCast__inlfunc_Softmax_token_4_X_ReduceMax", builder["reduceMax"](env.get("InsertedPrecisionFreeCast__1280"), {"axes":[3],"keepDimensions":true}));
  env.set("InsertedPrecisionFreeCast__inlfunc_Softmax_token_4_X_Sub", builder["sub"](env.get("InsertedPrecisionFreeCast__1280"), env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_4_X_ReduceMax"), {}));
  env.set("InsertedPrecisionFreeCast__inlfunc_Softmax_token_4_X_Exp", builder["exp"](env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_4_X_Sub"), {}));
  env.set("InsertedPrecisionFreeCast__inlfunc_Softmax_token_4_X_ReduceSum", builder["reduceSum"](env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_4_X_Exp"), {"axes":[3],"keepDimensions":true}));
  env.set("InsertedPrecisionFreeCast__1281", builder["div"](env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_4_X_Exp"), env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_4_X_ReduceSum"), {}));
  env.set("InsertedPrecisionFreeCast__1282", builder["matmul"](env.get("InsertedPrecisionFreeCast__1281"), env.get("InsertedPrecisionFreeCast__1214"), {"label":"/model/layers.5/attn/GroupQueryAttention_/Attention/qkv/matmul_2_945"}));
  env.set("_1282", builder.cast(env.get("InsertedPrecisionFreeCast__1282"), "float16"));
  env.set("_1283", builder["transpose"](env.get("_1282"), {"label":"/model/layers.5/attn/GroupQueryAttention_/Attention/qkv/transpose_946","permutation":[0,2,1,3]}));
  env.set("_1284", builder.reshape(env.get("_1283"), [1,128,2048]));
  env.set("InsertedPrecisionFreeCast__1284", builder.cast(env.get("_1284"), "float32"));
  env.set("InsertedPrecisionFreeCast__1285", builder["matmul"](env.get("InsertedPrecisionFreeCast__1284"), env.get("InsertedPrecisionFreeCast__452"), {"label":"/model/layers.5/attn/o_proj/MatMul_Q4_matmul_948"}));
  env.set("_1289", builder["add"](env.get("_1288"), env.get("InsertedPrecisionFreeCast__1285"), {"label":"/model/layers.5/post_attention_layernorm/SkipLayerNorm_add_skip_952"}));
  env.set("_1312", builder.cast(env.get("_1289"), "float16"));
  env.set("_1313", builder.cast(env.get("_1312"), "float32"));
  env.set("_438", builder.dequantizeLinear(env.get("_435"), env.get("_436"), env.get("_437"), {"axis":2,"blockSize":32,"label":"/model/layers.5/mlp/down_proj/MatMul_Q4_dequantizeLinear_195"}));
  env.set("_439", builder.reshape(env.get("_438"), [2048,5632]));
  env.set("_440", builder["transpose"](env.get("_439"), {"label":"/model/layers.5/mlp/down_proj/MatMul_Q4_transpose_dequantizeLinear_197","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__440", builder.cast(env.get("_440"), "float32"));
  env.set("_1303", builder.dequantizeLinear(env.get("_1300"), env.get("_1301"), env.get("_1302"), {"axis":2,"blockSize":32,"label":"/model/layers.5/mlp/gate_proj/MatMul_Q4_dequantizeLinear_962"}));
  env.set("_1304", builder.reshape(env.get("_1303"), [5632,2048]));
  env.set("_1305", builder["transpose"](env.get("_1304"), {"label":"/model/layers.5/mlp/gate_proj/MatMul_Q4_transpose_dequantizeLinear_964","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__1305", builder.cast(env.get("_1305"), "float32"));
  env.set("_1296", builder.cast(env.get("_1295"), "float32"));
  env.set("_1290", builder["pow"](env.get("_1289"), env.get("_583"), {"label":"/model/layers.5/post_attention_layernorm/SkipLayerNorm_pow_953"}));
  env.set("_1291", builder["reduceMean"](env.get("_1290"), {"axes":[2],"keepDimensions":true,"label":"/model/layers.5/post_attention_layernorm/SkipLayerNorm_reduceMean_954"}));
  env.set("_1292", builder["add"](env.get("_1291"), env.get("_586"), {"label":"/model/layers.5/post_attention_layernorm/SkipLayerNorm_add_955"}));
  env.set("_1293", builder["sqrt"](env.get("_1292"), {"label":"/model/layers.5/post_attention_layernorm/SkipLayerNorm_sqrt_956"}));
  env.set("_1294", builder["div"](env.get("_1289"), env.get("_1293"), {"label":"/model/layers.5/post_attention_layernorm/SkipLayerNorm_div_957"}));
  env.set("_1297", builder["mul"](env.get("_1296"), env.get("_1294"), {"label":"/model/layers.5/post_attention_layernorm/SkipLayerNorm_mul_959"}));
  env.set("InsertedPrecisionFreeCast__1306", builder["matmul"](env.get("_1297"), env.get("InsertedPrecisionFreeCast__1305"), {"label":"/model/layers.5/mlp/gate_proj/MatMul_Q4_matmul_965"}));
  env.set("InsertedPrecisionFreeCast__1307", builder["sigmoid"](env.get("InsertedPrecisionFreeCast__1306"), {"label":"/model/layers.5/mlp/act_fn/Sigmoid_966"}));
  env.set("InsertedPrecisionFreeCast__1308", builder["mul"](env.get("InsertedPrecisionFreeCast__1306"), env.get("InsertedPrecisionFreeCast__1307"), {"label":"/model/layers.5/mlp/act_fn/Mul_967"}));
  env.set("_444", builder.dequantizeLinear(env.get("_441"), env.get("_442"), env.get("_443"), {"axis":2,"blockSize":32,"label":"/model/layers.5/mlp/up_proj/MatMul_Q4_dequantizeLinear_198"}));
  env.set("_445", builder.reshape(env.get("_444"), [5632,2048]));
  env.set("_446", builder["transpose"](env.get("_445"), {"label":"/model/layers.5/mlp/up_proj/MatMul_Q4_transpose_dequantizeLinear_200","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__446", builder.cast(env.get("_446"), "float32"));
  env.set("InsertedPrecisionFreeCast__1299", builder["matmul"](env.get("_1297"), env.get("InsertedPrecisionFreeCast__446"), {"label":"/model/layers.5/mlp/up_proj/MatMul_Q4_matmul_961"}));
  env.set("InsertedPrecisionFreeCast__1309", builder["mul"](env.get("InsertedPrecisionFreeCast__1308"), env.get("InsertedPrecisionFreeCast__1299"), {"label":"/model/layers.5/mlp/Mul_968"}));
  env.set("InsertedPrecisionFreeCast__1310", builder["matmul"](env.get("InsertedPrecisionFreeCast__1309"), env.get("InsertedPrecisionFreeCast__440"), {"label":"/model/layers.5/mlp/down_proj/MatMul_Q4_matmul_969"}));
  env.set("_1314", builder["add"](env.get("_1313"), env.get("InsertedPrecisionFreeCast__1310"), {"label":"/model/layers.6/input_layernorm/SkipLayerNorm_add_skip_973"}));
  env.set("_1406", builder.cast(env.get("_1314"), "float16"));
  env.set("_1407", builder.cast(env.get("_1406"), "float32"));
  env.set("_426", builder.dequantizeLinear(env.get("_423"), env.get("_424"), env.get("_425"), {"axis":2,"blockSize":32,"label":"/model/layers.6/attn/o_proj/MatMul_Q4_dequantizeLinear_189"}));
  env.set("_427", builder.reshape(env.get("_426"), [2048,2048]));
  env.set("_428", builder["transpose"](env.get("_427"), {"label":"/model/layers.6/attn/o_proj/MatMul_Q4_transpose_dequantizeLinear_191","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__428", builder.cast(env.get("_428"), "float32"));
  env.set("Inserted_456", builder.cast(env.get("_601"), "uint8"));
  env.set("_1326", builder["where"](env.get("Inserted_456"), env.get("_602"), env.get("_600"), {"label":"/model/layers.6/attn/GroupQueryAttention_/GQA/scatter/where_984"}));
  env.set("_1327", builder["add"](env.get("_604"), env.get("_1326"), {"label":"/model/layers.6/attn/GroupQueryAttention_/GQA/right_constant/add_986"}));
  env.set("_1328", builder.concat([env.get("_606"), env.get("_1327")], 1, {"label":"/model/layers.6/attn/GroupQueryAttention_/GQA/concat_for_pre_scatter_indices_987"}));
  env.set("_1329", builder.reshape(env.get("_1328"), [1,128,4,3]));
  env.set("Inserted_458", builder.cast(env.get("_1329"), "int64"));
  env.set("Inserted_460", builder["max"](env.get("Inserted_458"), env.get("Inserted_459"), {"label":"Inserted_Max_991"}));
  env.set("Inserted_462", builder["min"](env.get("Inserted_460"), env.get("Inserted_461"), {"label":"Inserted_Min_992"}));
  env.set("_432", builder.dequantizeLinear(env.get("_429"), env.get("_430"), env.get("_431"), {"axis":2,"blockSize":32,"label":"/model/layers.6/attn/v_proj/MatMul_Q4_dequantizeLinear_192"}));
  env.set("_433", builder.reshape(env.get("_432"), [256,2048]));
  env.set("_434", builder["transpose"](env.get("_433"), {"label":"/model/layers.6/attn/v_proj/MatMul_Q4_transpose_dequantizeLinear_194","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__434", builder.cast(env.get("_434"), "float32"));
  env.set("_1321", builder.cast(env.get("_1320"), "float32"));
  env.set("_1315", builder["pow"](env.get("_1314"), env.get("_583"), {"label":"/model/layers.6/input_layernorm/SkipLayerNorm_pow_974"}));
  env.set("_1316", builder["reduceMean"](env.get("_1315"), {"axes":[2],"keepDimensions":true,"label":"/model/layers.6/input_layernorm/SkipLayerNorm_reduceMean_975"}));
  env.set("_1317", builder["add"](env.get("_1316"), env.get("_586"), {"label":"/model/layers.6/input_layernorm/SkipLayerNorm_add_976"}));
  env.set("_1318", builder["sqrt"](env.get("_1317"), {"label":"/model/layers.6/input_layernorm/SkipLayerNorm_sqrt_977"}));
  env.set("_1319", builder["div"](env.get("_1314"), env.get("_1318"), {"label":"/model/layers.6/input_layernorm/SkipLayerNorm_div_978"}));
  env.set("_1322", builder["mul"](env.get("_1321"), env.get("_1319"), {"label":"/model/layers.6/input_layernorm/SkipLayerNorm_mul_980"}));
  env.set("InsertedPrecisionFreeCast__1324", builder["matmul"](env.get("_1322"), env.get("InsertedPrecisionFreeCast__434"), {"label":"/model/layers.6/attn/v_proj/MatMul_Q4_matmul_982"}));
  env.set("_1324", builder.cast(env.get("InsertedPrecisionFreeCast__1324"), "float16"));
  env.set("_1325", builder.reshape(env.get("_1324"), [1,128,4,64]));
  env.set("present_6_value_13", builder["scatterND"](env.get("past_key_values_6_value_1330"), env.get("Inserted_462"), env.get("_1325"), {"label":"/model/layers.6/attn/GroupQueryAttention_/GQA/present_value/ScatterND_989"}));
  env.set("_1331", builder.reshape(env.get("present_6_value_13"), [1,4,1,512,64]));
  env.set("_1332", builder.expand(env.get("_1331"), [1,4,8,512,64], {"label":"/model/layers.6/attn/GroupQueryAttention_/GQA/true_present_value/expand_994"}));
  env.set("_1333", builder.reshape(env.get("_1332"), [1,32,512,64]));
  env.set("InsertedPrecisionFreeCast__1333", builder.cast(env.get("_1333"), "float32"));
  env.set("Inserted_487", builder.cast(env.get("_1329"), "int64"));
  env.set("Inserted_489", builder["max"](env.get("Inserted_487"), env.get("Inserted_488"), {"label":"Inserted_Max_1029"}));
  env.set("Inserted_491", builder["min"](env.get("Inserted_489"), env.get("Inserted_490"), {"label":"Inserted_Min_1030"}));
  env.set("Inserted_480", builder["clamp"](env.get("position_ids_625"), {"label":"Inserted_Clip_1020","maxValue":2047,"minValue":-2048}));
  env.set("_1359", builder["gather"](env.get("_644"), env.get("Inserted_480"), {"axis":0,"label":"/model/layers.6/attn/k_rotary/RotaryEmbedding_gather_cos_1019"}));
  env.set("_1360", builder.reshape(env.get("_1359"), [1,128,1,1,32]));
  env.set("InsertedPrecisionFreeCast__1360", builder.cast(env.get("_1360"), "float32"));
  env.set("_1346", builder.dequantizeLinear(env.get("_1343"), env.get("_1344"), env.get("_1345"), {"axis":2,"blockSize":32,"label":"/model/layers.6/attn/k_proj/MatMul_Q4_dequantizeLinear_1008"}));
  env.set("_1347", builder.reshape(env.get("_1346"), [256,2048]));
  env.set("_1348", builder["transpose"](env.get("_1347"), {"label":"/model/layers.6/attn/k_proj/MatMul_Q4_transpose_dequantizeLinear_1010","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__1348", builder.cast(env.get("_1348"), "float32"));
  env.set("InsertedPrecisionFreeCast__1349", builder["matmul"](env.get("_1322"), env.get("InsertedPrecisionFreeCast__1348"), {"label":"/model/layers.6/attn/k_proj/MatMul_Q4_matmul_1011"}));
  env.set("_1349", builder.cast(env.get("InsertedPrecisionFreeCast__1349"), "float16"));
  env.set("_1350", builder.reshape(env.get("_1349"), [1,128,4,64]));
  env.set("_1351", builder.reshape(env.get("_1350"), [1,128,4,2,32]));
  env.set("InsertedPrecisionFreeCast__1351", builder.cast(env.get("_1351"), "float32"));
  env.set("InsertedPrecisionFreeCast__1361", builder["mul"](env.get("InsertedPrecisionFreeCast__1351"), env.get("InsertedPrecisionFreeCast__1360"), {"label":"/model/layers.6/attn/k_rotary/RotaryEmbedding_mul_cos_1022"}));
  env.set("InsertedPrecisionFreeCast__1362", builder.reshape(env.get("InsertedPrecisionFreeCast__1361"), [1,128,4,64]));
  env.set("InsertedPrecisionFreeCast__1356", builder.cast(env.get("_1356"), "float32"));
  env.set("Inserted_471", builder["clamp"](env.get("position_ids_625"), {"label":"Inserted_Clip_1006","maxValue":2047,"minValue":-2048}));
  env.set("_1341", builder["gather"](env.get("_624"), env.get("Inserted_471"), {"axis":0,"label":"/model/layers.6/attn/k_rotary/RotaryEmbedding_gather_sin_1005"}));
  env.set("_1342", builder.reshape(env.get("_1341"), [1,128,1,1,32]));
  env.set("InsertedPrecisionFreeCast__1342", builder.cast(env.get("_1342"), "float32"));
  {
    const tmp = builder.split(env.get("_1351"), 2, {"axis":3,"label":"/model/layers.6/attn/k_rotary/RotaryEmbedding_split_partial_input0_1014"});
    env.set("_1352", tmp[0]);
    env.set("_1353", tmp[1]);
  }
  env.set("_1354", builder.concat([env.get("_1353"), env.get("_1352")], 3, {"label":"/model/layers.6/attn/k_rotary/RotaryEmbedding_concat_partial_input0_1015"}));
  env.set("InsertedPrecisionFreeCast__1354", builder.cast(env.get("_1354"), "float32"));
  env.set("InsertedPrecisionFreeCast__1355", builder["mul"](env.get("InsertedPrecisionFreeCast__1354"), env.get("InsertedPrecisionFreeCast__1342"), {"label":"/model/layers.6/attn/k_rotary/RotaryEmbedding_mul_sin_1016"}));
  env.set("InsertedPrecisionFreeCast__1357", builder["mul"](env.get("InsertedPrecisionFreeCast__1355"), env.get("InsertedPrecisionFreeCast__1356"), {"label":"/model/layers.6/attn/k_rotary/RotaryEmbedding_mul_sign_1017"}));
  env.set("InsertedPrecisionFreeCast__1358", builder.reshape(env.get("InsertedPrecisionFreeCast__1357"), [1,128,4,64]));
  env.set("InsertedPrecisionFreeCast__1363", builder["add"](env.get("InsertedPrecisionFreeCast__1362"), env.get("InsertedPrecisionFreeCast__1358"), {"label":"/model/layers.6/attn/k_rotary/RotaryEmbedding_add_mul_cos_sin_1024"}));
  env.set("_1363", builder.cast(env.get("InsertedPrecisionFreeCast__1363"), "float16"));
  env.set("_1364", builder.reshape(env.get("_1363"), [1,128,256]));
  env.set("_1365", builder.reshape(env.get("_1364"), [1,128,4,64]));
  env.set("present_6_key_12", builder["scatterND"](env.get("past_key_values_6_key_1366"), env.get("Inserted_491"), env.get("_1365"), {"label":"/model/layers.6/attn/GroupQueryAttention_/GQA/present_key/ScatterND_1027"}));
  env.set("_1367", builder.reshape(env.get("present_6_key_12"), [1,4,1,512,64]));
  env.set("_1368", builder.expand(env.get("_1367"), [1,4,8,512,64], {"label":"/model/layers.6/attn/GroupQueryAttention_/GQA/true_present_key/expand_1032"}));
  env.set("_1369", builder.reshape(env.get("_1368"), [1,32,512,64]));
  env.set("_1370", builder["transpose"](env.get("_1369"), {"label":"/model/layers.6/attn/GroupQueryAttention_/GQA/present_key/transpose_1034","permutation":[0,1,3,2]}));
  env.set("InsertedPrecisionFreeCast__1370", builder.cast(env.get("_1370"), "float32"));
  env.set("Inserted_504", builder["clamp"](env.get("position_ids_625"), {"label":"Inserted_Clip_1050","maxValue":2047,"minValue":-2048}));
  env.set("_1389", builder["gather"](env.get("_644"), env.get("Inserted_504"), {"axis":0,"label":"/model/layers.6/attn/q_rotary/RotaryEmbedding_gather_cos_1049"}));
  env.set("_1390", builder.reshape(env.get("_1389"), [1,128,1,1,32]));
  env.set("InsertedPrecisionFreeCast__1390", builder.cast(env.get("_1390"), "float32"));
  env.set("_1376", builder.dequantizeLinear(env.get("_1373"), env.get("_1374"), env.get("_1375"), {"axis":2,"blockSize":32,"label":"/model/layers.6/attn/q_proj/MatMul_Q4_dequantizeLinear_1038"}));
  env.set("_1377", builder.reshape(env.get("_1376"), [2048,2048]));
  env.set("_1378", builder["transpose"](env.get("_1377"), {"label":"/model/layers.6/attn/q_proj/MatMul_Q4_transpose_dequantizeLinear_1040","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__1378", builder.cast(env.get("_1378"), "float32"));
  env.set("InsertedPrecisionFreeCast__1379", builder["matmul"](env.get("_1322"), env.get("InsertedPrecisionFreeCast__1378"), {"label":"/model/layers.6/attn/q_proj/MatMul_Q4_matmul_1041"}));
  env.set("_1379", builder.cast(env.get("InsertedPrecisionFreeCast__1379"), "float16"));
  env.set("_1380", builder.reshape(env.get("_1379"), [1,128,32,64]));
  env.set("_1381", builder.reshape(env.get("_1380"), [1,128,32,2,32]));
  env.set("InsertedPrecisionFreeCast__1381", builder.cast(env.get("_1381"), "float32"));
  env.set("InsertedPrecisionFreeCast__1391", builder["mul"](env.get("InsertedPrecisionFreeCast__1381"), env.get("InsertedPrecisionFreeCast__1390"), {"label":"/model/layers.6/attn/q_rotary/RotaryEmbedding_mul_cos_1052"}));
  env.set("InsertedPrecisionFreeCast__1392", builder.reshape(env.get("InsertedPrecisionFreeCast__1391"), [1,128,32,64]));
  env.set("InsertedPrecisionFreeCast__1386", builder.cast(env.get("_1386"), "float32"));
  env.set("Inserted_495", builder["clamp"](env.get("position_ids_625"), {"label":"Inserted_Clip_1036","maxValue":2047,"minValue":-2048}));
  env.set("_1371", builder["gather"](env.get("_624"), env.get("Inserted_495"), {"axis":0,"label":"/model/layers.6/attn/q_rotary/RotaryEmbedding_gather_sin_1035"}));
  env.set("_1372", builder.reshape(env.get("_1371"), [1,128,1,1,32]));
  env.set("InsertedPrecisionFreeCast__1372", builder.cast(env.get("_1372"), "float32"));
  {
    const tmp = builder.split(env.get("_1381"), 2, {"axis":3,"label":"/model/layers.6/attn/q_rotary/RotaryEmbedding_split_partial_input0_1044"});
    env.set("_1382", tmp[0]);
    env.set("_1383", tmp[1]);
  }
  env.set("_1384", builder.concat([env.get("_1383"), env.get("_1382")], 3, {"label":"/model/layers.6/attn/q_rotary/RotaryEmbedding_concat_partial_input0_1045"}));
  env.set("InsertedPrecisionFreeCast__1384", builder.cast(env.get("_1384"), "float32"));
  env.set("InsertedPrecisionFreeCast__1385", builder["mul"](env.get("InsertedPrecisionFreeCast__1384"), env.get("InsertedPrecisionFreeCast__1372"), {"label":"/model/layers.6/attn/q_rotary/RotaryEmbedding_mul_sin_1046"}));
  env.set("InsertedPrecisionFreeCast__1387", builder["mul"](env.get("InsertedPrecisionFreeCast__1385"), env.get("InsertedPrecisionFreeCast__1386"), {"label":"/model/layers.6/attn/q_rotary/RotaryEmbedding_mul_sign_1047"}));
  env.set("InsertedPrecisionFreeCast__1388", builder.reshape(env.get("InsertedPrecisionFreeCast__1387"), [1,128,32,64]));
  env.set("InsertedPrecisionFreeCast__1393", builder["add"](env.get("InsertedPrecisionFreeCast__1392"), env.get("InsertedPrecisionFreeCast__1388"), {"label":"/model/layers.6/attn/q_rotary/RotaryEmbedding_add_mul_cos_sin_1054"}));
  env.set("_1393", builder.cast(env.get("InsertedPrecisionFreeCast__1393"), "float16"));
  env.set("_1394", builder.reshape(env.get("_1393"), [1,128,2048]));
  env.set("_1395", builder.reshape(env.get("_1394"), [1,128,32,64]));
  env.set("_1396", builder["transpose"](env.get("_1395"), {"label":"/model/layers.6/attn/GroupQueryAttention_/GQA/query/transpose_1057","permutation":[0,2,1,3]}));
  env.set("InsertedPrecisionFreeCast__1396", builder.cast(env.get("_1396"), "float32"));
  env.set("InsertedPrecisionFreeCast__1397", builder["matmul"](env.get("InsertedPrecisionFreeCast__1396"), env.get("InsertedPrecisionFreeCast__1370"), {"label":"/model/layers.6/attn/GroupQueryAttention_/Attention/qkv/matmul_1_1058"}));
  env.set("InsertedPrecisionFreeCast__1398", builder["mul"](env.get("InsertedPrecisionFreeCast__1397"), env.get("InsertedPrecisionFreeCast__684"), {"label":"/model/layers.6/attn/GroupQueryAttention_/Attention/qkv/div_1059"}));
  env.set("_1337", builder.expand(env.get("_617"), [1,32,128,512], {"label":"/model/layers.6/attn/GroupQueryAttention_/GQA/GQA_mask_shape_ones/expand_999"}));
  env.set("_1338", builder.cumulativeSum(env.get("_1337"), 3, {"exclusive":true,"label":"/model/layers.6/attn/GroupQueryAttention_range_of_mask_shape_1000"}));
  env.set("_1334", builder["add"](env.get("_613"), env.get("_1326"), {"label":"/model/layers.6/attn/GroupQueryAttention_/GQA/attn_mask/add_996"}));
  env.set("_1335", builder.expand(env.get("_1334"), [512,128], {"label":"/model/layers.6/attn/GroupQueryAttention_/GQA/expand_neq_right_997"}));
  env.set("_1336", builder["transpose"](env.get("_1335"), {"label":"/model/layers.6/attn/GroupQueryAttention_/GQA/neq_right/transpose_998","permutation":[1,0]}));
  env.set("Inserted_469", builder["lesser"](env.get("_1338"), env.get("_1336"), {"label":"/model/layers.6/attn/GroupQueryAttention_/GQA/attn_mask/condition_1_1001"}));
  env.set("InsertedPrecisionFreeCast__1340", builder["where"](env.get("Inserted_469"), env.get("InsertedPrecisionFreeCast__621"), env.get("InsertedPrecisionFreeCast__622"), {"label":"/model/layers.6/attn/GroupQueryAttention_/GQA/attn_mask/where_1003"}));
  env.set("InsertedPrecisionFreeCast__1399", builder["add"](env.get("InsertedPrecisionFreeCast__1398"), env.get("InsertedPrecisionFreeCast__1340"), {"label":"/model/layers.6/attn/GroupQueryAttention_/Attention/attn_mask/softmax_input_1060"}));
  env.set("InsertedPrecisionFreeCast__inlfunc_Softmax_token_5_X_ReduceMax", builder["reduceMax"](env.get("InsertedPrecisionFreeCast__1399"), {"axes":[3],"keepDimensions":true}));
  env.set("InsertedPrecisionFreeCast__inlfunc_Softmax_token_5_X_Sub", builder["sub"](env.get("InsertedPrecisionFreeCast__1399"), env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_5_X_ReduceMax"), {}));
  env.set("InsertedPrecisionFreeCast__inlfunc_Softmax_token_5_X_Exp", builder["exp"](env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_5_X_Sub"), {}));
  env.set("InsertedPrecisionFreeCast__inlfunc_Softmax_token_5_X_ReduceSum", builder["reduceSum"](env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_5_X_Exp"), {"axes":[3],"keepDimensions":true}));
  env.set("InsertedPrecisionFreeCast__1400", builder["div"](env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_5_X_Exp"), env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_5_X_ReduceSum"), {}));
  env.set("InsertedPrecisionFreeCast__1401", builder["matmul"](env.get("InsertedPrecisionFreeCast__1400"), env.get("InsertedPrecisionFreeCast__1333"), {"label":"/model/layers.6/attn/GroupQueryAttention_/Attention/qkv/matmul_2_1062"}));
  env.set("_1401", builder.cast(env.get("InsertedPrecisionFreeCast__1401"), "float16"));
  env.set("_1402", builder["transpose"](env.get("_1401"), {"label":"/model/layers.6/attn/GroupQueryAttention_/Attention/qkv/transpose_1063","permutation":[0,2,1,3]}));
  env.set("_1403", builder.reshape(env.get("_1402"), [1,128,2048]));
  env.set("InsertedPrecisionFreeCast__1403", builder.cast(env.get("_1403"), "float32"));
  env.set("InsertedPrecisionFreeCast__1404", builder["matmul"](env.get("InsertedPrecisionFreeCast__1403"), env.get("InsertedPrecisionFreeCast__428"), {"label":"/model/layers.6/attn/o_proj/MatMul_Q4_matmul_1065"}));
  env.set("_1408", builder["add"](env.get("_1407"), env.get("InsertedPrecisionFreeCast__1404"), {"label":"/model/layers.6/post_attention_layernorm/SkipLayerNorm_add_skip_1069"}));
  env.set("_1431", builder.cast(env.get("_1408"), "float16"));
  env.set("_1432", builder.cast(env.get("_1431"), "float32"));
  env.set("_414", builder.dequantizeLinear(env.get("_411"), env.get("_412"), env.get("_413"), {"axis":2,"blockSize":32,"label":"/model/layers.6/mlp/down_proj/MatMul_Q4_dequantizeLinear_183"}));
  env.set("_415", builder.reshape(env.get("_414"), [2048,5632]));
  env.set("_416", builder["transpose"](env.get("_415"), {"label":"/model/layers.6/mlp/down_proj/MatMul_Q4_transpose_dequantizeLinear_185","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__416", builder.cast(env.get("_416"), "float32"));
  env.set("_1422", builder.dequantizeLinear(env.get("_1419"), env.get("_1420"), env.get("_1421"), {"axis":2,"blockSize":32,"label":"/model/layers.6/mlp/gate_proj/MatMul_Q4_dequantizeLinear_1079"}));
  env.set("_1423", builder.reshape(env.get("_1422"), [5632,2048]));
  env.set("_1424", builder["transpose"](env.get("_1423"), {"label":"/model/layers.6/mlp/gate_proj/MatMul_Q4_transpose_dequantizeLinear_1081","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__1424", builder.cast(env.get("_1424"), "float32"));
  env.set("_1415", builder.cast(env.get("_1414"), "float32"));
  env.set("_1409", builder["pow"](env.get("_1408"), env.get("_583"), {"label":"/model/layers.6/post_attention_layernorm/SkipLayerNorm_pow_1070"}));
  env.set("_1410", builder["reduceMean"](env.get("_1409"), {"axes":[2],"keepDimensions":true,"label":"/model/layers.6/post_attention_layernorm/SkipLayerNorm_reduceMean_1071"}));
  env.set("_1411", builder["add"](env.get("_1410"), env.get("_586"), {"label":"/model/layers.6/post_attention_layernorm/SkipLayerNorm_add_1072"}));
  env.set("_1412", builder["sqrt"](env.get("_1411"), {"label":"/model/layers.6/post_attention_layernorm/SkipLayerNorm_sqrt_1073"}));
  env.set("_1413", builder["div"](env.get("_1408"), env.get("_1412"), {"label":"/model/layers.6/post_attention_layernorm/SkipLayerNorm_div_1074"}));
  env.set("_1416", builder["mul"](env.get("_1415"), env.get("_1413"), {"label":"/model/layers.6/post_attention_layernorm/SkipLayerNorm_mul_1076"}));
  env.set("InsertedPrecisionFreeCast__1425", builder["matmul"](env.get("_1416"), env.get("InsertedPrecisionFreeCast__1424"), {"label":"/model/layers.6/mlp/gate_proj/MatMul_Q4_matmul_1082"}));
  env.set("InsertedPrecisionFreeCast__1426", builder["sigmoid"](env.get("InsertedPrecisionFreeCast__1425"), {"label":"/model/layers.6/mlp/act_fn/Sigmoid_1083"}));
  env.set("InsertedPrecisionFreeCast__1427", builder["mul"](env.get("InsertedPrecisionFreeCast__1425"), env.get("InsertedPrecisionFreeCast__1426"), {"label":"/model/layers.6/mlp/act_fn/Mul_1084"}));
  env.set("_420", builder.dequantizeLinear(env.get("_417"), env.get("_418"), env.get("_419"), {"axis":2,"blockSize":32,"label":"/model/layers.6/mlp/up_proj/MatMul_Q4_dequantizeLinear_186"}));
  env.set("_421", builder.reshape(env.get("_420"), [5632,2048]));
  env.set("_422", builder["transpose"](env.get("_421"), {"label":"/model/layers.6/mlp/up_proj/MatMul_Q4_transpose_dequantizeLinear_188","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__422", builder.cast(env.get("_422"), "float32"));
  env.set("InsertedPrecisionFreeCast__1418", builder["matmul"](env.get("_1416"), env.get("InsertedPrecisionFreeCast__422"), {"label":"/model/layers.6/mlp/up_proj/MatMul_Q4_matmul_1078"}));
  env.set("InsertedPrecisionFreeCast__1428", builder["mul"](env.get("InsertedPrecisionFreeCast__1427"), env.get("InsertedPrecisionFreeCast__1418"), {"label":"/model/layers.6/mlp/Mul_1085"}));
  env.set("InsertedPrecisionFreeCast__1429", builder["matmul"](env.get("InsertedPrecisionFreeCast__1428"), env.get("InsertedPrecisionFreeCast__416"), {"label":"/model/layers.6/mlp/down_proj/MatMul_Q4_matmul_1086"}));
  env.set("_1433", builder["add"](env.get("_1432"), env.get("InsertedPrecisionFreeCast__1429"), {"label":"/model/layers.7/input_layernorm/SkipLayerNorm_add_skip_1090"}));
  env.set("_1525", builder.cast(env.get("_1433"), "float16"));
  env.set("_1526", builder.cast(env.get("_1525"), "float32"));
  env.set("_402", builder.dequantizeLinear(env.get("_399"), env.get("_400"), env.get("_401"), {"axis":2,"blockSize":32,"label":"/model/layers.7/attn/o_proj/MatMul_Q4_dequantizeLinear_177"}));
  env.set("_403", builder.reshape(env.get("_402"), [2048,2048]));
  env.set("_404", builder["transpose"](env.get("_403"), {"label":"/model/layers.7/attn/o_proj/MatMul_Q4_transpose_dequantizeLinear_179","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__404", builder.cast(env.get("_404"), "float32"));
  env.set("Inserted_516", builder.cast(env.get("_601"), "uint8"));
  env.set("_1445", builder["where"](env.get("Inserted_516"), env.get("_602"), env.get("_600"), {"label":"/model/layers.7/attn/GroupQueryAttention_/GQA/scatter/where_1101"}));
  env.set("_1446", builder["add"](env.get("_604"), env.get("_1445"), {"label":"/model/layers.7/attn/GroupQueryAttention_/GQA/right_constant/add_1103"}));
  env.set("_1447", builder.concat([env.get("_606"), env.get("_1446")], 1, {"label":"/model/layers.7/attn/GroupQueryAttention_/GQA/concat_for_pre_scatter_indices_1104"}));
  env.set("_1448", builder.reshape(env.get("_1447"), [1,128,4,3]));
  env.set("Inserted_518", builder.cast(env.get("_1448"), "int64"));
  env.set("Inserted_520", builder["max"](env.get("Inserted_518"), env.get("Inserted_519"), {"label":"Inserted_Max_1108"}));
  env.set("Inserted_522", builder["min"](env.get("Inserted_520"), env.get("Inserted_521"), {"label":"Inserted_Min_1109"}));
  env.set("_408", builder.dequantizeLinear(env.get("_405"), env.get("_406"), env.get("_407"), {"axis":2,"blockSize":32,"label":"/model/layers.7/attn/v_proj/MatMul_Q4_dequantizeLinear_180"}));
  env.set("_409", builder.reshape(env.get("_408"), [256,2048]));
  env.set("_410", builder["transpose"](env.get("_409"), {"label":"/model/layers.7/attn/v_proj/MatMul_Q4_transpose_dequantizeLinear_182","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__410", builder.cast(env.get("_410"), "float32"));
  env.set("_1440", builder.cast(env.get("_1439"), "float32"));
  env.set("_1434", builder["pow"](env.get("_1433"), env.get("_583"), {"label":"/model/layers.7/input_layernorm/SkipLayerNorm_pow_1091"}));
  env.set("_1435", builder["reduceMean"](env.get("_1434"), {"axes":[2],"keepDimensions":true,"label":"/model/layers.7/input_layernorm/SkipLayerNorm_reduceMean_1092"}));
  env.set("_1436", builder["add"](env.get("_1435"), env.get("_586"), {"label":"/model/layers.7/input_layernorm/SkipLayerNorm_add_1093"}));
  env.set("_1437", builder["sqrt"](env.get("_1436"), {"label":"/model/layers.7/input_layernorm/SkipLayerNorm_sqrt_1094"}));
  env.set("_1438", builder["div"](env.get("_1433"), env.get("_1437"), {"label":"/model/layers.7/input_layernorm/SkipLayerNorm_div_1095"}));
  env.set("_1441", builder["mul"](env.get("_1440"), env.get("_1438"), {"label":"/model/layers.7/input_layernorm/SkipLayerNorm_mul_1097"}));
  env.set("InsertedPrecisionFreeCast__1443", builder["matmul"](env.get("_1441"), env.get("InsertedPrecisionFreeCast__410"), {"label":"/model/layers.7/attn/v_proj/MatMul_Q4_matmul_1099"}));
  env.set("_1443", builder.cast(env.get("InsertedPrecisionFreeCast__1443"), "float16"));
  env.set("_1444", builder.reshape(env.get("_1443"), [1,128,4,64]));
  env.set("present_7_value_15", builder["scatterND"](env.get("past_key_values_7_value_1449"), env.get("Inserted_522"), env.get("_1444"), {"label":"/model/layers.7/attn/GroupQueryAttention_/GQA/present_value/ScatterND_1106"}));
  env.set("_1450", builder.reshape(env.get("present_7_value_15"), [1,4,1,512,64]));
  env.set("_1451", builder.expand(env.get("_1450"), [1,4,8,512,64], {"label":"/model/layers.7/attn/GroupQueryAttention_/GQA/true_present_value/expand_1111"}));
  env.set("_1452", builder.reshape(env.get("_1451"), [1,32,512,64]));
  env.set("InsertedPrecisionFreeCast__1452", builder.cast(env.get("_1452"), "float32"));
  env.set("Inserted_547", builder.cast(env.get("_1448"), "int64"));
  env.set("Inserted_549", builder["max"](env.get("Inserted_547"), env.get("Inserted_548"), {"label":"Inserted_Max_1146"}));
  env.set("Inserted_551", builder["min"](env.get("Inserted_549"), env.get("Inserted_550"), {"label":"Inserted_Min_1147"}));
  env.set("Inserted_540", builder["clamp"](env.get("position_ids_625"), {"label":"Inserted_Clip_1137","maxValue":2047,"minValue":-2048}));
  env.set("_1478", builder["gather"](env.get("_644"), env.get("Inserted_540"), {"axis":0,"label":"/model/layers.7/attn/k_rotary/RotaryEmbedding_gather_cos_1136"}));
  env.set("_1479", builder.reshape(env.get("_1478"), [1,128,1,1,32]));
  env.set("InsertedPrecisionFreeCast__1479", builder.cast(env.get("_1479"), "float32"));
  env.set("_1465", builder.dequantizeLinear(env.get("_1462"), env.get("_1463"), env.get("_1464"), {"axis":2,"blockSize":32,"label":"/model/layers.7/attn/k_proj/MatMul_Q4_dequantizeLinear_1125"}));
  env.set("_1466", builder.reshape(env.get("_1465"), [256,2048]));
  env.set("_1467", builder["transpose"](env.get("_1466"), {"label":"/model/layers.7/attn/k_proj/MatMul_Q4_transpose_dequantizeLinear_1127","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__1467", builder.cast(env.get("_1467"), "float32"));
  env.set("InsertedPrecisionFreeCast__1468", builder["matmul"](env.get("_1441"), env.get("InsertedPrecisionFreeCast__1467"), {"label":"/model/layers.7/attn/k_proj/MatMul_Q4_matmul_1128"}));
  env.set("_1468", builder.cast(env.get("InsertedPrecisionFreeCast__1468"), "float16"));
  env.set("_1469", builder.reshape(env.get("_1468"), [1,128,4,64]));
  env.set("_1470", builder.reshape(env.get("_1469"), [1,128,4,2,32]));
  env.set("InsertedPrecisionFreeCast__1470", builder.cast(env.get("_1470"), "float32"));
  env.set("InsertedPrecisionFreeCast__1480", builder["mul"](env.get("InsertedPrecisionFreeCast__1470"), env.get("InsertedPrecisionFreeCast__1479"), {"label":"/model/layers.7/attn/k_rotary/RotaryEmbedding_mul_cos_1139"}));
  env.set("InsertedPrecisionFreeCast__1481", builder.reshape(env.get("InsertedPrecisionFreeCast__1480"), [1,128,4,64]));
  env.set("InsertedPrecisionFreeCast__1475", builder.cast(env.get("_1475"), "float32"));
  env.set("Inserted_531", builder["clamp"](env.get("position_ids_625"), {"label":"Inserted_Clip_1123","maxValue":2047,"minValue":-2048}));
  env.set("_1460", builder["gather"](env.get("_624"), env.get("Inserted_531"), {"axis":0,"label":"/model/layers.7/attn/k_rotary/RotaryEmbedding_gather_sin_1122"}));
  env.set("_1461", builder.reshape(env.get("_1460"), [1,128,1,1,32]));
  env.set("InsertedPrecisionFreeCast__1461", builder.cast(env.get("_1461"), "float32"));
  {
    const tmp = builder.split(env.get("_1470"), 2, {"axis":3,"label":"/model/layers.7/attn/k_rotary/RotaryEmbedding_split_partial_input0_1131"});
    env.set("_1471", tmp[0]);
    env.set("_1472", tmp[1]);
  }
  env.set("_1473", builder.concat([env.get("_1472"), env.get("_1471")], 3, {"label":"/model/layers.7/attn/k_rotary/RotaryEmbedding_concat_partial_input0_1132"}));
  env.set("InsertedPrecisionFreeCast__1473", builder.cast(env.get("_1473"), "float32"));
  env.set("InsertedPrecisionFreeCast__1474", builder["mul"](env.get("InsertedPrecisionFreeCast__1473"), env.get("InsertedPrecisionFreeCast__1461"), {"label":"/model/layers.7/attn/k_rotary/RotaryEmbedding_mul_sin_1133"}));
  env.set("InsertedPrecisionFreeCast__1476", builder["mul"](env.get("InsertedPrecisionFreeCast__1474"), env.get("InsertedPrecisionFreeCast__1475"), {"label":"/model/layers.7/attn/k_rotary/RotaryEmbedding_mul_sign_1134"}));
  env.set("InsertedPrecisionFreeCast__1477", builder.reshape(env.get("InsertedPrecisionFreeCast__1476"), [1,128,4,64]));
  env.set("InsertedPrecisionFreeCast__1482", builder["add"](env.get("InsertedPrecisionFreeCast__1481"), env.get("InsertedPrecisionFreeCast__1477"), {"label":"/model/layers.7/attn/k_rotary/RotaryEmbedding_add_mul_cos_sin_1141"}));
  env.set("_1482", builder.cast(env.get("InsertedPrecisionFreeCast__1482"), "float16"));
  env.set("_1483", builder.reshape(env.get("_1482"), [1,128,256]));
  env.set("_1484", builder.reshape(env.get("_1483"), [1,128,4,64]));
  env.set("present_7_key_14", builder["scatterND"](env.get("past_key_values_7_key_1485"), env.get("Inserted_551"), env.get("_1484"), {"label":"/model/layers.7/attn/GroupQueryAttention_/GQA/present_key/ScatterND_1144"}));
  env.set("_1486", builder.reshape(env.get("present_7_key_14"), [1,4,1,512,64]));
  env.set("_1487", builder.expand(env.get("_1486"), [1,4,8,512,64], {"label":"/model/layers.7/attn/GroupQueryAttention_/GQA/true_present_key/expand_1149"}));
  env.set("_1488", builder.reshape(env.get("_1487"), [1,32,512,64]));
  env.set("_1489", builder["transpose"](env.get("_1488"), {"label":"/model/layers.7/attn/GroupQueryAttention_/GQA/present_key/transpose_1151","permutation":[0,1,3,2]}));
  env.set("InsertedPrecisionFreeCast__1489", builder.cast(env.get("_1489"), "float32"));
  env.set("Inserted_564", builder["clamp"](env.get("position_ids_625"), {"label":"Inserted_Clip_1167","maxValue":2047,"minValue":-2048}));
  env.set("_1508", builder["gather"](env.get("_644"), env.get("Inserted_564"), {"axis":0,"label":"/model/layers.7/attn/q_rotary/RotaryEmbedding_gather_cos_1166"}));
  env.set("_1509", builder.reshape(env.get("_1508"), [1,128,1,1,32]));
  env.set("InsertedPrecisionFreeCast__1509", builder.cast(env.get("_1509"), "float32"));
  env.set("_1495", builder.dequantizeLinear(env.get("_1492"), env.get("_1493"), env.get("_1494"), {"axis":2,"blockSize":32,"label":"/model/layers.7/attn/q_proj/MatMul_Q4_dequantizeLinear_1155"}));
  env.set("_1496", builder.reshape(env.get("_1495"), [2048,2048]));
  env.set("_1497", builder["transpose"](env.get("_1496"), {"label":"/model/layers.7/attn/q_proj/MatMul_Q4_transpose_dequantizeLinear_1157","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__1497", builder.cast(env.get("_1497"), "float32"));
  env.set("InsertedPrecisionFreeCast__1498", builder["matmul"](env.get("_1441"), env.get("InsertedPrecisionFreeCast__1497"), {"label":"/model/layers.7/attn/q_proj/MatMul_Q4_matmul_1158"}));
  env.set("_1498", builder.cast(env.get("InsertedPrecisionFreeCast__1498"), "float16"));
  env.set("_1499", builder.reshape(env.get("_1498"), [1,128,32,64]));
  env.set("_1500", builder.reshape(env.get("_1499"), [1,128,32,2,32]));
  env.set("InsertedPrecisionFreeCast__1500", builder.cast(env.get("_1500"), "float32"));
  env.set("InsertedPrecisionFreeCast__1510", builder["mul"](env.get("InsertedPrecisionFreeCast__1500"), env.get("InsertedPrecisionFreeCast__1509"), {"label":"/model/layers.7/attn/q_rotary/RotaryEmbedding_mul_cos_1169"}));
  env.set("InsertedPrecisionFreeCast__1511", builder.reshape(env.get("InsertedPrecisionFreeCast__1510"), [1,128,32,64]));
  env.set("InsertedPrecisionFreeCast__1505", builder.cast(env.get("_1505"), "float32"));
  env.set("Inserted_555", builder["clamp"](env.get("position_ids_625"), {"label":"Inserted_Clip_1153","maxValue":2047,"minValue":-2048}));
  env.set("_1490", builder["gather"](env.get("_624"), env.get("Inserted_555"), {"axis":0,"label":"/model/layers.7/attn/q_rotary/RotaryEmbedding_gather_sin_1152"}));
  env.set("_1491", builder.reshape(env.get("_1490"), [1,128,1,1,32]));
  env.set("InsertedPrecisionFreeCast__1491", builder.cast(env.get("_1491"), "float32"));
  {
    const tmp = builder.split(env.get("_1500"), 2, {"axis":3,"label":"/model/layers.7/attn/q_rotary/RotaryEmbedding_split_partial_input0_1161"});
    env.set("_1501", tmp[0]);
    env.set("_1502", tmp[1]);
  }
  env.set("_1503", builder.concat([env.get("_1502"), env.get("_1501")], 3, {"label":"/model/layers.7/attn/q_rotary/RotaryEmbedding_concat_partial_input0_1162"}));
  env.set("InsertedPrecisionFreeCast__1503", builder.cast(env.get("_1503"), "float32"));
  env.set("InsertedPrecisionFreeCast__1504", builder["mul"](env.get("InsertedPrecisionFreeCast__1503"), env.get("InsertedPrecisionFreeCast__1491"), {"label":"/model/layers.7/attn/q_rotary/RotaryEmbedding_mul_sin_1163"}));
  env.set("InsertedPrecisionFreeCast__1506", builder["mul"](env.get("InsertedPrecisionFreeCast__1504"), env.get("InsertedPrecisionFreeCast__1505"), {"label":"/model/layers.7/attn/q_rotary/RotaryEmbedding_mul_sign_1164"}));
  env.set("InsertedPrecisionFreeCast__1507", builder.reshape(env.get("InsertedPrecisionFreeCast__1506"), [1,128,32,64]));
  env.set("InsertedPrecisionFreeCast__1512", builder["add"](env.get("InsertedPrecisionFreeCast__1511"), env.get("InsertedPrecisionFreeCast__1507"), {"label":"/model/layers.7/attn/q_rotary/RotaryEmbedding_add_mul_cos_sin_1171"}));
  env.set("_1512", builder.cast(env.get("InsertedPrecisionFreeCast__1512"), "float16"));
  env.set("_1513", builder.reshape(env.get("_1512"), [1,128,2048]));
  env.set("_1514", builder.reshape(env.get("_1513"), [1,128,32,64]));
  env.set("_1515", builder["transpose"](env.get("_1514"), {"label":"/model/layers.7/attn/GroupQueryAttention_/GQA/query/transpose_1174","permutation":[0,2,1,3]}));
  env.set("InsertedPrecisionFreeCast__1515", builder.cast(env.get("_1515"), "float32"));
  env.set("InsertedPrecisionFreeCast__1516", builder["matmul"](env.get("InsertedPrecisionFreeCast__1515"), env.get("InsertedPrecisionFreeCast__1489"), {"label":"/model/layers.7/attn/GroupQueryAttention_/Attention/qkv/matmul_1_1175"}));
  env.set("InsertedPrecisionFreeCast__1517", builder["mul"](env.get("InsertedPrecisionFreeCast__1516"), env.get("InsertedPrecisionFreeCast__684"), {"label":"/model/layers.7/attn/GroupQueryAttention_/Attention/qkv/div_1176"}));
  env.set("_1456", builder.expand(env.get("_617"), [1,32,128,512], {"label":"/model/layers.7/attn/GroupQueryAttention_/GQA/GQA_mask_shape_ones/expand_1116"}));
  env.set("_1457", builder.cumulativeSum(env.get("_1456"), 3, {"exclusive":true,"label":"/model/layers.7/attn/GroupQueryAttention_range_of_mask_shape_1117"}));
  env.set("_1453", builder["add"](env.get("_613"), env.get("_1445"), {"label":"/model/layers.7/attn/GroupQueryAttention_/GQA/attn_mask/add_1113"}));
  env.set("_1454", builder.expand(env.get("_1453"), [512,128], {"label":"/model/layers.7/attn/GroupQueryAttention_/GQA/expand_neq_right_1114"}));
  env.set("_1455", builder["transpose"](env.get("_1454"), {"label":"/model/layers.7/attn/GroupQueryAttention_/GQA/neq_right/transpose_1115","permutation":[1,0]}));
  env.set("Inserted_529", builder["lesser"](env.get("_1457"), env.get("_1455"), {"label":"/model/layers.7/attn/GroupQueryAttention_/GQA/attn_mask/condition_1_1118"}));
  env.set("InsertedPrecisionFreeCast__1459", builder["where"](env.get("Inserted_529"), env.get("InsertedPrecisionFreeCast__621"), env.get("InsertedPrecisionFreeCast__622"), {"label":"/model/layers.7/attn/GroupQueryAttention_/GQA/attn_mask/where_1120"}));
  env.set("InsertedPrecisionFreeCast__1518", builder["add"](env.get("InsertedPrecisionFreeCast__1517"), env.get("InsertedPrecisionFreeCast__1459"), {"label":"/model/layers.7/attn/GroupQueryAttention_/Attention/attn_mask/softmax_input_1177"}));
  env.set("InsertedPrecisionFreeCast__inlfunc_Softmax_token_6_X_ReduceMax", builder["reduceMax"](env.get("InsertedPrecisionFreeCast__1518"), {"axes":[3],"keepDimensions":true}));
  env.set("InsertedPrecisionFreeCast__inlfunc_Softmax_token_6_X_Sub", builder["sub"](env.get("InsertedPrecisionFreeCast__1518"), env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_6_X_ReduceMax"), {}));
  env.set("InsertedPrecisionFreeCast__inlfunc_Softmax_token_6_X_Exp", builder["exp"](env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_6_X_Sub"), {}));
  env.set("InsertedPrecisionFreeCast__inlfunc_Softmax_token_6_X_ReduceSum", builder["reduceSum"](env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_6_X_Exp"), {"axes":[3],"keepDimensions":true}));
  env.set("InsertedPrecisionFreeCast__1519", builder["div"](env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_6_X_Exp"), env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_6_X_ReduceSum"), {}));
  env.set("InsertedPrecisionFreeCast__1520", builder["matmul"](env.get("InsertedPrecisionFreeCast__1519"), env.get("InsertedPrecisionFreeCast__1452"), {"label":"/model/layers.7/attn/GroupQueryAttention_/Attention/qkv/matmul_2_1179"}));
  env.set("_1520", builder.cast(env.get("InsertedPrecisionFreeCast__1520"), "float16"));
  env.set("_1521", builder["transpose"](env.get("_1520"), {"label":"/model/layers.7/attn/GroupQueryAttention_/Attention/qkv/transpose_1180","permutation":[0,2,1,3]}));
  env.set("_1522", builder.reshape(env.get("_1521"), [1,128,2048]));
  env.set("InsertedPrecisionFreeCast__1522", builder.cast(env.get("_1522"), "float32"));
  env.set("InsertedPrecisionFreeCast__1523", builder["matmul"](env.get("InsertedPrecisionFreeCast__1522"), env.get("InsertedPrecisionFreeCast__404"), {"label":"/model/layers.7/attn/o_proj/MatMul_Q4_matmul_1182"}));
  env.set("_1527", builder["add"](env.get("_1526"), env.get("InsertedPrecisionFreeCast__1523"), {"label":"/model/layers.7/post_attention_layernorm/SkipLayerNorm_add_skip_1186"}));
  env.set("_1550", builder.cast(env.get("_1527"), "float16"));
  env.set("_1551", builder.cast(env.get("_1550"), "float32"));
  env.set("_390", builder.dequantizeLinear(env.get("_387"), env.get("_388"), env.get("_389"), {"axis":2,"blockSize":32,"label":"/model/layers.7/mlp/down_proj/MatMul_Q4_dequantizeLinear_171"}));
  env.set("_391", builder.reshape(env.get("_390"), [2048,5632]));
  env.set("_392", builder["transpose"](env.get("_391"), {"label":"/model/layers.7/mlp/down_proj/MatMul_Q4_transpose_dequantizeLinear_173","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__392", builder.cast(env.get("_392"), "float32"));
  env.set("_1541", builder.dequantizeLinear(env.get("_1538"), env.get("_1539"), env.get("_1540"), {"axis":2,"blockSize":32,"label":"/model/layers.7/mlp/gate_proj/MatMul_Q4_dequantizeLinear_1196"}));
  env.set("_1542", builder.reshape(env.get("_1541"), [5632,2048]));
  env.set("_1543", builder["transpose"](env.get("_1542"), {"label":"/model/layers.7/mlp/gate_proj/MatMul_Q4_transpose_dequantizeLinear_1198","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__1543", builder.cast(env.get("_1543"), "float32"));
  env.set("_1534", builder.cast(env.get("_1533"), "float32"));
  env.set("_1528", builder["pow"](env.get("_1527"), env.get("_583"), {"label":"/model/layers.7/post_attention_layernorm/SkipLayerNorm_pow_1187"}));
  env.set("_1529", builder["reduceMean"](env.get("_1528"), {"axes":[2],"keepDimensions":true,"label":"/model/layers.7/post_attention_layernorm/SkipLayerNorm_reduceMean_1188"}));
  env.set("_1530", builder["add"](env.get("_1529"), env.get("_586"), {"label":"/model/layers.7/post_attention_layernorm/SkipLayerNorm_add_1189"}));
  env.set("_1531", builder["sqrt"](env.get("_1530"), {"label":"/model/layers.7/post_attention_layernorm/SkipLayerNorm_sqrt_1190"}));
  env.set("_1532", builder["div"](env.get("_1527"), env.get("_1531"), {"label":"/model/layers.7/post_attention_layernorm/SkipLayerNorm_div_1191"}));
  env.set("_1535", builder["mul"](env.get("_1534"), env.get("_1532"), {"label":"/model/layers.7/post_attention_layernorm/SkipLayerNorm_mul_1193"}));
  env.set("InsertedPrecisionFreeCast__1544", builder["matmul"](env.get("_1535"), env.get("InsertedPrecisionFreeCast__1543"), {"label":"/model/layers.7/mlp/gate_proj/MatMul_Q4_matmul_1199"}));
  env.set("InsertedPrecisionFreeCast__1545", builder["sigmoid"](env.get("InsertedPrecisionFreeCast__1544"), {"label":"/model/layers.7/mlp/act_fn/Sigmoid_1200"}));
  env.set("InsertedPrecisionFreeCast__1546", builder["mul"](env.get("InsertedPrecisionFreeCast__1544"), env.get("InsertedPrecisionFreeCast__1545"), {"label":"/model/layers.7/mlp/act_fn/Mul_1201"}));
  env.set("_396", builder.dequantizeLinear(env.get("_393"), env.get("_394"), env.get("_395"), {"axis":2,"blockSize":32,"label":"/model/layers.7/mlp/up_proj/MatMul_Q4_dequantizeLinear_174"}));
  env.set("_397", builder.reshape(env.get("_396"), [5632,2048]));
  env.set("_398", builder["transpose"](env.get("_397"), {"label":"/model/layers.7/mlp/up_proj/MatMul_Q4_transpose_dequantizeLinear_176","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__398", builder.cast(env.get("_398"), "float32"));
  env.set("InsertedPrecisionFreeCast__1537", builder["matmul"](env.get("_1535"), env.get("InsertedPrecisionFreeCast__398"), {"label":"/model/layers.7/mlp/up_proj/MatMul_Q4_matmul_1195"}));
  env.set("InsertedPrecisionFreeCast__1547", builder["mul"](env.get("InsertedPrecisionFreeCast__1546"), env.get("InsertedPrecisionFreeCast__1537"), {"label":"/model/layers.7/mlp/Mul_1202"}));
  env.set("InsertedPrecisionFreeCast__1548", builder["matmul"](env.get("InsertedPrecisionFreeCast__1547"), env.get("InsertedPrecisionFreeCast__392"), {"label":"/model/layers.7/mlp/down_proj/MatMul_Q4_matmul_1203"}));
  env.set("_1552", builder["add"](env.get("_1551"), env.get("InsertedPrecisionFreeCast__1548"), {"label":"/model/layers.8/input_layernorm/SkipLayerNorm_add_skip_1207"}));
  env.set("_1644", builder.cast(env.get("_1552"), "float16"));
  env.set("_1645", builder.cast(env.get("_1644"), "float32"));
  env.set("_378", builder.dequantizeLinear(env.get("_375"), env.get("_376"), env.get("_377"), {"axis":2,"blockSize":32,"label":"/model/layers.8/attn/o_proj/MatMul_Q4_dequantizeLinear_165"}));
  env.set("_379", builder.reshape(env.get("_378"), [2048,2048]));
  env.set("_380", builder["transpose"](env.get("_379"), {"label":"/model/layers.8/attn/o_proj/MatMul_Q4_transpose_dequantizeLinear_167","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__380", builder.cast(env.get("_380"), "float32"));
  env.set("Inserted_576", builder.cast(env.get("_601"), "uint8"));
  env.set("_1564", builder["where"](env.get("Inserted_576"), env.get("_602"), env.get("_600"), {"label":"/model/layers.8/attn/GroupQueryAttention_/GQA/scatter/where_1218"}));
  env.set("_1565", builder["add"](env.get("_604"), env.get("_1564"), {"label":"/model/layers.8/attn/GroupQueryAttention_/GQA/right_constant/add_1220"}));
  env.set("_1566", builder.concat([env.get("_606"), env.get("_1565")], 1, {"label":"/model/layers.8/attn/GroupQueryAttention_/GQA/concat_for_pre_scatter_indices_1221"}));
  env.set("_1567", builder.reshape(env.get("_1566"), [1,128,4,3]));
  env.set("Inserted_578", builder.cast(env.get("_1567"), "int64"));
  env.set("Inserted_580", builder["max"](env.get("Inserted_578"), env.get("Inserted_579"), {"label":"Inserted_Max_1225"}));
  env.set("Inserted_582", builder["min"](env.get("Inserted_580"), env.get("Inserted_581"), {"label":"Inserted_Min_1226"}));
  env.set("_384", builder.dequantizeLinear(env.get("_381"), env.get("_382"), env.get("_383"), {"axis":2,"blockSize":32,"label":"/model/layers.8/attn/v_proj/MatMul_Q4_dequantizeLinear_168"}));
  env.set("_385", builder.reshape(env.get("_384"), [256,2048]));
  env.set("_386", builder["transpose"](env.get("_385"), {"label":"/model/layers.8/attn/v_proj/MatMul_Q4_transpose_dequantizeLinear_170","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__386", builder.cast(env.get("_386"), "float32"));
  env.set("_1559", builder.cast(env.get("_1558"), "float32"));
  env.set("_1553", builder["pow"](env.get("_1552"), env.get("_583"), {"label":"/model/layers.8/input_layernorm/SkipLayerNorm_pow_1208"}));
  env.set("_1554", builder["reduceMean"](env.get("_1553"), {"axes":[2],"keepDimensions":true,"label":"/model/layers.8/input_layernorm/SkipLayerNorm_reduceMean_1209"}));
  env.set("_1555", builder["add"](env.get("_1554"), env.get("_586"), {"label":"/model/layers.8/input_layernorm/SkipLayerNorm_add_1210"}));
  env.set("_1556", builder["sqrt"](env.get("_1555"), {"label":"/model/layers.8/input_layernorm/SkipLayerNorm_sqrt_1211"}));
  env.set("_1557", builder["div"](env.get("_1552"), env.get("_1556"), {"label":"/model/layers.8/input_layernorm/SkipLayerNorm_div_1212"}));
  env.set("_1560", builder["mul"](env.get("_1559"), env.get("_1557"), {"label":"/model/layers.8/input_layernorm/SkipLayerNorm_mul_1214"}));
  env.set("InsertedPrecisionFreeCast__1562", builder["matmul"](env.get("_1560"), env.get("InsertedPrecisionFreeCast__386"), {"label":"/model/layers.8/attn/v_proj/MatMul_Q4_matmul_1216"}));
  env.set("_1562", builder.cast(env.get("InsertedPrecisionFreeCast__1562"), "float16"));
  env.set("_1563", builder.reshape(env.get("_1562"), [1,128,4,64]));
  env.set("present_8_value_17", builder["scatterND"](env.get("past_key_values_8_value_1568"), env.get("Inserted_582"), env.get("_1563"), {"label":"/model/layers.8/attn/GroupQueryAttention_/GQA/present_value/ScatterND_1223"}));
  env.set("_1569", builder.reshape(env.get("present_8_value_17"), [1,4,1,512,64]));
  env.set("_1570", builder.expand(env.get("_1569"), [1,4,8,512,64], {"label":"/model/layers.8/attn/GroupQueryAttention_/GQA/true_present_value/expand_1228"}));
  env.set("_1571", builder.reshape(env.get("_1570"), [1,32,512,64]));
  env.set("InsertedPrecisionFreeCast__1571", builder.cast(env.get("_1571"), "float32"));
  env.set("Inserted_607", builder.cast(env.get("_1567"), "int64"));
  env.set("Inserted_609", builder["max"](env.get("Inserted_607"), env.get("Inserted_608"), {"label":"Inserted_Max_1263"}));
  env.set("Inserted_611", builder["min"](env.get("Inserted_609"), env.get("Inserted_610"), {"label":"Inserted_Min_1264"}));
  env.set("Inserted_600", builder["clamp"](env.get("position_ids_625"), {"label":"Inserted_Clip_1254","maxValue":2047,"minValue":-2048}));
  env.set("_1597", builder["gather"](env.get("_644"), env.get("Inserted_600"), {"axis":0,"label":"/model/layers.8/attn/k_rotary/RotaryEmbedding_gather_cos_1253"}));
  env.set("_1598", builder.reshape(env.get("_1597"), [1,128,1,1,32]));
  env.set("InsertedPrecisionFreeCast__1598", builder.cast(env.get("_1598"), "float32"));
  env.set("_1584", builder.dequantizeLinear(env.get("_1581"), env.get("_1582"), env.get("_1583"), {"axis":2,"blockSize":32,"label":"/model/layers.8/attn/k_proj/MatMul_Q4_dequantizeLinear_1242"}));
  env.set("_1585", builder.reshape(env.get("_1584"), [256,2048]));
  env.set("_1586", builder["transpose"](env.get("_1585"), {"label":"/model/layers.8/attn/k_proj/MatMul_Q4_transpose_dequantizeLinear_1244","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__1586", builder.cast(env.get("_1586"), "float32"));
  env.set("InsertedPrecisionFreeCast__1587", builder["matmul"](env.get("_1560"), env.get("InsertedPrecisionFreeCast__1586"), {"label":"/model/layers.8/attn/k_proj/MatMul_Q4_matmul_1245"}));
  env.set("_1587", builder.cast(env.get("InsertedPrecisionFreeCast__1587"), "float16"));
  env.set("_1588", builder.reshape(env.get("_1587"), [1,128,4,64]));
  env.set("_1589", builder.reshape(env.get("_1588"), [1,128,4,2,32]));
  env.set("InsertedPrecisionFreeCast__1589", builder.cast(env.get("_1589"), "float32"));
  env.set("InsertedPrecisionFreeCast__1599", builder["mul"](env.get("InsertedPrecisionFreeCast__1589"), env.get("InsertedPrecisionFreeCast__1598"), {"label":"/model/layers.8/attn/k_rotary/RotaryEmbedding_mul_cos_1256"}));
  env.set("InsertedPrecisionFreeCast__1600", builder.reshape(env.get("InsertedPrecisionFreeCast__1599"), [1,128,4,64]));
  env.set("InsertedPrecisionFreeCast__1594", builder.cast(env.get("_1594"), "float32"));
  env.set("Inserted_591", builder["clamp"](env.get("position_ids_625"), {"label":"Inserted_Clip_1240","maxValue":2047,"minValue":-2048}));
  env.set("_1579", builder["gather"](env.get("_624"), env.get("Inserted_591"), {"axis":0,"label":"/model/layers.8/attn/k_rotary/RotaryEmbedding_gather_sin_1239"}));
  env.set("_1580", builder.reshape(env.get("_1579"), [1,128,1,1,32]));
  env.set("InsertedPrecisionFreeCast__1580", builder.cast(env.get("_1580"), "float32"));
  {
    const tmp = builder.split(env.get("_1589"), 2, {"axis":3,"label":"/model/layers.8/attn/k_rotary/RotaryEmbedding_split_partial_input0_1248"});
    env.set("_1590", tmp[0]);
    env.set("_1591", tmp[1]);
  }
  env.set("_1592", builder.concat([env.get("_1591"), env.get("_1590")], 3, {"label":"/model/layers.8/attn/k_rotary/RotaryEmbedding_concat_partial_input0_1249"}));
  env.set("InsertedPrecisionFreeCast__1592", builder.cast(env.get("_1592"), "float32"));
  env.set("InsertedPrecisionFreeCast__1593", builder["mul"](env.get("InsertedPrecisionFreeCast__1592"), env.get("InsertedPrecisionFreeCast__1580"), {"label":"/model/layers.8/attn/k_rotary/RotaryEmbedding_mul_sin_1250"}));
  env.set("InsertedPrecisionFreeCast__1595", builder["mul"](env.get("InsertedPrecisionFreeCast__1593"), env.get("InsertedPrecisionFreeCast__1594"), {"label":"/model/layers.8/attn/k_rotary/RotaryEmbedding_mul_sign_1251"}));
  env.set("InsertedPrecisionFreeCast__1596", builder.reshape(env.get("InsertedPrecisionFreeCast__1595"), [1,128,4,64]));
  env.set("InsertedPrecisionFreeCast__1601", builder["add"](env.get("InsertedPrecisionFreeCast__1600"), env.get("InsertedPrecisionFreeCast__1596"), {"label":"/model/layers.8/attn/k_rotary/RotaryEmbedding_add_mul_cos_sin_1258"}));
  env.set("_1601", builder.cast(env.get("InsertedPrecisionFreeCast__1601"), "float16"));
  env.set("_1602", builder.reshape(env.get("_1601"), [1,128,256]));
  env.set("_1603", builder.reshape(env.get("_1602"), [1,128,4,64]));
  env.set("present_8_key_16", builder["scatterND"](env.get("past_key_values_8_key_1604"), env.get("Inserted_611"), env.get("_1603"), {"label":"/model/layers.8/attn/GroupQueryAttention_/GQA/present_key/ScatterND_1261"}));
  env.set("_1605", builder.reshape(env.get("present_8_key_16"), [1,4,1,512,64]));
  env.set("_1606", builder.expand(env.get("_1605"), [1,4,8,512,64], {"label":"/model/layers.8/attn/GroupQueryAttention_/GQA/true_present_key/expand_1266"}));
  env.set("_1607", builder.reshape(env.get("_1606"), [1,32,512,64]));
  env.set("_1608", builder["transpose"](env.get("_1607"), {"label":"/model/layers.8/attn/GroupQueryAttention_/GQA/present_key/transpose_1268","permutation":[0,1,3,2]}));
  env.set("InsertedPrecisionFreeCast__1608", builder.cast(env.get("_1608"), "float32"));
  env.set("Inserted_624", builder["clamp"](env.get("position_ids_625"), {"label":"Inserted_Clip_1284","maxValue":2047,"minValue":-2048}));
  env.set("_1627", builder["gather"](env.get("_644"), env.get("Inserted_624"), {"axis":0,"label":"/model/layers.8/attn/q_rotary/RotaryEmbedding_gather_cos_1283"}));
  env.set("_1628", builder.reshape(env.get("_1627"), [1,128,1,1,32]));
  env.set("InsertedPrecisionFreeCast__1628", builder.cast(env.get("_1628"), "float32"));
  env.set("_1614", builder.dequantizeLinear(env.get("_1611"), env.get("_1612"), env.get("_1613"), {"axis":2,"blockSize":32,"label":"/model/layers.8/attn/q_proj/MatMul_Q4_dequantizeLinear_1272"}));
  env.set("_1615", builder.reshape(env.get("_1614"), [2048,2048]));
  env.set("_1616", builder["transpose"](env.get("_1615"), {"label":"/model/layers.8/attn/q_proj/MatMul_Q4_transpose_dequantizeLinear_1274","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__1616", builder.cast(env.get("_1616"), "float32"));
  env.set("InsertedPrecisionFreeCast__1617", builder["matmul"](env.get("_1560"), env.get("InsertedPrecisionFreeCast__1616"), {"label":"/model/layers.8/attn/q_proj/MatMul_Q4_matmul_1275"}));
  env.set("_1617", builder.cast(env.get("InsertedPrecisionFreeCast__1617"), "float16"));
  env.set("_1618", builder.reshape(env.get("_1617"), [1,128,32,64]));
  env.set("_1619", builder.reshape(env.get("_1618"), [1,128,32,2,32]));
  env.set("InsertedPrecisionFreeCast__1619", builder.cast(env.get("_1619"), "float32"));
  env.set("InsertedPrecisionFreeCast__1629", builder["mul"](env.get("InsertedPrecisionFreeCast__1619"), env.get("InsertedPrecisionFreeCast__1628"), {"label":"/model/layers.8/attn/q_rotary/RotaryEmbedding_mul_cos_1286"}));
  env.set("InsertedPrecisionFreeCast__1630", builder.reshape(env.get("InsertedPrecisionFreeCast__1629"), [1,128,32,64]));
  env.set("InsertedPrecisionFreeCast__1624", builder.cast(env.get("_1624"), "float32"));
  env.set("Inserted_615", builder["clamp"](env.get("position_ids_625"), {"label":"Inserted_Clip_1270","maxValue":2047,"minValue":-2048}));
  env.set("_1609", builder["gather"](env.get("_624"), env.get("Inserted_615"), {"axis":0,"label":"/model/layers.8/attn/q_rotary/RotaryEmbedding_gather_sin_1269"}));
  env.set("_1610", builder.reshape(env.get("_1609"), [1,128,1,1,32]));
  env.set("InsertedPrecisionFreeCast__1610", builder.cast(env.get("_1610"), "float32"));
  {
    const tmp = builder.split(env.get("_1619"), 2, {"axis":3,"label":"/model/layers.8/attn/q_rotary/RotaryEmbedding_split_partial_input0_1278"});
    env.set("_1620", tmp[0]);
    env.set("_1621", tmp[1]);
  }
  env.set("_1622", builder.concat([env.get("_1621"), env.get("_1620")], 3, {"label":"/model/layers.8/attn/q_rotary/RotaryEmbedding_concat_partial_input0_1279"}));
  env.set("InsertedPrecisionFreeCast__1622", builder.cast(env.get("_1622"), "float32"));
  env.set("InsertedPrecisionFreeCast__1623", builder["mul"](env.get("InsertedPrecisionFreeCast__1622"), env.get("InsertedPrecisionFreeCast__1610"), {"label":"/model/layers.8/attn/q_rotary/RotaryEmbedding_mul_sin_1280"}));
  env.set("InsertedPrecisionFreeCast__1625", builder["mul"](env.get("InsertedPrecisionFreeCast__1623"), env.get("InsertedPrecisionFreeCast__1624"), {"label":"/model/layers.8/attn/q_rotary/RotaryEmbedding_mul_sign_1281"}));
  env.set("InsertedPrecisionFreeCast__1626", builder.reshape(env.get("InsertedPrecisionFreeCast__1625"), [1,128,32,64]));
  env.set("InsertedPrecisionFreeCast__1631", builder["add"](env.get("InsertedPrecisionFreeCast__1630"), env.get("InsertedPrecisionFreeCast__1626"), {"label":"/model/layers.8/attn/q_rotary/RotaryEmbedding_add_mul_cos_sin_1288"}));
  env.set("_1631", builder.cast(env.get("InsertedPrecisionFreeCast__1631"), "float16"));
  env.set("_1632", builder.reshape(env.get("_1631"), [1,128,2048]));
  env.set("_1633", builder.reshape(env.get("_1632"), [1,128,32,64]));
  env.set("_1634", builder["transpose"](env.get("_1633"), {"label":"/model/layers.8/attn/GroupQueryAttention_/GQA/query/transpose_1291","permutation":[0,2,1,3]}));
  env.set("InsertedPrecisionFreeCast__1634", builder.cast(env.get("_1634"), "float32"));
  env.set("InsertedPrecisionFreeCast__1635", builder["matmul"](env.get("InsertedPrecisionFreeCast__1634"), env.get("InsertedPrecisionFreeCast__1608"), {"label":"/model/layers.8/attn/GroupQueryAttention_/Attention/qkv/matmul_1_1292"}));
  env.set("InsertedPrecisionFreeCast__1636", builder["mul"](env.get("InsertedPrecisionFreeCast__1635"), env.get("InsertedPrecisionFreeCast__684"), {"label":"/model/layers.8/attn/GroupQueryAttention_/Attention/qkv/div_1293"}));
  env.set("_1575", builder.expand(env.get("_617"), [1,32,128,512], {"label":"/model/layers.8/attn/GroupQueryAttention_/GQA/GQA_mask_shape_ones/expand_1233"}));
  env.set("_1576", builder.cumulativeSum(env.get("_1575"), 3, {"exclusive":true,"label":"/model/layers.8/attn/GroupQueryAttention_range_of_mask_shape_1234"}));
  env.set("_1572", builder["add"](env.get("_613"), env.get("_1564"), {"label":"/model/layers.8/attn/GroupQueryAttention_/GQA/attn_mask/add_1230"}));
  env.set("_1573", builder.expand(env.get("_1572"), [512,128], {"label":"/model/layers.8/attn/GroupQueryAttention_/GQA/expand_neq_right_1231"}));
  env.set("_1574", builder["transpose"](env.get("_1573"), {"label":"/model/layers.8/attn/GroupQueryAttention_/GQA/neq_right/transpose_1232","permutation":[1,0]}));
  env.set("Inserted_589", builder["lesser"](env.get("_1576"), env.get("_1574"), {"label":"/model/layers.8/attn/GroupQueryAttention_/GQA/attn_mask/condition_1_1235"}));
  env.set("InsertedPrecisionFreeCast__1578", builder["where"](env.get("Inserted_589"), env.get("InsertedPrecisionFreeCast__621"), env.get("InsertedPrecisionFreeCast__622"), {"label":"/model/layers.8/attn/GroupQueryAttention_/GQA/attn_mask/where_1237"}));
  env.set("InsertedPrecisionFreeCast__1637", builder["add"](env.get("InsertedPrecisionFreeCast__1636"), env.get("InsertedPrecisionFreeCast__1578"), {"label":"/model/layers.8/attn/GroupQueryAttention_/Attention/attn_mask/softmax_input_1294"}));
  env.set("InsertedPrecisionFreeCast__inlfunc_Softmax_token_7_X_ReduceMax", builder["reduceMax"](env.get("InsertedPrecisionFreeCast__1637"), {"axes":[3],"keepDimensions":true}));
  env.set("InsertedPrecisionFreeCast__inlfunc_Softmax_token_7_X_Sub", builder["sub"](env.get("InsertedPrecisionFreeCast__1637"), env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_7_X_ReduceMax"), {}));
  env.set("InsertedPrecisionFreeCast__inlfunc_Softmax_token_7_X_Exp", builder["exp"](env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_7_X_Sub"), {}));
  env.set("InsertedPrecisionFreeCast__inlfunc_Softmax_token_7_X_ReduceSum", builder["reduceSum"](env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_7_X_Exp"), {"axes":[3],"keepDimensions":true}));
  env.set("InsertedPrecisionFreeCast__1638", builder["div"](env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_7_X_Exp"), env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_7_X_ReduceSum"), {}));
  env.set("InsertedPrecisionFreeCast__1639", builder["matmul"](env.get("InsertedPrecisionFreeCast__1638"), env.get("InsertedPrecisionFreeCast__1571"), {"label":"/model/layers.8/attn/GroupQueryAttention_/Attention/qkv/matmul_2_1296"}));
  env.set("_1639", builder.cast(env.get("InsertedPrecisionFreeCast__1639"), "float16"));
  env.set("_1640", builder["transpose"](env.get("_1639"), {"label":"/model/layers.8/attn/GroupQueryAttention_/Attention/qkv/transpose_1297","permutation":[0,2,1,3]}));
  env.set("_1641", builder.reshape(env.get("_1640"), [1,128,2048]));
  env.set("InsertedPrecisionFreeCast__1641", builder.cast(env.get("_1641"), "float32"));
  env.set("InsertedPrecisionFreeCast__1642", builder["matmul"](env.get("InsertedPrecisionFreeCast__1641"), env.get("InsertedPrecisionFreeCast__380"), {"label":"/model/layers.8/attn/o_proj/MatMul_Q4_matmul_1299"}));
  env.set("_1646", builder["add"](env.get("_1645"), env.get("InsertedPrecisionFreeCast__1642"), {"label":"/model/layers.8/post_attention_layernorm/SkipLayerNorm_add_skip_1303"}));
  env.set("_1669", builder.cast(env.get("_1646"), "float16"));
  env.set("_1670", builder.cast(env.get("_1669"), "float32"));
  env.set("_366", builder.dequantizeLinear(env.get("_363"), env.get("_364"), env.get("_365"), {"axis":2,"blockSize":32,"label":"/model/layers.8/mlp/down_proj/MatMul_Q4_dequantizeLinear_159"}));
  env.set("_367", builder.reshape(env.get("_366"), [2048,5632]));
  env.set("_368", builder["transpose"](env.get("_367"), {"label":"/model/layers.8/mlp/down_proj/MatMul_Q4_transpose_dequantizeLinear_161","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__368", builder.cast(env.get("_368"), "float32"));
  env.set("_1660", builder.dequantizeLinear(env.get("_1657"), env.get("_1658"), env.get("_1659"), {"axis":2,"blockSize":32,"label":"/model/layers.8/mlp/gate_proj/MatMul_Q4_dequantizeLinear_1313"}));
  env.set("_1661", builder.reshape(env.get("_1660"), [5632,2048]));
  env.set("_1662", builder["transpose"](env.get("_1661"), {"label":"/model/layers.8/mlp/gate_proj/MatMul_Q4_transpose_dequantizeLinear_1315","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__1662", builder.cast(env.get("_1662"), "float32"));
  env.set("_1653", builder.cast(env.get("_1652"), "float32"));
  env.set("_1647", builder["pow"](env.get("_1646"), env.get("_583"), {"label":"/model/layers.8/post_attention_layernorm/SkipLayerNorm_pow_1304"}));
  env.set("_1648", builder["reduceMean"](env.get("_1647"), {"axes":[2],"keepDimensions":true,"label":"/model/layers.8/post_attention_layernorm/SkipLayerNorm_reduceMean_1305"}));
  env.set("_1649", builder["add"](env.get("_1648"), env.get("_586"), {"label":"/model/layers.8/post_attention_layernorm/SkipLayerNorm_add_1306"}));
  env.set("_1650", builder["sqrt"](env.get("_1649"), {"label":"/model/layers.8/post_attention_layernorm/SkipLayerNorm_sqrt_1307"}));
  env.set("_1651", builder["div"](env.get("_1646"), env.get("_1650"), {"label":"/model/layers.8/post_attention_layernorm/SkipLayerNorm_div_1308"}));
  env.set("_1654", builder["mul"](env.get("_1653"), env.get("_1651"), {"label":"/model/layers.8/post_attention_layernorm/SkipLayerNorm_mul_1310"}));
  env.set("InsertedPrecisionFreeCast__1663", builder["matmul"](env.get("_1654"), env.get("InsertedPrecisionFreeCast__1662"), {"label":"/model/layers.8/mlp/gate_proj/MatMul_Q4_matmul_1316"}));
  env.set("InsertedPrecisionFreeCast__1664", builder["sigmoid"](env.get("InsertedPrecisionFreeCast__1663"), {"label":"/model/layers.8/mlp/act_fn/Sigmoid_1317"}));
  env.set("InsertedPrecisionFreeCast__1665", builder["mul"](env.get("InsertedPrecisionFreeCast__1663"), env.get("InsertedPrecisionFreeCast__1664"), {"label":"/model/layers.8/mlp/act_fn/Mul_1318"}));
  env.set("_372", builder.dequantizeLinear(env.get("_369"), env.get("_370"), env.get("_371"), {"axis":2,"blockSize":32,"label":"/model/layers.8/mlp/up_proj/MatMul_Q4_dequantizeLinear_162"}));
  env.set("_373", builder.reshape(env.get("_372"), [5632,2048]));
  env.set("_374", builder["transpose"](env.get("_373"), {"label":"/model/layers.8/mlp/up_proj/MatMul_Q4_transpose_dequantizeLinear_164","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__374", builder.cast(env.get("_374"), "float32"));
  env.set("InsertedPrecisionFreeCast__1656", builder["matmul"](env.get("_1654"), env.get("InsertedPrecisionFreeCast__374"), {"label":"/model/layers.8/mlp/up_proj/MatMul_Q4_matmul_1312"}));
  env.set("InsertedPrecisionFreeCast__1666", builder["mul"](env.get("InsertedPrecisionFreeCast__1665"), env.get("InsertedPrecisionFreeCast__1656"), {"label":"/model/layers.8/mlp/Mul_1319"}));
  env.set("InsertedPrecisionFreeCast__1667", builder["matmul"](env.get("InsertedPrecisionFreeCast__1666"), env.get("InsertedPrecisionFreeCast__368"), {"label":"/model/layers.8/mlp/down_proj/MatMul_Q4_matmul_1320"}));
  env.set("_1671", builder["add"](env.get("_1670"), env.get("InsertedPrecisionFreeCast__1667"), {"label":"/model/layers.9/input_layernorm/SkipLayerNorm_add_skip_1324"}));
  env.set("_1763", builder.cast(env.get("_1671"), "float16"));
  env.set("_1764", builder.cast(env.get("_1763"), "float32"));
  env.set("_354", builder.dequantizeLinear(env.get("_351"), env.get("_352"), env.get("_353"), {"axis":2,"blockSize":32,"label":"/model/layers.9/attn/o_proj/MatMul_Q4_dequantizeLinear_153"}));
  env.set("_355", builder.reshape(env.get("_354"), [2048,2048]));
  env.set("_356", builder["transpose"](env.get("_355"), {"label":"/model/layers.9/attn/o_proj/MatMul_Q4_transpose_dequantizeLinear_155","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__356", builder.cast(env.get("_356"), "float32"));
  env.set("Inserted_636", builder.cast(env.get("_601"), "uint8"));
  env.set("_1683", builder["where"](env.get("Inserted_636"), env.get("_602"), env.get("_600"), {"label":"/model/layers.9/attn/GroupQueryAttention_/GQA/scatter/where_1335"}));
  env.set("_1684", builder["add"](env.get("_604"), env.get("_1683"), {"label":"/model/layers.9/attn/GroupQueryAttention_/GQA/right_constant/add_1337"}));
  env.set("_1685", builder.concat([env.get("_606"), env.get("_1684")], 1, {"label":"/model/layers.9/attn/GroupQueryAttention_/GQA/concat_for_pre_scatter_indices_1338"}));
  env.set("_1686", builder.reshape(env.get("_1685"), [1,128,4,3]));
  env.set("Inserted_638", builder.cast(env.get("_1686"), "int64"));
  env.set("Inserted_640", builder["max"](env.get("Inserted_638"), env.get("Inserted_639"), {"label":"Inserted_Max_1342"}));
  env.set("Inserted_642", builder["min"](env.get("Inserted_640"), env.get("Inserted_641"), {"label":"Inserted_Min_1343"}));
  env.set("_360", builder.dequantizeLinear(env.get("_357"), env.get("_358"), env.get("_359"), {"axis":2,"blockSize":32,"label":"/model/layers.9/attn/v_proj/MatMul_Q4_dequantizeLinear_156"}));
  env.set("_361", builder.reshape(env.get("_360"), [256,2048]));
  env.set("_362", builder["transpose"](env.get("_361"), {"label":"/model/layers.9/attn/v_proj/MatMul_Q4_transpose_dequantizeLinear_158","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__362", builder.cast(env.get("_362"), "float32"));
  env.set("_1678", builder.cast(env.get("_1677"), "float32"));
  env.set("_1672", builder["pow"](env.get("_1671"), env.get("_583"), {"label":"/model/layers.9/input_layernorm/SkipLayerNorm_pow_1325"}));
  env.set("_1673", builder["reduceMean"](env.get("_1672"), {"axes":[2],"keepDimensions":true,"label":"/model/layers.9/input_layernorm/SkipLayerNorm_reduceMean_1326"}));
  env.set("_1674", builder["add"](env.get("_1673"), env.get("_586"), {"label":"/model/layers.9/input_layernorm/SkipLayerNorm_add_1327"}));
  env.set("_1675", builder["sqrt"](env.get("_1674"), {"label":"/model/layers.9/input_layernorm/SkipLayerNorm_sqrt_1328"}));
  env.set("_1676", builder["div"](env.get("_1671"), env.get("_1675"), {"label":"/model/layers.9/input_layernorm/SkipLayerNorm_div_1329"}));
  env.set("_1679", builder["mul"](env.get("_1678"), env.get("_1676"), {"label":"/model/layers.9/input_layernorm/SkipLayerNorm_mul_1331"}));
  env.set("InsertedPrecisionFreeCast__1681", builder["matmul"](env.get("_1679"), env.get("InsertedPrecisionFreeCast__362"), {"label":"/model/layers.9/attn/v_proj/MatMul_Q4_matmul_1333"}));
  env.set("_1681", builder.cast(env.get("InsertedPrecisionFreeCast__1681"), "float16"));
  env.set("_1682", builder.reshape(env.get("_1681"), [1,128,4,64]));
  env.set("present_9_value_19", builder["scatterND"](env.get("past_key_values_9_value_1687"), env.get("Inserted_642"), env.get("_1682"), {"label":"/model/layers.9/attn/GroupQueryAttention_/GQA/present_value/ScatterND_1340"}));
  env.set("_1688", builder.reshape(env.get("present_9_value_19"), [1,4,1,512,64]));
  env.set("_1689", builder.expand(env.get("_1688"), [1,4,8,512,64], {"label":"/model/layers.9/attn/GroupQueryAttention_/GQA/true_present_value/expand_1345"}));
  env.set("_1690", builder.reshape(env.get("_1689"), [1,32,512,64]));
  env.set("InsertedPrecisionFreeCast__1690", builder.cast(env.get("_1690"), "float32"));
  env.set("Inserted_667", builder.cast(env.get("_1686"), "int64"));
  env.set("Inserted_669", builder["max"](env.get("Inserted_667"), env.get("Inserted_668"), {"label":"Inserted_Max_1380"}));
  env.set("Inserted_671", builder["min"](env.get("Inserted_669"), env.get("Inserted_670"), {"label":"Inserted_Min_1381"}));
  env.set("Inserted_660", builder["clamp"](env.get("position_ids_625"), {"label":"Inserted_Clip_1371","maxValue":2047,"minValue":-2048}));
  env.set("_1716", builder["gather"](env.get("_644"), env.get("Inserted_660"), {"axis":0,"label":"/model/layers.9/attn/k_rotary/RotaryEmbedding_gather_cos_1370"}));
  env.set("_1717", builder.reshape(env.get("_1716"), [1,128,1,1,32]));
  env.set("InsertedPrecisionFreeCast__1717", builder.cast(env.get("_1717"), "float32"));
  env.set("_1703", builder.dequantizeLinear(env.get("_1700"), env.get("_1701"), env.get("_1702"), {"axis":2,"blockSize":32,"label":"/model/layers.9/attn/k_proj/MatMul_Q4_dequantizeLinear_1359"}));
  env.set("_1704", builder.reshape(env.get("_1703"), [256,2048]));
  env.set("_1705", builder["transpose"](env.get("_1704"), {"label":"/model/layers.9/attn/k_proj/MatMul_Q4_transpose_dequantizeLinear_1361","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__1705", builder.cast(env.get("_1705"), "float32"));
  env.set("InsertedPrecisionFreeCast__1706", builder["matmul"](env.get("_1679"), env.get("InsertedPrecisionFreeCast__1705"), {"label":"/model/layers.9/attn/k_proj/MatMul_Q4_matmul_1362"}));
  env.set("_1706", builder.cast(env.get("InsertedPrecisionFreeCast__1706"), "float16"));
  env.set("_1707", builder.reshape(env.get("_1706"), [1,128,4,64]));
  env.set("_1708", builder.reshape(env.get("_1707"), [1,128,4,2,32]));
  env.set("InsertedPrecisionFreeCast__1708", builder.cast(env.get("_1708"), "float32"));
  env.set("InsertedPrecisionFreeCast__1718", builder["mul"](env.get("InsertedPrecisionFreeCast__1708"), env.get("InsertedPrecisionFreeCast__1717"), {"label":"/model/layers.9/attn/k_rotary/RotaryEmbedding_mul_cos_1373"}));
  env.set("InsertedPrecisionFreeCast__1719", builder.reshape(env.get("InsertedPrecisionFreeCast__1718"), [1,128,4,64]));
  env.set("InsertedPrecisionFreeCast__1713", builder.cast(env.get("_1713"), "float32"));
  env.set("Inserted_651", builder["clamp"](env.get("position_ids_625"), {"label":"Inserted_Clip_1357","maxValue":2047,"minValue":-2048}));
  env.set("_1698", builder["gather"](env.get("_624"), env.get("Inserted_651"), {"axis":0,"label":"/model/layers.9/attn/k_rotary/RotaryEmbedding_gather_sin_1356"}));
  env.set("_1699", builder.reshape(env.get("_1698"), [1,128,1,1,32]));
  env.set("InsertedPrecisionFreeCast__1699", builder.cast(env.get("_1699"), "float32"));
  {
    const tmp = builder.split(env.get("_1708"), 2, {"axis":3,"label":"/model/layers.9/attn/k_rotary/RotaryEmbedding_split_partial_input0_1365"});
    env.set("_1709", tmp[0]);
    env.set("_1710", tmp[1]);
  }
  env.set("_1711", builder.concat([env.get("_1710"), env.get("_1709")], 3, {"label":"/model/layers.9/attn/k_rotary/RotaryEmbedding_concat_partial_input0_1366"}));
  env.set("InsertedPrecisionFreeCast__1711", builder.cast(env.get("_1711"), "float32"));
  env.set("InsertedPrecisionFreeCast__1712", builder["mul"](env.get("InsertedPrecisionFreeCast__1711"), env.get("InsertedPrecisionFreeCast__1699"), {"label":"/model/layers.9/attn/k_rotary/RotaryEmbedding_mul_sin_1367"}));
  env.set("InsertedPrecisionFreeCast__1714", builder["mul"](env.get("InsertedPrecisionFreeCast__1712"), env.get("InsertedPrecisionFreeCast__1713"), {"label":"/model/layers.9/attn/k_rotary/RotaryEmbedding_mul_sign_1368"}));
  env.set("InsertedPrecisionFreeCast__1715", builder.reshape(env.get("InsertedPrecisionFreeCast__1714"), [1,128,4,64]));
  env.set("InsertedPrecisionFreeCast__1720", builder["add"](env.get("InsertedPrecisionFreeCast__1719"), env.get("InsertedPrecisionFreeCast__1715"), {"label":"/model/layers.9/attn/k_rotary/RotaryEmbedding_add_mul_cos_sin_1375"}));
  env.set("_1720", builder.cast(env.get("InsertedPrecisionFreeCast__1720"), "float16"));
  env.set("_1721", builder.reshape(env.get("_1720"), [1,128,256]));
  env.set("_1722", builder.reshape(env.get("_1721"), [1,128,4,64]));
  env.set("present_9_key_18", builder["scatterND"](env.get("past_key_values_9_key_1723"), env.get("Inserted_671"), env.get("_1722"), {"label":"/model/layers.9/attn/GroupQueryAttention_/GQA/present_key/ScatterND_1378"}));
  env.set("_1724", builder.reshape(env.get("present_9_key_18"), [1,4,1,512,64]));
  env.set("_1725", builder.expand(env.get("_1724"), [1,4,8,512,64], {"label":"/model/layers.9/attn/GroupQueryAttention_/GQA/true_present_key/expand_1383"}));
  env.set("_1726", builder.reshape(env.get("_1725"), [1,32,512,64]));
  env.set("_1727", builder["transpose"](env.get("_1726"), {"label":"/model/layers.9/attn/GroupQueryAttention_/GQA/present_key/transpose_1385","permutation":[0,1,3,2]}));
  env.set("InsertedPrecisionFreeCast__1727", builder.cast(env.get("_1727"), "float32"));
  env.set("Inserted_684", builder["clamp"](env.get("position_ids_625"), {"label":"Inserted_Clip_1401","maxValue":2047,"minValue":-2048}));
  env.set("_1746", builder["gather"](env.get("_644"), env.get("Inserted_684"), {"axis":0,"label":"/model/layers.9/attn/q_rotary/RotaryEmbedding_gather_cos_1400"}));
  env.set("_1747", builder.reshape(env.get("_1746"), [1,128,1,1,32]));
  env.set("InsertedPrecisionFreeCast__1747", builder.cast(env.get("_1747"), "float32"));
  env.set("_1733", builder.dequantizeLinear(env.get("_1730"), env.get("_1731"), env.get("_1732"), {"axis":2,"blockSize":32,"label":"/model/layers.9/attn/q_proj/MatMul_Q4_dequantizeLinear_1389"}));
  env.set("_1734", builder.reshape(env.get("_1733"), [2048,2048]));
  env.set("_1735", builder["transpose"](env.get("_1734"), {"label":"/model/layers.9/attn/q_proj/MatMul_Q4_transpose_dequantizeLinear_1391","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__1735", builder.cast(env.get("_1735"), "float32"));
  env.set("InsertedPrecisionFreeCast__1736", builder["matmul"](env.get("_1679"), env.get("InsertedPrecisionFreeCast__1735"), {"label":"/model/layers.9/attn/q_proj/MatMul_Q4_matmul_1392"}));
  env.set("_1736", builder.cast(env.get("InsertedPrecisionFreeCast__1736"), "float16"));
  env.set("_1737", builder.reshape(env.get("_1736"), [1,128,32,64]));
  env.set("_1738", builder.reshape(env.get("_1737"), [1,128,32,2,32]));
  env.set("InsertedPrecisionFreeCast__1738", builder.cast(env.get("_1738"), "float32"));
  env.set("InsertedPrecisionFreeCast__1748", builder["mul"](env.get("InsertedPrecisionFreeCast__1738"), env.get("InsertedPrecisionFreeCast__1747"), {"label":"/model/layers.9/attn/q_rotary/RotaryEmbedding_mul_cos_1403"}));
  env.set("InsertedPrecisionFreeCast__1749", builder.reshape(env.get("InsertedPrecisionFreeCast__1748"), [1,128,32,64]));
  env.set("InsertedPrecisionFreeCast__1743", builder.cast(env.get("_1743"), "float32"));
  env.set("Inserted_675", builder["clamp"](env.get("position_ids_625"), {"label":"Inserted_Clip_1387","maxValue":2047,"minValue":-2048}));
  env.set("_1728", builder["gather"](env.get("_624"), env.get("Inserted_675"), {"axis":0,"label":"/model/layers.9/attn/q_rotary/RotaryEmbedding_gather_sin_1386"}));
  env.set("_1729", builder.reshape(env.get("_1728"), [1,128,1,1,32]));
  env.set("InsertedPrecisionFreeCast__1729", builder.cast(env.get("_1729"), "float32"));
  {
    const tmp = builder.split(env.get("_1738"), 2, {"axis":3,"label":"/model/layers.9/attn/q_rotary/RotaryEmbedding_split_partial_input0_1395"});
    env.set("_1739", tmp[0]);
    env.set("_1740", tmp[1]);
  }
  env.set("_1741", builder.concat([env.get("_1740"), env.get("_1739")], 3, {"label":"/model/layers.9/attn/q_rotary/RotaryEmbedding_concat_partial_input0_1396"}));
  env.set("InsertedPrecisionFreeCast__1741", builder.cast(env.get("_1741"), "float32"));
  env.set("InsertedPrecisionFreeCast__1742", builder["mul"](env.get("InsertedPrecisionFreeCast__1741"), env.get("InsertedPrecisionFreeCast__1729"), {"label":"/model/layers.9/attn/q_rotary/RotaryEmbedding_mul_sin_1397"}));
  env.set("InsertedPrecisionFreeCast__1744", builder["mul"](env.get("InsertedPrecisionFreeCast__1742"), env.get("InsertedPrecisionFreeCast__1743"), {"label":"/model/layers.9/attn/q_rotary/RotaryEmbedding_mul_sign_1398"}));
  env.set("InsertedPrecisionFreeCast__1745", builder.reshape(env.get("InsertedPrecisionFreeCast__1744"), [1,128,32,64]));
  env.set("InsertedPrecisionFreeCast__1750", builder["add"](env.get("InsertedPrecisionFreeCast__1749"), env.get("InsertedPrecisionFreeCast__1745"), {"label":"/model/layers.9/attn/q_rotary/RotaryEmbedding_add_mul_cos_sin_1405"}));
  env.set("_1750", builder.cast(env.get("InsertedPrecisionFreeCast__1750"), "float16"));
  env.set("_1751", builder.reshape(env.get("_1750"), [1,128,2048]));
  env.set("_1752", builder.reshape(env.get("_1751"), [1,128,32,64]));
  env.set("_1753", builder["transpose"](env.get("_1752"), {"label":"/model/layers.9/attn/GroupQueryAttention_/GQA/query/transpose_1408","permutation":[0,2,1,3]}));
  env.set("InsertedPrecisionFreeCast__1753", builder.cast(env.get("_1753"), "float32"));
  env.set("InsertedPrecisionFreeCast__1754", builder["matmul"](env.get("InsertedPrecisionFreeCast__1753"), env.get("InsertedPrecisionFreeCast__1727"), {"label":"/model/layers.9/attn/GroupQueryAttention_/Attention/qkv/matmul_1_1409"}));
  env.set("InsertedPrecisionFreeCast__1755", builder["mul"](env.get("InsertedPrecisionFreeCast__1754"), env.get("InsertedPrecisionFreeCast__684"), {"label":"/model/layers.9/attn/GroupQueryAttention_/Attention/qkv/div_1410"}));
  env.set("_1694", builder.expand(env.get("_617"), [1,32,128,512], {"label":"/model/layers.9/attn/GroupQueryAttention_/GQA/GQA_mask_shape_ones/expand_1350"}));
  env.set("_1695", builder.cumulativeSum(env.get("_1694"), 3, {"exclusive":true,"label":"/model/layers.9/attn/GroupQueryAttention_range_of_mask_shape_1351"}));
  env.set("_1691", builder["add"](env.get("_613"), env.get("_1683"), {"label":"/model/layers.9/attn/GroupQueryAttention_/GQA/attn_mask/add_1347"}));
  env.set("_1692", builder.expand(env.get("_1691"), [512,128], {"label":"/model/layers.9/attn/GroupQueryAttention_/GQA/expand_neq_right_1348"}));
  env.set("_1693", builder["transpose"](env.get("_1692"), {"label":"/model/layers.9/attn/GroupQueryAttention_/GQA/neq_right/transpose_1349","permutation":[1,0]}));
  env.set("Inserted_649", builder["lesser"](env.get("_1695"), env.get("_1693"), {"label":"/model/layers.9/attn/GroupQueryAttention_/GQA/attn_mask/condition_1_1352"}));
  env.set("InsertedPrecisionFreeCast__1697", builder["where"](env.get("Inserted_649"), env.get("InsertedPrecisionFreeCast__621"), env.get("InsertedPrecisionFreeCast__622"), {"label":"/model/layers.9/attn/GroupQueryAttention_/GQA/attn_mask/where_1354"}));
  env.set("InsertedPrecisionFreeCast__1756", builder["add"](env.get("InsertedPrecisionFreeCast__1755"), env.get("InsertedPrecisionFreeCast__1697"), {"label":"/model/layers.9/attn/GroupQueryAttention_/Attention/attn_mask/softmax_input_1411"}));
  env.set("InsertedPrecisionFreeCast__inlfunc_Softmax_token_8_X_ReduceMax", builder["reduceMax"](env.get("InsertedPrecisionFreeCast__1756"), {"axes":[3],"keepDimensions":true}));
  env.set("InsertedPrecisionFreeCast__inlfunc_Softmax_token_8_X_Sub", builder["sub"](env.get("InsertedPrecisionFreeCast__1756"), env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_8_X_ReduceMax"), {}));
  env.set("InsertedPrecisionFreeCast__inlfunc_Softmax_token_8_X_Exp", builder["exp"](env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_8_X_Sub"), {}));
  env.set("InsertedPrecisionFreeCast__inlfunc_Softmax_token_8_X_ReduceSum", builder["reduceSum"](env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_8_X_Exp"), {"axes":[3],"keepDimensions":true}));
  env.set("InsertedPrecisionFreeCast__1757", builder["div"](env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_8_X_Exp"), env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_8_X_ReduceSum"), {}));
  env.set("InsertedPrecisionFreeCast__1758", builder["matmul"](env.get("InsertedPrecisionFreeCast__1757"), env.get("InsertedPrecisionFreeCast__1690"), {"label":"/model/layers.9/attn/GroupQueryAttention_/Attention/qkv/matmul_2_1413"}));
  env.set("_1758", builder.cast(env.get("InsertedPrecisionFreeCast__1758"), "float16"));
  env.set("_1759", builder["transpose"](env.get("_1758"), {"label":"/model/layers.9/attn/GroupQueryAttention_/Attention/qkv/transpose_1414","permutation":[0,2,1,3]}));
  env.set("_1760", builder.reshape(env.get("_1759"), [1,128,2048]));
  env.set("InsertedPrecisionFreeCast__1760", builder.cast(env.get("_1760"), "float32"));
  env.set("InsertedPrecisionFreeCast__1761", builder["matmul"](env.get("InsertedPrecisionFreeCast__1760"), env.get("InsertedPrecisionFreeCast__356"), {"label":"/model/layers.9/attn/o_proj/MatMul_Q4_matmul_1416"}));
  env.set("_1765", builder["add"](env.get("_1764"), env.get("InsertedPrecisionFreeCast__1761"), {"label":"/model/layers.9/post_attention_layernorm/SkipLayerNorm_add_skip_1420"}));
  env.set("_1788", builder.cast(env.get("_1765"), "float16"));
  env.set("_1789", builder.cast(env.get("_1788"), "float32"));
  env.set("_342", builder.dequantizeLinear(env.get("_339"), env.get("_340"), env.get("_341"), {"axis":2,"blockSize":32,"label":"/model/layers.9/mlp/down_proj/MatMul_Q4_dequantizeLinear_147"}));
  env.set("_343", builder.reshape(env.get("_342"), [2048,5632]));
  env.set("_344", builder["transpose"](env.get("_343"), {"label":"/model/layers.9/mlp/down_proj/MatMul_Q4_transpose_dequantizeLinear_149","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__344", builder.cast(env.get("_344"), "float32"));
  env.set("_1779", builder.dequantizeLinear(env.get("_1776"), env.get("_1777"), env.get("_1778"), {"axis":2,"blockSize":32,"label":"/model/layers.9/mlp/gate_proj/MatMul_Q4_dequantizeLinear_1430"}));
  env.set("_1780", builder.reshape(env.get("_1779"), [5632,2048]));
  env.set("_1781", builder["transpose"](env.get("_1780"), {"label":"/model/layers.9/mlp/gate_proj/MatMul_Q4_transpose_dequantizeLinear_1432","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__1781", builder.cast(env.get("_1781"), "float32"));
  env.set("_1772", builder.cast(env.get("_1771"), "float32"));
  env.set("_1766", builder["pow"](env.get("_1765"), env.get("_583"), {"label":"/model/layers.9/post_attention_layernorm/SkipLayerNorm_pow_1421"}));
  env.set("_1767", builder["reduceMean"](env.get("_1766"), {"axes":[2],"keepDimensions":true,"label":"/model/layers.9/post_attention_layernorm/SkipLayerNorm_reduceMean_1422"}));
  env.set("_1768", builder["add"](env.get("_1767"), env.get("_586"), {"label":"/model/layers.9/post_attention_layernorm/SkipLayerNorm_add_1423"}));
  env.set("_1769", builder["sqrt"](env.get("_1768"), {"label":"/model/layers.9/post_attention_layernorm/SkipLayerNorm_sqrt_1424"}));
  env.set("_1770", builder["div"](env.get("_1765"), env.get("_1769"), {"label":"/model/layers.9/post_attention_layernorm/SkipLayerNorm_div_1425"}));
  env.set("_1773", builder["mul"](env.get("_1772"), env.get("_1770"), {"label":"/model/layers.9/post_attention_layernorm/SkipLayerNorm_mul_1427"}));
  env.set("InsertedPrecisionFreeCast__1782", builder["matmul"](env.get("_1773"), env.get("InsertedPrecisionFreeCast__1781"), {"label":"/model/layers.9/mlp/gate_proj/MatMul_Q4_matmul_1433"}));
  env.set("InsertedPrecisionFreeCast__1783", builder["sigmoid"](env.get("InsertedPrecisionFreeCast__1782"), {"label":"/model/layers.9/mlp/act_fn/Sigmoid_1434"}));
  env.set("InsertedPrecisionFreeCast__1784", builder["mul"](env.get("InsertedPrecisionFreeCast__1782"), env.get("InsertedPrecisionFreeCast__1783"), {"label":"/model/layers.9/mlp/act_fn/Mul_1435"}));
  env.set("_348", builder.dequantizeLinear(env.get("_345"), env.get("_346"), env.get("_347"), {"axis":2,"blockSize":32,"label":"/model/layers.9/mlp/up_proj/MatMul_Q4_dequantizeLinear_150"}));
  env.set("_349", builder.reshape(env.get("_348"), [5632,2048]));
  env.set("_350", builder["transpose"](env.get("_349"), {"label":"/model/layers.9/mlp/up_proj/MatMul_Q4_transpose_dequantizeLinear_152","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__350", builder.cast(env.get("_350"), "float32"));
  env.set("InsertedPrecisionFreeCast__1775", builder["matmul"](env.get("_1773"), env.get("InsertedPrecisionFreeCast__350"), {"label":"/model/layers.9/mlp/up_proj/MatMul_Q4_matmul_1429"}));
  env.set("InsertedPrecisionFreeCast__1785", builder["mul"](env.get("InsertedPrecisionFreeCast__1784"), env.get("InsertedPrecisionFreeCast__1775"), {"label":"/model/layers.9/mlp/Mul_1436"}));
  env.set("InsertedPrecisionFreeCast__1786", builder["matmul"](env.get("InsertedPrecisionFreeCast__1785"), env.get("InsertedPrecisionFreeCast__344"), {"label":"/model/layers.9/mlp/down_proj/MatMul_Q4_matmul_1437"}));
  env.set("_1790", builder["add"](env.get("_1789"), env.get("InsertedPrecisionFreeCast__1786"), {"label":"/model/layers.10/input_layernorm/SkipLayerNorm_add_skip_1441"}));
  env.set("_1882", builder.cast(env.get("_1790"), "float16"));
  env.set("_1883", builder.cast(env.get("_1882"), "float32"));
  env.set("_330", builder.dequantizeLinear(env.get("_327"), env.get("_328"), env.get("_329"), {"axis":2,"blockSize":32,"label":"/model/layers.10/attn/o_proj/MatMul_Q4_dequantizeLinear_141"}));
  env.set("_331", builder.reshape(env.get("_330"), [2048,2048]));
  env.set("_332", builder["transpose"](env.get("_331"), {"label":"/model/layers.10/attn/o_proj/MatMul_Q4_transpose_dequantizeLinear_143","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__332", builder.cast(env.get("_332"), "float32"));
  env.set("Inserted_696", builder.cast(env.get("_601"), "uint8"));
  env.set("_1802", builder["where"](env.get("Inserted_696"), env.get("_602"), env.get("_600"), {"label":"/model/layers.10/attn/GroupQueryAttention_/GQA/scatter/where_1452"}));
  env.set("_1803", builder["add"](env.get("_604"), env.get("_1802"), {"label":"/model/layers.10/attn/GroupQueryAttention_/GQA/right_constant/add_1454"}));
  env.set("_1804", builder.concat([env.get("_606"), env.get("_1803")], 1, {"label":"/model/layers.10/attn/GroupQueryAttention_/GQA/concat_for_pre_scatter_indices_1455"}));
  env.set("_1805", builder.reshape(env.get("_1804"), [1,128,4,3]));
  env.set("Inserted_698", builder.cast(env.get("_1805"), "int64"));
  env.set("Inserted_700", builder["max"](env.get("Inserted_698"), env.get("Inserted_699"), {"label":"Inserted_Max_1459"}));
  env.set("Inserted_702", builder["min"](env.get("Inserted_700"), env.get("Inserted_701"), {"label":"Inserted_Min_1460"}));
  env.set("_336", builder.dequantizeLinear(env.get("_333"), env.get("_334"), env.get("_335"), {"axis":2,"blockSize":32,"label":"/model/layers.10/attn/v_proj/MatMul_Q4_dequantizeLinear_144"}));
  env.set("_337", builder.reshape(env.get("_336"), [256,2048]));
  env.set("_338", builder["transpose"](env.get("_337"), {"label":"/model/layers.10/attn/v_proj/MatMul_Q4_transpose_dequantizeLinear_146","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__338", builder.cast(env.get("_338"), "float32"));
  env.set("_1797", builder.cast(env.get("_1796"), "float32"));
  env.set("_1791", builder["pow"](env.get("_1790"), env.get("_583"), {"label":"/model/layers.10/input_layernorm/SkipLayerNorm_pow_1442"}));
  env.set("_1792", builder["reduceMean"](env.get("_1791"), {"axes":[2],"keepDimensions":true,"label":"/model/layers.10/input_layernorm/SkipLayerNorm_reduceMean_1443"}));
  env.set("_1793", builder["add"](env.get("_1792"), env.get("_586"), {"label":"/model/layers.10/input_layernorm/SkipLayerNorm_add_1444"}));
  env.set("_1794", builder["sqrt"](env.get("_1793"), {"label":"/model/layers.10/input_layernorm/SkipLayerNorm_sqrt_1445"}));
  env.set("_1795", builder["div"](env.get("_1790"), env.get("_1794"), {"label":"/model/layers.10/input_layernorm/SkipLayerNorm_div_1446"}));
  env.set("_1798", builder["mul"](env.get("_1797"), env.get("_1795"), {"label":"/model/layers.10/input_layernorm/SkipLayerNorm_mul_1448"}));
  env.set("InsertedPrecisionFreeCast__1800", builder["matmul"](env.get("_1798"), env.get("InsertedPrecisionFreeCast__338"), {"label":"/model/layers.10/attn/v_proj/MatMul_Q4_matmul_1450"}));
  env.set("_1800", builder.cast(env.get("InsertedPrecisionFreeCast__1800"), "float16"));
  env.set("_1801", builder.reshape(env.get("_1800"), [1,128,4,64]));
  env.set("present_10_value_21", builder["scatterND"](env.get("past_key_values_10_value_1806"), env.get("Inserted_702"), env.get("_1801"), {"label":"/model/layers.10/attn/GroupQueryAttention_/GQA/present_value/ScatterND_1457"}));
  env.set("_1807", builder.reshape(env.get("present_10_value_21"), [1,4,1,512,64]));
  env.set("_1808", builder.expand(env.get("_1807"), [1,4,8,512,64], {"label":"/model/layers.10/attn/GroupQueryAttention_/GQA/true_present_value/expand_1462"}));
  env.set("_1809", builder.reshape(env.get("_1808"), [1,32,512,64]));
  env.set("InsertedPrecisionFreeCast__1809", builder.cast(env.get("_1809"), "float32"));
  env.set("Inserted_727", builder.cast(env.get("_1805"), "int64"));
  env.set("Inserted_729", builder["max"](env.get("Inserted_727"), env.get("Inserted_728"), {"label":"Inserted_Max_1497"}));
  env.set("Inserted_731", builder["min"](env.get("Inserted_729"), env.get("Inserted_730"), {"label":"Inserted_Min_1498"}));
  env.set("Inserted_720", builder["clamp"](env.get("position_ids_625"), {"label":"Inserted_Clip_1488","maxValue":2047,"minValue":-2048}));
  env.set("_1835", builder["gather"](env.get("_644"), env.get("Inserted_720"), {"axis":0,"label":"/model/layers.10/attn/k_rotary/RotaryEmbedding_gather_cos_1487"}));
  env.set("_1836", builder.reshape(env.get("_1835"), [1,128,1,1,32]));
  env.set("InsertedPrecisionFreeCast__1836", builder.cast(env.get("_1836"), "float32"));
  env.set("_1822", builder.dequantizeLinear(env.get("_1819"), env.get("_1820"), env.get("_1821"), {"axis":2,"blockSize":32,"label":"/model/layers.10/attn/k_proj/MatMul_Q4_dequantizeLinear_1476"}));
  env.set("_1823", builder.reshape(env.get("_1822"), [256,2048]));
  env.set("_1824", builder["transpose"](env.get("_1823"), {"label":"/model/layers.10/attn/k_proj/MatMul_Q4_transpose_dequantizeLinear_1478","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__1824", builder.cast(env.get("_1824"), "float32"));
  env.set("InsertedPrecisionFreeCast__1825", builder["matmul"](env.get("_1798"), env.get("InsertedPrecisionFreeCast__1824"), {"label":"/model/layers.10/attn/k_proj/MatMul_Q4_matmul_1479"}));
  env.set("_1825", builder.cast(env.get("InsertedPrecisionFreeCast__1825"), "float16"));
  env.set("_1826", builder.reshape(env.get("_1825"), [1,128,4,64]));
  env.set("_1827", builder.reshape(env.get("_1826"), [1,128,4,2,32]));
  env.set("InsertedPrecisionFreeCast__1827", builder.cast(env.get("_1827"), "float32"));
  env.set("InsertedPrecisionFreeCast__1837", builder["mul"](env.get("InsertedPrecisionFreeCast__1827"), env.get("InsertedPrecisionFreeCast__1836"), {"label":"/model/layers.10/attn/k_rotary/RotaryEmbedding_mul_cos_1490"}));
  env.set("InsertedPrecisionFreeCast__1838", builder.reshape(env.get("InsertedPrecisionFreeCast__1837"), [1,128,4,64]));
  env.set("InsertedPrecisionFreeCast__1832", builder.cast(env.get("_1832"), "float32"));
  env.set("Inserted_711", builder["clamp"](env.get("position_ids_625"), {"label":"Inserted_Clip_1474","maxValue":2047,"minValue":-2048}));
  env.set("_1817", builder["gather"](env.get("_624"), env.get("Inserted_711"), {"axis":0,"label":"/model/layers.10/attn/k_rotary/RotaryEmbedding_gather_sin_1473"}));
  env.set("_1818", builder.reshape(env.get("_1817"), [1,128,1,1,32]));
  env.set("InsertedPrecisionFreeCast__1818", builder.cast(env.get("_1818"), "float32"));
  {
    const tmp = builder.split(env.get("_1827"), 2, {"axis":3,"label":"/model/layers.10/attn/k_rotary/RotaryEmbedding_split_partial_input0_1482"});
    env.set("_1828", tmp[0]);
    env.set("_1829", tmp[1]);
  }
  env.set("_1830", builder.concat([env.get("_1829"), env.get("_1828")], 3, {"label":"/model/layers.10/attn/k_rotary/RotaryEmbedding_concat_partial_input0_1483"}));
  env.set("InsertedPrecisionFreeCast__1830", builder.cast(env.get("_1830"), "float32"));
  env.set("InsertedPrecisionFreeCast__1831", builder["mul"](env.get("InsertedPrecisionFreeCast__1830"), env.get("InsertedPrecisionFreeCast__1818"), {"label":"/model/layers.10/attn/k_rotary/RotaryEmbedding_mul_sin_1484"}));
  env.set("InsertedPrecisionFreeCast__1833", builder["mul"](env.get("InsertedPrecisionFreeCast__1831"), env.get("InsertedPrecisionFreeCast__1832"), {"label":"/model/layers.10/attn/k_rotary/RotaryEmbedding_mul_sign_1485"}));
  env.set("InsertedPrecisionFreeCast__1834", builder.reshape(env.get("InsertedPrecisionFreeCast__1833"), [1,128,4,64]));
  env.set("InsertedPrecisionFreeCast__1839", builder["add"](env.get("InsertedPrecisionFreeCast__1838"), env.get("InsertedPrecisionFreeCast__1834"), {"label":"/model/layers.10/attn/k_rotary/RotaryEmbedding_add_mul_cos_sin_1492"}));
  env.set("_1839", builder.cast(env.get("InsertedPrecisionFreeCast__1839"), "float16"));
  env.set("_1840", builder.reshape(env.get("_1839"), [1,128,256]));
  env.set("_1841", builder.reshape(env.get("_1840"), [1,128,4,64]));
  env.set("present_10_key_20", builder["scatterND"](env.get("past_key_values_10_key_1842"), env.get("Inserted_731"), env.get("_1841"), {"label":"/model/layers.10/attn/GroupQueryAttention_/GQA/present_key/ScatterND_1495"}));
  env.set("_1843", builder.reshape(env.get("present_10_key_20"), [1,4,1,512,64]));
  env.set("_1844", builder.expand(env.get("_1843"), [1,4,8,512,64], {"label":"/model/layers.10/attn/GroupQueryAttention_/GQA/true_present_key/expand_1500"}));
  env.set("_1845", builder.reshape(env.get("_1844"), [1,32,512,64]));
  env.set("_1846", builder["transpose"](env.get("_1845"), {"label":"/model/layers.10/attn/GroupQueryAttention_/GQA/present_key/transpose_1502","permutation":[0,1,3,2]}));
  env.set("InsertedPrecisionFreeCast__1846", builder.cast(env.get("_1846"), "float32"));
  env.set("Inserted_744", builder["clamp"](env.get("position_ids_625"), {"label":"Inserted_Clip_1518","maxValue":2047,"minValue":-2048}));
  env.set("_1865", builder["gather"](env.get("_644"), env.get("Inserted_744"), {"axis":0,"label":"/model/layers.10/attn/q_rotary/RotaryEmbedding_gather_cos_1517"}));
  env.set("_1866", builder.reshape(env.get("_1865"), [1,128,1,1,32]));
  env.set("InsertedPrecisionFreeCast__1866", builder.cast(env.get("_1866"), "float32"));
  env.set("_1852", builder.dequantizeLinear(env.get("_1849"), env.get("_1850"), env.get("_1851"), {"axis":2,"blockSize":32,"label":"/model/layers.10/attn/q_proj/MatMul_Q4_dequantizeLinear_1506"}));
  env.set("_1853", builder.reshape(env.get("_1852"), [2048,2048]));
  env.set("_1854", builder["transpose"](env.get("_1853"), {"label":"/model/layers.10/attn/q_proj/MatMul_Q4_transpose_dequantizeLinear_1508","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__1854", builder.cast(env.get("_1854"), "float32"));
  env.set("InsertedPrecisionFreeCast__1855", builder["matmul"](env.get("_1798"), env.get("InsertedPrecisionFreeCast__1854"), {"label":"/model/layers.10/attn/q_proj/MatMul_Q4_matmul_1509"}));
  env.set("_1855", builder.cast(env.get("InsertedPrecisionFreeCast__1855"), "float16"));
  env.set("_1856", builder.reshape(env.get("_1855"), [1,128,32,64]));
  env.set("_1857", builder.reshape(env.get("_1856"), [1,128,32,2,32]));
  env.set("InsertedPrecisionFreeCast__1857", builder.cast(env.get("_1857"), "float32"));
  env.set("InsertedPrecisionFreeCast__1867", builder["mul"](env.get("InsertedPrecisionFreeCast__1857"), env.get("InsertedPrecisionFreeCast__1866"), {"label":"/model/layers.10/attn/q_rotary/RotaryEmbedding_mul_cos_1520"}));
  env.set("InsertedPrecisionFreeCast__1868", builder.reshape(env.get("InsertedPrecisionFreeCast__1867"), [1,128,32,64]));
  env.set("InsertedPrecisionFreeCast__1862", builder.cast(env.get("_1862"), "float32"));
  env.set("Inserted_735", builder["clamp"](env.get("position_ids_625"), {"label":"Inserted_Clip_1504","maxValue":2047,"minValue":-2048}));
  env.set("_1847", builder["gather"](env.get("_624"), env.get("Inserted_735"), {"axis":0,"label":"/model/layers.10/attn/q_rotary/RotaryEmbedding_gather_sin_1503"}));
  env.set("_1848", builder.reshape(env.get("_1847"), [1,128,1,1,32]));
  env.set("InsertedPrecisionFreeCast__1848", builder.cast(env.get("_1848"), "float32"));
  {
    const tmp = builder.split(env.get("_1857"), 2, {"axis":3,"label":"/model/layers.10/attn/q_rotary/RotaryEmbedding_split_partial_input0_1512"});
    env.set("_1858", tmp[0]);
    env.set("_1859", tmp[1]);
  }
  env.set("_1860", builder.concat([env.get("_1859"), env.get("_1858")], 3, {"label":"/model/layers.10/attn/q_rotary/RotaryEmbedding_concat_partial_input0_1513"}));
  env.set("InsertedPrecisionFreeCast__1860", builder.cast(env.get("_1860"), "float32"));
  env.set("InsertedPrecisionFreeCast__1861", builder["mul"](env.get("InsertedPrecisionFreeCast__1860"), env.get("InsertedPrecisionFreeCast__1848"), {"label":"/model/layers.10/attn/q_rotary/RotaryEmbedding_mul_sin_1514"}));
  env.set("InsertedPrecisionFreeCast__1863", builder["mul"](env.get("InsertedPrecisionFreeCast__1861"), env.get("InsertedPrecisionFreeCast__1862"), {"label":"/model/layers.10/attn/q_rotary/RotaryEmbedding_mul_sign_1515"}));
  env.set("InsertedPrecisionFreeCast__1864", builder.reshape(env.get("InsertedPrecisionFreeCast__1863"), [1,128,32,64]));
  env.set("InsertedPrecisionFreeCast__1869", builder["add"](env.get("InsertedPrecisionFreeCast__1868"), env.get("InsertedPrecisionFreeCast__1864"), {"label":"/model/layers.10/attn/q_rotary/RotaryEmbedding_add_mul_cos_sin_1522"}));
  env.set("_1869", builder.cast(env.get("InsertedPrecisionFreeCast__1869"), "float16"));
  env.set("_1870", builder.reshape(env.get("_1869"), [1,128,2048]));
  env.set("_1871", builder.reshape(env.get("_1870"), [1,128,32,64]));
  env.set("_1872", builder["transpose"](env.get("_1871"), {"label":"/model/layers.10/attn/GroupQueryAttention_/GQA/query/transpose_1525","permutation":[0,2,1,3]}));
  env.set("InsertedPrecisionFreeCast__1872", builder.cast(env.get("_1872"), "float32"));
  env.set("InsertedPrecisionFreeCast__1873", builder["matmul"](env.get("InsertedPrecisionFreeCast__1872"), env.get("InsertedPrecisionFreeCast__1846"), {"label":"/model/layers.10/attn/GroupQueryAttention_/Attention/qkv/matmul_1_1526"}));
  env.set("InsertedPrecisionFreeCast__1874", builder["mul"](env.get("InsertedPrecisionFreeCast__1873"), env.get("InsertedPrecisionFreeCast__684"), {"label":"/model/layers.10/attn/GroupQueryAttention_/Attention/qkv/div_1527"}));
  env.set("_1813", builder.expand(env.get("_617"), [1,32,128,512], {"label":"/model/layers.10/attn/GroupQueryAttention_/GQA/GQA_mask_shape_ones/expand_1467"}));
  env.set("_1814", builder.cumulativeSum(env.get("_1813"), 3, {"exclusive":true,"label":"/model/layers.10/attn/GroupQueryAttention_range_of_mask_shape_1468"}));
  env.set("_1810", builder["add"](env.get("_613"), env.get("_1802"), {"label":"/model/layers.10/attn/GroupQueryAttention_/GQA/attn_mask/add_1464"}));
  env.set("_1811", builder.expand(env.get("_1810"), [512,128], {"label":"/model/layers.10/attn/GroupQueryAttention_/GQA/expand_neq_right_1465"}));
  env.set("_1812", builder["transpose"](env.get("_1811"), {"label":"/model/layers.10/attn/GroupQueryAttention_/GQA/neq_right/transpose_1466","permutation":[1,0]}));
  env.set("Inserted_709", builder["lesser"](env.get("_1814"), env.get("_1812"), {"label":"/model/layers.10/attn/GroupQueryAttention_/GQA/attn_mask/condition_1_1469"}));
  env.set("InsertedPrecisionFreeCast__1816", builder["where"](env.get("Inserted_709"), env.get("InsertedPrecisionFreeCast__621"), env.get("InsertedPrecisionFreeCast__622"), {"label":"/model/layers.10/attn/GroupQueryAttention_/GQA/attn_mask/where_1471"}));
  env.set("InsertedPrecisionFreeCast__1875", builder["add"](env.get("InsertedPrecisionFreeCast__1874"), env.get("InsertedPrecisionFreeCast__1816"), {"label":"/model/layers.10/attn/GroupQueryAttention_/Attention/attn_mask/softmax_input_1528"}));
  env.set("InsertedPrecisionFreeCast__inlfunc_Softmax_token_9_X_ReduceMax", builder["reduceMax"](env.get("InsertedPrecisionFreeCast__1875"), {"axes":[3],"keepDimensions":true}));
  env.set("InsertedPrecisionFreeCast__inlfunc_Softmax_token_9_X_Sub", builder["sub"](env.get("InsertedPrecisionFreeCast__1875"), env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_9_X_ReduceMax"), {}));
  env.set("InsertedPrecisionFreeCast__inlfunc_Softmax_token_9_X_Exp", builder["exp"](env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_9_X_Sub"), {}));
  env.set("InsertedPrecisionFreeCast__inlfunc_Softmax_token_9_X_ReduceSum", builder["reduceSum"](env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_9_X_Exp"), {"axes":[3],"keepDimensions":true}));
  env.set("InsertedPrecisionFreeCast__1876", builder["div"](env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_9_X_Exp"), env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_9_X_ReduceSum"), {}));
  env.set("InsertedPrecisionFreeCast__1877", builder["matmul"](env.get("InsertedPrecisionFreeCast__1876"), env.get("InsertedPrecisionFreeCast__1809"), {"label":"/model/layers.10/attn/GroupQueryAttention_/Attention/qkv/matmul_2_1530"}));
  env.set("_1877", builder.cast(env.get("InsertedPrecisionFreeCast__1877"), "float16"));
  env.set("_1878", builder["transpose"](env.get("_1877"), {"label":"/model/layers.10/attn/GroupQueryAttention_/Attention/qkv/transpose_1531","permutation":[0,2,1,3]}));
  env.set("_1879", builder.reshape(env.get("_1878"), [1,128,2048]));
  env.set("InsertedPrecisionFreeCast__1879", builder.cast(env.get("_1879"), "float32"));
  env.set("InsertedPrecisionFreeCast__1880", builder["matmul"](env.get("InsertedPrecisionFreeCast__1879"), env.get("InsertedPrecisionFreeCast__332"), {"label":"/model/layers.10/attn/o_proj/MatMul_Q4_matmul_1533"}));
  env.set("_1884", builder["add"](env.get("_1883"), env.get("InsertedPrecisionFreeCast__1880"), {"label":"/model/layers.10/post_attention_layernorm/SkipLayerNorm_add_skip_1537"}));
  env.set("_1907", builder.cast(env.get("_1884"), "float16"));
  env.set("_1908", builder.cast(env.get("_1907"), "float32"));
  env.set("_318", builder.dequantizeLinear(env.get("_315"), env.get("_316"), env.get("_317"), {"axis":2,"blockSize":32,"label":"/model/layers.10/mlp/down_proj/MatMul_Q4_dequantizeLinear_135"}));
  env.set("_319", builder.reshape(env.get("_318"), [2048,5632]));
  env.set("_320", builder["transpose"](env.get("_319"), {"label":"/model/layers.10/mlp/down_proj/MatMul_Q4_transpose_dequantizeLinear_137","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__320", builder.cast(env.get("_320"), "float32"));
  env.set("_1898", builder.dequantizeLinear(env.get("_1895"), env.get("_1896"), env.get("_1897"), {"axis":2,"blockSize":32,"label":"/model/layers.10/mlp/gate_proj/MatMul_Q4_dequantizeLinear_1547"}));
  env.set("_1899", builder.reshape(env.get("_1898"), [5632,2048]));
  env.set("_1900", builder["transpose"](env.get("_1899"), {"label":"/model/layers.10/mlp/gate_proj/MatMul_Q4_transpose_dequantizeLinear_1549","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__1900", builder.cast(env.get("_1900"), "float32"));
  env.set("_1891", builder.cast(env.get("_1890"), "float32"));
  env.set("_1885", builder["pow"](env.get("_1884"), env.get("_583"), {"label":"/model/layers.10/post_attention_layernorm/SkipLayerNorm_pow_1538"}));
  env.set("_1886", builder["reduceMean"](env.get("_1885"), {"axes":[2],"keepDimensions":true,"label":"/model/layers.10/post_attention_layernorm/SkipLayerNorm_reduceMean_1539"}));
  env.set("_1887", builder["add"](env.get("_1886"), env.get("_586"), {"label":"/model/layers.10/post_attention_layernorm/SkipLayerNorm_add_1540"}));
  env.set("_1888", builder["sqrt"](env.get("_1887"), {"label":"/model/layers.10/post_attention_layernorm/SkipLayerNorm_sqrt_1541"}));
  env.set("_1889", builder["div"](env.get("_1884"), env.get("_1888"), {"label":"/model/layers.10/post_attention_layernorm/SkipLayerNorm_div_1542"}));
  env.set("_1892", builder["mul"](env.get("_1891"), env.get("_1889"), {"label":"/model/layers.10/post_attention_layernorm/SkipLayerNorm_mul_1544"}));
  env.set("InsertedPrecisionFreeCast__1901", builder["matmul"](env.get("_1892"), env.get("InsertedPrecisionFreeCast__1900"), {"label":"/model/layers.10/mlp/gate_proj/MatMul_Q4_matmul_1550"}));
  env.set("InsertedPrecisionFreeCast__1902", builder["sigmoid"](env.get("InsertedPrecisionFreeCast__1901"), {"label":"/model/layers.10/mlp/act_fn/Sigmoid_1551"}));
  env.set("InsertedPrecisionFreeCast__1903", builder["mul"](env.get("InsertedPrecisionFreeCast__1901"), env.get("InsertedPrecisionFreeCast__1902"), {"label":"/model/layers.10/mlp/act_fn/Mul_1552"}));
  env.set("_324", builder.dequantizeLinear(env.get("_321"), env.get("_322"), env.get("_323"), {"axis":2,"blockSize":32,"label":"/model/layers.10/mlp/up_proj/MatMul_Q4_dequantizeLinear_138"}));
  env.set("_325", builder.reshape(env.get("_324"), [5632,2048]));
  env.set("_326", builder["transpose"](env.get("_325"), {"label":"/model/layers.10/mlp/up_proj/MatMul_Q4_transpose_dequantizeLinear_140","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__326", builder.cast(env.get("_326"), "float32"));
  env.set("InsertedPrecisionFreeCast__1894", builder["matmul"](env.get("_1892"), env.get("InsertedPrecisionFreeCast__326"), {"label":"/model/layers.10/mlp/up_proj/MatMul_Q4_matmul_1546"}));
  env.set("InsertedPrecisionFreeCast__1904", builder["mul"](env.get("InsertedPrecisionFreeCast__1903"), env.get("InsertedPrecisionFreeCast__1894"), {"label":"/model/layers.10/mlp/Mul_1553"}));
  env.set("InsertedPrecisionFreeCast__1905", builder["matmul"](env.get("InsertedPrecisionFreeCast__1904"), env.get("InsertedPrecisionFreeCast__320"), {"label":"/model/layers.10/mlp/down_proj/MatMul_Q4_matmul_1554"}));
  env.set("_1909", builder["add"](env.get("_1908"), env.get("InsertedPrecisionFreeCast__1905"), {"label":"/model/layers.11/input_layernorm/SkipLayerNorm_add_skip_1558"}));
  env.set("_2001", builder.cast(env.get("_1909"), "float16"));
  env.set("_2002", builder.cast(env.get("_2001"), "float32"));
  env.set("_306", builder.dequantizeLinear(env.get("_303"), env.get("_304"), env.get("_305"), {"axis":2,"blockSize":32,"label":"/model/layers.11/attn/o_proj/MatMul_Q4_dequantizeLinear_129"}));
  env.set("_307", builder.reshape(env.get("_306"), [2048,2048]));
  env.set("_308", builder["transpose"](env.get("_307"), {"label":"/model/layers.11/attn/o_proj/MatMul_Q4_transpose_dequantizeLinear_131","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__308", builder.cast(env.get("_308"), "float32"));
  env.set("Inserted_756", builder.cast(env.get("_601"), "uint8"));
  env.set("_1921", builder["where"](env.get("Inserted_756"), env.get("_602"), env.get("_600"), {"label":"/model/layers.11/attn/GroupQueryAttention_/GQA/scatter/where_1569"}));
  env.set("_1922", builder["add"](env.get("_604"), env.get("_1921"), {"label":"/model/layers.11/attn/GroupQueryAttention_/GQA/right_constant/add_1571"}));
  env.set("_1923", builder.concat([env.get("_606"), env.get("_1922")], 1, {"label":"/model/layers.11/attn/GroupQueryAttention_/GQA/concat_for_pre_scatter_indices_1572"}));
  env.set("_1924", builder.reshape(env.get("_1923"), [1,128,4,3]));
  env.set("Inserted_758", builder.cast(env.get("_1924"), "int64"));
  env.set("Inserted_760", builder["max"](env.get("Inserted_758"), env.get("Inserted_759"), {"label":"Inserted_Max_1576"}));
  env.set("Inserted_762", builder["min"](env.get("Inserted_760"), env.get("Inserted_761"), {"label":"Inserted_Min_1577"}));
  env.set("_312", builder.dequantizeLinear(env.get("_309"), env.get("_310"), env.get("_311"), {"axis":2,"blockSize":32,"label":"/model/layers.11/attn/v_proj/MatMul_Q4_dequantizeLinear_132"}));
  env.set("_313", builder.reshape(env.get("_312"), [256,2048]));
  env.set("_314", builder["transpose"](env.get("_313"), {"label":"/model/layers.11/attn/v_proj/MatMul_Q4_transpose_dequantizeLinear_134","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__314", builder.cast(env.get("_314"), "float32"));
  env.set("_1916", builder.cast(env.get("_1915"), "float32"));
  env.set("_1910", builder["pow"](env.get("_1909"), env.get("_583"), {"label":"/model/layers.11/input_layernorm/SkipLayerNorm_pow_1559"}));
  env.set("_1911", builder["reduceMean"](env.get("_1910"), {"axes":[2],"keepDimensions":true,"label":"/model/layers.11/input_layernorm/SkipLayerNorm_reduceMean_1560"}));
  env.set("_1912", builder["add"](env.get("_1911"), env.get("_586"), {"label":"/model/layers.11/input_layernorm/SkipLayerNorm_add_1561"}));
  env.set("_1913", builder["sqrt"](env.get("_1912"), {"label":"/model/layers.11/input_layernorm/SkipLayerNorm_sqrt_1562"}));
  env.set("_1914", builder["div"](env.get("_1909"), env.get("_1913"), {"label":"/model/layers.11/input_layernorm/SkipLayerNorm_div_1563"}));
  env.set("_1917", builder["mul"](env.get("_1916"), env.get("_1914"), {"label":"/model/layers.11/input_layernorm/SkipLayerNorm_mul_1565"}));
  env.set("InsertedPrecisionFreeCast__1919", builder["matmul"](env.get("_1917"), env.get("InsertedPrecisionFreeCast__314"), {"label":"/model/layers.11/attn/v_proj/MatMul_Q4_matmul_1567"}));
  env.set("_1919", builder.cast(env.get("InsertedPrecisionFreeCast__1919"), "float16"));
  env.set("_1920", builder.reshape(env.get("_1919"), [1,128,4,64]));
  env.set("present_11_value_23", builder["scatterND"](env.get("past_key_values_11_value_1925"), env.get("Inserted_762"), env.get("_1920"), {"label":"/model/layers.11/attn/GroupQueryAttention_/GQA/present_value/ScatterND_1574"}));
  env.set("_1926", builder.reshape(env.get("present_11_value_23"), [1,4,1,512,64]));
  env.set("_1927", builder.expand(env.get("_1926"), [1,4,8,512,64], {"label":"/model/layers.11/attn/GroupQueryAttention_/GQA/true_present_value/expand_1579"}));
  env.set("_1928", builder.reshape(env.get("_1927"), [1,32,512,64]));
  env.set("InsertedPrecisionFreeCast__1928", builder.cast(env.get("_1928"), "float32"));
  env.set("Inserted_787", builder.cast(env.get("_1924"), "int64"));
  env.set("Inserted_789", builder["max"](env.get("Inserted_787"), env.get("Inserted_788"), {"label":"Inserted_Max_1614"}));
  env.set("Inserted_791", builder["min"](env.get("Inserted_789"), env.get("Inserted_790"), {"label":"Inserted_Min_1615"}));
  env.set("Inserted_780", builder["clamp"](env.get("position_ids_625"), {"label":"Inserted_Clip_1605","maxValue":2047,"minValue":-2048}));
  env.set("_1954", builder["gather"](env.get("_644"), env.get("Inserted_780"), {"axis":0,"label":"/model/layers.11/attn/k_rotary/RotaryEmbedding_gather_cos_1604"}));
  env.set("_1955", builder.reshape(env.get("_1954"), [1,128,1,1,32]));
  env.set("InsertedPrecisionFreeCast__1955", builder.cast(env.get("_1955"), "float32"));
  env.set("_1941", builder.dequantizeLinear(env.get("_1938"), env.get("_1939"), env.get("_1940"), {"axis":2,"blockSize":32,"label":"/model/layers.11/attn/k_proj/MatMul_Q4_dequantizeLinear_1593"}));
  env.set("_1942", builder.reshape(env.get("_1941"), [256,2048]));
  env.set("_1943", builder["transpose"](env.get("_1942"), {"label":"/model/layers.11/attn/k_proj/MatMul_Q4_transpose_dequantizeLinear_1595","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__1943", builder.cast(env.get("_1943"), "float32"));
  env.set("InsertedPrecisionFreeCast__1944", builder["matmul"](env.get("_1917"), env.get("InsertedPrecisionFreeCast__1943"), {"label":"/model/layers.11/attn/k_proj/MatMul_Q4_matmul_1596"}));
  env.set("_1944", builder.cast(env.get("InsertedPrecisionFreeCast__1944"), "float16"));
  env.set("_1945", builder.reshape(env.get("_1944"), [1,128,4,64]));
  env.set("_1946", builder.reshape(env.get("_1945"), [1,128,4,2,32]));
  env.set("InsertedPrecisionFreeCast__1946", builder.cast(env.get("_1946"), "float32"));
  env.set("InsertedPrecisionFreeCast__1956", builder["mul"](env.get("InsertedPrecisionFreeCast__1946"), env.get("InsertedPrecisionFreeCast__1955"), {"label":"/model/layers.11/attn/k_rotary/RotaryEmbedding_mul_cos_1607"}));
  env.set("InsertedPrecisionFreeCast__1957", builder.reshape(env.get("InsertedPrecisionFreeCast__1956"), [1,128,4,64]));
  env.set("InsertedPrecisionFreeCast__1951", builder.cast(env.get("_1951"), "float32"));
  env.set("Inserted_771", builder["clamp"](env.get("position_ids_625"), {"label":"Inserted_Clip_1591","maxValue":2047,"minValue":-2048}));
  env.set("_1936", builder["gather"](env.get("_624"), env.get("Inserted_771"), {"axis":0,"label":"/model/layers.11/attn/k_rotary/RotaryEmbedding_gather_sin_1590"}));
  env.set("_1937", builder.reshape(env.get("_1936"), [1,128,1,1,32]));
  env.set("InsertedPrecisionFreeCast__1937", builder.cast(env.get("_1937"), "float32"));
  {
    const tmp = builder.split(env.get("_1946"), 2, {"axis":3,"label":"/model/layers.11/attn/k_rotary/RotaryEmbedding_split_partial_input0_1599"});
    env.set("_1947", tmp[0]);
    env.set("_1948", tmp[1]);
  }
  env.set("_1949", builder.concat([env.get("_1948"), env.get("_1947")], 3, {"label":"/model/layers.11/attn/k_rotary/RotaryEmbedding_concat_partial_input0_1600"}));
  env.set("InsertedPrecisionFreeCast__1949", builder.cast(env.get("_1949"), "float32"));
  env.set("InsertedPrecisionFreeCast__1950", builder["mul"](env.get("InsertedPrecisionFreeCast__1949"), env.get("InsertedPrecisionFreeCast__1937"), {"label":"/model/layers.11/attn/k_rotary/RotaryEmbedding_mul_sin_1601"}));
  env.set("InsertedPrecisionFreeCast__1952", builder["mul"](env.get("InsertedPrecisionFreeCast__1950"), env.get("InsertedPrecisionFreeCast__1951"), {"label":"/model/layers.11/attn/k_rotary/RotaryEmbedding_mul_sign_1602"}));
  env.set("InsertedPrecisionFreeCast__1953", builder.reshape(env.get("InsertedPrecisionFreeCast__1952"), [1,128,4,64]));
  env.set("InsertedPrecisionFreeCast__1958", builder["add"](env.get("InsertedPrecisionFreeCast__1957"), env.get("InsertedPrecisionFreeCast__1953"), {"label":"/model/layers.11/attn/k_rotary/RotaryEmbedding_add_mul_cos_sin_1609"}));
  env.set("_1958", builder.cast(env.get("InsertedPrecisionFreeCast__1958"), "float16"));
  env.set("_1959", builder.reshape(env.get("_1958"), [1,128,256]));
  env.set("_1960", builder.reshape(env.get("_1959"), [1,128,4,64]));
  env.set("present_11_key_22", builder["scatterND"](env.get("past_key_values_11_key_1961"), env.get("Inserted_791"), env.get("_1960"), {"label":"/model/layers.11/attn/GroupQueryAttention_/GQA/present_key/ScatterND_1612"}));
  env.set("_1962", builder.reshape(env.get("present_11_key_22"), [1,4,1,512,64]));
  env.set("_1963", builder.expand(env.get("_1962"), [1,4,8,512,64], {"label":"/model/layers.11/attn/GroupQueryAttention_/GQA/true_present_key/expand_1617"}));
  env.set("_1964", builder.reshape(env.get("_1963"), [1,32,512,64]));
  env.set("_1965", builder["transpose"](env.get("_1964"), {"label":"/model/layers.11/attn/GroupQueryAttention_/GQA/present_key/transpose_1619","permutation":[0,1,3,2]}));
  env.set("InsertedPrecisionFreeCast__1965", builder.cast(env.get("_1965"), "float32"));
  env.set("Inserted_804", builder["clamp"](env.get("position_ids_625"), {"label":"Inserted_Clip_1635","maxValue":2047,"minValue":-2048}));
  env.set("_1984", builder["gather"](env.get("_644"), env.get("Inserted_804"), {"axis":0,"label":"/model/layers.11/attn/q_rotary/RotaryEmbedding_gather_cos_1634"}));
  env.set("_1985", builder.reshape(env.get("_1984"), [1,128,1,1,32]));
  env.set("InsertedPrecisionFreeCast__1985", builder.cast(env.get("_1985"), "float32"));
  env.set("_1971", builder.dequantizeLinear(env.get("_1968"), env.get("_1969"), env.get("_1970"), {"axis":2,"blockSize":32,"label":"/model/layers.11/attn/q_proj/MatMul_Q4_dequantizeLinear_1623"}));
  env.set("_1972", builder.reshape(env.get("_1971"), [2048,2048]));
  env.set("_1973", builder["transpose"](env.get("_1972"), {"label":"/model/layers.11/attn/q_proj/MatMul_Q4_transpose_dequantizeLinear_1625","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__1973", builder.cast(env.get("_1973"), "float32"));
  env.set("InsertedPrecisionFreeCast__1974", builder["matmul"](env.get("_1917"), env.get("InsertedPrecisionFreeCast__1973"), {"label":"/model/layers.11/attn/q_proj/MatMul_Q4_matmul_1626"}));
  env.set("_1974", builder.cast(env.get("InsertedPrecisionFreeCast__1974"), "float16"));
  env.set("_1975", builder.reshape(env.get("_1974"), [1,128,32,64]));
  env.set("_1976", builder.reshape(env.get("_1975"), [1,128,32,2,32]));
  env.set("InsertedPrecisionFreeCast__1976", builder.cast(env.get("_1976"), "float32"));
  env.set("InsertedPrecisionFreeCast__1986", builder["mul"](env.get("InsertedPrecisionFreeCast__1976"), env.get("InsertedPrecisionFreeCast__1985"), {"label":"/model/layers.11/attn/q_rotary/RotaryEmbedding_mul_cos_1637"}));
  env.set("InsertedPrecisionFreeCast__1987", builder.reshape(env.get("InsertedPrecisionFreeCast__1986"), [1,128,32,64]));
  env.set("InsertedPrecisionFreeCast__1981", builder.cast(env.get("_1981"), "float32"));
  env.set("Inserted_795", builder["clamp"](env.get("position_ids_625"), {"label":"Inserted_Clip_1621","maxValue":2047,"minValue":-2048}));
  env.set("_1966", builder["gather"](env.get("_624"), env.get("Inserted_795"), {"axis":0,"label":"/model/layers.11/attn/q_rotary/RotaryEmbedding_gather_sin_1620"}));
  env.set("_1967", builder.reshape(env.get("_1966"), [1,128,1,1,32]));
  env.set("InsertedPrecisionFreeCast__1967", builder.cast(env.get("_1967"), "float32"));
  {
    const tmp = builder.split(env.get("_1976"), 2, {"axis":3,"label":"/model/layers.11/attn/q_rotary/RotaryEmbedding_split_partial_input0_1629"});
    env.set("_1977", tmp[0]);
    env.set("_1978", tmp[1]);
  }
  env.set("_1979", builder.concat([env.get("_1978"), env.get("_1977")], 3, {"label":"/model/layers.11/attn/q_rotary/RotaryEmbedding_concat_partial_input0_1630"}));
  env.set("InsertedPrecisionFreeCast__1979", builder.cast(env.get("_1979"), "float32"));
  env.set("InsertedPrecisionFreeCast__1980", builder["mul"](env.get("InsertedPrecisionFreeCast__1979"), env.get("InsertedPrecisionFreeCast__1967"), {"label":"/model/layers.11/attn/q_rotary/RotaryEmbedding_mul_sin_1631"}));
  env.set("InsertedPrecisionFreeCast__1982", builder["mul"](env.get("InsertedPrecisionFreeCast__1980"), env.get("InsertedPrecisionFreeCast__1981"), {"label":"/model/layers.11/attn/q_rotary/RotaryEmbedding_mul_sign_1632"}));
  env.set("InsertedPrecisionFreeCast__1983", builder.reshape(env.get("InsertedPrecisionFreeCast__1982"), [1,128,32,64]));
  env.set("InsertedPrecisionFreeCast__1988", builder["add"](env.get("InsertedPrecisionFreeCast__1987"), env.get("InsertedPrecisionFreeCast__1983"), {"label":"/model/layers.11/attn/q_rotary/RotaryEmbedding_add_mul_cos_sin_1639"}));
  env.set("_1988", builder.cast(env.get("InsertedPrecisionFreeCast__1988"), "float16"));
  env.set("_1989", builder.reshape(env.get("_1988"), [1,128,2048]));
  env.set("_1990", builder.reshape(env.get("_1989"), [1,128,32,64]));
  env.set("_1991", builder["transpose"](env.get("_1990"), {"label":"/model/layers.11/attn/GroupQueryAttention_/GQA/query/transpose_1642","permutation":[0,2,1,3]}));
  env.set("InsertedPrecisionFreeCast__1991", builder.cast(env.get("_1991"), "float32"));
  env.set("InsertedPrecisionFreeCast__1992", builder["matmul"](env.get("InsertedPrecisionFreeCast__1991"), env.get("InsertedPrecisionFreeCast__1965"), {"label":"/model/layers.11/attn/GroupQueryAttention_/Attention/qkv/matmul_1_1643"}));
  env.set("InsertedPrecisionFreeCast__1993", builder["mul"](env.get("InsertedPrecisionFreeCast__1992"), env.get("InsertedPrecisionFreeCast__684"), {"label":"/model/layers.11/attn/GroupQueryAttention_/Attention/qkv/div_1644"}));
  env.set("_1932", builder.expand(env.get("_617"), [1,32,128,512], {"label":"/model/layers.11/attn/GroupQueryAttention_/GQA/GQA_mask_shape_ones/expand_1584"}));
  env.set("_1933", builder.cumulativeSum(env.get("_1932"), 3, {"exclusive":true,"label":"/model/layers.11/attn/GroupQueryAttention_range_of_mask_shape_1585"}));
  env.set("_1929", builder["add"](env.get("_613"), env.get("_1921"), {"label":"/model/layers.11/attn/GroupQueryAttention_/GQA/attn_mask/add_1581"}));
  env.set("_1930", builder.expand(env.get("_1929"), [512,128], {"label":"/model/layers.11/attn/GroupQueryAttention_/GQA/expand_neq_right_1582"}));
  env.set("_1931", builder["transpose"](env.get("_1930"), {"label":"/model/layers.11/attn/GroupQueryAttention_/GQA/neq_right/transpose_1583","permutation":[1,0]}));
  env.set("Inserted_769", builder["lesser"](env.get("_1933"), env.get("_1931"), {"label":"/model/layers.11/attn/GroupQueryAttention_/GQA/attn_mask/condition_1_1586"}));
  env.set("InsertedPrecisionFreeCast__1935", builder["where"](env.get("Inserted_769"), env.get("InsertedPrecisionFreeCast__621"), env.get("InsertedPrecisionFreeCast__622"), {"label":"/model/layers.11/attn/GroupQueryAttention_/GQA/attn_mask/where_1588"}));
  env.set("InsertedPrecisionFreeCast__1994", builder["add"](env.get("InsertedPrecisionFreeCast__1993"), env.get("InsertedPrecisionFreeCast__1935"), {"label":"/model/layers.11/attn/GroupQueryAttention_/Attention/attn_mask/softmax_input_1645"}));
  env.set("InsertedPrecisionFreeCast__inlfunc_Softmax_token_10_X_ReduceMax", builder["reduceMax"](env.get("InsertedPrecisionFreeCast__1994"), {"axes":[3],"keepDimensions":true}));
  env.set("InsertedPrecisionFreeCast__inlfunc_Softmax_token_10_X_Sub", builder["sub"](env.get("InsertedPrecisionFreeCast__1994"), env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_10_X_ReduceMax"), {}));
  env.set("InsertedPrecisionFreeCast__inlfunc_Softmax_token_10_X_Exp", builder["exp"](env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_10_X_Sub"), {}));
  env.set("InsertedPrecisionFreeCast__inlfunc_Softmax_token_10_X_ReduceSum", builder["reduceSum"](env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_10_X_Exp"), {"axes":[3],"keepDimensions":true}));
  env.set("InsertedPrecisionFreeCast__1995", builder["div"](env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_10_X_Exp"), env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_10_X_ReduceSum"), {}));
  env.set("InsertedPrecisionFreeCast__1996", builder["matmul"](env.get("InsertedPrecisionFreeCast__1995"), env.get("InsertedPrecisionFreeCast__1928"), {"label":"/model/layers.11/attn/GroupQueryAttention_/Attention/qkv/matmul_2_1647"}));
  env.set("_1996", builder.cast(env.get("InsertedPrecisionFreeCast__1996"), "float16"));
  env.set("_1997", builder["transpose"](env.get("_1996"), {"label":"/model/layers.11/attn/GroupQueryAttention_/Attention/qkv/transpose_1648","permutation":[0,2,1,3]}));
  env.set("_1998", builder.reshape(env.get("_1997"), [1,128,2048]));
  env.set("InsertedPrecisionFreeCast__1998", builder.cast(env.get("_1998"), "float32"));
  env.set("InsertedPrecisionFreeCast__1999", builder["matmul"](env.get("InsertedPrecisionFreeCast__1998"), env.get("InsertedPrecisionFreeCast__308"), {"label":"/model/layers.11/attn/o_proj/MatMul_Q4_matmul_1650"}));
  env.set("_2003", builder["add"](env.get("_2002"), env.get("InsertedPrecisionFreeCast__1999"), {"label":"/model/layers.11/post_attention_layernorm/SkipLayerNorm_add_skip_1654"}));
  env.set("_2026", builder.cast(env.get("_2003"), "float16"));
  env.set("_2027", builder.cast(env.get("_2026"), "float32"));
  env.set("_294", builder.dequantizeLinear(env.get("_291"), env.get("_292"), env.get("_293"), {"axis":2,"blockSize":32,"label":"/model/layers.11/mlp/down_proj/MatMul_Q4_dequantizeLinear_123"}));
  env.set("_295", builder.reshape(env.get("_294"), [2048,5632]));
  env.set("_296", builder["transpose"](env.get("_295"), {"label":"/model/layers.11/mlp/down_proj/MatMul_Q4_transpose_dequantizeLinear_125","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__296", builder.cast(env.get("_296"), "float32"));
  env.set("_2017", builder.dequantizeLinear(env.get("_2014"), env.get("_2015"), env.get("_2016"), {"axis":2,"blockSize":32,"label":"/model/layers.11/mlp/gate_proj/MatMul_Q4_dequantizeLinear_1664"}));
  env.set("_2018", builder.reshape(env.get("_2017"), [5632,2048]));
  env.set("_2019", builder["transpose"](env.get("_2018"), {"label":"/model/layers.11/mlp/gate_proj/MatMul_Q4_transpose_dequantizeLinear_1666","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__2019", builder.cast(env.get("_2019"), "float32"));
  env.set("_2010", builder.cast(env.get("_2009"), "float32"));
  env.set("_2004", builder["pow"](env.get("_2003"), env.get("_583"), {"label":"/model/layers.11/post_attention_layernorm/SkipLayerNorm_pow_1655"}));
  env.set("_2005", builder["reduceMean"](env.get("_2004"), {"axes":[2],"keepDimensions":true,"label":"/model/layers.11/post_attention_layernorm/SkipLayerNorm_reduceMean_1656"}));
  env.set("_2006", builder["add"](env.get("_2005"), env.get("_586"), {"label":"/model/layers.11/post_attention_layernorm/SkipLayerNorm_add_1657"}));
  env.set("_2007", builder["sqrt"](env.get("_2006"), {"label":"/model/layers.11/post_attention_layernorm/SkipLayerNorm_sqrt_1658"}));
  env.set("_2008", builder["div"](env.get("_2003"), env.get("_2007"), {"label":"/model/layers.11/post_attention_layernorm/SkipLayerNorm_div_1659"}));
  env.set("_2011", builder["mul"](env.get("_2010"), env.get("_2008"), {"label":"/model/layers.11/post_attention_layernorm/SkipLayerNorm_mul_1661"}));
  env.set("InsertedPrecisionFreeCast__2020", builder["matmul"](env.get("_2011"), env.get("InsertedPrecisionFreeCast__2019"), {"label":"/model/layers.11/mlp/gate_proj/MatMul_Q4_matmul_1667"}));
  env.set("InsertedPrecisionFreeCast__2021", builder["sigmoid"](env.get("InsertedPrecisionFreeCast__2020"), {"label":"/model/layers.11/mlp/act_fn/Sigmoid_1668"}));
  env.set("InsertedPrecisionFreeCast__2022", builder["mul"](env.get("InsertedPrecisionFreeCast__2020"), env.get("InsertedPrecisionFreeCast__2021"), {"label":"/model/layers.11/mlp/act_fn/Mul_1669"}));
  env.set("_300", builder.dequantizeLinear(env.get("_297"), env.get("_298"), env.get("_299"), {"axis":2,"blockSize":32,"label":"/model/layers.11/mlp/up_proj/MatMul_Q4_dequantizeLinear_126"}));
  env.set("_301", builder.reshape(env.get("_300"), [5632,2048]));
  env.set("_302", builder["transpose"](env.get("_301"), {"label":"/model/layers.11/mlp/up_proj/MatMul_Q4_transpose_dequantizeLinear_128","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__302", builder.cast(env.get("_302"), "float32"));
  env.set("InsertedPrecisionFreeCast__2013", builder["matmul"](env.get("_2011"), env.get("InsertedPrecisionFreeCast__302"), {"label":"/model/layers.11/mlp/up_proj/MatMul_Q4_matmul_1663"}));
  env.set("InsertedPrecisionFreeCast__2023", builder["mul"](env.get("InsertedPrecisionFreeCast__2022"), env.get("InsertedPrecisionFreeCast__2013"), {"label":"/model/layers.11/mlp/Mul_1670"}));
  env.set("InsertedPrecisionFreeCast__2024", builder["matmul"](env.get("InsertedPrecisionFreeCast__2023"), env.get("InsertedPrecisionFreeCast__296"), {"label":"/model/layers.11/mlp/down_proj/MatMul_Q4_matmul_1671"}));
  env.set("_2028", builder["add"](env.get("_2027"), env.get("InsertedPrecisionFreeCast__2024"), {"label":"/model/layers.12/input_layernorm/SkipLayerNorm_add_skip_1675"}));
  env.set("_2120", builder.cast(env.get("_2028"), "float16"));
  env.set("_2121", builder.cast(env.get("_2120"), "float32"));
  env.set("_282", builder.dequantizeLinear(env.get("_279"), env.get("_280"), env.get("_281"), {"axis":2,"blockSize":32,"label":"/model/layers.12/attn/o_proj/MatMul_Q4_dequantizeLinear_117"}));
  env.set("_283", builder.reshape(env.get("_282"), [2048,2048]));
  env.set("_284", builder["transpose"](env.get("_283"), {"label":"/model/layers.12/attn/o_proj/MatMul_Q4_transpose_dequantizeLinear_119","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__284", builder.cast(env.get("_284"), "float32"));
  env.set("Inserted_816", builder.cast(env.get("_601"), "uint8"));
  env.set("_2040", builder["where"](env.get("Inserted_816"), env.get("_602"), env.get("_600"), {"label":"/model/layers.12/attn/GroupQueryAttention_/GQA/scatter/where_1686"}));
  env.set("_2041", builder["add"](env.get("_604"), env.get("_2040"), {"label":"/model/layers.12/attn/GroupQueryAttention_/GQA/right_constant/add_1688"}));
  env.set("_2042", builder.concat([env.get("_606"), env.get("_2041")], 1, {"label":"/model/layers.12/attn/GroupQueryAttention_/GQA/concat_for_pre_scatter_indices_1689"}));
  env.set("_2043", builder.reshape(env.get("_2042"), [1,128,4,3]));
  env.set("Inserted_818", builder.cast(env.get("_2043"), "int64"));
  env.set("Inserted_820", builder["max"](env.get("Inserted_818"), env.get("Inserted_819"), {"label":"Inserted_Max_1693"}));
  env.set("Inserted_822", builder["min"](env.get("Inserted_820"), env.get("Inserted_821"), {"label":"Inserted_Min_1694"}));
  env.set("_288", builder.dequantizeLinear(env.get("_285"), env.get("_286"), env.get("_287"), {"axis":2,"blockSize":32,"label":"/model/layers.12/attn/v_proj/MatMul_Q4_dequantizeLinear_120"}));
  env.set("_289", builder.reshape(env.get("_288"), [256,2048]));
  env.set("_290", builder["transpose"](env.get("_289"), {"label":"/model/layers.12/attn/v_proj/MatMul_Q4_transpose_dequantizeLinear_122","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__290", builder.cast(env.get("_290"), "float32"));
  env.set("_2035", builder.cast(env.get("_2034"), "float32"));
  env.set("_2029", builder["pow"](env.get("_2028"), env.get("_583"), {"label":"/model/layers.12/input_layernorm/SkipLayerNorm_pow_1676"}));
  env.set("_2030", builder["reduceMean"](env.get("_2029"), {"axes":[2],"keepDimensions":true,"label":"/model/layers.12/input_layernorm/SkipLayerNorm_reduceMean_1677"}));
  env.set("_2031", builder["add"](env.get("_2030"), env.get("_586"), {"label":"/model/layers.12/input_layernorm/SkipLayerNorm_add_1678"}));
  env.set("_2032", builder["sqrt"](env.get("_2031"), {"label":"/model/layers.12/input_layernorm/SkipLayerNorm_sqrt_1679"}));
  env.set("_2033", builder["div"](env.get("_2028"), env.get("_2032"), {"label":"/model/layers.12/input_layernorm/SkipLayerNorm_div_1680"}));
  env.set("_2036", builder["mul"](env.get("_2035"), env.get("_2033"), {"label":"/model/layers.12/input_layernorm/SkipLayerNorm_mul_1682"}));
  env.set("InsertedPrecisionFreeCast__2038", builder["matmul"](env.get("_2036"), env.get("InsertedPrecisionFreeCast__290"), {"label":"/model/layers.12/attn/v_proj/MatMul_Q4_matmul_1684"}));
  env.set("_2038", builder.cast(env.get("InsertedPrecisionFreeCast__2038"), "float16"));
  env.set("_2039", builder.reshape(env.get("_2038"), [1,128,4,64]));
  env.set("present_12_value_25", builder["scatterND"](env.get("past_key_values_12_value_2044"), env.get("Inserted_822"), env.get("_2039"), {"label":"/model/layers.12/attn/GroupQueryAttention_/GQA/present_value/ScatterND_1691"}));
  env.set("_2045", builder.reshape(env.get("present_12_value_25"), [1,4,1,512,64]));
  env.set("_2046", builder.expand(env.get("_2045"), [1,4,8,512,64], {"label":"/model/layers.12/attn/GroupQueryAttention_/GQA/true_present_value/expand_1696"}));
  env.set("_2047", builder.reshape(env.get("_2046"), [1,32,512,64]));
  env.set("InsertedPrecisionFreeCast__2047", builder.cast(env.get("_2047"), "float32"));
  env.set("Inserted_847", builder.cast(env.get("_2043"), "int64"));
  env.set("Inserted_849", builder["max"](env.get("Inserted_847"), env.get("Inserted_848"), {"label":"Inserted_Max_1731"}));
  env.set("Inserted_851", builder["min"](env.get("Inserted_849"), env.get("Inserted_850"), {"label":"Inserted_Min_1732"}));
  env.set("Inserted_840", builder["clamp"](env.get("position_ids_625"), {"label":"Inserted_Clip_1722","maxValue":2047,"minValue":-2048}));
  env.set("_2073", builder["gather"](env.get("_644"), env.get("Inserted_840"), {"axis":0,"label":"/model/layers.12/attn/k_rotary/RotaryEmbedding_gather_cos_1721"}));
  env.set("_2074", builder.reshape(env.get("_2073"), [1,128,1,1,32]));
  env.set("InsertedPrecisionFreeCast__2074", builder.cast(env.get("_2074"), "float32"));
  env.set("_2060", builder.dequantizeLinear(env.get("_2057"), env.get("_2058"), env.get("_2059"), {"axis":2,"blockSize":32,"label":"/model/layers.12/attn/k_proj/MatMul_Q4_dequantizeLinear_1710"}));
  env.set("_2061", builder.reshape(env.get("_2060"), [256,2048]));
  env.set("_2062", builder["transpose"](env.get("_2061"), {"label":"/model/layers.12/attn/k_proj/MatMul_Q4_transpose_dequantizeLinear_1712","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__2062", builder.cast(env.get("_2062"), "float32"));
  env.set("InsertedPrecisionFreeCast__2063", builder["matmul"](env.get("_2036"), env.get("InsertedPrecisionFreeCast__2062"), {"label":"/model/layers.12/attn/k_proj/MatMul_Q4_matmul_1713"}));
  env.set("_2063", builder.cast(env.get("InsertedPrecisionFreeCast__2063"), "float16"));
  env.set("_2064", builder.reshape(env.get("_2063"), [1,128,4,64]));
  env.set("_2065", builder.reshape(env.get("_2064"), [1,128,4,2,32]));
  env.set("InsertedPrecisionFreeCast__2065", builder.cast(env.get("_2065"), "float32"));
  env.set("InsertedPrecisionFreeCast__2075", builder["mul"](env.get("InsertedPrecisionFreeCast__2065"), env.get("InsertedPrecisionFreeCast__2074"), {"label":"/model/layers.12/attn/k_rotary/RotaryEmbedding_mul_cos_1724"}));
  env.set("InsertedPrecisionFreeCast__2076", builder.reshape(env.get("InsertedPrecisionFreeCast__2075"), [1,128,4,64]));
  env.set("InsertedPrecisionFreeCast__2070", builder.cast(env.get("_2070"), "float32"));
  env.set("Inserted_831", builder["clamp"](env.get("position_ids_625"), {"label":"Inserted_Clip_1708","maxValue":2047,"minValue":-2048}));
  env.set("_2055", builder["gather"](env.get("_624"), env.get("Inserted_831"), {"axis":0,"label":"/model/layers.12/attn/k_rotary/RotaryEmbedding_gather_sin_1707"}));
  env.set("_2056", builder.reshape(env.get("_2055"), [1,128,1,1,32]));
  env.set("InsertedPrecisionFreeCast__2056", builder.cast(env.get("_2056"), "float32"));
  {
    const tmp = builder.split(env.get("_2065"), 2, {"axis":3,"label":"/model/layers.12/attn/k_rotary/RotaryEmbedding_split_partial_input0_1716"});
    env.set("_2066", tmp[0]);
    env.set("_2067", tmp[1]);
  }
  env.set("_2068", builder.concat([env.get("_2067"), env.get("_2066")], 3, {"label":"/model/layers.12/attn/k_rotary/RotaryEmbedding_concat_partial_input0_1717"}));
  env.set("InsertedPrecisionFreeCast__2068", builder.cast(env.get("_2068"), "float32"));
  env.set("InsertedPrecisionFreeCast__2069", builder["mul"](env.get("InsertedPrecisionFreeCast__2068"), env.get("InsertedPrecisionFreeCast__2056"), {"label":"/model/layers.12/attn/k_rotary/RotaryEmbedding_mul_sin_1718"}));
  env.set("InsertedPrecisionFreeCast__2071", builder["mul"](env.get("InsertedPrecisionFreeCast__2069"), env.get("InsertedPrecisionFreeCast__2070"), {"label":"/model/layers.12/attn/k_rotary/RotaryEmbedding_mul_sign_1719"}));
  env.set("InsertedPrecisionFreeCast__2072", builder.reshape(env.get("InsertedPrecisionFreeCast__2071"), [1,128,4,64]));
  env.set("InsertedPrecisionFreeCast__2077", builder["add"](env.get("InsertedPrecisionFreeCast__2076"), env.get("InsertedPrecisionFreeCast__2072"), {"label":"/model/layers.12/attn/k_rotary/RotaryEmbedding_add_mul_cos_sin_1726"}));
  env.set("_2077", builder.cast(env.get("InsertedPrecisionFreeCast__2077"), "float16"));
  env.set("_2078", builder.reshape(env.get("_2077"), [1,128,256]));
  env.set("_2079", builder.reshape(env.get("_2078"), [1,128,4,64]));
  env.set("present_12_key_24", builder["scatterND"](env.get("past_key_values_12_key_2080"), env.get("Inserted_851"), env.get("_2079"), {"label":"/model/layers.12/attn/GroupQueryAttention_/GQA/present_key/ScatterND_1729"}));
  env.set("_2081", builder.reshape(env.get("present_12_key_24"), [1,4,1,512,64]));
  env.set("_2082", builder.expand(env.get("_2081"), [1,4,8,512,64], {"label":"/model/layers.12/attn/GroupQueryAttention_/GQA/true_present_key/expand_1734"}));
  env.set("_2083", builder.reshape(env.get("_2082"), [1,32,512,64]));
  env.set("_2084", builder["transpose"](env.get("_2083"), {"label":"/model/layers.12/attn/GroupQueryAttention_/GQA/present_key/transpose_1736","permutation":[0,1,3,2]}));
  env.set("InsertedPrecisionFreeCast__2084", builder.cast(env.get("_2084"), "float32"));
  env.set("Inserted_864", builder["clamp"](env.get("position_ids_625"), {"label":"Inserted_Clip_1752","maxValue":2047,"minValue":-2048}));
  env.set("_2103", builder["gather"](env.get("_644"), env.get("Inserted_864"), {"axis":0,"label":"/model/layers.12/attn/q_rotary/RotaryEmbedding_gather_cos_1751"}));
  env.set("_2104", builder.reshape(env.get("_2103"), [1,128,1,1,32]));
  env.set("InsertedPrecisionFreeCast__2104", builder.cast(env.get("_2104"), "float32"));
  env.set("_2090", builder.dequantizeLinear(env.get("_2087"), env.get("_2088"), env.get("_2089"), {"axis":2,"blockSize":32,"label":"/model/layers.12/attn/q_proj/MatMul_Q4_dequantizeLinear_1740"}));
  env.set("_2091", builder.reshape(env.get("_2090"), [2048,2048]));
  env.set("_2092", builder["transpose"](env.get("_2091"), {"label":"/model/layers.12/attn/q_proj/MatMul_Q4_transpose_dequantizeLinear_1742","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__2092", builder.cast(env.get("_2092"), "float32"));
  env.set("InsertedPrecisionFreeCast__2093", builder["matmul"](env.get("_2036"), env.get("InsertedPrecisionFreeCast__2092"), {"label":"/model/layers.12/attn/q_proj/MatMul_Q4_matmul_1743"}));
  env.set("_2093", builder.cast(env.get("InsertedPrecisionFreeCast__2093"), "float16"));
  env.set("_2094", builder.reshape(env.get("_2093"), [1,128,32,64]));
  env.set("_2095", builder.reshape(env.get("_2094"), [1,128,32,2,32]));
  env.set("InsertedPrecisionFreeCast__2095", builder.cast(env.get("_2095"), "float32"));
  env.set("InsertedPrecisionFreeCast__2105", builder["mul"](env.get("InsertedPrecisionFreeCast__2095"), env.get("InsertedPrecisionFreeCast__2104"), {"label":"/model/layers.12/attn/q_rotary/RotaryEmbedding_mul_cos_1754"}));
  env.set("InsertedPrecisionFreeCast__2106", builder.reshape(env.get("InsertedPrecisionFreeCast__2105"), [1,128,32,64]));
  env.set("InsertedPrecisionFreeCast__2100", builder.cast(env.get("_2100"), "float32"));
  env.set("Inserted_855", builder["clamp"](env.get("position_ids_625"), {"label":"Inserted_Clip_1738","maxValue":2047,"minValue":-2048}));
  env.set("_2085", builder["gather"](env.get("_624"), env.get("Inserted_855"), {"axis":0,"label":"/model/layers.12/attn/q_rotary/RotaryEmbedding_gather_sin_1737"}));
  env.set("_2086", builder.reshape(env.get("_2085"), [1,128,1,1,32]));
  env.set("InsertedPrecisionFreeCast__2086", builder.cast(env.get("_2086"), "float32"));
  {
    const tmp = builder.split(env.get("_2095"), 2, {"axis":3,"label":"/model/layers.12/attn/q_rotary/RotaryEmbedding_split_partial_input0_1746"});
    env.set("_2096", tmp[0]);
    env.set("_2097", tmp[1]);
  }
  env.set("_2098", builder.concat([env.get("_2097"), env.get("_2096")], 3, {"label":"/model/layers.12/attn/q_rotary/RotaryEmbedding_concat_partial_input0_1747"}));
  env.set("InsertedPrecisionFreeCast__2098", builder.cast(env.get("_2098"), "float32"));
  env.set("InsertedPrecisionFreeCast__2099", builder["mul"](env.get("InsertedPrecisionFreeCast__2098"), env.get("InsertedPrecisionFreeCast__2086"), {"label":"/model/layers.12/attn/q_rotary/RotaryEmbedding_mul_sin_1748"}));
  env.set("InsertedPrecisionFreeCast__2101", builder["mul"](env.get("InsertedPrecisionFreeCast__2099"), env.get("InsertedPrecisionFreeCast__2100"), {"label":"/model/layers.12/attn/q_rotary/RotaryEmbedding_mul_sign_1749"}));
  env.set("InsertedPrecisionFreeCast__2102", builder.reshape(env.get("InsertedPrecisionFreeCast__2101"), [1,128,32,64]));
  env.set("InsertedPrecisionFreeCast__2107", builder["add"](env.get("InsertedPrecisionFreeCast__2106"), env.get("InsertedPrecisionFreeCast__2102"), {"label":"/model/layers.12/attn/q_rotary/RotaryEmbedding_add_mul_cos_sin_1756"}));
  env.set("_2107", builder.cast(env.get("InsertedPrecisionFreeCast__2107"), "float16"));
  env.set("_2108", builder.reshape(env.get("_2107"), [1,128,2048]));
  env.set("_2109", builder.reshape(env.get("_2108"), [1,128,32,64]));
  env.set("_2110", builder["transpose"](env.get("_2109"), {"label":"/model/layers.12/attn/GroupQueryAttention_/GQA/query/transpose_1759","permutation":[0,2,1,3]}));
  env.set("InsertedPrecisionFreeCast__2110", builder.cast(env.get("_2110"), "float32"));
  env.set("InsertedPrecisionFreeCast__2111", builder["matmul"](env.get("InsertedPrecisionFreeCast__2110"), env.get("InsertedPrecisionFreeCast__2084"), {"label":"/model/layers.12/attn/GroupQueryAttention_/Attention/qkv/matmul_1_1760"}));
  env.set("InsertedPrecisionFreeCast__2112", builder["mul"](env.get("InsertedPrecisionFreeCast__2111"), env.get("InsertedPrecisionFreeCast__684"), {"label":"/model/layers.12/attn/GroupQueryAttention_/Attention/qkv/div_1761"}));
  env.set("_2051", builder.expand(env.get("_617"), [1,32,128,512], {"label":"/model/layers.12/attn/GroupQueryAttention_/GQA/GQA_mask_shape_ones/expand_1701"}));
  env.set("_2052", builder.cumulativeSum(env.get("_2051"), 3, {"exclusive":true,"label":"/model/layers.12/attn/GroupQueryAttention_range_of_mask_shape_1702"}));
  env.set("_2048", builder["add"](env.get("_613"), env.get("_2040"), {"label":"/model/layers.12/attn/GroupQueryAttention_/GQA/attn_mask/add_1698"}));
  env.set("_2049", builder.expand(env.get("_2048"), [512,128], {"label":"/model/layers.12/attn/GroupQueryAttention_/GQA/expand_neq_right_1699"}));
  env.set("_2050", builder["transpose"](env.get("_2049"), {"label":"/model/layers.12/attn/GroupQueryAttention_/GQA/neq_right/transpose_1700","permutation":[1,0]}));
  env.set("Inserted_829", builder["lesser"](env.get("_2052"), env.get("_2050"), {"label":"/model/layers.12/attn/GroupQueryAttention_/GQA/attn_mask/condition_1_1703"}));
  env.set("InsertedPrecisionFreeCast__2054", builder["where"](env.get("Inserted_829"), env.get("InsertedPrecisionFreeCast__621"), env.get("InsertedPrecisionFreeCast__622"), {"label":"/model/layers.12/attn/GroupQueryAttention_/GQA/attn_mask/where_1705"}));
  env.set("InsertedPrecisionFreeCast__2113", builder["add"](env.get("InsertedPrecisionFreeCast__2112"), env.get("InsertedPrecisionFreeCast__2054"), {"label":"/model/layers.12/attn/GroupQueryAttention_/Attention/attn_mask/softmax_input_1762"}));
  env.set("InsertedPrecisionFreeCast__inlfunc_Softmax_token_11_X_ReduceMax", builder["reduceMax"](env.get("InsertedPrecisionFreeCast__2113"), {"axes":[3],"keepDimensions":true}));
  env.set("InsertedPrecisionFreeCast__inlfunc_Softmax_token_11_X_Sub", builder["sub"](env.get("InsertedPrecisionFreeCast__2113"), env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_11_X_ReduceMax"), {}));
  env.set("InsertedPrecisionFreeCast__inlfunc_Softmax_token_11_X_Exp", builder["exp"](env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_11_X_Sub"), {}));
  env.set("InsertedPrecisionFreeCast__inlfunc_Softmax_token_11_X_ReduceSum", builder["reduceSum"](env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_11_X_Exp"), {"axes":[3],"keepDimensions":true}));
  env.set("InsertedPrecisionFreeCast__2114", builder["div"](env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_11_X_Exp"), env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_11_X_ReduceSum"), {}));
  env.set("InsertedPrecisionFreeCast__2115", builder["matmul"](env.get("InsertedPrecisionFreeCast__2114"), env.get("InsertedPrecisionFreeCast__2047"), {"label":"/model/layers.12/attn/GroupQueryAttention_/Attention/qkv/matmul_2_1764"}));
  env.set("_2115", builder.cast(env.get("InsertedPrecisionFreeCast__2115"), "float16"));
  env.set("_2116", builder["transpose"](env.get("_2115"), {"label":"/model/layers.12/attn/GroupQueryAttention_/Attention/qkv/transpose_1765","permutation":[0,2,1,3]}));
  env.set("_2117", builder.reshape(env.get("_2116"), [1,128,2048]));
  env.set("InsertedPrecisionFreeCast__2117", builder.cast(env.get("_2117"), "float32"));
  env.set("InsertedPrecisionFreeCast__2118", builder["matmul"](env.get("InsertedPrecisionFreeCast__2117"), env.get("InsertedPrecisionFreeCast__284"), {"label":"/model/layers.12/attn/o_proj/MatMul_Q4_matmul_1767"}));
  env.set("_2122", builder["add"](env.get("_2121"), env.get("InsertedPrecisionFreeCast__2118"), {"label":"/model/layers.12/post_attention_layernorm/SkipLayerNorm_add_skip_1771"}));
  env.set("_2145", builder.cast(env.get("_2122"), "float16"));
  env.set("_2146", builder.cast(env.get("_2145"), "float32"));
  env.set("_270", builder.dequantizeLinear(env.get("_267"), env.get("_268"), env.get("_269"), {"axis":2,"blockSize":32,"label":"/model/layers.12/mlp/down_proj/MatMul_Q4_dequantizeLinear_111"}));
  env.set("_271", builder.reshape(env.get("_270"), [2048,5632]));
  env.set("_272", builder["transpose"](env.get("_271"), {"label":"/model/layers.12/mlp/down_proj/MatMul_Q4_transpose_dequantizeLinear_113","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__272", builder.cast(env.get("_272"), "float32"));
  env.set("_2136", builder.dequantizeLinear(env.get("_2133"), env.get("_2134"), env.get("_2135"), {"axis":2,"blockSize":32,"label":"/model/layers.12/mlp/gate_proj/MatMul_Q4_dequantizeLinear_1781"}));
  env.set("_2137", builder.reshape(env.get("_2136"), [5632,2048]));
  env.set("_2138", builder["transpose"](env.get("_2137"), {"label":"/model/layers.12/mlp/gate_proj/MatMul_Q4_transpose_dequantizeLinear_1783","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__2138", builder.cast(env.get("_2138"), "float32"));
  env.set("_2129", builder.cast(env.get("_2128"), "float32"));
  env.set("_2123", builder["pow"](env.get("_2122"), env.get("_583"), {"label":"/model/layers.12/post_attention_layernorm/SkipLayerNorm_pow_1772"}));
  env.set("_2124", builder["reduceMean"](env.get("_2123"), {"axes":[2],"keepDimensions":true,"label":"/model/layers.12/post_attention_layernorm/SkipLayerNorm_reduceMean_1773"}));
  env.set("_2125", builder["add"](env.get("_2124"), env.get("_586"), {"label":"/model/layers.12/post_attention_layernorm/SkipLayerNorm_add_1774"}));
  env.set("_2126", builder["sqrt"](env.get("_2125"), {"label":"/model/layers.12/post_attention_layernorm/SkipLayerNorm_sqrt_1775"}));
  env.set("_2127", builder["div"](env.get("_2122"), env.get("_2126"), {"label":"/model/layers.12/post_attention_layernorm/SkipLayerNorm_div_1776"}));
  env.set("_2130", builder["mul"](env.get("_2129"), env.get("_2127"), {"label":"/model/layers.12/post_attention_layernorm/SkipLayerNorm_mul_1778"}));
  env.set("InsertedPrecisionFreeCast__2139", builder["matmul"](env.get("_2130"), env.get("InsertedPrecisionFreeCast__2138"), {"label":"/model/layers.12/mlp/gate_proj/MatMul_Q4_matmul_1784"}));
  env.set("InsertedPrecisionFreeCast__2140", builder["sigmoid"](env.get("InsertedPrecisionFreeCast__2139"), {"label":"/model/layers.12/mlp/act_fn/Sigmoid_1785"}));
  env.set("InsertedPrecisionFreeCast__2141", builder["mul"](env.get("InsertedPrecisionFreeCast__2139"), env.get("InsertedPrecisionFreeCast__2140"), {"label":"/model/layers.12/mlp/act_fn/Mul_1786"}));
  env.set("_276", builder.dequantizeLinear(env.get("_273"), env.get("_274"), env.get("_275"), {"axis":2,"blockSize":32,"label":"/model/layers.12/mlp/up_proj/MatMul_Q4_dequantizeLinear_114"}));
  env.set("_277", builder.reshape(env.get("_276"), [5632,2048]));
  env.set("_278", builder["transpose"](env.get("_277"), {"label":"/model/layers.12/mlp/up_proj/MatMul_Q4_transpose_dequantizeLinear_116","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__278", builder.cast(env.get("_278"), "float32"));
  env.set("InsertedPrecisionFreeCast__2132", builder["matmul"](env.get("_2130"), env.get("InsertedPrecisionFreeCast__278"), {"label":"/model/layers.12/mlp/up_proj/MatMul_Q4_matmul_1780"}));
  env.set("InsertedPrecisionFreeCast__2142", builder["mul"](env.get("InsertedPrecisionFreeCast__2141"), env.get("InsertedPrecisionFreeCast__2132"), {"label":"/model/layers.12/mlp/Mul_1787"}));
  env.set("InsertedPrecisionFreeCast__2143", builder["matmul"](env.get("InsertedPrecisionFreeCast__2142"), env.get("InsertedPrecisionFreeCast__272"), {"label":"/model/layers.12/mlp/down_proj/MatMul_Q4_matmul_1788"}));
  env.set("_2147", builder["add"](env.get("_2146"), env.get("InsertedPrecisionFreeCast__2143"), {"label":"/model/layers.13/input_layernorm/SkipLayerNorm_add_skip_1792"}));
  env.set("_2239", builder.cast(env.get("_2147"), "float16"));
  env.set("_2240", builder.cast(env.get("_2239"), "float32"));
  env.set("_258", builder.dequantizeLinear(env.get("_255"), env.get("_256"), env.get("_257"), {"axis":2,"blockSize":32,"label":"/model/layers.13/attn/o_proj/MatMul_Q4_dequantizeLinear_105"}));
  env.set("_259", builder.reshape(env.get("_258"), [2048,2048]));
  env.set("_260", builder["transpose"](env.get("_259"), {"label":"/model/layers.13/attn/o_proj/MatMul_Q4_transpose_dequantizeLinear_107","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__260", builder.cast(env.get("_260"), "float32"));
  env.set("Inserted_876", builder.cast(env.get("_601"), "uint8"));
  env.set("_2159", builder["where"](env.get("Inserted_876"), env.get("_602"), env.get("_600"), {"label":"/model/layers.13/attn/GroupQueryAttention_/GQA/scatter/where_1803"}));
  env.set("_2160", builder["add"](env.get("_604"), env.get("_2159"), {"label":"/model/layers.13/attn/GroupQueryAttention_/GQA/right_constant/add_1805"}));
  env.set("_2161", builder.concat([env.get("_606"), env.get("_2160")], 1, {"label":"/model/layers.13/attn/GroupQueryAttention_/GQA/concat_for_pre_scatter_indices_1806"}));
  env.set("_2162", builder.reshape(env.get("_2161"), [1,128,4,3]));
  env.set("Inserted_878", builder.cast(env.get("_2162"), "int64"));
  env.set("Inserted_880", builder["max"](env.get("Inserted_878"), env.get("Inserted_879"), {"label":"Inserted_Max_1810"}));
  env.set("Inserted_882", builder["min"](env.get("Inserted_880"), env.get("Inserted_881"), {"label":"Inserted_Min_1811"}));
  env.set("_264", builder.dequantizeLinear(env.get("_261"), env.get("_262"), env.get("_263"), {"axis":2,"blockSize":32,"label":"/model/layers.13/attn/v_proj/MatMul_Q4_dequantizeLinear_108"}));
  env.set("_265", builder.reshape(env.get("_264"), [256,2048]));
  env.set("_266", builder["transpose"](env.get("_265"), {"label":"/model/layers.13/attn/v_proj/MatMul_Q4_transpose_dequantizeLinear_110","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__266", builder.cast(env.get("_266"), "float32"));
  env.set("_2154", builder.cast(env.get("_2153"), "float32"));
  env.set("_2148", builder["pow"](env.get("_2147"), env.get("_583"), {"label":"/model/layers.13/input_layernorm/SkipLayerNorm_pow_1793"}));
  env.set("_2149", builder["reduceMean"](env.get("_2148"), {"axes":[2],"keepDimensions":true,"label":"/model/layers.13/input_layernorm/SkipLayerNorm_reduceMean_1794"}));
  env.set("_2150", builder["add"](env.get("_2149"), env.get("_586"), {"label":"/model/layers.13/input_layernorm/SkipLayerNorm_add_1795"}));
  env.set("_2151", builder["sqrt"](env.get("_2150"), {"label":"/model/layers.13/input_layernorm/SkipLayerNorm_sqrt_1796"}));
  env.set("_2152", builder["div"](env.get("_2147"), env.get("_2151"), {"label":"/model/layers.13/input_layernorm/SkipLayerNorm_div_1797"}));
  env.set("_2155", builder["mul"](env.get("_2154"), env.get("_2152"), {"label":"/model/layers.13/input_layernorm/SkipLayerNorm_mul_1799"}));
  env.set("InsertedPrecisionFreeCast__2157", builder["matmul"](env.get("_2155"), env.get("InsertedPrecisionFreeCast__266"), {"label":"/model/layers.13/attn/v_proj/MatMul_Q4_matmul_1801"}));
  env.set("_2157", builder.cast(env.get("InsertedPrecisionFreeCast__2157"), "float16"));
  env.set("_2158", builder.reshape(env.get("_2157"), [1,128,4,64]));
  env.set("present_13_value_27", builder["scatterND"](env.get("past_key_values_13_value_2163"), env.get("Inserted_882"), env.get("_2158"), {"label":"/model/layers.13/attn/GroupQueryAttention_/GQA/present_value/ScatterND_1808"}));
  env.set("_2164", builder.reshape(env.get("present_13_value_27"), [1,4,1,512,64]));
  env.set("_2165", builder.expand(env.get("_2164"), [1,4,8,512,64], {"label":"/model/layers.13/attn/GroupQueryAttention_/GQA/true_present_value/expand_1813"}));
  env.set("_2166", builder.reshape(env.get("_2165"), [1,32,512,64]));
  env.set("InsertedPrecisionFreeCast__2166", builder.cast(env.get("_2166"), "float32"));
  env.set("Inserted_907", builder.cast(env.get("_2162"), "int64"));
  env.set("Inserted_909", builder["max"](env.get("Inserted_907"), env.get("Inserted_908"), {"label":"Inserted_Max_1848"}));
  env.set("Inserted_911", builder["min"](env.get("Inserted_909"), env.get("Inserted_910"), {"label":"Inserted_Min_1849"}));
  env.set("Inserted_900", builder["clamp"](env.get("position_ids_625"), {"label":"Inserted_Clip_1839","maxValue":2047,"minValue":-2048}));
  env.set("_2192", builder["gather"](env.get("_644"), env.get("Inserted_900"), {"axis":0,"label":"/model/layers.13/attn/k_rotary/RotaryEmbedding_gather_cos_1838"}));
  env.set("_2193", builder.reshape(env.get("_2192"), [1,128,1,1,32]));
  env.set("InsertedPrecisionFreeCast__2193", builder.cast(env.get("_2193"), "float32"));
  env.set("_2179", builder.dequantizeLinear(env.get("_2176"), env.get("_2177"), env.get("_2178"), {"axis":2,"blockSize":32,"label":"/model/layers.13/attn/k_proj/MatMul_Q4_dequantizeLinear_1827"}));
  env.set("_2180", builder.reshape(env.get("_2179"), [256,2048]));
  env.set("_2181", builder["transpose"](env.get("_2180"), {"label":"/model/layers.13/attn/k_proj/MatMul_Q4_transpose_dequantizeLinear_1829","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__2181", builder.cast(env.get("_2181"), "float32"));
  env.set("InsertedPrecisionFreeCast__2182", builder["matmul"](env.get("_2155"), env.get("InsertedPrecisionFreeCast__2181"), {"label":"/model/layers.13/attn/k_proj/MatMul_Q4_matmul_1830"}));
  env.set("_2182", builder.cast(env.get("InsertedPrecisionFreeCast__2182"), "float16"));
  env.set("_2183", builder.reshape(env.get("_2182"), [1,128,4,64]));
  env.set("_2184", builder.reshape(env.get("_2183"), [1,128,4,2,32]));
  env.set("InsertedPrecisionFreeCast__2184", builder.cast(env.get("_2184"), "float32"));
  env.set("InsertedPrecisionFreeCast__2194", builder["mul"](env.get("InsertedPrecisionFreeCast__2184"), env.get("InsertedPrecisionFreeCast__2193"), {"label":"/model/layers.13/attn/k_rotary/RotaryEmbedding_mul_cos_1841"}));
  env.set("InsertedPrecisionFreeCast__2195", builder.reshape(env.get("InsertedPrecisionFreeCast__2194"), [1,128,4,64]));
  env.set("InsertedPrecisionFreeCast__2189", builder.cast(env.get("_2189"), "float32"));
  env.set("Inserted_891", builder["clamp"](env.get("position_ids_625"), {"label":"Inserted_Clip_1825","maxValue":2047,"minValue":-2048}));
  env.set("_2174", builder["gather"](env.get("_624"), env.get("Inserted_891"), {"axis":0,"label":"/model/layers.13/attn/k_rotary/RotaryEmbedding_gather_sin_1824"}));
  env.set("_2175", builder.reshape(env.get("_2174"), [1,128,1,1,32]));
  env.set("InsertedPrecisionFreeCast__2175", builder.cast(env.get("_2175"), "float32"));
  {
    const tmp = builder.split(env.get("_2184"), 2, {"axis":3,"label":"/model/layers.13/attn/k_rotary/RotaryEmbedding_split_partial_input0_1833"});
    env.set("_2185", tmp[0]);
    env.set("_2186", tmp[1]);
  }
  env.set("_2187", builder.concat([env.get("_2186"), env.get("_2185")], 3, {"label":"/model/layers.13/attn/k_rotary/RotaryEmbedding_concat_partial_input0_1834"}));
  env.set("InsertedPrecisionFreeCast__2187", builder.cast(env.get("_2187"), "float32"));
  env.set("InsertedPrecisionFreeCast__2188", builder["mul"](env.get("InsertedPrecisionFreeCast__2187"), env.get("InsertedPrecisionFreeCast__2175"), {"label":"/model/layers.13/attn/k_rotary/RotaryEmbedding_mul_sin_1835"}));
  env.set("InsertedPrecisionFreeCast__2190", builder["mul"](env.get("InsertedPrecisionFreeCast__2188"), env.get("InsertedPrecisionFreeCast__2189"), {"label":"/model/layers.13/attn/k_rotary/RotaryEmbedding_mul_sign_1836"}));
  env.set("InsertedPrecisionFreeCast__2191", builder.reshape(env.get("InsertedPrecisionFreeCast__2190"), [1,128,4,64]));
  env.set("InsertedPrecisionFreeCast__2196", builder["add"](env.get("InsertedPrecisionFreeCast__2195"), env.get("InsertedPrecisionFreeCast__2191"), {"label":"/model/layers.13/attn/k_rotary/RotaryEmbedding_add_mul_cos_sin_1843"}));
  env.set("_2196", builder.cast(env.get("InsertedPrecisionFreeCast__2196"), "float16"));
  env.set("_2197", builder.reshape(env.get("_2196"), [1,128,256]));
  env.set("_2198", builder.reshape(env.get("_2197"), [1,128,4,64]));
  env.set("present_13_key_26", builder["scatterND"](env.get("past_key_values_13_key_2199"), env.get("Inserted_911"), env.get("_2198"), {"label":"/model/layers.13/attn/GroupQueryAttention_/GQA/present_key/ScatterND_1846"}));
  env.set("_2200", builder.reshape(env.get("present_13_key_26"), [1,4,1,512,64]));
  env.set("_2201", builder.expand(env.get("_2200"), [1,4,8,512,64], {"label":"/model/layers.13/attn/GroupQueryAttention_/GQA/true_present_key/expand_1851"}));
  env.set("_2202", builder.reshape(env.get("_2201"), [1,32,512,64]));
  env.set("_2203", builder["transpose"](env.get("_2202"), {"label":"/model/layers.13/attn/GroupQueryAttention_/GQA/present_key/transpose_1853","permutation":[0,1,3,2]}));
  env.set("InsertedPrecisionFreeCast__2203", builder.cast(env.get("_2203"), "float32"));
  env.set("Inserted_924", builder["clamp"](env.get("position_ids_625"), {"label":"Inserted_Clip_1869","maxValue":2047,"minValue":-2048}));
  env.set("_2222", builder["gather"](env.get("_644"), env.get("Inserted_924"), {"axis":0,"label":"/model/layers.13/attn/q_rotary/RotaryEmbedding_gather_cos_1868"}));
  env.set("_2223", builder.reshape(env.get("_2222"), [1,128,1,1,32]));
  env.set("InsertedPrecisionFreeCast__2223", builder.cast(env.get("_2223"), "float32"));
  env.set("_2209", builder.dequantizeLinear(env.get("_2206"), env.get("_2207"), env.get("_2208"), {"axis":2,"blockSize":32,"label":"/model/layers.13/attn/q_proj/MatMul_Q4_dequantizeLinear_1857"}));
  env.set("_2210", builder.reshape(env.get("_2209"), [2048,2048]));
  env.set("_2211", builder["transpose"](env.get("_2210"), {"label":"/model/layers.13/attn/q_proj/MatMul_Q4_transpose_dequantizeLinear_1859","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__2211", builder.cast(env.get("_2211"), "float32"));
  env.set("InsertedPrecisionFreeCast__2212", builder["matmul"](env.get("_2155"), env.get("InsertedPrecisionFreeCast__2211"), {"label":"/model/layers.13/attn/q_proj/MatMul_Q4_matmul_1860"}));
  env.set("_2212", builder.cast(env.get("InsertedPrecisionFreeCast__2212"), "float16"));
  env.set("_2213", builder.reshape(env.get("_2212"), [1,128,32,64]));
  env.set("_2214", builder.reshape(env.get("_2213"), [1,128,32,2,32]));
  env.set("InsertedPrecisionFreeCast__2214", builder.cast(env.get("_2214"), "float32"));
  env.set("InsertedPrecisionFreeCast__2224", builder["mul"](env.get("InsertedPrecisionFreeCast__2214"), env.get("InsertedPrecisionFreeCast__2223"), {"label":"/model/layers.13/attn/q_rotary/RotaryEmbedding_mul_cos_1871"}));
  env.set("InsertedPrecisionFreeCast__2225", builder.reshape(env.get("InsertedPrecisionFreeCast__2224"), [1,128,32,64]));
  env.set("InsertedPrecisionFreeCast__2219", builder.cast(env.get("_2219"), "float32"));
  env.set("Inserted_915", builder["clamp"](env.get("position_ids_625"), {"label":"Inserted_Clip_1855","maxValue":2047,"minValue":-2048}));
  env.set("_2204", builder["gather"](env.get("_624"), env.get("Inserted_915"), {"axis":0,"label":"/model/layers.13/attn/q_rotary/RotaryEmbedding_gather_sin_1854"}));
  env.set("_2205", builder.reshape(env.get("_2204"), [1,128,1,1,32]));
  env.set("InsertedPrecisionFreeCast__2205", builder.cast(env.get("_2205"), "float32"));
  {
    const tmp = builder.split(env.get("_2214"), 2, {"axis":3,"label":"/model/layers.13/attn/q_rotary/RotaryEmbedding_split_partial_input0_1863"});
    env.set("_2215", tmp[0]);
    env.set("_2216", tmp[1]);
  }
  env.set("_2217", builder.concat([env.get("_2216"), env.get("_2215")], 3, {"label":"/model/layers.13/attn/q_rotary/RotaryEmbedding_concat_partial_input0_1864"}));
  env.set("InsertedPrecisionFreeCast__2217", builder.cast(env.get("_2217"), "float32"));
  env.set("InsertedPrecisionFreeCast__2218", builder["mul"](env.get("InsertedPrecisionFreeCast__2217"), env.get("InsertedPrecisionFreeCast__2205"), {"label":"/model/layers.13/attn/q_rotary/RotaryEmbedding_mul_sin_1865"}));
  env.set("InsertedPrecisionFreeCast__2220", builder["mul"](env.get("InsertedPrecisionFreeCast__2218"), env.get("InsertedPrecisionFreeCast__2219"), {"label":"/model/layers.13/attn/q_rotary/RotaryEmbedding_mul_sign_1866"}));
  env.set("InsertedPrecisionFreeCast__2221", builder.reshape(env.get("InsertedPrecisionFreeCast__2220"), [1,128,32,64]));
  env.set("InsertedPrecisionFreeCast__2226", builder["add"](env.get("InsertedPrecisionFreeCast__2225"), env.get("InsertedPrecisionFreeCast__2221"), {"label":"/model/layers.13/attn/q_rotary/RotaryEmbedding_add_mul_cos_sin_1873"}));
  env.set("_2226", builder.cast(env.get("InsertedPrecisionFreeCast__2226"), "float16"));
  env.set("_2227", builder.reshape(env.get("_2226"), [1,128,2048]));
  env.set("_2228", builder.reshape(env.get("_2227"), [1,128,32,64]));
  env.set("_2229", builder["transpose"](env.get("_2228"), {"label":"/model/layers.13/attn/GroupQueryAttention_/GQA/query/transpose_1876","permutation":[0,2,1,3]}));
  env.set("InsertedPrecisionFreeCast__2229", builder.cast(env.get("_2229"), "float32"));
  env.set("InsertedPrecisionFreeCast__2230", builder["matmul"](env.get("InsertedPrecisionFreeCast__2229"), env.get("InsertedPrecisionFreeCast__2203"), {"label":"/model/layers.13/attn/GroupQueryAttention_/Attention/qkv/matmul_1_1877"}));
  env.set("InsertedPrecisionFreeCast__2231", builder["mul"](env.get("InsertedPrecisionFreeCast__2230"), env.get("InsertedPrecisionFreeCast__684"), {"label":"/model/layers.13/attn/GroupQueryAttention_/Attention/qkv/div_1878"}));
  env.set("_2170", builder.expand(env.get("_617"), [1,32,128,512], {"label":"/model/layers.13/attn/GroupQueryAttention_/GQA/GQA_mask_shape_ones/expand_1818"}));
  env.set("_2171", builder.cumulativeSum(env.get("_2170"), 3, {"exclusive":true,"label":"/model/layers.13/attn/GroupQueryAttention_range_of_mask_shape_1819"}));
  env.set("_2167", builder["add"](env.get("_613"), env.get("_2159"), {"label":"/model/layers.13/attn/GroupQueryAttention_/GQA/attn_mask/add_1815"}));
  env.set("_2168", builder.expand(env.get("_2167"), [512,128], {"label":"/model/layers.13/attn/GroupQueryAttention_/GQA/expand_neq_right_1816"}));
  env.set("_2169", builder["transpose"](env.get("_2168"), {"label":"/model/layers.13/attn/GroupQueryAttention_/GQA/neq_right/transpose_1817","permutation":[1,0]}));
  env.set("Inserted_889", builder["lesser"](env.get("_2171"), env.get("_2169"), {"label":"/model/layers.13/attn/GroupQueryAttention_/GQA/attn_mask/condition_1_1820"}));
  env.set("InsertedPrecisionFreeCast__2173", builder["where"](env.get("Inserted_889"), env.get("InsertedPrecisionFreeCast__621"), env.get("InsertedPrecisionFreeCast__622"), {"label":"/model/layers.13/attn/GroupQueryAttention_/GQA/attn_mask/where_1822"}));
  env.set("InsertedPrecisionFreeCast__2232", builder["add"](env.get("InsertedPrecisionFreeCast__2231"), env.get("InsertedPrecisionFreeCast__2173"), {"label":"/model/layers.13/attn/GroupQueryAttention_/Attention/attn_mask/softmax_input_1879"}));
  env.set("InsertedPrecisionFreeCast__inlfunc_Softmax_token_12_X_ReduceMax", builder["reduceMax"](env.get("InsertedPrecisionFreeCast__2232"), {"axes":[3],"keepDimensions":true}));
  env.set("InsertedPrecisionFreeCast__inlfunc_Softmax_token_12_X_Sub", builder["sub"](env.get("InsertedPrecisionFreeCast__2232"), env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_12_X_ReduceMax"), {}));
  env.set("InsertedPrecisionFreeCast__inlfunc_Softmax_token_12_X_Exp", builder["exp"](env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_12_X_Sub"), {}));
  env.set("InsertedPrecisionFreeCast__inlfunc_Softmax_token_12_X_ReduceSum", builder["reduceSum"](env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_12_X_Exp"), {"axes":[3],"keepDimensions":true}));
  env.set("InsertedPrecisionFreeCast__2233", builder["div"](env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_12_X_Exp"), env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_12_X_ReduceSum"), {}));
  env.set("InsertedPrecisionFreeCast__2234", builder["matmul"](env.get("InsertedPrecisionFreeCast__2233"), env.get("InsertedPrecisionFreeCast__2166"), {"label":"/model/layers.13/attn/GroupQueryAttention_/Attention/qkv/matmul_2_1881"}));
  env.set("_2234", builder.cast(env.get("InsertedPrecisionFreeCast__2234"), "float16"));
  env.set("_2235", builder["transpose"](env.get("_2234"), {"label":"/model/layers.13/attn/GroupQueryAttention_/Attention/qkv/transpose_1882","permutation":[0,2,1,3]}));
  env.set("_2236", builder.reshape(env.get("_2235"), [1,128,2048]));
  env.set("InsertedPrecisionFreeCast__2236", builder.cast(env.get("_2236"), "float32"));
  env.set("InsertedPrecisionFreeCast__2237", builder["matmul"](env.get("InsertedPrecisionFreeCast__2236"), env.get("InsertedPrecisionFreeCast__260"), {"label":"/model/layers.13/attn/o_proj/MatMul_Q4_matmul_1884"}));
  env.set("_2241", builder["add"](env.get("_2240"), env.get("InsertedPrecisionFreeCast__2237"), {"label":"/model/layers.13/post_attention_layernorm/SkipLayerNorm_add_skip_1888"}));
  env.set("_2264", builder.cast(env.get("_2241"), "float16"));
  env.set("_2265", builder.cast(env.get("_2264"), "float32"));
  env.set("_246", builder.dequantizeLinear(env.get("_243"), env.get("_244"), env.get("_245"), {"axis":2,"blockSize":32,"label":"/model/layers.13/mlp/down_proj/MatMul_Q4_dequantizeLinear_99"}));
  env.set("_247", builder.reshape(env.get("_246"), [2048,5632]));
  env.set("_248", builder["transpose"](env.get("_247"), {"label":"/model/layers.13/mlp/down_proj/MatMul_Q4_transpose_dequantizeLinear_101","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__248", builder.cast(env.get("_248"), "float32"));
  env.set("_2255", builder.dequantizeLinear(env.get("_2252"), env.get("_2253"), env.get("_2254"), {"axis":2,"blockSize":32,"label":"/model/layers.13/mlp/gate_proj/MatMul_Q4_dequantizeLinear_1898"}));
  env.set("_2256", builder.reshape(env.get("_2255"), [5632,2048]));
  env.set("_2257", builder["transpose"](env.get("_2256"), {"label":"/model/layers.13/mlp/gate_proj/MatMul_Q4_transpose_dequantizeLinear_1900","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__2257", builder.cast(env.get("_2257"), "float32"));
  env.set("_2248", builder.cast(env.get("_2247"), "float32"));
  env.set("_2242", builder["pow"](env.get("_2241"), env.get("_583"), {"label":"/model/layers.13/post_attention_layernorm/SkipLayerNorm_pow_1889"}));
  env.set("_2243", builder["reduceMean"](env.get("_2242"), {"axes":[2],"keepDimensions":true,"label":"/model/layers.13/post_attention_layernorm/SkipLayerNorm_reduceMean_1890"}));
  env.set("_2244", builder["add"](env.get("_2243"), env.get("_586"), {"label":"/model/layers.13/post_attention_layernorm/SkipLayerNorm_add_1891"}));
  env.set("_2245", builder["sqrt"](env.get("_2244"), {"label":"/model/layers.13/post_attention_layernorm/SkipLayerNorm_sqrt_1892"}));
  env.set("_2246", builder["div"](env.get("_2241"), env.get("_2245"), {"label":"/model/layers.13/post_attention_layernorm/SkipLayerNorm_div_1893"}));
  env.set("_2249", builder["mul"](env.get("_2248"), env.get("_2246"), {"label":"/model/layers.13/post_attention_layernorm/SkipLayerNorm_mul_1895"}));
  env.set("InsertedPrecisionFreeCast__2258", builder["matmul"](env.get("_2249"), env.get("InsertedPrecisionFreeCast__2257"), {"label":"/model/layers.13/mlp/gate_proj/MatMul_Q4_matmul_1901"}));
  env.set("InsertedPrecisionFreeCast__2259", builder["sigmoid"](env.get("InsertedPrecisionFreeCast__2258"), {"label":"/model/layers.13/mlp/act_fn/Sigmoid_1902"}));
  env.set("InsertedPrecisionFreeCast__2260", builder["mul"](env.get("InsertedPrecisionFreeCast__2258"), env.get("InsertedPrecisionFreeCast__2259"), {"label":"/model/layers.13/mlp/act_fn/Mul_1903"}));
  env.set("_252", builder.dequantizeLinear(env.get("_249"), env.get("_250"), env.get("_251"), {"axis":2,"blockSize":32,"label":"/model/layers.13/mlp/up_proj/MatMul_Q4_dequantizeLinear_102"}));
  env.set("_253", builder.reshape(env.get("_252"), [5632,2048]));
  env.set("_254", builder["transpose"](env.get("_253"), {"label":"/model/layers.13/mlp/up_proj/MatMul_Q4_transpose_dequantizeLinear_104","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__254", builder.cast(env.get("_254"), "float32"));
  env.set("InsertedPrecisionFreeCast__2251", builder["matmul"](env.get("_2249"), env.get("InsertedPrecisionFreeCast__254"), {"label":"/model/layers.13/mlp/up_proj/MatMul_Q4_matmul_1897"}));
  env.set("InsertedPrecisionFreeCast__2261", builder["mul"](env.get("InsertedPrecisionFreeCast__2260"), env.get("InsertedPrecisionFreeCast__2251"), {"label":"/model/layers.13/mlp/Mul_1904"}));
  env.set("InsertedPrecisionFreeCast__2262", builder["matmul"](env.get("InsertedPrecisionFreeCast__2261"), env.get("InsertedPrecisionFreeCast__248"), {"label":"/model/layers.13/mlp/down_proj/MatMul_Q4_matmul_1905"}));
  env.set("_2266", builder["add"](env.get("_2265"), env.get("InsertedPrecisionFreeCast__2262"), {"label":"/model/layers.14/input_layernorm/SkipLayerNorm_add_skip_1909"}));
  env.set("_2358", builder.cast(env.get("_2266"), "float16"));
  env.set("_2359", builder.cast(env.get("_2358"), "float32"));
  env.set("_234", builder.dequantizeLinear(env.get("_231"), env.get("_232"), env.get("_233"), {"axis":2,"blockSize":32,"label":"/model/layers.14/attn/o_proj/MatMul_Q4_dequantizeLinear_93"}));
  env.set("_235", builder.reshape(env.get("_234"), [2048,2048]));
  env.set("_236", builder["transpose"](env.get("_235"), {"label":"/model/layers.14/attn/o_proj/MatMul_Q4_transpose_dequantizeLinear_95","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__236", builder.cast(env.get("_236"), "float32"));
  env.set("Inserted_936", builder.cast(env.get("_601"), "uint8"));
  env.set("_2278", builder["where"](env.get("Inserted_936"), env.get("_602"), env.get("_600"), {"label":"/model/layers.14/attn/GroupQueryAttention_/GQA/scatter/where_1920"}));
  env.set("_2279", builder["add"](env.get("_604"), env.get("_2278"), {"label":"/model/layers.14/attn/GroupQueryAttention_/GQA/right_constant/add_1922"}));
  env.set("_2280", builder.concat([env.get("_606"), env.get("_2279")], 1, {"label":"/model/layers.14/attn/GroupQueryAttention_/GQA/concat_for_pre_scatter_indices_1923"}));
  env.set("_2281", builder.reshape(env.get("_2280"), [1,128,4,3]));
  env.set("Inserted_938", builder.cast(env.get("_2281"), "int64"));
  env.set("Inserted_940", builder["max"](env.get("Inserted_938"), env.get("Inserted_939"), {"label":"Inserted_Max_1927"}));
  env.set("Inserted_942", builder["min"](env.get("Inserted_940"), env.get("Inserted_941"), {"label":"Inserted_Min_1928"}));
  env.set("_240", builder.dequantizeLinear(env.get("_237"), env.get("_238"), env.get("_239"), {"axis":2,"blockSize":32,"label":"/model/layers.14/attn/v_proj/MatMul_Q4_dequantizeLinear_96"}));
  env.set("_241", builder.reshape(env.get("_240"), [256,2048]));
  env.set("_242", builder["transpose"](env.get("_241"), {"label":"/model/layers.14/attn/v_proj/MatMul_Q4_transpose_dequantizeLinear_98","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__242", builder.cast(env.get("_242"), "float32"));
  env.set("_2273", builder.cast(env.get("_2272"), "float32"));
  env.set("_2267", builder["pow"](env.get("_2266"), env.get("_583"), {"label":"/model/layers.14/input_layernorm/SkipLayerNorm_pow_1910"}));
  env.set("_2268", builder["reduceMean"](env.get("_2267"), {"axes":[2],"keepDimensions":true,"label":"/model/layers.14/input_layernorm/SkipLayerNorm_reduceMean_1911"}));
  env.set("_2269", builder["add"](env.get("_2268"), env.get("_586"), {"label":"/model/layers.14/input_layernorm/SkipLayerNorm_add_1912"}));
  env.set("_2270", builder["sqrt"](env.get("_2269"), {"label":"/model/layers.14/input_layernorm/SkipLayerNorm_sqrt_1913"}));
  env.set("_2271", builder["div"](env.get("_2266"), env.get("_2270"), {"label":"/model/layers.14/input_layernorm/SkipLayerNorm_div_1914"}));
  env.set("_2274", builder["mul"](env.get("_2273"), env.get("_2271"), {"label":"/model/layers.14/input_layernorm/SkipLayerNorm_mul_1916"}));
  env.set("InsertedPrecisionFreeCast__2276", builder["matmul"](env.get("_2274"), env.get("InsertedPrecisionFreeCast__242"), {"label":"/model/layers.14/attn/v_proj/MatMul_Q4_matmul_1918"}));
  env.set("_2276", builder.cast(env.get("InsertedPrecisionFreeCast__2276"), "float16"));
  env.set("_2277", builder.reshape(env.get("_2276"), [1,128,4,64]));
  env.set("present_14_value_29", builder["scatterND"](env.get("past_key_values_14_value_2282"), env.get("Inserted_942"), env.get("_2277"), {"label":"/model/layers.14/attn/GroupQueryAttention_/GQA/present_value/ScatterND_1925"}));
  env.set("_2283", builder.reshape(env.get("present_14_value_29"), [1,4,1,512,64]));
  env.set("_2284", builder.expand(env.get("_2283"), [1,4,8,512,64], {"label":"/model/layers.14/attn/GroupQueryAttention_/GQA/true_present_value/expand_1930"}));
  env.set("_2285", builder.reshape(env.get("_2284"), [1,32,512,64]));
  env.set("InsertedPrecisionFreeCast__2285", builder.cast(env.get("_2285"), "float32"));
  env.set("Inserted_967", builder.cast(env.get("_2281"), "int64"));
  env.set("Inserted_969", builder["max"](env.get("Inserted_967"), env.get("Inserted_968"), {"label":"Inserted_Max_1965"}));
  env.set("Inserted_971", builder["min"](env.get("Inserted_969"), env.get("Inserted_970"), {"label":"Inserted_Min_1966"}));
  env.set("Inserted_960", builder["clamp"](env.get("position_ids_625"), {"label":"Inserted_Clip_1956","maxValue":2047,"minValue":-2048}));
  env.set("_2311", builder["gather"](env.get("_644"), env.get("Inserted_960"), {"axis":0,"label":"/model/layers.14/attn/k_rotary/RotaryEmbedding_gather_cos_1955"}));
  env.set("_2312", builder.reshape(env.get("_2311"), [1,128,1,1,32]));
  env.set("InsertedPrecisionFreeCast__2312", builder.cast(env.get("_2312"), "float32"));
  env.set("_2298", builder.dequantizeLinear(env.get("_2295"), env.get("_2296"), env.get("_2297"), {"axis":2,"blockSize":32,"label":"/model/layers.14/attn/k_proj/MatMul_Q4_dequantizeLinear_1944"}));
  env.set("_2299", builder.reshape(env.get("_2298"), [256,2048]));
  env.set("_2300", builder["transpose"](env.get("_2299"), {"label":"/model/layers.14/attn/k_proj/MatMul_Q4_transpose_dequantizeLinear_1946","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__2300", builder.cast(env.get("_2300"), "float32"));
  env.set("InsertedPrecisionFreeCast__2301", builder["matmul"](env.get("_2274"), env.get("InsertedPrecisionFreeCast__2300"), {"label":"/model/layers.14/attn/k_proj/MatMul_Q4_matmul_1947"}));
  env.set("_2301", builder.cast(env.get("InsertedPrecisionFreeCast__2301"), "float16"));
  env.set("_2302", builder.reshape(env.get("_2301"), [1,128,4,64]));
  env.set("_2303", builder.reshape(env.get("_2302"), [1,128,4,2,32]));
  env.set("InsertedPrecisionFreeCast__2303", builder.cast(env.get("_2303"), "float32"));
  env.set("InsertedPrecisionFreeCast__2313", builder["mul"](env.get("InsertedPrecisionFreeCast__2303"), env.get("InsertedPrecisionFreeCast__2312"), {"label":"/model/layers.14/attn/k_rotary/RotaryEmbedding_mul_cos_1958"}));
  env.set("InsertedPrecisionFreeCast__2314", builder.reshape(env.get("InsertedPrecisionFreeCast__2313"), [1,128,4,64]));
  env.set("InsertedPrecisionFreeCast__2308", builder.cast(env.get("_2308"), "float32"));
  env.set("Inserted_951", builder["clamp"](env.get("position_ids_625"), {"label":"Inserted_Clip_1942","maxValue":2047,"minValue":-2048}));
  env.set("_2293", builder["gather"](env.get("_624"), env.get("Inserted_951"), {"axis":0,"label":"/model/layers.14/attn/k_rotary/RotaryEmbedding_gather_sin_1941"}));
  env.set("_2294", builder.reshape(env.get("_2293"), [1,128,1,1,32]));
  env.set("InsertedPrecisionFreeCast__2294", builder.cast(env.get("_2294"), "float32"));
  {
    const tmp = builder.split(env.get("_2303"), 2, {"axis":3,"label":"/model/layers.14/attn/k_rotary/RotaryEmbedding_split_partial_input0_1950"});
    env.set("_2304", tmp[0]);
    env.set("_2305", tmp[1]);
  }
  env.set("_2306", builder.concat([env.get("_2305"), env.get("_2304")], 3, {"label":"/model/layers.14/attn/k_rotary/RotaryEmbedding_concat_partial_input0_1951"}));
  env.set("InsertedPrecisionFreeCast__2306", builder.cast(env.get("_2306"), "float32"));
  env.set("InsertedPrecisionFreeCast__2307", builder["mul"](env.get("InsertedPrecisionFreeCast__2306"), env.get("InsertedPrecisionFreeCast__2294"), {"label":"/model/layers.14/attn/k_rotary/RotaryEmbedding_mul_sin_1952"}));
  env.set("InsertedPrecisionFreeCast__2309", builder["mul"](env.get("InsertedPrecisionFreeCast__2307"), env.get("InsertedPrecisionFreeCast__2308"), {"label":"/model/layers.14/attn/k_rotary/RotaryEmbedding_mul_sign_1953"}));
  env.set("InsertedPrecisionFreeCast__2310", builder.reshape(env.get("InsertedPrecisionFreeCast__2309"), [1,128,4,64]));
  env.set("InsertedPrecisionFreeCast__2315", builder["add"](env.get("InsertedPrecisionFreeCast__2314"), env.get("InsertedPrecisionFreeCast__2310"), {"label":"/model/layers.14/attn/k_rotary/RotaryEmbedding_add_mul_cos_sin_1960"}));
  env.set("_2315", builder.cast(env.get("InsertedPrecisionFreeCast__2315"), "float16"));
  env.set("_2316", builder.reshape(env.get("_2315"), [1,128,256]));
  env.set("_2317", builder.reshape(env.get("_2316"), [1,128,4,64]));
  env.set("present_14_key_28", builder["scatterND"](env.get("past_key_values_14_key_2318"), env.get("Inserted_971"), env.get("_2317"), {"label":"/model/layers.14/attn/GroupQueryAttention_/GQA/present_key/ScatterND_1963"}));
  env.set("_2319", builder.reshape(env.get("present_14_key_28"), [1,4,1,512,64]));
  env.set("_2320", builder.expand(env.get("_2319"), [1,4,8,512,64], {"label":"/model/layers.14/attn/GroupQueryAttention_/GQA/true_present_key/expand_1968"}));
  env.set("_2321", builder.reshape(env.get("_2320"), [1,32,512,64]));
  env.set("_2322", builder["transpose"](env.get("_2321"), {"label":"/model/layers.14/attn/GroupQueryAttention_/GQA/present_key/transpose_1970","permutation":[0,1,3,2]}));
  env.set("InsertedPrecisionFreeCast__2322", builder.cast(env.get("_2322"), "float32"));
  env.set("Inserted_984", builder["clamp"](env.get("position_ids_625"), {"label":"Inserted_Clip_1986","maxValue":2047,"minValue":-2048}));
  env.set("_2341", builder["gather"](env.get("_644"), env.get("Inserted_984"), {"axis":0,"label":"/model/layers.14/attn/q_rotary/RotaryEmbedding_gather_cos_1985"}));
  env.set("_2342", builder.reshape(env.get("_2341"), [1,128,1,1,32]));
  env.set("InsertedPrecisionFreeCast__2342", builder.cast(env.get("_2342"), "float32"));
  env.set("_2328", builder.dequantizeLinear(env.get("_2325"), env.get("_2326"), env.get("_2327"), {"axis":2,"blockSize":32,"label":"/model/layers.14/attn/q_proj/MatMul_Q4_dequantizeLinear_1974"}));
  env.set("_2329", builder.reshape(env.get("_2328"), [2048,2048]));
  env.set("_2330", builder["transpose"](env.get("_2329"), {"label":"/model/layers.14/attn/q_proj/MatMul_Q4_transpose_dequantizeLinear_1976","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__2330", builder.cast(env.get("_2330"), "float32"));
  env.set("InsertedPrecisionFreeCast__2331", builder["matmul"](env.get("_2274"), env.get("InsertedPrecisionFreeCast__2330"), {"label":"/model/layers.14/attn/q_proj/MatMul_Q4_matmul_1977"}));
  env.set("_2331", builder.cast(env.get("InsertedPrecisionFreeCast__2331"), "float16"));
  env.set("_2332", builder.reshape(env.get("_2331"), [1,128,32,64]));
  env.set("_2333", builder.reshape(env.get("_2332"), [1,128,32,2,32]));
  env.set("InsertedPrecisionFreeCast__2333", builder.cast(env.get("_2333"), "float32"));
  env.set("InsertedPrecisionFreeCast__2343", builder["mul"](env.get("InsertedPrecisionFreeCast__2333"), env.get("InsertedPrecisionFreeCast__2342"), {"label":"/model/layers.14/attn/q_rotary/RotaryEmbedding_mul_cos_1988"}));
  env.set("InsertedPrecisionFreeCast__2344", builder.reshape(env.get("InsertedPrecisionFreeCast__2343"), [1,128,32,64]));
  env.set("InsertedPrecisionFreeCast__2338", builder.cast(env.get("_2338"), "float32"));
  env.set("Inserted_975", builder["clamp"](env.get("position_ids_625"), {"label":"Inserted_Clip_1972","maxValue":2047,"minValue":-2048}));
  env.set("_2323", builder["gather"](env.get("_624"), env.get("Inserted_975"), {"axis":0,"label":"/model/layers.14/attn/q_rotary/RotaryEmbedding_gather_sin_1971"}));
  env.set("_2324", builder.reshape(env.get("_2323"), [1,128,1,1,32]));
  env.set("InsertedPrecisionFreeCast__2324", builder.cast(env.get("_2324"), "float32"));
  {
    const tmp = builder.split(env.get("_2333"), 2, {"axis":3,"label":"/model/layers.14/attn/q_rotary/RotaryEmbedding_split_partial_input0_1980"});
    env.set("_2334", tmp[0]);
    env.set("_2335", tmp[1]);
  }
  env.set("_2336", builder.concat([env.get("_2335"), env.get("_2334")], 3, {"label":"/model/layers.14/attn/q_rotary/RotaryEmbedding_concat_partial_input0_1981"}));
  env.set("InsertedPrecisionFreeCast__2336", builder.cast(env.get("_2336"), "float32"));
  env.set("InsertedPrecisionFreeCast__2337", builder["mul"](env.get("InsertedPrecisionFreeCast__2336"), env.get("InsertedPrecisionFreeCast__2324"), {"label":"/model/layers.14/attn/q_rotary/RotaryEmbedding_mul_sin_1982"}));
  env.set("InsertedPrecisionFreeCast__2339", builder["mul"](env.get("InsertedPrecisionFreeCast__2337"), env.get("InsertedPrecisionFreeCast__2338"), {"label":"/model/layers.14/attn/q_rotary/RotaryEmbedding_mul_sign_1983"}));
  env.set("InsertedPrecisionFreeCast__2340", builder.reshape(env.get("InsertedPrecisionFreeCast__2339"), [1,128,32,64]));
  env.set("InsertedPrecisionFreeCast__2345", builder["add"](env.get("InsertedPrecisionFreeCast__2344"), env.get("InsertedPrecisionFreeCast__2340"), {"label":"/model/layers.14/attn/q_rotary/RotaryEmbedding_add_mul_cos_sin_1990"}));
  env.set("_2345", builder.cast(env.get("InsertedPrecisionFreeCast__2345"), "float16"));
  env.set("_2346", builder.reshape(env.get("_2345"), [1,128,2048]));
  env.set("_2347", builder.reshape(env.get("_2346"), [1,128,32,64]));
  env.set("_2348", builder["transpose"](env.get("_2347"), {"label":"/model/layers.14/attn/GroupQueryAttention_/GQA/query/transpose_1993","permutation":[0,2,1,3]}));
  env.set("InsertedPrecisionFreeCast__2348", builder.cast(env.get("_2348"), "float32"));
  env.set("InsertedPrecisionFreeCast__2349", builder["matmul"](env.get("InsertedPrecisionFreeCast__2348"), env.get("InsertedPrecisionFreeCast__2322"), {"label":"/model/layers.14/attn/GroupQueryAttention_/Attention/qkv/matmul_1_1994"}));
  env.set("InsertedPrecisionFreeCast__2350", builder["mul"](env.get("InsertedPrecisionFreeCast__2349"), env.get("InsertedPrecisionFreeCast__684"), {"label":"/model/layers.14/attn/GroupQueryAttention_/Attention/qkv/div_1995"}));
  env.set("_2289", builder.expand(env.get("_617"), [1,32,128,512], {"label":"/model/layers.14/attn/GroupQueryAttention_/GQA/GQA_mask_shape_ones/expand_1935"}));
  env.set("_2290", builder.cumulativeSum(env.get("_2289"), 3, {"exclusive":true,"label":"/model/layers.14/attn/GroupQueryAttention_range_of_mask_shape_1936"}));
  env.set("_2286", builder["add"](env.get("_613"), env.get("_2278"), {"label":"/model/layers.14/attn/GroupQueryAttention_/GQA/attn_mask/add_1932"}));
  env.set("_2287", builder.expand(env.get("_2286"), [512,128], {"label":"/model/layers.14/attn/GroupQueryAttention_/GQA/expand_neq_right_1933"}));
  env.set("_2288", builder["transpose"](env.get("_2287"), {"label":"/model/layers.14/attn/GroupQueryAttention_/GQA/neq_right/transpose_1934","permutation":[1,0]}));
  env.set("Inserted_949", builder["lesser"](env.get("_2290"), env.get("_2288"), {"label":"/model/layers.14/attn/GroupQueryAttention_/GQA/attn_mask/condition_1_1937"}));
  env.set("InsertedPrecisionFreeCast__2292", builder["where"](env.get("Inserted_949"), env.get("InsertedPrecisionFreeCast__621"), env.get("InsertedPrecisionFreeCast__622"), {"label":"/model/layers.14/attn/GroupQueryAttention_/GQA/attn_mask/where_1939"}));
  env.set("InsertedPrecisionFreeCast__2351", builder["add"](env.get("InsertedPrecisionFreeCast__2350"), env.get("InsertedPrecisionFreeCast__2292"), {"label":"/model/layers.14/attn/GroupQueryAttention_/Attention/attn_mask/softmax_input_1996"}));
  env.set("InsertedPrecisionFreeCast__inlfunc_Softmax_token_13_X_ReduceMax", builder["reduceMax"](env.get("InsertedPrecisionFreeCast__2351"), {"axes":[3],"keepDimensions":true}));
  env.set("InsertedPrecisionFreeCast__inlfunc_Softmax_token_13_X_Sub", builder["sub"](env.get("InsertedPrecisionFreeCast__2351"), env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_13_X_ReduceMax"), {}));
  env.set("InsertedPrecisionFreeCast__inlfunc_Softmax_token_13_X_Exp", builder["exp"](env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_13_X_Sub"), {}));
  env.set("InsertedPrecisionFreeCast__inlfunc_Softmax_token_13_X_ReduceSum", builder["reduceSum"](env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_13_X_Exp"), {"axes":[3],"keepDimensions":true}));
  env.set("InsertedPrecisionFreeCast__2352", builder["div"](env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_13_X_Exp"), env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_13_X_ReduceSum"), {}));
  env.set("InsertedPrecisionFreeCast__2353", builder["matmul"](env.get("InsertedPrecisionFreeCast__2352"), env.get("InsertedPrecisionFreeCast__2285"), {"label":"/model/layers.14/attn/GroupQueryAttention_/Attention/qkv/matmul_2_1998"}));
  env.set("_2353", builder.cast(env.get("InsertedPrecisionFreeCast__2353"), "float16"));
  env.set("_2354", builder["transpose"](env.get("_2353"), {"label":"/model/layers.14/attn/GroupQueryAttention_/Attention/qkv/transpose_1999","permutation":[0,2,1,3]}));
  env.set("_2355", builder.reshape(env.get("_2354"), [1,128,2048]));
  env.set("InsertedPrecisionFreeCast__2355", builder.cast(env.get("_2355"), "float32"));
  env.set("InsertedPrecisionFreeCast__2356", builder["matmul"](env.get("InsertedPrecisionFreeCast__2355"), env.get("InsertedPrecisionFreeCast__236"), {"label":"/model/layers.14/attn/o_proj/MatMul_Q4_matmul_2001"}));
  env.set("_2360", builder["add"](env.get("_2359"), env.get("InsertedPrecisionFreeCast__2356"), {"label":"/model/layers.14/post_attention_layernorm/SkipLayerNorm_add_skip_2005"}));
  env.set("_2383", builder.cast(env.get("_2360"), "float16"));
  env.set("_2384", builder.cast(env.get("_2383"), "float32"));
  env.set("_222", builder.dequantizeLinear(env.get("_219"), env.get("_220"), env.get("_221"), {"axis":2,"blockSize":32,"label":"/model/layers.14/mlp/down_proj/MatMul_Q4_dequantizeLinear_87"}));
  env.set("_223", builder.reshape(env.get("_222"), [2048,5632]));
  env.set("_224", builder["transpose"](env.get("_223"), {"label":"/model/layers.14/mlp/down_proj/MatMul_Q4_transpose_dequantizeLinear_89","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__224", builder.cast(env.get("_224"), "float32"));
  env.set("_2374", builder.dequantizeLinear(env.get("_2371"), env.get("_2372"), env.get("_2373"), {"axis":2,"blockSize":32,"label":"/model/layers.14/mlp/gate_proj/MatMul_Q4_dequantizeLinear_2015"}));
  env.set("_2375", builder.reshape(env.get("_2374"), [5632,2048]));
  env.set("_2376", builder["transpose"](env.get("_2375"), {"label":"/model/layers.14/mlp/gate_proj/MatMul_Q4_transpose_dequantizeLinear_2017","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__2376", builder.cast(env.get("_2376"), "float32"));
  env.set("_2367", builder.cast(env.get("_2366"), "float32"));
  env.set("_2361", builder["pow"](env.get("_2360"), env.get("_583"), {"label":"/model/layers.14/post_attention_layernorm/SkipLayerNorm_pow_2006"}));
  env.set("_2362", builder["reduceMean"](env.get("_2361"), {"axes":[2],"keepDimensions":true,"label":"/model/layers.14/post_attention_layernorm/SkipLayerNorm_reduceMean_2007"}));
  env.set("_2363", builder["add"](env.get("_2362"), env.get("_586"), {"label":"/model/layers.14/post_attention_layernorm/SkipLayerNorm_add_2008"}));
  env.set("_2364", builder["sqrt"](env.get("_2363"), {"label":"/model/layers.14/post_attention_layernorm/SkipLayerNorm_sqrt_2009"}));
  env.set("_2365", builder["div"](env.get("_2360"), env.get("_2364"), {"label":"/model/layers.14/post_attention_layernorm/SkipLayerNorm_div_2010"}));
  env.set("_2368", builder["mul"](env.get("_2367"), env.get("_2365"), {"label":"/model/layers.14/post_attention_layernorm/SkipLayerNorm_mul_2012"}));
  env.set("InsertedPrecisionFreeCast__2377", builder["matmul"](env.get("_2368"), env.get("InsertedPrecisionFreeCast__2376"), {"label":"/model/layers.14/mlp/gate_proj/MatMul_Q4_matmul_2018"}));
  env.set("InsertedPrecisionFreeCast__2378", builder["sigmoid"](env.get("InsertedPrecisionFreeCast__2377"), {"label":"/model/layers.14/mlp/act_fn/Sigmoid_2019"}));
  env.set("InsertedPrecisionFreeCast__2379", builder["mul"](env.get("InsertedPrecisionFreeCast__2377"), env.get("InsertedPrecisionFreeCast__2378"), {"label":"/model/layers.14/mlp/act_fn/Mul_2020"}));
  env.set("_228", builder.dequantizeLinear(env.get("_225"), env.get("_226"), env.get("_227"), {"axis":2,"blockSize":32,"label":"/model/layers.14/mlp/up_proj/MatMul_Q4_dequantizeLinear_90"}));
  env.set("_229", builder.reshape(env.get("_228"), [5632,2048]));
  env.set("_230", builder["transpose"](env.get("_229"), {"label":"/model/layers.14/mlp/up_proj/MatMul_Q4_transpose_dequantizeLinear_92","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__230", builder.cast(env.get("_230"), "float32"));
  env.set("InsertedPrecisionFreeCast__2370", builder["matmul"](env.get("_2368"), env.get("InsertedPrecisionFreeCast__230"), {"label":"/model/layers.14/mlp/up_proj/MatMul_Q4_matmul_2014"}));
  env.set("InsertedPrecisionFreeCast__2380", builder["mul"](env.get("InsertedPrecisionFreeCast__2379"), env.get("InsertedPrecisionFreeCast__2370"), {"label":"/model/layers.14/mlp/Mul_2021"}));
  env.set("InsertedPrecisionFreeCast__2381", builder["matmul"](env.get("InsertedPrecisionFreeCast__2380"), env.get("InsertedPrecisionFreeCast__224"), {"label":"/model/layers.14/mlp/down_proj/MatMul_Q4_matmul_2022"}));
  env.set("_2385", builder["add"](env.get("_2384"), env.get("InsertedPrecisionFreeCast__2381"), {"label":"/model/layers.15/input_layernorm/SkipLayerNorm_add_skip_2026"}));
  env.set("_2477", builder.cast(env.get("_2385"), "float16"));
  env.set("_2478", builder.cast(env.get("_2477"), "float32"));
  env.set("_210", builder.dequantizeLinear(env.get("_207"), env.get("_208"), env.get("_209"), {"axis":2,"blockSize":32,"label":"/model/layers.15/attn/o_proj/MatMul_Q4_dequantizeLinear_81"}));
  env.set("_211", builder.reshape(env.get("_210"), [2048,2048]));
  env.set("_212", builder["transpose"](env.get("_211"), {"label":"/model/layers.15/attn/o_proj/MatMul_Q4_transpose_dequantizeLinear_83","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__212", builder.cast(env.get("_212"), "float32"));
  env.set("Inserted_996", builder.cast(env.get("_601"), "uint8"));
  env.set("_2397", builder["where"](env.get("Inserted_996"), env.get("_602"), env.get("_600"), {"label":"/model/layers.15/attn/GroupQueryAttention_/GQA/scatter/where_2037"}));
  env.set("_2398", builder["add"](env.get("_604"), env.get("_2397"), {"label":"/model/layers.15/attn/GroupQueryAttention_/GQA/right_constant/add_2039"}));
  env.set("_2399", builder.concat([env.get("_606"), env.get("_2398")], 1, {"label":"/model/layers.15/attn/GroupQueryAttention_/GQA/concat_for_pre_scatter_indices_2040"}));
  env.set("_2400", builder.reshape(env.get("_2399"), [1,128,4,3]));
  env.set("Inserted_998", builder.cast(env.get("_2400"), "int64"));
  env.set("Inserted_1000", builder["max"](env.get("Inserted_998"), env.get("Inserted_999"), {"label":"Inserted_Max_2044"}));
  env.set("Inserted_1002", builder["min"](env.get("Inserted_1000"), env.get("Inserted_1001"), {"label":"Inserted_Min_2045"}));
  env.set("_216", builder.dequantizeLinear(env.get("_213"), env.get("_214"), env.get("_215"), {"axis":2,"blockSize":32,"label":"/model/layers.15/attn/v_proj/MatMul_Q4_dequantizeLinear_84"}));
  env.set("_217", builder.reshape(env.get("_216"), [256,2048]));
  env.set("_218", builder["transpose"](env.get("_217"), {"label":"/model/layers.15/attn/v_proj/MatMul_Q4_transpose_dequantizeLinear_86","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__218", builder.cast(env.get("_218"), "float32"));
  env.set("_2392", builder.cast(env.get("_2391"), "float32"));
  env.set("_2386", builder["pow"](env.get("_2385"), env.get("_583"), {"label":"/model/layers.15/input_layernorm/SkipLayerNorm_pow_2027"}));
  env.set("_2387", builder["reduceMean"](env.get("_2386"), {"axes":[2],"keepDimensions":true,"label":"/model/layers.15/input_layernorm/SkipLayerNorm_reduceMean_2028"}));
  env.set("_2388", builder["add"](env.get("_2387"), env.get("_586"), {"label":"/model/layers.15/input_layernorm/SkipLayerNorm_add_2029"}));
  env.set("_2389", builder["sqrt"](env.get("_2388"), {"label":"/model/layers.15/input_layernorm/SkipLayerNorm_sqrt_2030"}));
  env.set("_2390", builder["div"](env.get("_2385"), env.get("_2389"), {"label":"/model/layers.15/input_layernorm/SkipLayerNorm_div_2031"}));
  env.set("_2393", builder["mul"](env.get("_2392"), env.get("_2390"), {"label":"/model/layers.15/input_layernorm/SkipLayerNorm_mul_2033"}));
  env.set("InsertedPrecisionFreeCast__2395", builder["matmul"](env.get("_2393"), env.get("InsertedPrecisionFreeCast__218"), {"label":"/model/layers.15/attn/v_proj/MatMul_Q4_matmul_2035"}));
  env.set("_2395", builder.cast(env.get("InsertedPrecisionFreeCast__2395"), "float16"));
  env.set("_2396", builder.reshape(env.get("_2395"), [1,128,4,64]));
  env.set("present_15_value_31", builder["scatterND"](env.get("past_key_values_15_value_2401"), env.get("Inserted_1002"), env.get("_2396"), {"label":"/model/layers.15/attn/GroupQueryAttention_/GQA/present_value/ScatterND_2042"}));
  env.set("_2402", builder.reshape(env.get("present_15_value_31"), [1,4,1,512,64]));
  env.set("_2403", builder.expand(env.get("_2402"), [1,4,8,512,64], {"label":"/model/layers.15/attn/GroupQueryAttention_/GQA/true_present_value/expand_2047"}));
  env.set("_2404", builder.reshape(env.get("_2403"), [1,32,512,64]));
  env.set("InsertedPrecisionFreeCast__2404", builder.cast(env.get("_2404"), "float32"));
  env.set("Inserted_1027", builder.cast(env.get("_2400"), "int64"));
  env.set("Inserted_1029", builder["max"](env.get("Inserted_1027"), env.get("Inserted_1028"), {"label":"Inserted_Max_2082"}));
  env.set("Inserted_1031", builder["min"](env.get("Inserted_1029"), env.get("Inserted_1030"), {"label":"Inserted_Min_2083"}));
  env.set("Inserted_1020", builder["clamp"](env.get("position_ids_625"), {"label":"Inserted_Clip_2073","maxValue":2047,"minValue":-2048}));
  env.set("_2430", builder["gather"](env.get("_644"), env.get("Inserted_1020"), {"axis":0,"label":"/model/layers.15/attn/k_rotary/RotaryEmbedding_gather_cos_2072"}));
  env.set("_2431", builder.reshape(env.get("_2430"), [1,128,1,1,32]));
  env.set("InsertedPrecisionFreeCast__2431", builder.cast(env.get("_2431"), "float32"));
  env.set("_2417", builder.dequantizeLinear(env.get("_2414"), env.get("_2415"), env.get("_2416"), {"axis":2,"blockSize":32,"label":"/model/layers.15/attn/k_proj/MatMul_Q4_dequantizeLinear_2061"}));
  env.set("_2418", builder.reshape(env.get("_2417"), [256,2048]));
  env.set("_2419", builder["transpose"](env.get("_2418"), {"label":"/model/layers.15/attn/k_proj/MatMul_Q4_transpose_dequantizeLinear_2063","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__2419", builder.cast(env.get("_2419"), "float32"));
  env.set("InsertedPrecisionFreeCast__2420", builder["matmul"](env.get("_2393"), env.get("InsertedPrecisionFreeCast__2419"), {"label":"/model/layers.15/attn/k_proj/MatMul_Q4_matmul_2064"}));
  env.set("_2420", builder.cast(env.get("InsertedPrecisionFreeCast__2420"), "float16"));
  env.set("_2421", builder.reshape(env.get("_2420"), [1,128,4,64]));
  env.set("_2422", builder.reshape(env.get("_2421"), [1,128,4,2,32]));
  env.set("InsertedPrecisionFreeCast__2422", builder.cast(env.get("_2422"), "float32"));
  env.set("InsertedPrecisionFreeCast__2432", builder["mul"](env.get("InsertedPrecisionFreeCast__2422"), env.get("InsertedPrecisionFreeCast__2431"), {"label":"/model/layers.15/attn/k_rotary/RotaryEmbedding_mul_cos_2075"}));
  env.set("InsertedPrecisionFreeCast__2433", builder.reshape(env.get("InsertedPrecisionFreeCast__2432"), [1,128,4,64]));
  env.set("InsertedPrecisionFreeCast__2427", builder.cast(env.get("_2427"), "float32"));
  env.set("Inserted_1011", builder["clamp"](env.get("position_ids_625"), {"label":"Inserted_Clip_2059","maxValue":2047,"minValue":-2048}));
  env.set("_2412", builder["gather"](env.get("_624"), env.get("Inserted_1011"), {"axis":0,"label":"/model/layers.15/attn/k_rotary/RotaryEmbedding_gather_sin_2058"}));
  env.set("_2413", builder.reshape(env.get("_2412"), [1,128,1,1,32]));
  env.set("InsertedPrecisionFreeCast__2413", builder.cast(env.get("_2413"), "float32"));
  {
    const tmp = builder.split(env.get("_2422"), 2, {"axis":3,"label":"/model/layers.15/attn/k_rotary/RotaryEmbedding_split_partial_input0_2067"});
    env.set("_2423", tmp[0]);
    env.set("_2424", tmp[1]);
  }
  env.set("_2425", builder.concat([env.get("_2424"), env.get("_2423")], 3, {"label":"/model/layers.15/attn/k_rotary/RotaryEmbedding_concat_partial_input0_2068"}));
  env.set("InsertedPrecisionFreeCast__2425", builder.cast(env.get("_2425"), "float32"));
  env.set("InsertedPrecisionFreeCast__2426", builder["mul"](env.get("InsertedPrecisionFreeCast__2425"), env.get("InsertedPrecisionFreeCast__2413"), {"label":"/model/layers.15/attn/k_rotary/RotaryEmbedding_mul_sin_2069"}));
  env.set("InsertedPrecisionFreeCast__2428", builder["mul"](env.get("InsertedPrecisionFreeCast__2426"), env.get("InsertedPrecisionFreeCast__2427"), {"label":"/model/layers.15/attn/k_rotary/RotaryEmbedding_mul_sign_2070"}));
  env.set("InsertedPrecisionFreeCast__2429", builder.reshape(env.get("InsertedPrecisionFreeCast__2428"), [1,128,4,64]));
  env.set("InsertedPrecisionFreeCast__2434", builder["add"](env.get("InsertedPrecisionFreeCast__2433"), env.get("InsertedPrecisionFreeCast__2429"), {"label":"/model/layers.15/attn/k_rotary/RotaryEmbedding_add_mul_cos_sin_2077"}));
  env.set("_2434", builder.cast(env.get("InsertedPrecisionFreeCast__2434"), "float16"));
  env.set("_2435", builder.reshape(env.get("_2434"), [1,128,256]));
  env.set("_2436", builder.reshape(env.get("_2435"), [1,128,4,64]));
  env.set("present_15_key_30", builder["scatterND"](env.get("past_key_values_15_key_2437"), env.get("Inserted_1031"), env.get("_2436"), {"label":"/model/layers.15/attn/GroupQueryAttention_/GQA/present_key/ScatterND_2080"}));
  env.set("_2438", builder.reshape(env.get("present_15_key_30"), [1,4,1,512,64]));
  env.set("_2439", builder.expand(env.get("_2438"), [1,4,8,512,64], {"label":"/model/layers.15/attn/GroupQueryAttention_/GQA/true_present_key/expand_2085"}));
  env.set("_2440", builder.reshape(env.get("_2439"), [1,32,512,64]));
  env.set("_2441", builder["transpose"](env.get("_2440"), {"label":"/model/layers.15/attn/GroupQueryAttention_/GQA/present_key/transpose_2087","permutation":[0,1,3,2]}));
  env.set("InsertedPrecisionFreeCast__2441", builder.cast(env.get("_2441"), "float32"));
  env.set("Inserted_1044", builder["clamp"](env.get("position_ids_625"), {"label":"Inserted_Clip_2103","maxValue":2047,"minValue":-2048}));
  env.set("_2460", builder["gather"](env.get("_644"), env.get("Inserted_1044"), {"axis":0,"label":"/model/layers.15/attn/q_rotary/RotaryEmbedding_gather_cos_2102"}));
  env.set("_2461", builder.reshape(env.get("_2460"), [1,128,1,1,32]));
  env.set("InsertedPrecisionFreeCast__2461", builder.cast(env.get("_2461"), "float32"));
  env.set("_2447", builder.dequantizeLinear(env.get("_2444"), env.get("_2445"), env.get("_2446"), {"axis":2,"blockSize":32,"label":"/model/layers.15/attn/q_proj/MatMul_Q4_dequantizeLinear_2091"}));
  env.set("_2448", builder.reshape(env.get("_2447"), [2048,2048]));
  env.set("_2449", builder["transpose"](env.get("_2448"), {"label":"/model/layers.15/attn/q_proj/MatMul_Q4_transpose_dequantizeLinear_2093","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__2449", builder.cast(env.get("_2449"), "float32"));
  env.set("InsertedPrecisionFreeCast__2450", builder["matmul"](env.get("_2393"), env.get("InsertedPrecisionFreeCast__2449"), {"label":"/model/layers.15/attn/q_proj/MatMul_Q4_matmul_2094"}));
  env.set("_2450", builder.cast(env.get("InsertedPrecisionFreeCast__2450"), "float16"));
  env.set("_2451", builder.reshape(env.get("_2450"), [1,128,32,64]));
  env.set("_2452", builder.reshape(env.get("_2451"), [1,128,32,2,32]));
  env.set("InsertedPrecisionFreeCast__2452", builder.cast(env.get("_2452"), "float32"));
  env.set("InsertedPrecisionFreeCast__2462", builder["mul"](env.get("InsertedPrecisionFreeCast__2452"), env.get("InsertedPrecisionFreeCast__2461"), {"label":"/model/layers.15/attn/q_rotary/RotaryEmbedding_mul_cos_2105"}));
  env.set("InsertedPrecisionFreeCast__2463", builder.reshape(env.get("InsertedPrecisionFreeCast__2462"), [1,128,32,64]));
  env.set("InsertedPrecisionFreeCast__2457", builder.cast(env.get("_2457"), "float32"));
  env.set("Inserted_1035", builder["clamp"](env.get("position_ids_625"), {"label":"Inserted_Clip_2089","maxValue":2047,"minValue":-2048}));
  env.set("_2442", builder["gather"](env.get("_624"), env.get("Inserted_1035"), {"axis":0,"label":"/model/layers.15/attn/q_rotary/RotaryEmbedding_gather_sin_2088"}));
  env.set("_2443", builder.reshape(env.get("_2442"), [1,128,1,1,32]));
  env.set("InsertedPrecisionFreeCast__2443", builder.cast(env.get("_2443"), "float32"));
  {
    const tmp = builder.split(env.get("_2452"), 2, {"axis":3,"label":"/model/layers.15/attn/q_rotary/RotaryEmbedding_split_partial_input0_2097"});
    env.set("_2453", tmp[0]);
    env.set("_2454", tmp[1]);
  }
  env.set("_2455", builder.concat([env.get("_2454"), env.get("_2453")], 3, {"label":"/model/layers.15/attn/q_rotary/RotaryEmbedding_concat_partial_input0_2098"}));
  env.set("InsertedPrecisionFreeCast__2455", builder.cast(env.get("_2455"), "float32"));
  env.set("InsertedPrecisionFreeCast__2456", builder["mul"](env.get("InsertedPrecisionFreeCast__2455"), env.get("InsertedPrecisionFreeCast__2443"), {"label":"/model/layers.15/attn/q_rotary/RotaryEmbedding_mul_sin_2099"}));
  env.set("InsertedPrecisionFreeCast__2458", builder["mul"](env.get("InsertedPrecisionFreeCast__2456"), env.get("InsertedPrecisionFreeCast__2457"), {"label":"/model/layers.15/attn/q_rotary/RotaryEmbedding_mul_sign_2100"}));
  env.set("InsertedPrecisionFreeCast__2459", builder.reshape(env.get("InsertedPrecisionFreeCast__2458"), [1,128,32,64]));
  env.set("InsertedPrecisionFreeCast__2464", builder["add"](env.get("InsertedPrecisionFreeCast__2463"), env.get("InsertedPrecisionFreeCast__2459"), {"label":"/model/layers.15/attn/q_rotary/RotaryEmbedding_add_mul_cos_sin_2107"}));
  env.set("_2464", builder.cast(env.get("InsertedPrecisionFreeCast__2464"), "float16"));
  env.set("_2465", builder.reshape(env.get("_2464"), [1,128,2048]));
  env.set("_2466", builder.reshape(env.get("_2465"), [1,128,32,64]));
  env.set("_2467", builder["transpose"](env.get("_2466"), {"label":"/model/layers.15/attn/GroupQueryAttention_/GQA/query/transpose_2110","permutation":[0,2,1,3]}));
  env.set("InsertedPrecisionFreeCast__2467", builder.cast(env.get("_2467"), "float32"));
  env.set("InsertedPrecisionFreeCast__2468", builder["matmul"](env.get("InsertedPrecisionFreeCast__2467"), env.get("InsertedPrecisionFreeCast__2441"), {"label":"/model/layers.15/attn/GroupQueryAttention_/Attention/qkv/matmul_1_2111"}));
  env.set("InsertedPrecisionFreeCast__2469", builder["mul"](env.get("InsertedPrecisionFreeCast__2468"), env.get("InsertedPrecisionFreeCast__684"), {"label":"/model/layers.15/attn/GroupQueryAttention_/Attention/qkv/div_2112"}));
  env.set("_2408", builder.expand(env.get("_617"), [1,32,128,512], {"label":"/model/layers.15/attn/GroupQueryAttention_/GQA/GQA_mask_shape_ones/expand_2052"}));
  env.set("_2409", builder.cumulativeSum(env.get("_2408"), 3, {"exclusive":true,"label":"/model/layers.15/attn/GroupQueryAttention_range_of_mask_shape_2053"}));
  env.set("_2405", builder["add"](env.get("_613"), env.get("_2397"), {"label":"/model/layers.15/attn/GroupQueryAttention_/GQA/attn_mask/add_2049"}));
  env.set("_2406", builder.expand(env.get("_2405"), [512,128], {"label":"/model/layers.15/attn/GroupQueryAttention_/GQA/expand_neq_right_2050"}));
  env.set("_2407", builder["transpose"](env.get("_2406"), {"label":"/model/layers.15/attn/GroupQueryAttention_/GQA/neq_right/transpose_2051","permutation":[1,0]}));
  env.set("Inserted_1009", builder["lesser"](env.get("_2409"), env.get("_2407"), {"label":"/model/layers.15/attn/GroupQueryAttention_/GQA/attn_mask/condition_1_2054"}));
  env.set("InsertedPrecisionFreeCast__2411", builder["where"](env.get("Inserted_1009"), env.get("InsertedPrecisionFreeCast__621"), env.get("InsertedPrecisionFreeCast__622"), {"label":"/model/layers.15/attn/GroupQueryAttention_/GQA/attn_mask/where_2056"}));
  env.set("InsertedPrecisionFreeCast__2470", builder["add"](env.get("InsertedPrecisionFreeCast__2469"), env.get("InsertedPrecisionFreeCast__2411"), {"label":"/model/layers.15/attn/GroupQueryAttention_/Attention/attn_mask/softmax_input_2113"}));
  env.set("InsertedPrecisionFreeCast__inlfunc_Softmax_token_14_X_ReduceMax", builder["reduceMax"](env.get("InsertedPrecisionFreeCast__2470"), {"axes":[3],"keepDimensions":true}));
  env.set("InsertedPrecisionFreeCast__inlfunc_Softmax_token_14_X_Sub", builder["sub"](env.get("InsertedPrecisionFreeCast__2470"), env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_14_X_ReduceMax"), {}));
  env.set("InsertedPrecisionFreeCast__inlfunc_Softmax_token_14_X_Exp", builder["exp"](env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_14_X_Sub"), {}));
  env.set("InsertedPrecisionFreeCast__inlfunc_Softmax_token_14_X_ReduceSum", builder["reduceSum"](env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_14_X_Exp"), {"axes":[3],"keepDimensions":true}));
  env.set("InsertedPrecisionFreeCast__2471", builder["div"](env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_14_X_Exp"), env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_14_X_ReduceSum"), {}));
  env.set("InsertedPrecisionFreeCast__2472", builder["matmul"](env.get("InsertedPrecisionFreeCast__2471"), env.get("InsertedPrecisionFreeCast__2404"), {"label":"/model/layers.15/attn/GroupQueryAttention_/Attention/qkv/matmul_2_2115"}));
  env.set("_2472", builder.cast(env.get("InsertedPrecisionFreeCast__2472"), "float16"));
  env.set("_2473", builder["transpose"](env.get("_2472"), {"label":"/model/layers.15/attn/GroupQueryAttention_/Attention/qkv/transpose_2116","permutation":[0,2,1,3]}));
  env.set("_2474", builder.reshape(env.get("_2473"), [1,128,2048]));
  env.set("InsertedPrecisionFreeCast__2474", builder.cast(env.get("_2474"), "float32"));
  env.set("InsertedPrecisionFreeCast__2475", builder["matmul"](env.get("InsertedPrecisionFreeCast__2474"), env.get("InsertedPrecisionFreeCast__212"), {"label":"/model/layers.15/attn/o_proj/MatMul_Q4_matmul_2118"}));
  env.set("_2479", builder["add"](env.get("_2478"), env.get("InsertedPrecisionFreeCast__2475"), {"label":"/model/layers.15/post_attention_layernorm/SkipLayerNorm_add_skip_2122"}));
  env.set("_2502", builder.cast(env.get("_2479"), "float16"));
  env.set("_2503", builder.cast(env.get("_2502"), "float32"));
  env.set("_198", builder.dequantizeLinear(env.get("_195"), env.get("_196"), env.get("_197"), {"axis":2,"blockSize":32,"label":"/model/layers.15/mlp/down_proj/MatMul_Q4_dequantizeLinear_75"}));
  env.set("_199", builder.reshape(env.get("_198"), [2048,5632]));
  env.set("_200", builder["transpose"](env.get("_199"), {"label":"/model/layers.15/mlp/down_proj/MatMul_Q4_transpose_dequantizeLinear_77","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__200", builder.cast(env.get("_200"), "float32"));
  env.set("_2493", builder.dequantizeLinear(env.get("_2490"), env.get("_2491"), env.get("_2492"), {"axis":2,"blockSize":32,"label":"/model/layers.15/mlp/gate_proj/MatMul_Q4_dequantizeLinear_2132"}));
  env.set("_2494", builder.reshape(env.get("_2493"), [5632,2048]));
  env.set("_2495", builder["transpose"](env.get("_2494"), {"label":"/model/layers.15/mlp/gate_proj/MatMul_Q4_transpose_dequantizeLinear_2134","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__2495", builder.cast(env.get("_2495"), "float32"));
  env.set("_2486", builder.cast(env.get("_2485"), "float32"));
  env.set("_2480", builder["pow"](env.get("_2479"), env.get("_583"), {"label":"/model/layers.15/post_attention_layernorm/SkipLayerNorm_pow_2123"}));
  env.set("_2481", builder["reduceMean"](env.get("_2480"), {"axes":[2],"keepDimensions":true,"label":"/model/layers.15/post_attention_layernorm/SkipLayerNorm_reduceMean_2124"}));
  env.set("_2482", builder["add"](env.get("_2481"), env.get("_586"), {"label":"/model/layers.15/post_attention_layernorm/SkipLayerNorm_add_2125"}));
  env.set("_2483", builder["sqrt"](env.get("_2482"), {"label":"/model/layers.15/post_attention_layernorm/SkipLayerNorm_sqrt_2126"}));
  env.set("_2484", builder["div"](env.get("_2479"), env.get("_2483"), {"label":"/model/layers.15/post_attention_layernorm/SkipLayerNorm_div_2127"}));
  env.set("_2487", builder["mul"](env.get("_2486"), env.get("_2484"), {"label":"/model/layers.15/post_attention_layernorm/SkipLayerNorm_mul_2129"}));
  env.set("InsertedPrecisionFreeCast__2496", builder["matmul"](env.get("_2487"), env.get("InsertedPrecisionFreeCast__2495"), {"label":"/model/layers.15/mlp/gate_proj/MatMul_Q4_matmul_2135"}));
  env.set("InsertedPrecisionFreeCast__2497", builder["sigmoid"](env.get("InsertedPrecisionFreeCast__2496"), {"label":"/model/layers.15/mlp/act_fn/Sigmoid_2136"}));
  env.set("InsertedPrecisionFreeCast__2498", builder["mul"](env.get("InsertedPrecisionFreeCast__2496"), env.get("InsertedPrecisionFreeCast__2497"), {"label":"/model/layers.15/mlp/act_fn/Mul_2137"}));
  env.set("_204", builder.dequantizeLinear(env.get("_201"), env.get("_202"), env.get("_203"), {"axis":2,"blockSize":32,"label":"/model/layers.15/mlp/up_proj/MatMul_Q4_dequantizeLinear_78"}));
  env.set("_205", builder.reshape(env.get("_204"), [5632,2048]));
  env.set("_206", builder["transpose"](env.get("_205"), {"label":"/model/layers.15/mlp/up_proj/MatMul_Q4_transpose_dequantizeLinear_80","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__206", builder.cast(env.get("_206"), "float32"));
  env.set("InsertedPrecisionFreeCast__2489", builder["matmul"](env.get("_2487"), env.get("InsertedPrecisionFreeCast__206"), {"label":"/model/layers.15/mlp/up_proj/MatMul_Q4_matmul_2131"}));
  env.set("InsertedPrecisionFreeCast__2499", builder["mul"](env.get("InsertedPrecisionFreeCast__2498"), env.get("InsertedPrecisionFreeCast__2489"), {"label":"/model/layers.15/mlp/Mul_2138"}));
  env.set("InsertedPrecisionFreeCast__2500", builder["matmul"](env.get("InsertedPrecisionFreeCast__2499"), env.get("InsertedPrecisionFreeCast__200"), {"label":"/model/layers.15/mlp/down_proj/MatMul_Q4_matmul_2139"}));
  env.set("_2504", builder["add"](env.get("_2503"), env.get("InsertedPrecisionFreeCast__2500"), {"label":"/model/layers.16/input_layernorm/SkipLayerNorm_add_skip_2143"}));
  env.set("_2596", builder.cast(env.get("_2504"), "float16"));
  env.set("_2597", builder.cast(env.get("_2596"), "float32"));
  env.set("_186", builder.dequantizeLinear(env.get("_183"), env.get("_184"), env.get("_185"), {"axis":2,"blockSize":32,"label":"/model/layers.16/attn/o_proj/MatMul_Q4_dequantizeLinear_69"}));
  env.set("_187", builder.reshape(env.get("_186"), [2048,2048]));
  env.set("_188", builder["transpose"](env.get("_187"), {"label":"/model/layers.16/attn/o_proj/MatMul_Q4_transpose_dequantizeLinear_71","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__188", builder.cast(env.get("_188"), "float32"));
  env.set("Inserted_1056", builder.cast(env.get("_601"), "uint8"));
  env.set("_2516", builder["where"](env.get("Inserted_1056"), env.get("_602"), env.get("_600"), {"label":"/model/layers.16/attn/GroupQueryAttention_/GQA/scatter/where_2154"}));
  env.set("_2517", builder["add"](env.get("_604"), env.get("_2516"), {"label":"/model/layers.16/attn/GroupQueryAttention_/GQA/right_constant/add_2156"}));
  env.set("_2518", builder.concat([env.get("_606"), env.get("_2517")], 1, {"label":"/model/layers.16/attn/GroupQueryAttention_/GQA/concat_for_pre_scatter_indices_2157"}));
  env.set("_2519", builder.reshape(env.get("_2518"), [1,128,4,3]));
  env.set("Inserted_1058", builder.cast(env.get("_2519"), "int64"));
  env.set("Inserted_1060", builder["max"](env.get("Inserted_1058"), env.get("Inserted_1059"), {"label":"Inserted_Max_2161"}));
  env.set("Inserted_1062", builder["min"](env.get("Inserted_1060"), env.get("Inserted_1061"), {"label":"Inserted_Min_2162"}));
  env.set("_192", builder.dequantizeLinear(env.get("_189"), env.get("_190"), env.get("_191"), {"axis":2,"blockSize":32,"label":"/model/layers.16/attn/v_proj/MatMul_Q4_dequantizeLinear_72"}));
  env.set("_193", builder.reshape(env.get("_192"), [256,2048]));
  env.set("_194", builder["transpose"](env.get("_193"), {"label":"/model/layers.16/attn/v_proj/MatMul_Q4_transpose_dequantizeLinear_74","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__194", builder.cast(env.get("_194"), "float32"));
  env.set("_2511", builder.cast(env.get("_2510"), "float32"));
  env.set("_2505", builder["pow"](env.get("_2504"), env.get("_583"), {"label":"/model/layers.16/input_layernorm/SkipLayerNorm_pow_2144"}));
  env.set("_2506", builder["reduceMean"](env.get("_2505"), {"axes":[2],"keepDimensions":true,"label":"/model/layers.16/input_layernorm/SkipLayerNorm_reduceMean_2145"}));
  env.set("_2507", builder["add"](env.get("_2506"), env.get("_586"), {"label":"/model/layers.16/input_layernorm/SkipLayerNorm_add_2146"}));
  env.set("_2508", builder["sqrt"](env.get("_2507"), {"label":"/model/layers.16/input_layernorm/SkipLayerNorm_sqrt_2147"}));
  env.set("_2509", builder["div"](env.get("_2504"), env.get("_2508"), {"label":"/model/layers.16/input_layernorm/SkipLayerNorm_div_2148"}));
  env.set("_2512", builder["mul"](env.get("_2511"), env.get("_2509"), {"label":"/model/layers.16/input_layernorm/SkipLayerNorm_mul_2150"}));
  env.set("InsertedPrecisionFreeCast__2514", builder["matmul"](env.get("_2512"), env.get("InsertedPrecisionFreeCast__194"), {"label":"/model/layers.16/attn/v_proj/MatMul_Q4_matmul_2152"}));
  env.set("_2514", builder.cast(env.get("InsertedPrecisionFreeCast__2514"), "float16"));
  env.set("_2515", builder.reshape(env.get("_2514"), [1,128,4,64]));
  env.set("present_16_value_33", builder["scatterND"](env.get("past_key_values_16_value_2520"), env.get("Inserted_1062"), env.get("_2515"), {"label":"/model/layers.16/attn/GroupQueryAttention_/GQA/present_value/ScatterND_2159"}));
  env.set("_2521", builder.reshape(env.get("present_16_value_33"), [1,4,1,512,64]));
  env.set("_2522", builder.expand(env.get("_2521"), [1,4,8,512,64], {"label":"/model/layers.16/attn/GroupQueryAttention_/GQA/true_present_value/expand_2164"}));
  env.set("_2523", builder.reshape(env.get("_2522"), [1,32,512,64]));
  env.set("InsertedPrecisionFreeCast__2523", builder.cast(env.get("_2523"), "float32"));
  env.set("Inserted_1087", builder.cast(env.get("_2519"), "int64"));
  env.set("Inserted_1089", builder["max"](env.get("Inserted_1087"), env.get("Inserted_1088"), {"label":"Inserted_Max_2199"}));
  env.set("Inserted_1091", builder["min"](env.get("Inserted_1089"), env.get("Inserted_1090"), {"label":"Inserted_Min_2200"}));
  env.set("Inserted_1080", builder["clamp"](env.get("position_ids_625"), {"label":"Inserted_Clip_2190","maxValue":2047,"minValue":-2048}));
  env.set("_2549", builder["gather"](env.get("_644"), env.get("Inserted_1080"), {"axis":0,"label":"/model/layers.16/attn/k_rotary/RotaryEmbedding_gather_cos_2189"}));
  env.set("_2550", builder.reshape(env.get("_2549"), [1,128,1,1,32]));
  env.set("InsertedPrecisionFreeCast__2550", builder.cast(env.get("_2550"), "float32"));
  env.set("_2536", builder.dequantizeLinear(env.get("_2533"), env.get("_2534"), env.get("_2535"), {"axis":2,"blockSize":32,"label":"/model/layers.16/attn/k_proj/MatMul_Q4_dequantizeLinear_2178"}));
  env.set("_2537", builder.reshape(env.get("_2536"), [256,2048]));
  env.set("_2538", builder["transpose"](env.get("_2537"), {"label":"/model/layers.16/attn/k_proj/MatMul_Q4_transpose_dequantizeLinear_2180","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__2538", builder.cast(env.get("_2538"), "float32"));
  env.set("InsertedPrecisionFreeCast__2539", builder["matmul"](env.get("_2512"), env.get("InsertedPrecisionFreeCast__2538"), {"label":"/model/layers.16/attn/k_proj/MatMul_Q4_matmul_2181"}));
  env.set("_2539", builder.cast(env.get("InsertedPrecisionFreeCast__2539"), "float16"));
  env.set("_2540", builder.reshape(env.get("_2539"), [1,128,4,64]));
  env.set("_2541", builder.reshape(env.get("_2540"), [1,128,4,2,32]));
  env.set("InsertedPrecisionFreeCast__2541", builder.cast(env.get("_2541"), "float32"));
  env.set("InsertedPrecisionFreeCast__2551", builder["mul"](env.get("InsertedPrecisionFreeCast__2541"), env.get("InsertedPrecisionFreeCast__2550"), {"label":"/model/layers.16/attn/k_rotary/RotaryEmbedding_mul_cos_2192"}));
  env.set("InsertedPrecisionFreeCast__2552", builder.reshape(env.get("InsertedPrecisionFreeCast__2551"), [1,128,4,64]));
  env.set("InsertedPrecisionFreeCast__2546", builder.cast(env.get("_2546"), "float32"));
  env.set("Inserted_1071", builder["clamp"](env.get("position_ids_625"), {"label":"Inserted_Clip_2176","maxValue":2047,"minValue":-2048}));
  env.set("_2531", builder["gather"](env.get("_624"), env.get("Inserted_1071"), {"axis":0,"label":"/model/layers.16/attn/k_rotary/RotaryEmbedding_gather_sin_2175"}));
  env.set("_2532", builder.reshape(env.get("_2531"), [1,128,1,1,32]));
  env.set("InsertedPrecisionFreeCast__2532", builder.cast(env.get("_2532"), "float32"));
  {
    const tmp = builder.split(env.get("_2541"), 2, {"axis":3,"label":"/model/layers.16/attn/k_rotary/RotaryEmbedding_split_partial_input0_2184"});
    env.set("_2542", tmp[0]);
    env.set("_2543", tmp[1]);
  }
  env.set("_2544", builder.concat([env.get("_2543"), env.get("_2542")], 3, {"label":"/model/layers.16/attn/k_rotary/RotaryEmbedding_concat_partial_input0_2185"}));
  env.set("InsertedPrecisionFreeCast__2544", builder.cast(env.get("_2544"), "float32"));
  env.set("InsertedPrecisionFreeCast__2545", builder["mul"](env.get("InsertedPrecisionFreeCast__2544"), env.get("InsertedPrecisionFreeCast__2532"), {"label":"/model/layers.16/attn/k_rotary/RotaryEmbedding_mul_sin_2186"}));
  env.set("InsertedPrecisionFreeCast__2547", builder["mul"](env.get("InsertedPrecisionFreeCast__2545"), env.get("InsertedPrecisionFreeCast__2546"), {"label":"/model/layers.16/attn/k_rotary/RotaryEmbedding_mul_sign_2187"}));
  env.set("InsertedPrecisionFreeCast__2548", builder.reshape(env.get("InsertedPrecisionFreeCast__2547"), [1,128,4,64]));
  env.set("InsertedPrecisionFreeCast__2553", builder["add"](env.get("InsertedPrecisionFreeCast__2552"), env.get("InsertedPrecisionFreeCast__2548"), {"label":"/model/layers.16/attn/k_rotary/RotaryEmbedding_add_mul_cos_sin_2194"}));
  env.set("_2553", builder.cast(env.get("InsertedPrecisionFreeCast__2553"), "float16"));
  env.set("_2554", builder.reshape(env.get("_2553"), [1,128,256]));
  env.set("_2555", builder.reshape(env.get("_2554"), [1,128,4,64]));
  env.set("present_16_key_32", builder["scatterND"](env.get("past_key_values_16_key_2556"), env.get("Inserted_1091"), env.get("_2555"), {"label":"/model/layers.16/attn/GroupQueryAttention_/GQA/present_key/ScatterND_2197"}));
  env.set("_2557", builder.reshape(env.get("present_16_key_32"), [1,4,1,512,64]));
  env.set("_2558", builder.expand(env.get("_2557"), [1,4,8,512,64], {"label":"/model/layers.16/attn/GroupQueryAttention_/GQA/true_present_key/expand_2202"}));
  env.set("_2559", builder.reshape(env.get("_2558"), [1,32,512,64]));
  env.set("_2560", builder["transpose"](env.get("_2559"), {"label":"/model/layers.16/attn/GroupQueryAttention_/GQA/present_key/transpose_2204","permutation":[0,1,3,2]}));
  env.set("InsertedPrecisionFreeCast__2560", builder.cast(env.get("_2560"), "float32"));
  env.set("Inserted_1104", builder["clamp"](env.get("position_ids_625"), {"label":"Inserted_Clip_2220","maxValue":2047,"minValue":-2048}));
  env.set("_2579", builder["gather"](env.get("_644"), env.get("Inserted_1104"), {"axis":0,"label":"/model/layers.16/attn/q_rotary/RotaryEmbedding_gather_cos_2219"}));
  env.set("_2580", builder.reshape(env.get("_2579"), [1,128,1,1,32]));
  env.set("InsertedPrecisionFreeCast__2580", builder.cast(env.get("_2580"), "float32"));
  env.set("_2566", builder.dequantizeLinear(env.get("_2563"), env.get("_2564"), env.get("_2565"), {"axis":2,"blockSize":32,"label":"/model/layers.16/attn/q_proj/MatMul_Q4_dequantizeLinear_2208"}));
  env.set("_2567", builder.reshape(env.get("_2566"), [2048,2048]));
  env.set("_2568", builder["transpose"](env.get("_2567"), {"label":"/model/layers.16/attn/q_proj/MatMul_Q4_transpose_dequantizeLinear_2210","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__2568", builder.cast(env.get("_2568"), "float32"));
  env.set("InsertedPrecisionFreeCast__2569", builder["matmul"](env.get("_2512"), env.get("InsertedPrecisionFreeCast__2568"), {"label":"/model/layers.16/attn/q_proj/MatMul_Q4_matmul_2211"}));
  env.set("_2569", builder.cast(env.get("InsertedPrecisionFreeCast__2569"), "float16"));
  env.set("_2570", builder.reshape(env.get("_2569"), [1,128,32,64]));
  env.set("_2571", builder.reshape(env.get("_2570"), [1,128,32,2,32]));
  env.set("InsertedPrecisionFreeCast__2571", builder.cast(env.get("_2571"), "float32"));
  env.set("InsertedPrecisionFreeCast__2581", builder["mul"](env.get("InsertedPrecisionFreeCast__2571"), env.get("InsertedPrecisionFreeCast__2580"), {"label":"/model/layers.16/attn/q_rotary/RotaryEmbedding_mul_cos_2222"}));
  env.set("InsertedPrecisionFreeCast__2582", builder.reshape(env.get("InsertedPrecisionFreeCast__2581"), [1,128,32,64]));
  env.set("InsertedPrecisionFreeCast__2576", builder.cast(env.get("_2576"), "float32"));
  env.set("Inserted_1095", builder["clamp"](env.get("position_ids_625"), {"label":"Inserted_Clip_2206","maxValue":2047,"minValue":-2048}));
  env.set("_2561", builder["gather"](env.get("_624"), env.get("Inserted_1095"), {"axis":0,"label":"/model/layers.16/attn/q_rotary/RotaryEmbedding_gather_sin_2205"}));
  env.set("_2562", builder.reshape(env.get("_2561"), [1,128,1,1,32]));
  env.set("InsertedPrecisionFreeCast__2562", builder.cast(env.get("_2562"), "float32"));
  {
    const tmp = builder.split(env.get("_2571"), 2, {"axis":3,"label":"/model/layers.16/attn/q_rotary/RotaryEmbedding_split_partial_input0_2214"});
    env.set("_2572", tmp[0]);
    env.set("_2573", tmp[1]);
  }
  env.set("_2574", builder.concat([env.get("_2573"), env.get("_2572")], 3, {"label":"/model/layers.16/attn/q_rotary/RotaryEmbedding_concat_partial_input0_2215"}));
  env.set("InsertedPrecisionFreeCast__2574", builder.cast(env.get("_2574"), "float32"));
  env.set("InsertedPrecisionFreeCast__2575", builder["mul"](env.get("InsertedPrecisionFreeCast__2574"), env.get("InsertedPrecisionFreeCast__2562"), {"label":"/model/layers.16/attn/q_rotary/RotaryEmbedding_mul_sin_2216"}));
  env.set("InsertedPrecisionFreeCast__2577", builder["mul"](env.get("InsertedPrecisionFreeCast__2575"), env.get("InsertedPrecisionFreeCast__2576"), {"label":"/model/layers.16/attn/q_rotary/RotaryEmbedding_mul_sign_2217"}));
  env.set("InsertedPrecisionFreeCast__2578", builder.reshape(env.get("InsertedPrecisionFreeCast__2577"), [1,128,32,64]));
  env.set("InsertedPrecisionFreeCast__2583", builder["add"](env.get("InsertedPrecisionFreeCast__2582"), env.get("InsertedPrecisionFreeCast__2578"), {"label":"/model/layers.16/attn/q_rotary/RotaryEmbedding_add_mul_cos_sin_2224"}));
  env.set("_2583", builder.cast(env.get("InsertedPrecisionFreeCast__2583"), "float16"));
  env.set("_2584", builder.reshape(env.get("_2583"), [1,128,2048]));
  env.set("_2585", builder.reshape(env.get("_2584"), [1,128,32,64]));
  env.set("_2586", builder["transpose"](env.get("_2585"), {"label":"/model/layers.16/attn/GroupQueryAttention_/GQA/query/transpose_2227","permutation":[0,2,1,3]}));
  env.set("InsertedPrecisionFreeCast__2586", builder.cast(env.get("_2586"), "float32"));
  env.set("InsertedPrecisionFreeCast__2587", builder["matmul"](env.get("InsertedPrecisionFreeCast__2586"), env.get("InsertedPrecisionFreeCast__2560"), {"label":"/model/layers.16/attn/GroupQueryAttention_/Attention/qkv/matmul_1_2228"}));
  env.set("InsertedPrecisionFreeCast__2588", builder["mul"](env.get("InsertedPrecisionFreeCast__2587"), env.get("InsertedPrecisionFreeCast__684"), {"label":"/model/layers.16/attn/GroupQueryAttention_/Attention/qkv/div_2229"}));
  env.set("_2527", builder.expand(env.get("_617"), [1,32,128,512], {"label":"/model/layers.16/attn/GroupQueryAttention_/GQA/GQA_mask_shape_ones/expand_2169"}));
  env.set("_2528", builder.cumulativeSum(env.get("_2527"), 3, {"exclusive":true,"label":"/model/layers.16/attn/GroupQueryAttention_range_of_mask_shape_2170"}));
  env.set("_2524", builder["add"](env.get("_613"), env.get("_2516"), {"label":"/model/layers.16/attn/GroupQueryAttention_/GQA/attn_mask/add_2166"}));
  env.set("_2525", builder.expand(env.get("_2524"), [512,128], {"label":"/model/layers.16/attn/GroupQueryAttention_/GQA/expand_neq_right_2167"}));
  env.set("_2526", builder["transpose"](env.get("_2525"), {"label":"/model/layers.16/attn/GroupQueryAttention_/GQA/neq_right/transpose_2168","permutation":[1,0]}));
  env.set("Inserted_1069", builder["lesser"](env.get("_2528"), env.get("_2526"), {"label":"/model/layers.16/attn/GroupQueryAttention_/GQA/attn_mask/condition_1_2171"}));
  env.set("InsertedPrecisionFreeCast__2530", builder["where"](env.get("Inserted_1069"), env.get("InsertedPrecisionFreeCast__621"), env.get("InsertedPrecisionFreeCast__622"), {"label":"/model/layers.16/attn/GroupQueryAttention_/GQA/attn_mask/where_2173"}));
  env.set("InsertedPrecisionFreeCast__2589", builder["add"](env.get("InsertedPrecisionFreeCast__2588"), env.get("InsertedPrecisionFreeCast__2530"), {"label":"/model/layers.16/attn/GroupQueryAttention_/Attention/attn_mask/softmax_input_2230"}));
  env.set("InsertedPrecisionFreeCast__inlfunc_Softmax_token_15_X_ReduceMax", builder["reduceMax"](env.get("InsertedPrecisionFreeCast__2589"), {"axes":[3],"keepDimensions":true}));
  env.set("InsertedPrecisionFreeCast__inlfunc_Softmax_token_15_X_Sub", builder["sub"](env.get("InsertedPrecisionFreeCast__2589"), env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_15_X_ReduceMax"), {}));
  env.set("InsertedPrecisionFreeCast__inlfunc_Softmax_token_15_X_Exp", builder["exp"](env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_15_X_Sub"), {}));
  env.set("InsertedPrecisionFreeCast__inlfunc_Softmax_token_15_X_ReduceSum", builder["reduceSum"](env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_15_X_Exp"), {"axes":[3],"keepDimensions":true}));
  env.set("InsertedPrecisionFreeCast__2590", builder["div"](env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_15_X_Exp"), env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_15_X_ReduceSum"), {}));
  env.set("InsertedPrecisionFreeCast__2591", builder["matmul"](env.get("InsertedPrecisionFreeCast__2590"), env.get("InsertedPrecisionFreeCast__2523"), {"label":"/model/layers.16/attn/GroupQueryAttention_/Attention/qkv/matmul_2_2232"}));
  env.set("_2591", builder.cast(env.get("InsertedPrecisionFreeCast__2591"), "float16"));
  env.set("_2592", builder["transpose"](env.get("_2591"), {"label":"/model/layers.16/attn/GroupQueryAttention_/Attention/qkv/transpose_2233","permutation":[0,2,1,3]}));
  env.set("_2593", builder.reshape(env.get("_2592"), [1,128,2048]));
  env.set("InsertedPrecisionFreeCast__2593", builder.cast(env.get("_2593"), "float32"));
  env.set("InsertedPrecisionFreeCast__2594", builder["matmul"](env.get("InsertedPrecisionFreeCast__2593"), env.get("InsertedPrecisionFreeCast__188"), {"label":"/model/layers.16/attn/o_proj/MatMul_Q4_matmul_2235"}));
  env.set("_2598", builder["add"](env.get("_2597"), env.get("InsertedPrecisionFreeCast__2594"), {"label":"/model/layers.16/post_attention_layernorm/SkipLayerNorm_add_skip_2239"}));
  env.set("_2621", builder.cast(env.get("_2598"), "float16"));
  env.set("_2622", builder.cast(env.get("_2621"), "float32"));
  env.set("_174", builder.dequantizeLinear(env.get("_171"), env.get("_172"), env.get("_173"), {"axis":2,"blockSize":32,"label":"/model/layers.16/mlp/down_proj/MatMul_Q4_dequantizeLinear_63"}));
  env.set("_175", builder.reshape(env.get("_174"), [2048,5632]));
  env.set("_176", builder["transpose"](env.get("_175"), {"label":"/model/layers.16/mlp/down_proj/MatMul_Q4_transpose_dequantizeLinear_65","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__176", builder.cast(env.get("_176"), "float32"));
  env.set("_2612", builder.dequantizeLinear(env.get("_2609"), env.get("_2610"), env.get("_2611"), {"axis":2,"blockSize":32,"label":"/model/layers.16/mlp/gate_proj/MatMul_Q4_dequantizeLinear_2249"}));
  env.set("_2613", builder.reshape(env.get("_2612"), [5632,2048]));
  env.set("_2614", builder["transpose"](env.get("_2613"), {"label":"/model/layers.16/mlp/gate_proj/MatMul_Q4_transpose_dequantizeLinear_2251","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__2614", builder.cast(env.get("_2614"), "float32"));
  env.set("_2605", builder.cast(env.get("_2604"), "float32"));
  env.set("_2599", builder["pow"](env.get("_2598"), env.get("_583"), {"label":"/model/layers.16/post_attention_layernorm/SkipLayerNorm_pow_2240"}));
  env.set("_2600", builder["reduceMean"](env.get("_2599"), {"axes":[2],"keepDimensions":true,"label":"/model/layers.16/post_attention_layernorm/SkipLayerNorm_reduceMean_2241"}));
  env.set("_2601", builder["add"](env.get("_2600"), env.get("_586"), {"label":"/model/layers.16/post_attention_layernorm/SkipLayerNorm_add_2242"}));
  env.set("_2602", builder["sqrt"](env.get("_2601"), {"label":"/model/layers.16/post_attention_layernorm/SkipLayerNorm_sqrt_2243"}));
  env.set("_2603", builder["div"](env.get("_2598"), env.get("_2602"), {"label":"/model/layers.16/post_attention_layernorm/SkipLayerNorm_div_2244"}));
  env.set("_2606", builder["mul"](env.get("_2605"), env.get("_2603"), {"label":"/model/layers.16/post_attention_layernorm/SkipLayerNorm_mul_2246"}));
  env.set("InsertedPrecisionFreeCast__2615", builder["matmul"](env.get("_2606"), env.get("InsertedPrecisionFreeCast__2614"), {"label":"/model/layers.16/mlp/gate_proj/MatMul_Q4_matmul_2252"}));
  env.set("InsertedPrecisionFreeCast__2616", builder["sigmoid"](env.get("InsertedPrecisionFreeCast__2615"), {"label":"/model/layers.16/mlp/act_fn/Sigmoid_2253"}));
  env.set("InsertedPrecisionFreeCast__2617", builder["mul"](env.get("InsertedPrecisionFreeCast__2615"), env.get("InsertedPrecisionFreeCast__2616"), {"label":"/model/layers.16/mlp/act_fn/Mul_2254"}));
  env.set("_180", builder.dequantizeLinear(env.get("_177"), env.get("_178"), env.get("_179"), {"axis":2,"blockSize":32,"label":"/model/layers.16/mlp/up_proj/MatMul_Q4_dequantizeLinear_66"}));
  env.set("_181", builder.reshape(env.get("_180"), [5632,2048]));
  env.set("_182", builder["transpose"](env.get("_181"), {"label":"/model/layers.16/mlp/up_proj/MatMul_Q4_transpose_dequantizeLinear_68","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__182", builder.cast(env.get("_182"), "float32"));
  env.set("InsertedPrecisionFreeCast__2608", builder["matmul"](env.get("_2606"), env.get("InsertedPrecisionFreeCast__182"), {"label":"/model/layers.16/mlp/up_proj/MatMul_Q4_matmul_2248"}));
  env.set("InsertedPrecisionFreeCast__2618", builder["mul"](env.get("InsertedPrecisionFreeCast__2617"), env.get("InsertedPrecisionFreeCast__2608"), {"label":"/model/layers.16/mlp/Mul_2255"}));
  env.set("InsertedPrecisionFreeCast__2619", builder["matmul"](env.get("InsertedPrecisionFreeCast__2618"), env.get("InsertedPrecisionFreeCast__176"), {"label":"/model/layers.16/mlp/down_proj/MatMul_Q4_matmul_2256"}));
  env.set("_2623", builder["add"](env.get("_2622"), env.get("InsertedPrecisionFreeCast__2619"), {"label":"/model/layers.17/input_layernorm/SkipLayerNorm_add_skip_2260"}));
  env.set("_2715", builder.cast(env.get("_2623"), "float16"));
  env.set("_2716", builder.cast(env.get("_2715"), "float32"));
  env.set("_162", builder.dequantizeLinear(env.get("_159"), env.get("_160"), env.get("_161"), {"axis":2,"blockSize":32,"label":"/model/layers.17/attn/o_proj/MatMul_Q4_dequantizeLinear_57"}));
  env.set("_163", builder.reshape(env.get("_162"), [2048,2048]));
  env.set("_164", builder["transpose"](env.get("_163"), {"label":"/model/layers.17/attn/o_proj/MatMul_Q4_transpose_dequantizeLinear_59","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__164", builder.cast(env.get("_164"), "float32"));
  env.set("Inserted_1116", builder.cast(env.get("_601"), "uint8"));
  env.set("_2635", builder["where"](env.get("Inserted_1116"), env.get("_602"), env.get("_600"), {"label":"/model/layers.17/attn/GroupQueryAttention_/GQA/scatter/where_2271"}));
  env.set("_2636", builder["add"](env.get("_604"), env.get("_2635"), {"label":"/model/layers.17/attn/GroupQueryAttention_/GQA/right_constant/add_2273"}));
  env.set("_2637", builder.concat([env.get("_606"), env.get("_2636")], 1, {"label":"/model/layers.17/attn/GroupQueryAttention_/GQA/concat_for_pre_scatter_indices_2274"}));
  env.set("_2638", builder.reshape(env.get("_2637"), [1,128,4,3]));
  env.set("Inserted_1118", builder.cast(env.get("_2638"), "int64"));
  env.set("Inserted_1120", builder["max"](env.get("Inserted_1118"), env.get("Inserted_1119"), {"label":"Inserted_Max_2278"}));
  env.set("Inserted_1122", builder["min"](env.get("Inserted_1120"), env.get("Inserted_1121"), {"label":"Inserted_Min_2279"}));
  env.set("_168", builder.dequantizeLinear(env.get("_165"), env.get("_166"), env.get("_167"), {"axis":2,"blockSize":32,"label":"/model/layers.17/attn/v_proj/MatMul_Q4_dequantizeLinear_60"}));
  env.set("_169", builder.reshape(env.get("_168"), [256,2048]));
  env.set("_170", builder["transpose"](env.get("_169"), {"label":"/model/layers.17/attn/v_proj/MatMul_Q4_transpose_dequantizeLinear_62","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__170", builder.cast(env.get("_170"), "float32"));
  env.set("_2630", builder.cast(env.get("_2629"), "float32"));
  env.set("_2624", builder["pow"](env.get("_2623"), env.get("_583"), {"label":"/model/layers.17/input_layernorm/SkipLayerNorm_pow_2261"}));
  env.set("_2625", builder["reduceMean"](env.get("_2624"), {"axes":[2],"keepDimensions":true,"label":"/model/layers.17/input_layernorm/SkipLayerNorm_reduceMean_2262"}));
  env.set("_2626", builder["add"](env.get("_2625"), env.get("_586"), {"label":"/model/layers.17/input_layernorm/SkipLayerNorm_add_2263"}));
  env.set("_2627", builder["sqrt"](env.get("_2626"), {"label":"/model/layers.17/input_layernorm/SkipLayerNorm_sqrt_2264"}));
  env.set("_2628", builder["div"](env.get("_2623"), env.get("_2627"), {"label":"/model/layers.17/input_layernorm/SkipLayerNorm_div_2265"}));
  env.set("_2631", builder["mul"](env.get("_2630"), env.get("_2628"), {"label":"/model/layers.17/input_layernorm/SkipLayerNorm_mul_2267"}));
  env.set("InsertedPrecisionFreeCast__2633", builder["matmul"](env.get("_2631"), env.get("InsertedPrecisionFreeCast__170"), {"label":"/model/layers.17/attn/v_proj/MatMul_Q4_matmul_2269"}));
  env.set("_2633", builder.cast(env.get("InsertedPrecisionFreeCast__2633"), "float16"));
  env.set("_2634", builder.reshape(env.get("_2633"), [1,128,4,64]));
  env.set("present_17_value_35", builder["scatterND"](env.get("past_key_values_17_value_2639"), env.get("Inserted_1122"), env.get("_2634"), {"label":"/model/layers.17/attn/GroupQueryAttention_/GQA/present_value/ScatterND_2276"}));
  env.set("_2640", builder.reshape(env.get("present_17_value_35"), [1,4,1,512,64]));
  env.set("_2641", builder.expand(env.get("_2640"), [1,4,8,512,64], {"label":"/model/layers.17/attn/GroupQueryAttention_/GQA/true_present_value/expand_2281"}));
  env.set("_2642", builder.reshape(env.get("_2641"), [1,32,512,64]));
  env.set("InsertedPrecisionFreeCast__2642", builder.cast(env.get("_2642"), "float32"));
  env.set("Inserted_1147", builder.cast(env.get("_2638"), "int64"));
  env.set("Inserted_1149", builder["max"](env.get("Inserted_1147"), env.get("Inserted_1148"), {"label":"Inserted_Max_2316"}));
  env.set("Inserted_1151", builder["min"](env.get("Inserted_1149"), env.get("Inserted_1150"), {"label":"Inserted_Min_2317"}));
  env.set("Inserted_1140", builder["clamp"](env.get("position_ids_625"), {"label":"Inserted_Clip_2307","maxValue":2047,"minValue":-2048}));
  env.set("_2668", builder["gather"](env.get("_644"), env.get("Inserted_1140"), {"axis":0,"label":"/model/layers.17/attn/k_rotary/RotaryEmbedding_gather_cos_2306"}));
  env.set("_2669", builder.reshape(env.get("_2668"), [1,128,1,1,32]));
  env.set("InsertedPrecisionFreeCast__2669", builder.cast(env.get("_2669"), "float32"));
  env.set("_2655", builder.dequantizeLinear(env.get("_2652"), env.get("_2653"), env.get("_2654"), {"axis":2,"blockSize":32,"label":"/model/layers.17/attn/k_proj/MatMul_Q4_dequantizeLinear_2295"}));
  env.set("_2656", builder.reshape(env.get("_2655"), [256,2048]));
  env.set("_2657", builder["transpose"](env.get("_2656"), {"label":"/model/layers.17/attn/k_proj/MatMul_Q4_transpose_dequantizeLinear_2297","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__2657", builder.cast(env.get("_2657"), "float32"));
  env.set("InsertedPrecisionFreeCast__2658", builder["matmul"](env.get("_2631"), env.get("InsertedPrecisionFreeCast__2657"), {"label":"/model/layers.17/attn/k_proj/MatMul_Q4_matmul_2298"}));
  env.set("_2658", builder.cast(env.get("InsertedPrecisionFreeCast__2658"), "float16"));
  env.set("_2659", builder.reshape(env.get("_2658"), [1,128,4,64]));
  env.set("_2660", builder.reshape(env.get("_2659"), [1,128,4,2,32]));
  env.set("InsertedPrecisionFreeCast__2660", builder.cast(env.get("_2660"), "float32"));
  env.set("InsertedPrecisionFreeCast__2670", builder["mul"](env.get("InsertedPrecisionFreeCast__2660"), env.get("InsertedPrecisionFreeCast__2669"), {"label":"/model/layers.17/attn/k_rotary/RotaryEmbedding_mul_cos_2309"}));
  env.set("InsertedPrecisionFreeCast__2671", builder.reshape(env.get("InsertedPrecisionFreeCast__2670"), [1,128,4,64]));
  env.set("InsertedPrecisionFreeCast__2665", builder.cast(env.get("_2665"), "float32"));
  env.set("Inserted_1131", builder["clamp"](env.get("position_ids_625"), {"label":"Inserted_Clip_2293","maxValue":2047,"minValue":-2048}));
  env.set("_2650", builder["gather"](env.get("_624"), env.get("Inserted_1131"), {"axis":0,"label":"/model/layers.17/attn/k_rotary/RotaryEmbedding_gather_sin_2292"}));
  env.set("_2651", builder.reshape(env.get("_2650"), [1,128,1,1,32]));
  env.set("InsertedPrecisionFreeCast__2651", builder.cast(env.get("_2651"), "float32"));
  {
    const tmp = builder.split(env.get("_2660"), 2, {"axis":3,"label":"/model/layers.17/attn/k_rotary/RotaryEmbedding_split_partial_input0_2301"});
    env.set("_2661", tmp[0]);
    env.set("_2662", tmp[1]);
  }
  env.set("_2663", builder.concat([env.get("_2662"), env.get("_2661")], 3, {"label":"/model/layers.17/attn/k_rotary/RotaryEmbedding_concat_partial_input0_2302"}));
  env.set("InsertedPrecisionFreeCast__2663", builder.cast(env.get("_2663"), "float32"));
  env.set("InsertedPrecisionFreeCast__2664", builder["mul"](env.get("InsertedPrecisionFreeCast__2663"), env.get("InsertedPrecisionFreeCast__2651"), {"label":"/model/layers.17/attn/k_rotary/RotaryEmbedding_mul_sin_2303"}));
  env.set("InsertedPrecisionFreeCast__2666", builder["mul"](env.get("InsertedPrecisionFreeCast__2664"), env.get("InsertedPrecisionFreeCast__2665"), {"label":"/model/layers.17/attn/k_rotary/RotaryEmbedding_mul_sign_2304"}));
  env.set("InsertedPrecisionFreeCast__2667", builder.reshape(env.get("InsertedPrecisionFreeCast__2666"), [1,128,4,64]));
  env.set("InsertedPrecisionFreeCast__2672", builder["add"](env.get("InsertedPrecisionFreeCast__2671"), env.get("InsertedPrecisionFreeCast__2667"), {"label":"/model/layers.17/attn/k_rotary/RotaryEmbedding_add_mul_cos_sin_2311"}));
  env.set("_2672", builder.cast(env.get("InsertedPrecisionFreeCast__2672"), "float16"));
  env.set("_2673", builder.reshape(env.get("_2672"), [1,128,256]));
  env.set("_2674", builder.reshape(env.get("_2673"), [1,128,4,64]));
  env.set("present_17_key_34", builder["scatterND"](env.get("past_key_values_17_key_2675"), env.get("Inserted_1151"), env.get("_2674"), {"label":"/model/layers.17/attn/GroupQueryAttention_/GQA/present_key/ScatterND_2314"}));
  env.set("_2676", builder.reshape(env.get("present_17_key_34"), [1,4,1,512,64]));
  env.set("_2677", builder.expand(env.get("_2676"), [1,4,8,512,64], {"label":"/model/layers.17/attn/GroupQueryAttention_/GQA/true_present_key/expand_2319"}));
  env.set("_2678", builder.reshape(env.get("_2677"), [1,32,512,64]));
  env.set("_2679", builder["transpose"](env.get("_2678"), {"label":"/model/layers.17/attn/GroupQueryAttention_/GQA/present_key/transpose_2321","permutation":[0,1,3,2]}));
  env.set("InsertedPrecisionFreeCast__2679", builder.cast(env.get("_2679"), "float32"));
  env.set("Inserted_1164", builder["clamp"](env.get("position_ids_625"), {"label":"Inserted_Clip_2337","maxValue":2047,"minValue":-2048}));
  env.set("_2698", builder["gather"](env.get("_644"), env.get("Inserted_1164"), {"axis":0,"label":"/model/layers.17/attn/q_rotary/RotaryEmbedding_gather_cos_2336"}));
  env.set("_2699", builder.reshape(env.get("_2698"), [1,128,1,1,32]));
  env.set("InsertedPrecisionFreeCast__2699", builder.cast(env.get("_2699"), "float32"));
  env.set("_2685", builder.dequantizeLinear(env.get("_2682"), env.get("_2683"), env.get("_2684"), {"axis":2,"blockSize":32,"label":"/model/layers.17/attn/q_proj/MatMul_Q4_dequantizeLinear_2325"}));
  env.set("_2686", builder.reshape(env.get("_2685"), [2048,2048]));
  env.set("_2687", builder["transpose"](env.get("_2686"), {"label":"/model/layers.17/attn/q_proj/MatMul_Q4_transpose_dequantizeLinear_2327","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__2687", builder.cast(env.get("_2687"), "float32"));
  env.set("InsertedPrecisionFreeCast__2688", builder["matmul"](env.get("_2631"), env.get("InsertedPrecisionFreeCast__2687"), {"label":"/model/layers.17/attn/q_proj/MatMul_Q4_matmul_2328"}));
  env.set("_2688", builder.cast(env.get("InsertedPrecisionFreeCast__2688"), "float16"));
  env.set("_2689", builder.reshape(env.get("_2688"), [1,128,32,64]));
  env.set("_2690", builder.reshape(env.get("_2689"), [1,128,32,2,32]));
  env.set("InsertedPrecisionFreeCast__2690", builder.cast(env.get("_2690"), "float32"));
  env.set("InsertedPrecisionFreeCast__2700", builder["mul"](env.get("InsertedPrecisionFreeCast__2690"), env.get("InsertedPrecisionFreeCast__2699"), {"label":"/model/layers.17/attn/q_rotary/RotaryEmbedding_mul_cos_2339"}));
  env.set("InsertedPrecisionFreeCast__2701", builder.reshape(env.get("InsertedPrecisionFreeCast__2700"), [1,128,32,64]));
  env.set("InsertedPrecisionFreeCast__2695", builder.cast(env.get("_2695"), "float32"));
  env.set("Inserted_1155", builder["clamp"](env.get("position_ids_625"), {"label":"Inserted_Clip_2323","maxValue":2047,"minValue":-2048}));
  env.set("_2680", builder["gather"](env.get("_624"), env.get("Inserted_1155"), {"axis":0,"label":"/model/layers.17/attn/q_rotary/RotaryEmbedding_gather_sin_2322"}));
  env.set("_2681", builder.reshape(env.get("_2680"), [1,128,1,1,32]));
  env.set("InsertedPrecisionFreeCast__2681", builder.cast(env.get("_2681"), "float32"));
  {
    const tmp = builder.split(env.get("_2690"), 2, {"axis":3,"label":"/model/layers.17/attn/q_rotary/RotaryEmbedding_split_partial_input0_2331"});
    env.set("_2691", tmp[0]);
    env.set("_2692", tmp[1]);
  }
  env.set("_2693", builder.concat([env.get("_2692"), env.get("_2691")], 3, {"label":"/model/layers.17/attn/q_rotary/RotaryEmbedding_concat_partial_input0_2332"}));
  env.set("InsertedPrecisionFreeCast__2693", builder.cast(env.get("_2693"), "float32"));
  env.set("InsertedPrecisionFreeCast__2694", builder["mul"](env.get("InsertedPrecisionFreeCast__2693"), env.get("InsertedPrecisionFreeCast__2681"), {"label":"/model/layers.17/attn/q_rotary/RotaryEmbedding_mul_sin_2333"}));
  env.set("InsertedPrecisionFreeCast__2696", builder["mul"](env.get("InsertedPrecisionFreeCast__2694"), env.get("InsertedPrecisionFreeCast__2695"), {"label":"/model/layers.17/attn/q_rotary/RotaryEmbedding_mul_sign_2334"}));
  env.set("InsertedPrecisionFreeCast__2697", builder.reshape(env.get("InsertedPrecisionFreeCast__2696"), [1,128,32,64]));
  env.set("InsertedPrecisionFreeCast__2702", builder["add"](env.get("InsertedPrecisionFreeCast__2701"), env.get("InsertedPrecisionFreeCast__2697"), {"label":"/model/layers.17/attn/q_rotary/RotaryEmbedding_add_mul_cos_sin_2341"}));
  env.set("_2702", builder.cast(env.get("InsertedPrecisionFreeCast__2702"), "float16"));
  env.set("_2703", builder.reshape(env.get("_2702"), [1,128,2048]));
  env.set("_2704", builder.reshape(env.get("_2703"), [1,128,32,64]));
  env.set("_2705", builder["transpose"](env.get("_2704"), {"label":"/model/layers.17/attn/GroupQueryAttention_/GQA/query/transpose_2344","permutation":[0,2,1,3]}));
  env.set("InsertedPrecisionFreeCast__2705", builder.cast(env.get("_2705"), "float32"));
  env.set("InsertedPrecisionFreeCast__2706", builder["matmul"](env.get("InsertedPrecisionFreeCast__2705"), env.get("InsertedPrecisionFreeCast__2679"), {"label":"/model/layers.17/attn/GroupQueryAttention_/Attention/qkv/matmul_1_2345"}));
  env.set("InsertedPrecisionFreeCast__2707", builder["mul"](env.get("InsertedPrecisionFreeCast__2706"), env.get("InsertedPrecisionFreeCast__684"), {"label":"/model/layers.17/attn/GroupQueryAttention_/Attention/qkv/div_2346"}));
  env.set("_2646", builder.expand(env.get("_617"), [1,32,128,512], {"label":"/model/layers.17/attn/GroupQueryAttention_/GQA/GQA_mask_shape_ones/expand_2286"}));
  env.set("_2647", builder.cumulativeSum(env.get("_2646"), 3, {"exclusive":true,"label":"/model/layers.17/attn/GroupQueryAttention_range_of_mask_shape_2287"}));
  env.set("_2643", builder["add"](env.get("_613"), env.get("_2635"), {"label":"/model/layers.17/attn/GroupQueryAttention_/GQA/attn_mask/add_2283"}));
  env.set("_2644", builder.expand(env.get("_2643"), [512,128], {"label":"/model/layers.17/attn/GroupQueryAttention_/GQA/expand_neq_right_2284"}));
  env.set("_2645", builder["transpose"](env.get("_2644"), {"label":"/model/layers.17/attn/GroupQueryAttention_/GQA/neq_right/transpose_2285","permutation":[1,0]}));
  env.set("Inserted_1129", builder["lesser"](env.get("_2647"), env.get("_2645"), {"label":"/model/layers.17/attn/GroupQueryAttention_/GQA/attn_mask/condition_1_2288"}));
  env.set("InsertedPrecisionFreeCast__2649", builder["where"](env.get("Inserted_1129"), env.get("InsertedPrecisionFreeCast__621"), env.get("InsertedPrecisionFreeCast__622"), {"label":"/model/layers.17/attn/GroupQueryAttention_/GQA/attn_mask/where_2290"}));
  env.set("InsertedPrecisionFreeCast__2708", builder["add"](env.get("InsertedPrecisionFreeCast__2707"), env.get("InsertedPrecisionFreeCast__2649"), {"label":"/model/layers.17/attn/GroupQueryAttention_/Attention/attn_mask/softmax_input_2347"}));
  env.set("InsertedPrecisionFreeCast__inlfunc_Softmax_token_16_X_ReduceMax", builder["reduceMax"](env.get("InsertedPrecisionFreeCast__2708"), {"axes":[3],"keepDimensions":true}));
  env.set("InsertedPrecisionFreeCast__inlfunc_Softmax_token_16_X_Sub", builder["sub"](env.get("InsertedPrecisionFreeCast__2708"), env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_16_X_ReduceMax"), {}));
  env.set("InsertedPrecisionFreeCast__inlfunc_Softmax_token_16_X_Exp", builder["exp"](env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_16_X_Sub"), {}));
  env.set("InsertedPrecisionFreeCast__inlfunc_Softmax_token_16_X_ReduceSum", builder["reduceSum"](env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_16_X_Exp"), {"axes":[3],"keepDimensions":true}));
  env.set("InsertedPrecisionFreeCast__2709", builder["div"](env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_16_X_Exp"), env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_16_X_ReduceSum"), {}));
  env.set("InsertedPrecisionFreeCast__2710", builder["matmul"](env.get("InsertedPrecisionFreeCast__2709"), env.get("InsertedPrecisionFreeCast__2642"), {"label":"/model/layers.17/attn/GroupQueryAttention_/Attention/qkv/matmul_2_2349"}));
  env.set("_2710", builder.cast(env.get("InsertedPrecisionFreeCast__2710"), "float16"));
  env.set("_2711", builder["transpose"](env.get("_2710"), {"label":"/model/layers.17/attn/GroupQueryAttention_/Attention/qkv/transpose_2350","permutation":[0,2,1,3]}));
  env.set("_2712", builder.reshape(env.get("_2711"), [1,128,2048]));
  env.set("InsertedPrecisionFreeCast__2712", builder.cast(env.get("_2712"), "float32"));
  env.set("InsertedPrecisionFreeCast__2713", builder["matmul"](env.get("InsertedPrecisionFreeCast__2712"), env.get("InsertedPrecisionFreeCast__164"), {"label":"/model/layers.17/attn/o_proj/MatMul_Q4_matmul_2352"}));
  env.set("_2717", builder["add"](env.get("_2716"), env.get("InsertedPrecisionFreeCast__2713"), {"label":"/model/layers.17/post_attention_layernorm/SkipLayerNorm_add_skip_2356"}));
  env.set("_2740", builder.cast(env.get("_2717"), "float16"));
  env.set("_2741", builder.cast(env.get("_2740"), "float32"));
  env.set("_150", builder.dequantizeLinear(env.get("_147"), env.get("_148"), env.get("_149"), {"axis":2,"blockSize":32,"label":"/model/layers.17/mlp/down_proj/MatMul_Q4_dequantizeLinear_51"}));
  env.set("_151", builder.reshape(env.get("_150"), [2048,5632]));
  env.set("_152", builder["transpose"](env.get("_151"), {"label":"/model/layers.17/mlp/down_proj/MatMul_Q4_transpose_dequantizeLinear_53","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__152", builder.cast(env.get("_152"), "float32"));
  env.set("_2731", builder.dequantizeLinear(env.get("_2728"), env.get("_2729"), env.get("_2730"), {"axis":2,"blockSize":32,"label":"/model/layers.17/mlp/gate_proj/MatMul_Q4_dequantizeLinear_2366"}));
  env.set("_2732", builder.reshape(env.get("_2731"), [5632,2048]));
  env.set("_2733", builder["transpose"](env.get("_2732"), {"label":"/model/layers.17/mlp/gate_proj/MatMul_Q4_transpose_dequantizeLinear_2368","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__2733", builder.cast(env.get("_2733"), "float32"));
  env.set("_2724", builder.cast(env.get("_2723"), "float32"));
  env.set("_2718", builder["pow"](env.get("_2717"), env.get("_583"), {"label":"/model/layers.17/post_attention_layernorm/SkipLayerNorm_pow_2357"}));
  env.set("_2719", builder["reduceMean"](env.get("_2718"), {"axes":[2],"keepDimensions":true,"label":"/model/layers.17/post_attention_layernorm/SkipLayerNorm_reduceMean_2358"}));
  env.set("_2720", builder["add"](env.get("_2719"), env.get("_586"), {"label":"/model/layers.17/post_attention_layernorm/SkipLayerNorm_add_2359"}));
  env.set("_2721", builder["sqrt"](env.get("_2720"), {"label":"/model/layers.17/post_attention_layernorm/SkipLayerNorm_sqrt_2360"}));
  env.set("_2722", builder["div"](env.get("_2717"), env.get("_2721"), {"label":"/model/layers.17/post_attention_layernorm/SkipLayerNorm_div_2361"}));
  env.set("_2725", builder["mul"](env.get("_2724"), env.get("_2722"), {"label":"/model/layers.17/post_attention_layernorm/SkipLayerNorm_mul_2363"}));
  env.set("InsertedPrecisionFreeCast__2734", builder["matmul"](env.get("_2725"), env.get("InsertedPrecisionFreeCast__2733"), {"label":"/model/layers.17/mlp/gate_proj/MatMul_Q4_matmul_2369"}));
  env.set("InsertedPrecisionFreeCast__2735", builder["sigmoid"](env.get("InsertedPrecisionFreeCast__2734"), {"label":"/model/layers.17/mlp/act_fn/Sigmoid_2370"}));
  env.set("InsertedPrecisionFreeCast__2736", builder["mul"](env.get("InsertedPrecisionFreeCast__2734"), env.get("InsertedPrecisionFreeCast__2735"), {"label":"/model/layers.17/mlp/act_fn/Mul_2371"}));
  env.set("_156", builder.dequantizeLinear(env.get("_153"), env.get("_154"), env.get("_155"), {"axis":2,"blockSize":32,"label":"/model/layers.17/mlp/up_proj/MatMul_Q4_dequantizeLinear_54"}));
  env.set("_157", builder.reshape(env.get("_156"), [5632,2048]));
  env.set("_158", builder["transpose"](env.get("_157"), {"label":"/model/layers.17/mlp/up_proj/MatMul_Q4_transpose_dequantizeLinear_56","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__158", builder.cast(env.get("_158"), "float32"));
  env.set("InsertedPrecisionFreeCast__2727", builder["matmul"](env.get("_2725"), env.get("InsertedPrecisionFreeCast__158"), {"label":"/model/layers.17/mlp/up_proj/MatMul_Q4_matmul_2365"}));
  env.set("InsertedPrecisionFreeCast__2737", builder["mul"](env.get("InsertedPrecisionFreeCast__2736"), env.get("InsertedPrecisionFreeCast__2727"), {"label":"/model/layers.17/mlp/Mul_2372"}));
  env.set("InsertedPrecisionFreeCast__2738", builder["matmul"](env.get("InsertedPrecisionFreeCast__2737"), env.get("InsertedPrecisionFreeCast__152"), {"label":"/model/layers.17/mlp/down_proj/MatMul_Q4_matmul_2373"}));
  env.set("_2742", builder["add"](env.get("_2741"), env.get("InsertedPrecisionFreeCast__2738"), {"label":"/model/layers.18/input_layernorm/SkipLayerNorm_add_skip_2377"}));
  env.set("_2834", builder.cast(env.get("_2742"), "float16"));
  env.set("_2835", builder.cast(env.get("_2834"), "float32"));
  env.set("_138", builder.dequantizeLinear(env.get("_135"), env.get("_136"), env.get("_137"), {"axis":2,"blockSize":32,"label":"/model/layers.18/attn/o_proj/MatMul_Q4_dequantizeLinear_45"}));
  env.set("_139", builder.reshape(env.get("_138"), [2048,2048]));
  env.set("_140", builder["transpose"](env.get("_139"), {"label":"/model/layers.18/attn/o_proj/MatMul_Q4_transpose_dequantizeLinear_47","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__140", builder.cast(env.get("_140"), "float32"));
  env.set("Inserted_1176", builder.cast(env.get("_601"), "uint8"));
  env.set("_2754", builder["where"](env.get("Inserted_1176"), env.get("_602"), env.get("_600"), {"label":"/model/layers.18/attn/GroupQueryAttention_/GQA/scatter/where_2388"}));
  env.set("_2755", builder["add"](env.get("_604"), env.get("_2754"), {"label":"/model/layers.18/attn/GroupQueryAttention_/GQA/right_constant/add_2390"}));
  env.set("_2756", builder.concat([env.get("_606"), env.get("_2755")], 1, {"label":"/model/layers.18/attn/GroupQueryAttention_/GQA/concat_for_pre_scatter_indices_2391"}));
  env.set("_2757", builder.reshape(env.get("_2756"), [1,128,4,3]));
  env.set("Inserted_1178", builder.cast(env.get("_2757"), "int64"));
  env.set("Inserted_1180", builder["max"](env.get("Inserted_1178"), env.get("Inserted_1179"), {"label":"Inserted_Max_2395"}));
  env.set("Inserted_1182", builder["min"](env.get("Inserted_1180"), env.get("Inserted_1181"), {"label":"Inserted_Min_2396"}));
  env.set("_144", builder.dequantizeLinear(env.get("_141"), env.get("_142"), env.get("_143"), {"axis":2,"blockSize":32,"label":"/model/layers.18/attn/v_proj/MatMul_Q4_dequantizeLinear_48"}));
  env.set("_145", builder.reshape(env.get("_144"), [256,2048]));
  env.set("_146", builder["transpose"](env.get("_145"), {"label":"/model/layers.18/attn/v_proj/MatMul_Q4_transpose_dequantizeLinear_50","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__146", builder.cast(env.get("_146"), "float32"));
  env.set("_2749", builder.cast(env.get("_2748"), "float32"));
  env.set("_2743", builder["pow"](env.get("_2742"), env.get("_583"), {"label":"/model/layers.18/input_layernorm/SkipLayerNorm_pow_2378"}));
  env.set("_2744", builder["reduceMean"](env.get("_2743"), {"axes":[2],"keepDimensions":true,"label":"/model/layers.18/input_layernorm/SkipLayerNorm_reduceMean_2379"}));
  env.set("_2745", builder["add"](env.get("_2744"), env.get("_586"), {"label":"/model/layers.18/input_layernorm/SkipLayerNorm_add_2380"}));
  env.set("_2746", builder["sqrt"](env.get("_2745"), {"label":"/model/layers.18/input_layernorm/SkipLayerNorm_sqrt_2381"}));
  env.set("_2747", builder["div"](env.get("_2742"), env.get("_2746"), {"label":"/model/layers.18/input_layernorm/SkipLayerNorm_div_2382"}));
  env.set("_2750", builder["mul"](env.get("_2749"), env.get("_2747"), {"label":"/model/layers.18/input_layernorm/SkipLayerNorm_mul_2384"}));
  env.set("InsertedPrecisionFreeCast__2752", builder["matmul"](env.get("_2750"), env.get("InsertedPrecisionFreeCast__146"), {"label":"/model/layers.18/attn/v_proj/MatMul_Q4_matmul_2386"}));
  env.set("_2752", builder.cast(env.get("InsertedPrecisionFreeCast__2752"), "float16"));
  env.set("_2753", builder.reshape(env.get("_2752"), [1,128,4,64]));
  env.set("present_18_value_37", builder["scatterND"](env.get("past_key_values_18_value_2758"), env.get("Inserted_1182"), env.get("_2753"), {"label":"/model/layers.18/attn/GroupQueryAttention_/GQA/present_value/ScatterND_2393"}));
  env.set("_2759", builder.reshape(env.get("present_18_value_37"), [1,4,1,512,64]));
  env.set("_2760", builder.expand(env.get("_2759"), [1,4,8,512,64], {"label":"/model/layers.18/attn/GroupQueryAttention_/GQA/true_present_value/expand_2398"}));
  env.set("_2761", builder.reshape(env.get("_2760"), [1,32,512,64]));
  env.set("InsertedPrecisionFreeCast__2761", builder.cast(env.get("_2761"), "float32"));
  env.set("Inserted_1207", builder.cast(env.get("_2757"), "int64"));
  env.set("Inserted_1209", builder["max"](env.get("Inserted_1207"), env.get("Inserted_1208"), {"label":"Inserted_Max_2433"}));
  env.set("Inserted_1211", builder["min"](env.get("Inserted_1209"), env.get("Inserted_1210"), {"label":"Inserted_Min_2434"}));
  env.set("Inserted_1200", builder["clamp"](env.get("position_ids_625"), {"label":"Inserted_Clip_2424","maxValue":2047,"minValue":-2048}));
  env.set("_2787", builder["gather"](env.get("_644"), env.get("Inserted_1200"), {"axis":0,"label":"/model/layers.18/attn/k_rotary/RotaryEmbedding_gather_cos_2423"}));
  env.set("_2788", builder.reshape(env.get("_2787"), [1,128,1,1,32]));
  env.set("InsertedPrecisionFreeCast__2788", builder.cast(env.get("_2788"), "float32"));
  env.set("_2774", builder.dequantizeLinear(env.get("_2771"), env.get("_2772"), env.get("_2773"), {"axis":2,"blockSize":32,"label":"/model/layers.18/attn/k_proj/MatMul_Q4_dequantizeLinear_2412"}));
  env.set("_2775", builder.reshape(env.get("_2774"), [256,2048]));
  env.set("_2776", builder["transpose"](env.get("_2775"), {"label":"/model/layers.18/attn/k_proj/MatMul_Q4_transpose_dequantizeLinear_2414","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__2776", builder.cast(env.get("_2776"), "float32"));
  env.set("InsertedPrecisionFreeCast__2777", builder["matmul"](env.get("_2750"), env.get("InsertedPrecisionFreeCast__2776"), {"label":"/model/layers.18/attn/k_proj/MatMul_Q4_matmul_2415"}));
  env.set("_2777", builder.cast(env.get("InsertedPrecisionFreeCast__2777"), "float16"));
  env.set("_2778", builder.reshape(env.get("_2777"), [1,128,4,64]));
  env.set("_2779", builder.reshape(env.get("_2778"), [1,128,4,2,32]));
  env.set("InsertedPrecisionFreeCast__2779", builder.cast(env.get("_2779"), "float32"));
  env.set("InsertedPrecisionFreeCast__2789", builder["mul"](env.get("InsertedPrecisionFreeCast__2779"), env.get("InsertedPrecisionFreeCast__2788"), {"label":"/model/layers.18/attn/k_rotary/RotaryEmbedding_mul_cos_2426"}));
  env.set("InsertedPrecisionFreeCast__2790", builder.reshape(env.get("InsertedPrecisionFreeCast__2789"), [1,128,4,64]));
  env.set("InsertedPrecisionFreeCast__2784", builder.cast(env.get("_2784"), "float32"));
  env.set("Inserted_1191", builder["clamp"](env.get("position_ids_625"), {"label":"Inserted_Clip_2410","maxValue":2047,"minValue":-2048}));
  env.set("_2769", builder["gather"](env.get("_624"), env.get("Inserted_1191"), {"axis":0,"label":"/model/layers.18/attn/k_rotary/RotaryEmbedding_gather_sin_2409"}));
  env.set("_2770", builder.reshape(env.get("_2769"), [1,128,1,1,32]));
  env.set("InsertedPrecisionFreeCast__2770", builder.cast(env.get("_2770"), "float32"));
  {
    const tmp = builder.split(env.get("_2779"), 2, {"axis":3,"label":"/model/layers.18/attn/k_rotary/RotaryEmbedding_split_partial_input0_2418"});
    env.set("_2780", tmp[0]);
    env.set("_2781", tmp[1]);
  }
  env.set("_2782", builder.concat([env.get("_2781"), env.get("_2780")], 3, {"label":"/model/layers.18/attn/k_rotary/RotaryEmbedding_concat_partial_input0_2419"}));
  env.set("InsertedPrecisionFreeCast__2782", builder.cast(env.get("_2782"), "float32"));
  env.set("InsertedPrecisionFreeCast__2783", builder["mul"](env.get("InsertedPrecisionFreeCast__2782"), env.get("InsertedPrecisionFreeCast__2770"), {"label":"/model/layers.18/attn/k_rotary/RotaryEmbedding_mul_sin_2420"}));
  env.set("InsertedPrecisionFreeCast__2785", builder["mul"](env.get("InsertedPrecisionFreeCast__2783"), env.get("InsertedPrecisionFreeCast__2784"), {"label":"/model/layers.18/attn/k_rotary/RotaryEmbedding_mul_sign_2421"}));
  env.set("InsertedPrecisionFreeCast__2786", builder.reshape(env.get("InsertedPrecisionFreeCast__2785"), [1,128,4,64]));
  env.set("InsertedPrecisionFreeCast__2791", builder["add"](env.get("InsertedPrecisionFreeCast__2790"), env.get("InsertedPrecisionFreeCast__2786"), {"label":"/model/layers.18/attn/k_rotary/RotaryEmbedding_add_mul_cos_sin_2428"}));
  env.set("_2791", builder.cast(env.get("InsertedPrecisionFreeCast__2791"), "float16"));
  env.set("_2792", builder.reshape(env.get("_2791"), [1,128,256]));
  env.set("_2793", builder.reshape(env.get("_2792"), [1,128,4,64]));
  env.set("present_18_key_36", builder["scatterND"](env.get("past_key_values_18_key_2794"), env.get("Inserted_1211"), env.get("_2793"), {"label":"/model/layers.18/attn/GroupQueryAttention_/GQA/present_key/ScatterND_2431"}));
  env.set("_2795", builder.reshape(env.get("present_18_key_36"), [1,4,1,512,64]));
  env.set("_2796", builder.expand(env.get("_2795"), [1,4,8,512,64], {"label":"/model/layers.18/attn/GroupQueryAttention_/GQA/true_present_key/expand_2436"}));
  env.set("_2797", builder.reshape(env.get("_2796"), [1,32,512,64]));
  env.set("_2798", builder["transpose"](env.get("_2797"), {"label":"/model/layers.18/attn/GroupQueryAttention_/GQA/present_key/transpose_2438","permutation":[0,1,3,2]}));
  env.set("InsertedPrecisionFreeCast__2798", builder.cast(env.get("_2798"), "float32"));
  env.set("Inserted_1224", builder["clamp"](env.get("position_ids_625"), {"label":"Inserted_Clip_2454","maxValue":2047,"minValue":-2048}));
  env.set("_2817", builder["gather"](env.get("_644"), env.get("Inserted_1224"), {"axis":0,"label":"/model/layers.18/attn/q_rotary/RotaryEmbedding_gather_cos_2453"}));
  env.set("_2818", builder.reshape(env.get("_2817"), [1,128,1,1,32]));
  env.set("InsertedPrecisionFreeCast__2818", builder.cast(env.get("_2818"), "float32"));
  env.set("_2804", builder.dequantizeLinear(env.get("_2801"), env.get("_2802"), env.get("_2803"), {"axis":2,"blockSize":32,"label":"/model/layers.18/attn/q_proj/MatMul_Q4_dequantizeLinear_2442"}));
  env.set("_2805", builder.reshape(env.get("_2804"), [2048,2048]));
  env.set("_2806", builder["transpose"](env.get("_2805"), {"label":"/model/layers.18/attn/q_proj/MatMul_Q4_transpose_dequantizeLinear_2444","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__2806", builder.cast(env.get("_2806"), "float32"));
  env.set("InsertedPrecisionFreeCast__2807", builder["matmul"](env.get("_2750"), env.get("InsertedPrecisionFreeCast__2806"), {"label":"/model/layers.18/attn/q_proj/MatMul_Q4_matmul_2445"}));
  env.set("_2807", builder.cast(env.get("InsertedPrecisionFreeCast__2807"), "float16"));
  env.set("_2808", builder.reshape(env.get("_2807"), [1,128,32,64]));
  env.set("_2809", builder.reshape(env.get("_2808"), [1,128,32,2,32]));
  env.set("InsertedPrecisionFreeCast__2809", builder.cast(env.get("_2809"), "float32"));
  env.set("InsertedPrecisionFreeCast__2819", builder["mul"](env.get("InsertedPrecisionFreeCast__2809"), env.get("InsertedPrecisionFreeCast__2818"), {"label":"/model/layers.18/attn/q_rotary/RotaryEmbedding_mul_cos_2456"}));
  env.set("InsertedPrecisionFreeCast__2820", builder.reshape(env.get("InsertedPrecisionFreeCast__2819"), [1,128,32,64]));
  env.set("InsertedPrecisionFreeCast__2814", builder.cast(env.get("_2814"), "float32"));
  env.set("Inserted_1215", builder["clamp"](env.get("position_ids_625"), {"label":"Inserted_Clip_2440","maxValue":2047,"minValue":-2048}));
  env.set("_2799", builder["gather"](env.get("_624"), env.get("Inserted_1215"), {"axis":0,"label":"/model/layers.18/attn/q_rotary/RotaryEmbedding_gather_sin_2439"}));
  env.set("_2800", builder.reshape(env.get("_2799"), [1,128,1,1,32]));
  env.set("InsertedPrecisionFreeCast__2800", builder.cast(env.get("_2800"), "float32"));
  {
    const tmp = builder.split(env.get("_2809"), 2, {"axis":3,"label":"/model/layers.18/attn/q_rotary/RotaryEmbedding_split_partial_input0_2448"});
    env.set("_2810", tmp[0]);
    env.set("_2811", tmp[1]);
  }
  env.set("_2812", builder.concat([env.get("_2811"), env.get("_2810")], 3, {"label":"/model/layers.18/attn/q_rotary/RotaryEmbedding_concat_partial_input0_2449"}));
  env.set("InsertedPrecisionFreeCast__2812", builder.cast(env.get("_2812"), "float32"));
  env.set("InsertedPrecisionFreeCast__2813", builder["mul"](env.get("InsertedPrecisionFreeCast__2812"), env.get("InsertedPrecisionFreeCast__2800"), {"label":"/model/layers.18/attn/q_rotary/RotaryEmbedding_mul_sin_2450"}));
  env.set("InsertedPrecisionFreeCast__2815", builder["mul"](env.get("InsertedPrecisionFreeCast__2813"), env.get("InsertedPrecisionFreeCast__2814"), {"label":"/model/layers.18/attn/q_rotary/RotaryEmbedding_mul_sign_2451"}));
  env.set("InsertedPrecisionFreeCast__2816", builder.reshape(env.get("InsertedPrecisionFreeCast__2815"), [1,128,32,64]));
  env.set("InsertedPrecisionFreeCast__2821", builder["add"](env.get("InsertedPrecisionFreeCast__2820"), env.get("InsertedPrecisionFreeCast__2816"), {"label":"/model/layers.18/attn/q_rotary/RotaryEmbedding_add_mul_cos_sin_2458"}));
  env.set("_2821", builder.cast(env.get("InsertedPrecisionFreeCast__2821"), "float16"));
  env.set("_2822", builder.reshape(env.get("_2821"), [1,128,2048]));
  env.set("_2823", builder.reshape(env.get("_2822"), [1,128,32,64]));
  env.set("_2824", builder["transpose"](env.get("_2823"), {"label":"/model/layers.18/attn/GroupQueryAttention_/GQA/query/transpose_2461","permutation":[0,2,1,3]}));
  env.set("InsertedPrecisionFreeCast__2824", builder.cast(env.get("_2824"), "float32"));
  env.set("InsertedPrecisionFreeCast__2825", builder["matmul"](env.get("InsertedPrecisionFreeCast__2824"), env.get("InsertedPrecisionFreeCast__2798"), {"label":"/model/layers.18/attn/GroupQueryAttention_/Attention/qkv/matmul_1_2462"}));
  env.set("InsertedPrecisionFreeCast__2826", builder["mul"](env.get("InsertedPrecisionFreeCast__2825"), env.get("InsertedPrecisionFreeCast__684"), {"label":"/model/layers.18/attn/GroupQueryAttention_/Attention/qkv/div_2463"}));
  env.set("_2765", builder.expand(env.get("_617"), [1,32,128,512], {"label":"/model/layers.18/attn/GroupQueryAttention_/GQA/GQA_mask_shape_ones/expand_2403"}));
  env.set("_2766", builder.cumulativeSum(env.get("_2765"), 3, {"exclusive":true,"label":"/model/layers.18/attn/GroupQueryAttention_range_of_mask_shape_2404"}));
  env.set("_2762", builder["add"](env.get("_613"), env.get("_2754"), {"label":"/model/layers.18/attn/GroupQueryAttention_/GQA/attn_mask/add_2400"}));
  env.set("_2763", builder.expand(env.get("_2762"), [512,128], {"label":"/model/layers.18/attn/GroupQueryAttention_/GQA/expand_neq_right_2401"}));
  env.set("_2764", builder["transpose"](env.get("_2763"), {"label":"/model/layers.18/attn/GroupQueryAttention_/GQA/neq_right/transpose_2402","permutation":[1,0]}));
  env.set("Inserted_1189", builder["lesser"](env.get("_2766"), env.get("_2764"), {"label":"/model/layers.18/attn/GroupQueryAttention_/GQA/attn_mask/condition_1_2405"}));
  env.set("InsertedPrecisionFreeCast__2768", builder["where"](env.get("Inserted_1189"), env.get("InsertedPrecisionFreeCast__621"), env.get("InsertedPrecisionFreeCast__622"), {"label":"/model/layers.18/attn/GroupQueryAttention_/GQA/attn_mask/where_2407"}));
  env.set("InsertedPrecisionFreeCast__2827", builder["add"](env.get("InsertedPrecisionFreeCast__2826"), env.get("InsertedPrecisionFreeCast__2768"), {"label":"/model/layers.18/attn/GroupQueryAttention_/Attention/attn_mask/softmax_input_2464"}));
  env.set("InsertedPrecisionFreeCast__inlfunc_Softmax_token_17_X_ReduceMax", builder["reduceMax"](env.get("InsertedPrecisionFreeCast__2827"), {"axes":[3],"keepDimensions":true}));
  env.set("InsertedPrecisionFreeCast__inlfunc_Softmax_token_17_X_Sub", builder["sub"](env.get("InsertedPrecisionFreeCast__2827"), env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_17_X_ReduceMax"), {}));
  env.set("InsertedPrecisionFreeCast__inlfunc_Softmax_token_17_X_Exp", builder["exp"](env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_17_X_Sub"), {}));
  env.set("InsertedPrecisionFreeCast__inlfunc_Softmax_token_17_X_ReduceSum", builder["reduceSum"](env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_17_X_Exp"), {"axes":[3],"keepDimensions":true}));
  env.set("InsertedPrecisionFreeCast__2828", builder["div"](env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_17_X_Exp"), env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_17_X_ReduceSum"), {}));
  env.set("InsertedPrecisionFreeCast__2829", builder["matmul"](env.get("InsertedPrecisionFreeCast__2828"), env.get("InsertedPrecisionFreeCast__2761"), {"label":"/model/layers.18/attn/GroupQueryAttention_/Attention/qkv/matmul_2_2466"}));
  env.set("_2829", builder.cast(env.get("InsertedPrecisionFreeCast__2829"), "float16"));
  env.set("_2830", builder["transpose"](env.get("_2829"), {"label":"/model/layers.18/attn/GroupQueryAttention_/Attention/qkv/transpose_2467","permutation":[0,2,1,3]}));
  env.set("_2831", builder.reshape(env.get("_2830"), [1,128,2048]));
  env.set("InsertedPrecisionFreeCast__2831", builder.cast(env.get("_2831"), "float32"));
  env.set("InsertedPrecisionFreeCast__2832", builder["matmul"](env.get("InsertedPrecisionFreeCast__2831"), env.get("InsertedPrecisionFreeCast__140"), {"label":"/model/layers.18/attn/o_proj/MatMul_Q4_matmul_2469"}));
  env.set("_2836", builder["add"](env.get("_2835"), env.get("InsertedPrecisionFreeCast__2832"), {"label":"/model/layers.18/post_attention_layernorm/SkipLayerNorm_add_skip_2473"}));
  env.set("_2859", builder.cast(env.get("_2836"), "float16"));
  env.set("_2860", builder.cast(env.get("_2859"), "float32"));
  env.set("_126", builder.dequantizeLinear(env.get("_123"), env.get("_124"), env.get("_125"), {"axis":2,"blockSize":32,"label":"/model/layers.18/mlp/down_proj/MatMul_Q4_dequantizeLinear_39"}));
  env.set("_127", builder.reshape(env.get("_126"), [2048,5632]));
  env.set("_128", builder["transpose"](env.get("_127"), {"label":"/model/layers.18/mlp/down_proj/MatMul_Q4_transpose_dequantizeLinear_41","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__128", builder.cast(env.get("_128"), "float32"));
  env.set("_2850", builder.dequantizeLinear(env.get("_2847"), env.get("_2848"), env.get("_2849"), {"axis":2,"blockSize":32,"label":"/model/layers.18/mlp/gate_proj/MatMul_Q4_dequantizeLinear_2483"}));
  env.set("_2851", builder.reshape(env.get("_2850"), [5632,2048]));
  env.set("_2852", builder["transpose"](env.get("_2851"), {"label":"/model/layers.18/mlp/gate_proj/MatMul_Q4_transpose_dequantizeLinear_2485","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__2852", builder.cast(env.get("_2852"), "float32"));
  env.set("_2843", builder.cast(env.get("_2842"), "float32"));
  env.set("_2837", builder["pow"](env.get("_2836"), env.get("_583"), {"label":"/model/layers.18/post_attention_layernorm/SkipLayerNorm_pow_2474"}));
  env.set("_2838", builder["reduceMean"](env.get("_2837"), {"axes":[2],"keepDimensions":true,"label":"/model/layers.18/post_attention_layernorm/SkipLayerNorm_reduceMean_2475"}));
  env.set("_2839", builder["add"](env.get("_2838"), env.get("_586"), {"label":"/model/layers.18/post_attention_layernorm/SkipLayerNorm_add_2476"}));
  env.set("_2840", builder["sqrt"](env.get("_2839"), {"label":"/model/layers.18/post_attention_layernorm/SkipLayerNorm_sqrt_2477"}));
  env.set("_2841", builder["div"](env.get("_2836"), env.get("_2840"), {"label":"/model/layers.18/post_attention_layernorm/SkipLayerNorm_div_2478"}));
  env.set("_2844", builder["mul"](env.get("_2843"), env.get("_2841"), {"label":"/model/layers.18/post_attention_layernorm/SkipLayerNorm_mul_2480"}));
  env.set("InsertedPrecisionFreeCast__2853", builder["matmul"](env.get("_2844"), env.get("InsertedPrecisionFreeCast__2852"), {"label":"/model/layers.18/mlp/gate_proj/MatMul_Q4_matmul_2486"}));
  env.set("InsertedPrecisionFreeCast__2854", builder["sigmoid"](env.get("InsertedPrecisionFreeCast__2853"), {"label":"/model/layers.18/mlp/act_fn/Sigmoid_2487"}));
  env.set("InsertedPrecisionFreeCast__2855", builder["mul"](env.get("InsertedPrecisionFreeCast__2853"), env.get("InsertedPrecisionFreeCast__2854"), {"label":"/model/layers.18/mlp/act_fn/Mul_2488"}));
  env.set("_132", builder.dequantizeLinear(env.get("_129"), env.get("_130"), env.get("_131"), {"axis":2,"blockSize":32,"label":"/model/layers.18/mlp/up_proj/MatMul_Q4_dequantizeLinear_42"}));
  env.set("_133", builder.reshape(env.get("_132"), [5632,2048]));
  env.set("_134", builder["transpose"](env.get("_133"), {"label":"/model/layers.18/mlp/up_proj/MatMul_Q4_transpose_dequantizeLinear_44","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__134", builder.cast(env.get("_134"), "float32"));
  env.set("InsertedPrecisionFreeCast__2846", builder["matmul"](env.get("_2844"), env.get("InsertedPrecisionFreeCast__134"), {"label":"/model/layers.18/mlp/up_proj/MatMul_Q4_matmul_2482"}));
  env.set("InsertedPrecisionFreeCast__2856", builder["mul"](env.get("InsertedPrecisionFreeCast__2855"), env.get("InsertedPrecisionFreeCast__2846"), {"label":"/model/layers.18/mlp/Mul_2489"}));
  env.set("InsertedPrecisionFreeCast__2857", builder["matmul"](env.get("InsertedPrecisionFreeCast__2856"), env.get("InsertedPrecisionFreeCast__128"), {"label":"/model/layers.18/mlp/down_proj/MatMul_Q4_matmul_2490"}));
  env.set("_2861", builder["add"](env.get("_2860"), env.get("InsertedPrecisionFreeCast__2857"), {"label":"/model/layers.19/input_layernorm/SkipLayerNorm_add_skip_2494"}));
  env.set("_2953", builder.cast(env.get("_2861"), "float16"));
  env.set("_2954", builder.cast(env.get("_2953"), "float32"));
  env.set("_114", builder.dequantizeLinear(env.get("_111"), env.get("_112"), env.get("_113"), {"axis":2,"blockSize":32,"label":"/model/layers.19/attn/o_proj/MatMul_Q4_dequantizeLinear_33"}));
  env.set("_115", builder.reshape(env.get("_114"), [2048,2048]));
  env.set("_116", builder["transpose"](env.get("_115"), {"label":"/model/layers.19/attn/o_proj/MatMul_Q4_transpose_dequantizeLinear_35","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__116", builder.cast(env.get("_116"), "float32"));
  env.set("Inserted_1236", builder.cast(env.get("_601"), "uint8"));
  env.set("_2873", builder["where"](env.get("Inserted_1236"), env.get("_602"), env.get("_600"), {"label":"/model/layers.19/attn/GroupQueryAttention_/GQA/scatter/where_2505"}));
  env.set("_2874", builder["add"](env.get("_604"), env.get("_2873"), {"label":"/model/layers.19/attn/GroupQueryAttention_/GQA/right_constant/add_2507"}));
  env.set("_2875", builder.concat([env.get("_606"), env.get("_2874")], 1, {"label":"/model/layers.19/attn/GroupQueryAttention_/GQA/concat_for_pre_scatter_indices_2508"}));
  env.set("_2876", builder.reshape(env.get("_2875"), [1,128,4,3]));
  env.set("Inserted_1238", builder.cast(env.get("_2876"), "int64"));
  env.set("Inserted_1240", builder["max"](env.get("Inserted_1238"), env.get("Inserted_1239"), {"label":"Inserted_Max_2512"}));
  env.set("Inserted_1242", builder["min"](env.get("Inserted_1240"), env.get("Inserted_1241"), {"label":"Inserted_Min_2513"}));
  env.set("_120", builder.dequantizeLinear(env.get("_117"), env.get("_118"), env.get("_119"), {"axis":2,"blockSize":32,"label":"/model/layers.19/attn/v_proj/MatMul_Q4_dequantizeLinear_36"}));
  env.set("_121", builder.reshape(env.get("_120"), [256,2048]));
  env.set("_122", builder["transpose"](env.get("_121"), {"label":"/model/layers.19/attn/v_proj/MatMul_Q4_transpose_dequantizeLinear_38","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__122", builder.cast(env.get("_122"), "float32"));
  env.set("_2868", builder.cast(env.get("_2867"), "float32"));
  env.set("_2862", builder["pow"](env.get("_2861"), env.get("_583"), {"label":"/model/layers.19/input_layernorm/SkipLayerNorm_pow_2495"}));
  env.set("_2863", builder["reduceMean"](env.get("_2862"), {"axes":[2],"keepDimensions":true,"label":"/model/layers.19/input_layernorm/SkipLayerNorm_reduceMean_2496"}));
  env.set("_2864", builder["add"](env.get("_2863"), env.get("_586"), {"label":"/model/layers.19/input_layernorm/SkipLayerNorm_add_2497"}));
  env.set("_2865", builder["sqrt"](env.get("_2864"), {"label":"/model/layers.19/input_layernorm/SkipLayerNorm_sqrt_2498"}));
  env.set("_2866", builder["div"](env.get("_2861"), env.get("_2865"), {"label":"/model/layers.19/input_layernorm/SkipLayerNorm_div_2499"}));
  env.set("_2869", builder["mul"](env.get("_2868"), env.get("_2866"), {"label":"/model/layers.19/input_layernorm/SkipLayerNorm_mul_2501"}));
  env.set("InsertedPrecisionFreeCast__2871", builder["matmul"](env.get("_2869"), env.get("InsertedPrecisionFreeCast__122"), {"label":"/model/layers.19/attn/v_proj/MatMul_Q4_matmul_2503"}));
  env.set("_2871", builder.cast(env.get("InsertedPrecisionFreeCast__2871"), "float16"));
  env.set("_2872", builder.reshape(env.get("_2871"), [1,128,4,64]));
  env.set("present_19_value_39", builder["scatterND"](env.get("past_key_values_19_value_2877"), env.get("Inserted_1242"), env.get("_2872"), {"label":"/model/layers.19/attn/GroupQueryAttention_/GQA/present_value/ScatterND_2510"}));
  env.set("_2878", builder.reshape(env.get("present_19_value_39"), [1,4,1,512,64]));
  env.set("_2879", builder.expand(env.get("_2878"), [1,4,8,512,64], {"label":"/model/layers.19/attn/GroupQueryAttention_/GQA/true_present_value/expand_2515"}));
  env.set("_2880", builder.reshape(env.get("_2879"), [1,32,512,64]));
  env.set("InsertedPrecisionFreeCast__2880", builder.cast(env.get("_2880"), "float32"));
  env.set("Inserted_1267", builder.cast(env.get("_2876"), "int64"));
  env.set("Inserted_1269", builder["max"](env.get("Inserted_1267"), env.get("Inserted_1268"), {"label":"Inserted_Max_2550"}));
  env.set("Inserted_1271", builder["min"](env.get("Inserted_1269"), env.get("Inserted_1270"), {"label":"Inserted_Min_2551"}));
  env.set("Inserted_1260", builder["clamp"](env.get("position_ids_625"), {"label":"Inserted_Clip_2541","maxValue":2047,"minValue":-2048}));
  env.set("_2906", builder["gather"](env.get("_644"), env.get("Inserted_1260"), {"axis":0,"label":"/model/layers.19/attn/k_rotary/RotaryEmbedding_gather_cos_2540"}));
  env.set("_2907", builder.reshape(env.get("_2906"), [1,128,1,1,32]));
  env.set("InsertedPrecisionFreeCast__2907", builder.cast(env.get("_2907"), "float32"));
  env.set("_2893", builder.dequantizeLinear(env.get("_2890"), env.get("_2891"), env.get("_2892"), {"axis":2,"blockSize":32,"label":"/model/layers.19/attn/k_proj/MatMul_Q4_dequantizeLinear_2529"}));
  env.set("_2894", builder.reshape(env.get("_2893"), [256,2048]));
  env.set("_2895", builder["transpose"](env.get("_2894"), {"label":"/model/layers.19/attn/k_proj/MatMul_Q4_transpose_dequantizeLinear_2531","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__2895", builder.cast(env.get("_2895"), "float32"));
  env.set("InsertedPrecisionFreeCast__2896", builder["matmul"](env.get("_2869"), env.get("InsertedPrecisionFreeCast__2895"), {"label":"/model/layers.19/attn/k_proj/MatMul_Q4_matmul_2532"}));
  env.set("_2896", builder.cast(env.get("InsertedPrecisionFreeCast__2896"), "float16"));
  env.set("_2897", builder.reshape(env.get("_2896"), [1,128,4,64]));
  env.set("_2898", builder.reshape(env.get("_2897"), [1,128,4,2,32]));
  env.set("InsertedPrecisionFreeCast__2898", builder.cast(env.get("_2898"), "float32"));
  env.set("InsertedPrecisionFreeCast__2908", builder["mul"](env.get("InsertedPrecisionFreeCast__2898"), env.get("InsertedPrecisionFreeCast__2907"), {"label":"/model/layers.19/attn/k_rotary/RotaryEmbedding_mul_cos_2543"}));
  env.set("InsertedPrecisionFreeCast__2909", builder.reshape(env.get("InsertedPrecisionFreeCast__2908"), [1,128,4,64]));
  env.set("InsertedPrecisionFreeCast__2903", builder.cast(env.get("_2903"), "float32"));
  env.set("Inserted_1251", builder["clamp"](env.get("position_ids_625"), {"label":"Inserted_Clip_2527","maxValue":2047,"minValue":-2048}));
  env.set("_2888", builder["gather"](env.get("_624"), env.get("Inserted_1251"), {"axis":0,"label":"/model/layers.19/attn/k_rotary/RotaryEmbedding_gather_sin_2526"}));
  env.set("_2889", builder.reshape(env.get("_2888"), [1,128,1,1,32]));
  env.set("InsertedPrecisionFreeCast__2889", builder.cast(env.get("_2889"), "float32"));
  {
    const tmp = builder.split(env.get("_2898"), 2, {"axis":3,"label":"/model/layers.19/attn/k_rotary/RotaryEmbedding_split_partial_input0_2535"});
    env.set("_2899", tmp[0]);
    env.set("_2900", tmp[1]);
  }
  env.set("_2901", builder.concat([env.get("_2900"), env.get("_2899")], 3, {"label":"/model/layers.19/attn/k_rotary/RotaryEmbedding_concat_partial_input0_2536"}));
  env.set("InsertedPrecisionFreeCast__2901", builder.cast(env.get("_2901"), "float32"));
  env.set("InsertedPrecisionFreeCast__2902", builder["mul"](env.get("InsertedPrecisionFreeCast__2901"), env.get("InsertedPrecisionFreeCast__2889"), {"label":"/model/layers.19/attn/k_rotary/RotaryEmbedding_mul_sin_2537"}));
  env.set("InsertedPrecisionFreeCast__2904", builder["mul"](env.get("InsertedPrecisionFreeCast__2902"), env.get("InsertedPrecisionFreeCast__2903"), {"label":"/model/layers.19/attn/k_rotary/RotaryEmbedding_mul_sign_2538"}));
  env.set("InsertedPrecisionFreeCast__2905", builder.reshape(env.get("InsertedPrecisionFreeCast__2904"), [1,128,4,64]));
  env.set("InsertedPrecisionFreeCast__2910", builder["add"](env.get("InsertedPrecisionFreeCast__2909"), env.get("InsertedPrecisionFreeCast__2905"), {"label":"/model/layers.19/attn/k_rotary/RotaryEmbedding_add_mul_cos_sin_2545"}));
  env.set("_2910", builder.cast(env.get("InsertedPrecisionFreeCast__2910"), "float16"));
  env.set("_2911", builder.reshape(env.get("_2910"), [1,128,256]));
  env.set("_2912", builder.reshape(env.get("_2911"), [1,128,4,64]));
  env.set("present_19_key_38", builder["scatterND"](env.get("past_key_values_19_key_2913"), env.get("Inserted_1271"), env.get("_2912"), {"label":"/model/layers.19/attn/GroupQueryAttention_/GQA/present_key/ScatterND_2548"}));
  env.set("_2914", builder.reshape(env.get("present_19_key_38"), [1,4,1,512,64]));
  env.set("_2915", builder.expand(env.get("_2914"), [1,4,8,512,64], {"label":"/model/layers.19/attn/GroupQueryAttention_/GQA/true_present_key/expand_2553"}));
  env.set("_2916", builder.reshape(env.get("_2915"), [1,32,512,64]));
  env.set("_2917", builder["transpose"](env.get("_2916"), {"label":"/model/layers.19/attn/GroupQueryAttention_/GQA/present_key/transpose_2555","permutation":[0,1,3,2]}));
  env.set("InsertedPrecisionFreeCast__2917", builder.cast(env.get("_2917"), "float32"));
  env.set("Inserted_1284", builder["clamp"](env.get("position_ids_625"), {"label":"Inserted_Clip_2571","maxValue":2047,"minValue":-2048}));
  env.set("_2936", builder["gather"](env.get("_644"), env.get("Inserted_1284"), {"axis":0,"label":"/model/layers.19/attn/q_rotary/RotaryEmbedding_gather_cos_2570"}));
  env.set("_2937", builder.reshape(env.get("_2936"), [1,128,1,1,32]));
  env.set("InsertedPrecisionFreeCast__2937", builder.cast(env.get("_2937"), "float32"));
  env.set("_2923", builder.dequantizeLinear(env.get("_2920"), env.get("_2921"), env.get("_2922"), {"axis":2,"blockSize":32,"label":"/model/layers.19/attn/q_proj/MatMul_Q4_dequantizeLinear_2559"}));
  env.set("_2924", builder.reshape(env.get("_2923"), [2048,2048]));
  env.set("_2925", builder["transpose"](env.get("_2924"), {"label":"/model/layers.19/attn/q_proj/MatMul_Q4_transpose_dequantizeLinear_2561","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__2925", builder.cast(env.get("_2925"), "float32"));
  env.set("InsertedPrecisionFreeCast__2926", builder["matmul"](env.get("_2869"), env.get("InsertedPrecisionFreeCast__2925"), {"label":"/model/layers.19/attn/q_proj/MatMul_Q4_matmul_2562"}));
  env.set("_2926", builder.cast(env.get("InsertedPrecisionFreeCast__2926"), "float16"));
  env.set("_2927", builder.reshape(env.get("_2926"), [1,128,32,64]));
  env.set("_2928", builder.reshape(env.get("_2927"), [1,128,32,2,32]));
  env.set("InsertedPrecisionFreeCast__2928", builder.cast(env.get("_2928"), "float32"));
  env.set("InsertedPrecisionFreeCast__2938", builder["mul"](env.get("InsertedPrecisionFreeCast__2928"), env.get("InsertedPrecisionFreeCast__2937"), {"label":"/model/layers.19/attn/q_rotary/RotaryEmbedding_mul_cos_2573"}));
  env.set("InsertedPrecisionFreeCast__2939", builder.reshape(env.get("InsertedPrecisionFreeCast__2938"), [1,128,32,64]));
  env.set("InsertedPrecisionFreeCast__2933", builder.cast(env.get("_2933"), "float32"));
  env.set("Inserted_1275", builder["clamp"](env.get("position_ids_625"), {"label":"Inserted_Clip_2557","maxValue":2047,"minValue":-2048}));
  env.set("_2918", builder["gather"](env.get("_624"), env.get("Inserted_1275"), {"axis":0,"label":"/model/layers.19/attn/q_rotary/RotaryEmbedding_gather_sin_2556"}));
  env.set("_2919", builder.reshape(env.get("_2918"), [1,128,1,1,32]));
  env.set("InsertedPrecisionFreeCast__2919", builder.cast(env.get("_2919"), "float32"));
  {
    const tmp = builder.split(env.get("_2928"), 2, {"axis":3,"label":"/model/layers.19/attn/q_rotary/RotaryEmbedding_split_partial_input0_2565"});
    env.set("_2929", tmp[0]);
    env.set("_2930", tmp[1]);
  }
  env.set("_2931", builder.concat([env.get("_2930"), env.get("_2929")], 3, {"label":"/model/layers.19/attn/q_rotary/RotaryEmbedding_concat_partial_input0_2566"}));
  env.set("InsertedPrecisionFreeCast__2931", builder.cast(env.get("_2931"), "float32"));
  env.set("InsertedPrecisionFreeCast__2932", builder["mul"](env.get("InsertedPrecisionFreeCast__2931"), env.get("InsertedPrecisionFreeCast__2919"), {"label":"/model/layers.19/attn/q_rotary/RotaryEmbedding_mul_sin_2567"}));
  env.set("InsertedPrecisionFreeCast__2934", builder["mul"](env.get("InsertedPrecisionFreeCast__2932"), env.get("InsertedPrecisionFreeCast__2933"), {"label":"/model/layers.19/attn/q_rotary/RotaryEmbedding_mul_sign_2568"}));
  env.set("InsertedPrecisionFreeCast__2935", builder.reshape(env.get("InsertedPrecisionFreeCast__2934"), [1,128,32,64]));
  env.set("InsertedPrecisionFreeCast__2940", builder["add"](env.get("InsertedPrecisionFreeCast__2939"), env.get("InsertedPrecisionFreeCast__2935"), {"label":"/model/layers.19/attn/q_rotary/RotaryEmbedding_add_mul_cos_sin_2575"}));
  env.set("_2940", builder.cast(env.get("InsertedPrecisionFreeCast__2940"), "float16"));
  env.set("_2941", builder.reshape(env.get("_2940"), [1,128,2048]));
  env.set("_2942", builder.reshape(env.get("_2941"), [1,128,32,64]));
  env.set("_2943", builder["transpose"](env.get("_2942"), {"label":"/model/layers.19/attn/GroupQueryAttention_/GQA/query/transpose_2578","permutation":[0,2,1,3]}));
  env.set("InsertedPrecisionFreeCast__2943", builder.cast(env.get("_2943"), "float32"));
  env.set("InsertedPrecisionFreeCast__2944", builder["matmul"](env.get("InsertedPrecisionFreeCast__2943"), env.get("InsertedPrecisionFreeCast__2917"), {"label":"/model/layers.19/attn/GroupQueryAttention_/Attention/qkv/matmul_1_2579"}));
  env.set("InsertedPrecisionFreeCast__2945", builder["mul"](env.get("InsertedPrecisionFreeCast__2944"), env.get("InsertedPrecisionFreeCast__684"), {"label":"/model/layers.19/attn/GroupQueryAttention_/Attention/qkv/div_2580"}));
  env.set("_2884", builder.expand(env.get("_617"), [1,32,128,512], {"label":"/model/layers.19/attn/GroupQueryAttention_/GQA/GQA_mask_shape_ones/expand_2520"}));
  env.set("_2885", builder.cumulativeSum(env.get("_2884"), 3, {"exclusive":true,"label":"/model/layers.19/attn/GroupQueryAttention_range_of_mask_shape_2521"}));
  env.set("_2881", builder["add"](env.get("_613"), env.get("_2873"), {"label":"/model/layers.19/attn/GroupQueryAttention_/GQA/attn_mask/add_2517"}));
  env.set("_2882", builder.expand(env.get("_2881"), [512,128], {"label":"/model/layers.19/attn/GroupQueryAttention_/GQA/expand_neq_right_2518"}));
  env.set("_2883", builder["transpose"](env.get("_2882"), {"label":"/model/layers.19/attn/GroupQueryAttention_/GQA/neq_right/transpose_2519","permutation":[1,0]}));
  env.set("Inserted_1249", builder["lesser"](env.get("_2885"), env.get("_2883"), {"label":"/model/layers.19/attn/GroupQueryAttention_/GQA/attn_mask/condition_1_2522"}));
  env.set("InsertedPrecisionFreeCast__2887", builder["where"](env.get("Inserted_1249"), env.get("InsertedPrecisionFreeCast__621"), env.get("InsertedPrecisionFreeCast__622"), {"label":"/model/layers.19/attn/GroupQueryAttention_/GQA/attn_mask/where_2524"}));
  env.set("InsertedPrecisionFreeCast__2946", builder["add"](env.get("InsertedPrecisionFreeCast__2945"), env.get("InsertedPrecisionFreeCast__2887"), {"label":"/model/layers.19/attn/GroupQueryAttention_/Attention/attn_mask/softmax_input_2581"}));
  env.set("InsertedPrecisionFreeCast__inlfunc_Softmax_token_18_X_ReduceMax", builder["reduceMax"](env.get("InsertedPrecisionFreeCast__2946"), {"axes":[3],"keepDimensions":true}));
  env.set("InsertedPrecisionFreeCast__inlfunc_Softmax_token_18_X_Sub", builder["sub"](env.get("InsertedPrecisionFreeCast__2946"), env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_18_X_ReduceMax"), {}));
  env.set("InsertedPrecisionFreeCast__inlfunc_Softmax_token_18_X_Exp", builder["exp"](env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_18_X_Sub"), {}));
  env.set("InsertedPrecisionFreeCast__inlfunc_Softmax_token_18_X_ReduceSum", builder["reduceSum"](env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_18_X_Exp"), {"axes":[3],"keepDimensions":true}));
  env.set("InsertedPrecisionFreeCast__2947", builder["div"](env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_18_X_Exp"), env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_18_X_ReduceSum"), {}));
  env.set("InsertedPrecisionFreeCast__2948", builder["matmul"](env.get("InsertedPrecisionFreeCast__2947"), env.get("InsertedPrecisionFreeCast__2880"), {"label":"/model/layers.19/attn/GroupQueryAttention_/Attention/qkv/matmul_2_2583"}));
  env.set("_2948", builder.cast(env.get("InsertedPrecisionFreeCast__2948"), "float16"));
  env.set("_2949", builder["transpose"](env.get("_2948"), {"label":"/model/layers.19/attn/GroupQueryAttention_/Attention/qkv/transpose_2584","permutation":[0,2,1,3]}));
  env.set("_2950", builder.reshape(env.get("_2949"), [1,128,2048]));
  env.set("InsertedPrecisionFreeCast__2950", builder.cast(env.get("_2950"), "float32"));
  env.set("InsertedPrecisionFreeCast__2951", builder["matmul"](env.get("InsertedPrecisionFreeCast__2950"), env.get("InsertedPrecisionFreeCast__116"), {"label":"/model/layers.19/attn/o_proj/MatMul_Q4_matmul_2586"}));
  env.set("_2955", builder["add"](env.get("_2954"), env.get("InsertedPrecisionFreeCast__2951"), {"label":"/model/layers.19/post_attention_layernorm/SkipLayerNorm_add_skip_2590"}));
  env.set("_2978", builder.cast(env.get("_2955"), "float16"));
  env.set("_2979", builder.cast(env.get("_2978"), "float32"));
  env.set("_102", builder.dequantizeLinear(env.get("_99"), env.get("_100"), env.get("_101"), {"axis":2,"blockSize":32,"label":"/model/layers.19/mlp/down_proj/MatMul_Q4_dequantizeLinear_27"}));
  env.set("_103", builder.reshape(env.get("_102"), [2048,5632]));
  env.set("_104", builder["transpose"](env.get("_103"), {"label":"/model/layers.19/mlp/down_proj/MatMul_Q4_transpose_dequantizeLinear_29","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__104", builder.cast(env.get("_104"), "float32"));
  env.set("_2969", builder.dequantizeLinear(env.get("_2966"), env.get("_2967"), env.get("_2968"), {"axis":2,"blockSize":32,"label":"/model/layers.19/mlp/gate_proj/MatMul_Q4_dequantizeLinear_2600"}));
  env.set("_2970", builder.reshape(env.get("_2969"), [5632,2048]));
  env.set("_2971", builder["transpose"](env.get("_2970"), {"label":"/model/layers.19/mlp/gate_proj/MatMul_Q4_transpose_dequantizeLinear_2602","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__2971", builder.cast(env.get("_2971"), "float32"));
  env.set("_2962", builder.cast(env.get("_2961"), "float32"));
  env.set("_2956", builder["pow"](env.get("_2955"), env.get("_583"), {"label":"/model/layers.19/post_attention_layernorm/SkipLayerNorm_pow_2591"}));
  env.set("_2957", builder["reduceMean"](env.get("_2956"), {"axes":[2],"keepDimensions":true,"label":"/model/layers.19/post_attention_layernorm/SkipLayerNorm_reduceMean_2592"}));
  env.set("_2958", builder["add"](env.get("_2957"), env.get("_586"), {"label":"/model/layers.19/post_attention_layernorm/SkipLayerNorm_add_2593"}));
  env.set("_2959", builder["sqrt"](env.get("_2958"), {"label":"/model/layers.19/post_attention_layernorm/SkipLayerNorm_sqrt_2594"}));
  env.set("_2960", builder["div"](env.get("_2955"), env.get("_2959"), {"label":"/model/layers.19/post_attention_layernorm/SkipLayerNorm_div_2595"}));
  env.set("_2963", builder["mul"](env.get("_2962"), env.get("_2960"), {"label":"/model/layers.19/post_attention_layernorm/SkipLayerNorm_mul_2597"}));
  env.set("InsertedPrecisionFreeCast__2972", builder["matmul"](env.get("_2963"), env.get("InsertedPrecisionFreeCast__2971"), {"label":"/model/layers.19/mlp/gate_proj/MatMul_Q4_matmul_2603"}));
  env.set("InsertedPrecisionFreeCast__2973", builder["sigmoid"](env.get("InsertedPrecisionFreeCast__2972"), {"label":"/model/layers.19/mlp/act_fn/Sigmoid_2604"}));
  env.set("InsertedPrecisionFreeCast__2974", builder["mul"](env.get("InsertedPrecisionFreeCast__2972"), env.get("InsertedPrecisionFreeCast__2973"), {"label":"/model/layers.19/mlp/act_fn/Mul_2605"}));
  env.set("_108", builder.dequantizeLinear(env.get("_105"), env.get("_106"), env.get("_107"), {"axis":2,"blockSize":32,"label":"/model/layers.19/mlp/up_proj/MatMul_Q4_dequantizeLinear_30"}));
  env.set("_109", builder.reshape(env.get("_108"), [5632,2048]));
  env.set("_110", builder["transpose"](env.get("_109"), {"label":"/model/layers.19/mlp/up_proj/MatMul_Q4_transpose_dequantizeLinear_32","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__110", builder.cast(env.get("_110"), "float32"));
  env.set("InsertedPrecisionFreeCast__2965", builder["matmul"](env.get("_2963"), env.get("InsertedPrecisionFreeCast__110"), {"label":"/model/layers.19/mlp/up_proj/MatMul_Q4_matmul_2599"}));
  env.set("InsertedPrecisionFreeCast__2975", builder["mul"](env.get("InsertedPrecisionFreeCast__2974"), env.get("InsertedPrecisionFreeCast__2965"), {"label":"/model/layers.19/mlp/Mul_2606"}));
  env.set("InsertedPrecisionFreeCast__2976", builder["matmul"](env.get("InsertedPrecisionFreeCast__2975"), env.get("InsertedPrecisionFreeCast__104"), {"label":"/model/layers.19/mlp/down_proj/MatMul_Q4_matmul_2607"}));
  env.set("_2980", builder["add"](env.get("_2979"), env.get("InsertedPrecisionFreeCast__2976"), {"label":"/model/layers.20/input_layernorm/SkipLayerNorm_add_skip_2611"}));
  env.set("_3072", builder.cast(env.get("_2980"), "float16"));
  env.set("_3073", builder.cast(env.get("_3072"), "float32"));
  env.set("_90", builder.dequantizeLinear(env.get("_87"), env.get("_88"), env.get("_89"), {"axis":2,"blockSize":32,"label":"/model/layers.20/attn/o_proj/MatMul_Q4_dequantizeLinear_21"}));
  env.set("_91", builder.reshape(env.get("_90"), [2048,2048]));
  env.set("_92", builder["transpose"](env.get("_91"), {"label":"/model/layers.20/attn/o_proj/MatMul_Q4_transpose_dequantizeLinear_23","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__92", builder.cast(env.get("_92"), "float32"));
  env.set("Inserted_1296", builder.cast(env.get("_601"), "uint8"));
  env.set("_2992", builder["where"](env.get("Inserted_1296"), env.get("_602"), env.get("_600"), {"label":"/model/layers.20/attn/GroupQueryAttention_/GQA/scatter/where_2622"}));
  env.set("_2993", builder["add"](env.get("_604"), env.get("_2992"), {"label":"/model/layers.20/attn/GroupQueryAttention_/GQA/right_constant/add_2624"}));
  env.set("_2994", builder.concat([env.get("_606"), env.get("_2993")], 1, {"label":"/model/layers.20/attn/GroupQueryAttention_/GQA/concat_for_pre_scatter_indices_2625"}));
  env.set("_2995", builder.reshape(env.get("_2994"), [1,128,4,3]));
  env.set("Inserted_1298", builder.cast(env.get("_2995"), "int64"));
  env.set("Inserted_1300", builder["max"](env.get("Inserted_1298"), env.get("Inserted_1299"), {"label":"Inserted_Max_2629"}));
  env.set("Inserted_1302", builder["min"](env.get("Inserted_1300"), env.get("Inserted_1301"), {"label":"Inserted_Min_2630"}));
  env.set("_96", builder.dequantizeLinear(env.get("_93"), env.get("_94"), env.get("_95"), {"axis":2,"blockSize":32,"label":"/model/layers.20/attn/v_proj/MatMul_Q4_dequantizeLinear_24"}));
  env.set("_97", builder.reshape(env.get("_96"), [256,2048]));
  env.set("_98", builder["transpose"](env.get("_97"), {"label":"/model/layers.20/attn/v_proj/MatMul_Q4_transpose_dequantizeLinear_26","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__98", builder.cast(env.get("_98"), "float32"));
  env.set("_2987", builder.cast(env.get("_2986"), "float32"));
  env.set("_2981", builder["pow"](env.get("_2980"), env.get("_583"), {"label":"/model/layers.20/input_layernorm/SkipLayerNorm_pow_2612"}));
  env.set("_2982", builder["reduceMean"](env.get("_2981"), {"axes":[2],"keepDimensions":true,"label":"/model/layers.20/input_layernorm/SkipLayerNorm_reduceMean_2613"}));
  env.set("_2983", builder["add"](env.get("_2982"), env.get("_586"), {"label":"/model/layers.20/input_layernorm/SkipLayerNorm_add_2614"}));
  env.set("_2984", builder["sqrt"](env.get("_2983"), {"label":"/model/layers.20/input_layernorm/SkipLayerNorm_sqrt_2615"}));
  env.set("_2985", builder["div"](env.get("_2980"), env.get("_2984"), {"label":"/model/layers.20/input_layernorm/SkipLayerNorm_div_2616"}));
  env.set("_2988", builder["mul"](env.get("_2987"), env.get("_2985"), {"label":"/model/layers.20/input_layernorm/SkipLayerNorm_mul_2618"}));
  env.set("InsertedPrecisionFreeCast__2990", builder["matmul"](env.get("_2988"), env.get("InsertedPrecisionFreeCast__98"), {"label":"/model/layers.20/attn/v_proj/MatMul_Q4_matmul_2620"}));
  env.set("_2990", builder.cast(env.get("InsertedPrecisionFreeCast__2990"), "float16"));
  env.set("_2991", builder.reshape(env.get("_2990"), [1,128,4,64]));
  env.set("present_20_value_41", builder["scatterND"](env.get("past_key_values_20_value_2996"), env.get("Inserted_1302"), env.get("_2991"), {"label":"/model/layers.20/attn/GroupQueryAttention_/GQA/present_value/ScatterND_2627"}));
  env.set("_2997", builder.reshape(env.get("present_20_value_41"), [1,4,1,512,64]));
  env.set("_2998", builder.expand(env.get("_2997"), [1,4,8,512,64], {"label":"/model/layers.20/attn/GroupQueryAttention_/GQA/true_present_value/expand_2632"}));
  env.set("_2999", builder.reshape(env.get("_2998"), [1,32,512,64]));
  env.set("InsertedPrecisionFreeCast__2999", builder.cast(env.get("_2999"), "float32"));
  env.set("Inserted_1327", builder.cast(env.get("_2995"), "int64"));
  env.set("Inserted_1329", builder["max"](env.get("Inserted_1327"), env.get("Inserted_1328"), {"label":"Inserted_Max_2667"}));
  env.set("Inserted_1331", builder["min"](env.get("Inserted_1329"), env.get("Inserted_1330"), {"label":"Inserted_Min_2668"}));
  env.set("Inserted_1320", builder["clamp"](env.get("position_ids_625"), {"label":"Inserted_Clip_2658","maxValue":2047,"minValue":-2048}));
  env.set("_3025", builder["gather"](env.get("_644"), env.get("Inserted_1320"), {"axis":0,"label":"/model/layers.20/attn/k_rotary/RotaryEmbedding_gather_cos_2657"}));
  env.set("_3026", builder.reshape(env.get("_3025"), [1,128,1,1,32]));
  env.set("InsertedPrecisionFreeCast__3026", builder.cast(env.get("_3026"), "float32"));
  env.set("_3012", builder.dequantizeLinear(env.get("_3009"), env.get("_3010"), env.get("_3011"), {"axis":2,"blockSize":32,"label":"/model/layers.20/attn/k_proj/MatMul_Q4_dequantizeLinear_2646"}));
  env.set("_3013", builder.reshape(env.get("_3012"), [256,2048]));
  env.set("_3014", builder["transpose"](env.get("_3013"), {"label":"/model/layers.20/attn/k_proj/MatMul_Q4_transpose_dequantizeLinear_2648","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__3014", builder.cast(env.get("_3014"), "float32"));
  env.set("InsertedPrecisionFreeCast__3015", builder["matmul"](env.get("_2988"), env.get("InsertedPrecisionFreeCast__3014"), {"label":"/model/layers.20/attn/k_proj/MatMul_Q4_matmul_2649"}));
  env.set("_3015", builder.cast(env.get("InsertedPrecisionFreeCast__3015"), "float16"));
  env.set("_3016", builder.reshape(env.get("_3015"), [1,128,4,64]));
  env.set("_3017", builder.reshape(env.get("_3016"), [1,128,4,2,32]));
  env.set("InsertedPrecisionFreeCast__3017", builder.cast(env.get("_3017"), "float32"));
  env.set("InsertedPrecisionFreeCast__3027", builder["mul"](env.get("InsertedPrecisionFreeCast__3017"), env.get("InsertedPrecisionFreeCast__3026"), {"label":"/model/layers.20/attn/k_rotary/RotaryEmbedding_mul_cos_2660"}));
  env.set("InsertedPrecisionFreeCast__3028", builder.reshape(env.get("InsertedPrecisionFreeCast__3027"), [1,128,4,64]));
  env.set("InsertedPrecisionFreeCast__3022", builder.cast(env.get("_3022"), "float32"));
  env.set("Inserted_1311", builder["clamp"](env.get("position_ids_625"), {"label":"Inserted_Clip_2644","maxValue":2047,"minValue":-2048}));
  env.set("_3007", builder["gather"](env.get("_624"), env.get("Inserted_1311"), {"axis":0,"label":"/model/layers.20/attn/k_rotary/RotaryEmbedding_gather_sin_2643"}));
  env.set("_3008", builder.reshape(env.get("_3007"), [1,128,1,1,32]));
  env.set("InsertedPrecisionFreeCast__3008", builder.cast(env.get("_3008"), "float32"));
  {
    const tmp = builder.split(env.get("_3017"), 2, {"axis":3,"label":"/model/layers.20/attn/k_rotary/RotaryEmbedding_split_partial_input0_2652"});
    env.set("_3018", tmp[0]);
    env.set("_3019", tmp[1]);
  }
  env.set("_3020", builder.concat([env.get("_3019"), env.get("_3018")], 3, {"label":"/model/layers.20/attn/k_rotary/RotaryEmbedding_concat_partial_input0_2653"}));
  env.set("InsertedPrecisionFreeCast__3020", builder.cast(env.get("_3020"), "float32"));
  env.set("InsertedPrecisionFreeCast__3021", builder["mul"](env.get("InsertedPrecisionFreeCast__3020"), env.get("InsertedPrecisionFreeCast__3008"), {"label":"/model/layers.20/attn/k_rotary/RotaryEmbedding_mul_sin_2654"}));
  env.set("InsertedPrecisionFreeCast__3023", builder["mul"](env.get("InsertedPrecisionFreeCast__3021"), env.get("InsertedPrecisionFreeCast__3022"), {"label":"/model/layers.20/attn/k_rotary/RotaryEmbedding_mul_sign_2655"}));
  env.set("InsertedPrecisionFreeCast__3024", builder.reshape(env.get("InsertedPrecisionFreeCast__3023"), [1,128,4,64]));
  env.set("InsertedPrecisionFreeCast__3029", builder["add"](env.get("InsertedPrecisionFreeCast__3028"), env.get("InsertedPrecisionFreeCast__3024"), {"label":"/model/layers.20/attn/k_rotary/RotaryEmbedding_add_mul_cos_sin_2662"}));
  env.set("_3029", builder.cast(env.get("InsertedPrecisionFreeCast__3029"), "float16"));
  env.set("_3030", builder.reshape(env.get("_3029"), [1,128,256]));
  env.set("_3031", builder.reshape(env.get("_3030"), [1,128,4,64]));
  env.set("present_20_key_40", builder["scatterND"](env.get("past_key_values_20_key_3032"), env.get("Inserted_1331"), env.get("_3031"), {"label":"/model/layers.20/attn/GroupQueryAttention_/GQA/present_key/ScatterND_2665"}));
  env.set("_3033", builder.reshape(env.get("present_20_key_40"), [1,4,1,512,64]));
  env.set("_3034", builder.expand(env.get("_3033"), [1,4,8,512,64], {"label":"/model/layers.20/attn/GroupQueryAttention_/GQA/true_present_key/expand_2670"}));
  env.set("_3035", builder.reshape(env.get("_3034"), [1,32,512,64]));
  env.set("_3036", builder["transpose"](env.get("_3035"), {"label":"/model/layers.20/attn/GroupQueryAttention_/GQA/present_key/transpose_2672","permutation":[0,1,3,2]}));
  env.set("InsertedPrecisionFreeCast__3036", builder.cast(env.get("_3036"), "float32"));
  env.set("Inserted_1344", builder["clamp"](env.get("position_ids_625"), {"label":"Inserted_Clip_2688","maxValue":2047,"minValue":-2048}));
  env.set("_3055", builder["gather"](env.get("_644"), env.get("Inserted_1344"), {"axis":0,"label":"/model/layers.20/attn/q_rotary/RotaryEmbedding_gather_cos_2687"}));
  env.set("_3056", builder.reshape(env.get("_3055"), [1,128,1,1,32]));
  env.set("InsertedPrecisionFreeCast__3056", builder.cast(env.get("_3056"), "float32"));
  env.set("_3042", builder.dequantizeLinear(env.get("_3039"), env.get("_3040"), env.get("_3041"), {"axis":2,"blockSize":32,"label":"/model/layers.20/attn/q_proj/MatMul_Q4_dequantizeLinear_2676"}));
  env.set("_3043", builder.reshape(env.get("_3042"), [2048,2048]));
  env.set("_3044", builder["transpose"](env.get("_3043"), {"label":"/model/layers.20/attn/q_proj/MatMul_Q4_transpose_dequantizeLinear_2678","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__3044", builder.cast(env.get("_3044"), "float32"));
  env.set("InsertedPrecisionFreeCast__3045", builder["matmul"](env.get("_2988"), env.get("InsertedPrecisionFreeCast__3044"), {"label":"/model/layers.20/attn/q_proj/MatMul_Q4_matmul_2679"}));
  env.set("_3045", builder.cast(env.get("InsertedPrecisionFreeCast__3045"), "float16"));
  env.set("_3046", builder.reshape(env.get("_3045"), [1,128,32,64]));
  env.set("_3047", builder.reshape(env.get("_3046"), [1,128,32,2,32]));
  env.set("InsertedPrecisionFreeCast__3047", builder.cast(env.get("_3047"), "float32"));
  env.set("InsertedPrecisionFreeCast__3057", builder["mul"](env.get("InsertedPrecisionFreeCast__3047"), env.get("InsertedPrecisionFreeCast__3056"), {"label":"/model/layers.20/attn/q_rotary/RotaryEmbedding_mul_cos_2690"}));
  env.set("InsertedPrecisionFreeCast__3058", builder.reshape(env.get("InsertedPrecisionFreeCast__3057"), [1,128,32,64]));
  env.set("InsertedPrecisionFreeCast__3052", builder.cast(env.get("_3052"), "float32"));
  env.set("Inserted_1335", builder["clamp"](env.get("position_ids_625"), {"label":"Inserted_Clip_2674","maxValue":2047,"minValue":-2048}));
  env.set("_3037", builder["gather"](env.get("_624"), env.get("Inserted_1335"), {"axis":0,"label":"/model/layers.20/attn/q_rotary/RotaryEmbedding_gather_sin_2673"}));
  env.set("_3038", builder.reshape(env.get("_3037"), [1,128,1,1,32]));
  env.set("InsertedPrecisionFreeCast__3038", builder.cast(env.get("_3038"), "float32"));
  {
    const tmp = builder.split(env.get("_3047"), 2, {"axis":3,"label":"/model/layers.20/attn/q_rotary/RotaryEmbedding_split_partial_input0_2682"});
    env.set("_3048", tmp[0]);
    env.set("_3049", tmp[1]);
  }
  env.set("_3050", builder.concat([env.get("_3049"), env.get("_3048")], 3, {"label":"/model/layers.20/attn/q_rotary/RotaryEmbedding_concat_partial_input0_2683"}));
  env.set("InsertedPrecisionFreeCast__3050", builder.cast(env.get("_3050"), "float32"));
  env.set("InsertedPrecisionFreeCast__3051", builder["mul"](env.get("InsertedPrecisionFreeCast__3050"), env.get("InsertedPrecisionFreeCast__3038"), {"label":"/model/layers.20/attn/q_rotary/RotaryEmbedding_mul_sin_2684"}));
  env.set("InsertedPrecisionFreeCast__3053", builder["mul"](env.get("InsertedPrecisionFreeCast__3051"), env.get("InsertedPrecisionFreeCast__3052"), {"label":"/model/layers.20/attn/q_rotary/RotaryEmbedding_mul_sign_2685"}));
  env.set("InsertedPrecisionFreeCast__3054", builder.reshape(env.get("InsertedPrecisionFreeCast__3053"), [1,128,32,64]));
  env.set("InsertedPrecisionFreeCast__3059", builder["add"](env.get("InsertedPrecisionFreeCast__3058"), env.get("InsertedPrecisionFreeCast__3054"), {"label":"/model/layers.20/attn/q_rotary/RotaryEmbedding_add_mul_cos_sin_2692"}));
  env.set("_3059", builder.cast(env.get("InsertedPrecisionFreeCast__3059"), "float16"));
  env.set("_3060", builder.reshape(env.get("_3059"), [1,128,2048]));
  env.set("_3061", builder.reshape(env.get("_3060"), [1,128,32,64]));
  env.set("_3062", builder["transpose"](env.get("_3061"), {"label":"/model/layers.20/attn/GroupQueryAttention_/GQA/query/transpose_2695","permutation":[0,2,1,3]}));
  env.set("InsertedPrecisionFreeCast__3062", builder.cast(env.get("_3062"), "float32"));
  env.set("InsertedPrecisionFreeCast__3063", builder["matmul"](env.get("InsertedPrecisionFreeCast__3062"), env.get("InsertedPrecisionFreeCast__3036"), {"label":"/model/layers.20/attn/GroupQueryAttention_/Attention/qkv/matmul_1_2696"}));
  env.set("InsertedPrecisionFreeCast__3064", builder["mul"](env.get("InsertedPrecisionFreeCast__3063"), env.get("InsertedPrecisionFreeCast__684"), {"label":"/model/layers.20/attn/GroupQueryAttention_/Attention/qkv/div_2697"}));
  env.set("_3003", builder.expand(env.get("_617"), [1,32,128,512], {"label":"/model/layers.20/attn/GroupQueryAttention_/GQA/GQA_mask_shape_ones/expand_2637"}));
  env.set("_3004", builder.cumulativeSum(env.get("_3003"), 3, {"exclusive":true,"label":"/model/layers.20/attn/GroupQueryAttention_range_of_mask_shape_2638"}));
  env.set("_3000", builder["add"](env.get("_613"), env.get("_2992"), {"label":"/model/layers.20/attn/GroupQueryAttention_/GQA/attn_mask/add_2634"}));
  env.set("_3001", builder.expand(env.get("_3000"), [512,128], {"label":"/model/layers.20/attn/GroupQueryAttention_/GQA/expand_neq_right_2635"}));
  env.set("_3002", builder["transpose"](env.get("_3001"), {"label":"/model/layers.20/attn/GroupQueryAttention_/GQA/neq_right/transpose_2636","permutation":[1,0]}));
  env.set("Inserted_1309", builder["lesser"](env.get("_3004"), env.get("_3002"), {"label":"/model/layers.20/attn/GroupQueryAttention_/GQA/attn_mask/condition_1_2639"}));
  env.set("InsertedPrecisionFreeCast__3006", builder["where"](env.get("Inserted_1309"), env.get("InsertedPrecisionFreeCast__621"), env.get("InsertedPrecisionFreeCast__622"), {"label":"/model/layers.20/attn/GroupQueryAttention_/GQA/attn_mask/where_2641"}));
  env.set("InsertedPrecisionFreeCast__3065", builder["add"](env.get("InsertedPrecisionFreeCast__3064"), env.get("InsertedPrecisionFreeCast__3006"), {"label":"/model/layers.20/attn/GroupQueryAttention_/Attention/attn_mask/softmax_input_2698"}));
  env.set("InsertedPrecisionFreeCast__inlfunc_Softmax_token_19_X_ReduceMax", builder["reduceMax"](env.get("InsertedPrecisionFreeCast__3065"), {"axes":[3],"keepDimensions":true}));
  env.set("InsertedPrecisionFreeCast__inlfunc_Softmax_token_19_X_Sub", builder["sub"](env.get("InsertedPrecisionFreeCast__3065"), env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_19_X_ReduceMax"), {}));
  env.set("InsertedPrecisionFreeCast__inlfunc_Softmax_token_19_X_Exp", builder["exp"](env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_19_X_Sub"), {}));
  env.set("InsertedPrecisionFreeCast__inlfunc_Softmax_token_19_X_ReduceSum", builder["reduceSum"](env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_19_X_Exp"), {"axes":[3],"keepDimensions":true}));
  env.set("InsertedPrecisionFreeCast__3066", builder["div"](env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_19_X_Exp"), env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_19_X_ReduceSum"), {}));
  env.set("InsertedPrecisionFreeCast__3067", builder["matmul"](env.get("InsertedPrecisionFreeCast__3066"), env.get("InsertedPrecisionFreeCast__2999"), {"label":"/model/layers.20/attn/GroupQueryAttention_/Attention/qkv/matmul_2_2700"}));
  env.set("_3067", builder.cast(env.get("InsertedPrecisionFreeCast__3067"), "float16"));
  env.set("_3068", builder["transpose"](env.get("_3067"), {"label":"/model/layers.20/attn/GroupQueryAttention_/Attention/qkv/transpose_2701","permutation":[0,2,1,3]}));
  env.set("_3069", builder.reshape(env.get("_3068"), [1,128,2048]));
  env.set("InsertedPrecisionFreeCast__3069", builder.cast(env.get("_3069"), "float32"));
  env.set("InsertedPrecisionFreeCast__3070", builder["matmul"](env.get("InsertedPrecisionFreeCast__3069"), env.get("InsertedPrecisionFreeCast__92"), {"label":"/model/layers.20/attn/o_proj/MatMul_Q4_matmul_2703"}));
  env.set("_3074", builder["add"](env.get("_3073"), env.get("InsertedPrecisionFreeCast__3070"), {"label":"/model/layers.20/post_attention_layernorm/SkipLayerNorm_add_skip_2707"}));
  env.set("_3097", builder.cast(env.get("_3074"), "float16"));
  env.set("_3098", builder.cast(env.get("_3097"), "float32"));
  env.set("_78", builder.dequantizeLinear(env.get("_75"), env.get("_76"), env.get("_77"), {"axis":2,"blockSize":32,"label":"/model/layers.20/mlp/down_proj/MatMul_Q4_dequantizeLinear_15"}));
  env.set("_79", builder.reshape(env.get("_78"), [2048,5632]));
  env.set("_80", builder["transpose"](env.get("_79"), {"label":"/model/layers.20/mlp/down_proj/MatMul_Q4_transpose_dequantizeLinear_17","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__80", builder.cast(env.get("_80"), "float32"));
  env.set("_3088", builder.dequantizeLinear(env.get("_3085"), env.get("_3086"), env.get("_3087"), {"axis":2,"blockSize":32,"label":"/model/layers.20/mlp/gate_proj/MatMul_Q4_dequantizeLinear_2717"}));
  env.set("_3089", builder.reshape(env.get("_3088"), [5632,2048]));
  env.set("_3090", builder["transpose"](env.get("_3089"), {"label":"/model/layers.20/mlp/gate_proj/MatMul_Q4_transpose_dequantizeLinear_2719","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__3090", builder.cast(env.get("_3090"), "float32"));
  env.set("_3081", builder.cast(env.get("_3080"), "float32"));
  env.set("_3075", builder["pow"](env.get("_3074"), env.get("_583"), {"label":"/model/layers.20/post_attention_layernorm/SkipLayerNorm_pow_2708"}));
  env.set("_3076", builder["reduceMean"](env.get("_3075"), {"axes":[2],"keepDimensions":true,"label":"/model/layers.20/post_attention_layernorm/SkipLayerNorm_reduceMean_2709"}));
  env.set("_3077", builder["add"](env.get("_3076"), env.get("_586"), {"label":"/model/layers.20/post_attention_layernorm/SkipLayerNorm_add_2710"}));
  env.set("_3078", builder["sqrt"](env.get("_3077"), {"label":"/model/layers.20/post_attention_layernorm/SkipLayerNorm_sqrt_2711"}));
  env.set("_3079", builder["div"](env.get("_3074"), env.get("_3078"), {"label":"/model/layers.20/post_attention_layernorm/SkipLayerNorm_div_2712"}));
  env.set("_3082", builder["mul"](env.get("_3081"), env.get("_3079"), {"label":"/model/layers.20/post_attention_layernorm/SkipLayerNorm_mul_2714"}));
  env.set("InsertedPrecisionFreeCast__3091", builder["matmul"](env.get("_3082"), env.get("InsertedPrecisionFreeCast__3090"), {"label":"/model/layers.20/mlp/gate_proj/MatMul_Q4_matmul_2720"}));
  env.set("InsertedPrecisionFreeCast__3092", builder["sigmoid"](env.get("InsertedPrecisionFreeCast__3091"), {"label":"/model/layers.20/mlp/act_fn/Sigmoid_2721"}));
  env.set("InsertedPrecisionFreeCast__3093", builder["mul"](env.get("InsertedPrecisionFreeCast__3091"), env.get("InsertedPrecisionFreeCast__3092"), {"label":"/model/layers.20/mlp/act_fn/Mul_2722"}));
  env.set("_84", builder.dequantizeLinear(env.get("_81"), env.get("_82"), env.get("_83"), {"axis":2,"blockSize":32,"label":"/model/layers.20/mlp/up_proj/MatMul_Q4_dequantizeLinear_18"}));
  env.set("_85", builder.reshape(env.get("_84"), [5632,2048]));
  env.set("_86", builder["transpose"](env.get("_85"), {"label":"/model/layers.20/mlp/up_proj/MatMul_Q4_transpose_dequantizeLinear_20","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__86", builder.cast(env.get("_86"), "float32"));
  env.set("InsertedPrecisionFreeCast__3084", builder["matmul"](env.get("_3082"), env.get("InsertedPrecisionFreeCast__86"), {"label":"/model/layers.20/mlp/up_proj/MatMul_Q4_matmul_2716"}));
  env.set("InsertedPrecisionFreeCast__3094", builder["mul"](env.get("InsertedPrecisionFreeCast__3093"), env.get("InsertedPrecisionFreeCast__3084"), {"label":"/model/layers.20/mlp/Mul_2723"}));
  env.set("InsertedPrecisionFreeCast__3095", builder["matmul"](env.get("InsertedPrecisionFreeCast__3094"), env.get("InsertedPrecisionFreeCast__80"), {"label":"/model/layers.20/mlp/down_proj/MatMul_Q4_matmul_2724"}));
  env.set("_3099", builder["add"](env.get("_3098"), env.get("InsertedPrecisionFreeCast__3095"), {"label":"/model/layers.21/input_layernorm/SkipLayerNorm_add_skip_2728"}));
  env.set("_3191", builder.cast(env.get("_3099"), "float16"));
  env.set("_3192", builder.cast(env.get("_3191"), "float32"));
  env.set("_66", builder.dequantizeLinear(env.get("_63"), env.get("_64"), env.get("_65"), {"axis":2,"blockSize":32,"label":"/model/layers.21/attn/o_proj/MatMul_Q4_dequantizeLinear_9"}));
  env.set("_67", builder.reshape(env.get("_66"), [2048,2048]));
  env.set("_68", builder["transpose"](env.get("_67"), {"label":"/model/layers.21/attn/o_proj/MatMul_Q4_transpose_dequantizeLinear_11","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__68", builder.cast(env.get("_68"), "float32"));
  env.set("Inserted_1356", builder.cast(env.get("_601"), "uint8"));
  env.set("_3111", builder["where"](env.get("Inserted_1356"), env.get("_602"), env.get("_600"), {"label":"/model/layers.21/attn/GroupQueryAttention_/GQA/scatter/where_2739"}));
  env.set("_3112", builder["add"](env.get("_604"), env.get("_3111"), {"label":"/model/layers.21/attn/GroupQueryAttention_/GQA/right_constant/add_2741"}));
  env.set("_3113", builder.concat([env.get("_606"), env.get("_3112")], 1, {"label":"/model/layers.21/attn/GroupQueryAttention_/GQA/concat_for_pre_scatter_indices_2742"}));
  env.set("_3114", builder.reshape(env.get("_3113"), [1,128,4,3]));
  env.set("Inserted_1358", builder.cast(env.get("_3114"), "int64"));
  env.set("Inserted_1360", builder["max"](env.get("Inserted_1358"), env.get("Inserted_1359"), {"label":"Inserted_Max_2746"}));
  env.set("Inserted_1362", builder["min"](env.get("Inserted_1360"), env.get("Inserted_1361"), {"label":"Inserted_Min_2747"}));
  env.set("_72", builder.dequantizeLinear(env.get("_69"), env.get("_70"), env.get("_71"), {"axis":2,"blockSize":32,"label":"/model/layers.21/attn/v_proj/MatMul_Q4_dequantizeLinear_12"}));
  env.set("_73", builder.reshape(env.get("_72"), [256,2048]));
  env.set("_74", builder["transpose"](env.get("_73"), {"label":"/model/layers.21/attn/v_proj/MatMul_Q4_transpose_dequantizeLinear_14","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__74", builder.cast(env.get("_74"), "float32"));
  env.set("_3106", builder.cast(env.get("_3105"), "float32"));
  env.set("_3100", builder["pow"](env.get("_3099"), env.get("_583"), {"label":"/model/layers.21/input_layernorm/SkipLayerNorm_pow_2729"}));
  env.set("_3101", builder["reduceMean"](env.get("_3100"), {"axes":[2],"keepDimensions":true,"label":"/model/layers.21/input_layernorm/SkipLayerNorm_reduceMean_2730"}));
  env.set("_3102", builder["add"](env.get("_3101"), env.get("_586"), {"label":"/model/layers.21/input_layernorm/SkipLayerNorm_add_2731"}));
  env.set("_3103", builder["sqrt"](env.get("_3102"), {"label":"/model/layers.21/input_layernorm/SkipLayerNorm_sqrt_2732"}));
  env.set("_3104", builder["div"](env.get("_3099"), env.get("_3103"), {"label":"/model/layers.21/input_layernorm/SkipLayerNorm_div_2733"}));
  env.set("_3107", builder["mul"](env.get("_3106"), env.get("_3104"), {"label":"/model/layers.21/input_layernorm/SkipLayerNorm_mul_2735"}));
  env.set("InsertedPrecisionFreeCast__3109", builder["matmul"](env.get("_3107"), env.get("InsertedPrecisionFreeCast__74"), {"label":"/model/layers.21/attn/v_proj/MatMul_Q4_matmul_2737"}));
  env.set("_3109", builder.cast(env.get("InsertedPrecisionFreeCast__3109"), "float16"));
  env.set("_3110", builder.reshape(env.get("_3109"), [1,128,4,64]));
  env.set("present_21_value_43", builder["scatterND"](env.get("past_key_values_21_value_3115"), env.get("Inserted_1362"), env.get("_3110"), {"label":"/model/layers.21/attn/GroupQueryAttention_/GQA/present_value/ScatterND_2744"}));
  env.set("_3116", builder.reshape(env.get("present_21_value_43"), [1,4,1,512,64]));
  env.set("_3117", builder.expand(env.get("_3116"), [1,4,8,512,64], {"label":"/model/layers.21/attn/GroupQueryAttention_/GQA/true_present_value/expand_2749"}));
  env.set("_3118", builder.reshape(env.get("_3117"), [1,32,512,64]));
  env.set("InsertedPrecisionFreeCast__3118", builder.cast(env.get("_3118"), "float32"));
  env.set("Inserted_1387", builder.cast(env.get("_3114"), "int64"));
  env.set("Inserted_1389", builder["max"](env.get("Inserted_1387"), env.get("Inserted_1388"), {"label":"Inserted_Max_2784"}));
  env.set("Inserted_1391", builder["min"](env.get("Inserted_1389"), env.get("Inserted_1390"), {"label":"Inserted_Min_2785"}));
  env.set("Inserted_1380", builder["clamp"](env.get("position_ids_625"), {"label":"Inserted_Clip_2775","maxValue":2047,"minValue":-2048}));
  env.set("_3144", builder["gather"](env.get("_644"), env.get("Inserted_1380"), {"axis":0,"label":"/model/layers.21/attn/k_rotary/RotaryEmbedding_gather_cos_2774"}));
  env.set("_3145", builder.reshape(env.get("_3144"), [1,128,1,1,32]));
  env.set("InsertedPrecisionFreeCast__3145", builder.cast(env.get("_3145"), "float32"));
  env.set("_3131", builder.dequantizeLinear(env.get("_3128"), env.get("_3129"), env.get("_3130"), {"axis":2,"blockSize":32,"label":"/model/layers.21/attn/k_proj/MatMul_Q4_dequantizeLinear_2763"}));
  env.set("_3132", builder.reshape(env.get("_3131"), [256,2048]));
  env.set("_3133", builder["transpose"](env.get("_3132"), {"label":"/model/layers.21/attn/k_proj/MatMul_Q4_transpose_dequantizeLinear_2765","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__3133", builder.cast(env.get("_3133"), "float32"));
  env.set("InsertedPrecisionFreeCast__3134", builder["matmul"](env.get("_3107"), env.get("InsertedPrecisionFreeCast__3133"), {"label":"/model/layers.21/attn/k_proj/MatMul_Q4_matmul_2766"}));
  env.set("_3134", builder.cast(env.get("InsertedPrecisionFreeCast__3134"), "float16"));
  env.set("_3135", builder.reshape(env.get("_3134"), [1,128,4,64]));
  env.set("_3136", builder.reshape(env.get("_3135"), [1,128,4,2,32]));
  env.set("InsertedPrecisionFreeCast__3136", builder.cast(env.get("_3136"), "float32"));
  env.set("InsertedPrecisionFreeCast__3146", builder["mul"](env.get("InsertedPrecisionFreeCast__3136"), env.get("InsertedPrecisionFreeCast__3145"), {"label":"/model/layers.21/attn/k_rotary/RotaryEmbedding_mul_cos_2777"}));
  env.set("InsertedPrecisionFreeCast__3147", builder.reshape(env.get("InsertedPrecisionFreeCast__3146"), [1,128,4,64]));
  env.set("InsertedPrecisionFreeCast__3141", builder.cast(env.get("_3141"), "float32"));
  env.set("Inserted_1371", builder["clamp"](env.get("position_ids_625"), {"label":"Inserted_Clip_2761","maxValue":2047,"minValue":-2048}));
  env.set("_3126", builder["gather"](env.get("_624"), env.get("Inserted_1371"), {"axis":0,"label":"/model/layers.21/attn/k_rotary/RotaryEmbedding_gather_sin_2760"}));
  env.set("_3127", builder.reshape(env.get("_3126"), [1,128,1,1,32]));
  env.set("InsertedPrecisionFreeCast__3127", builder.cast(env.get("_3127"), "float32"));
  {
    const tmp = builder.split(env.get("_3136"), 2, {"axis":3,"label":"/model/layers.21/attn/k_rotary/RotaryEmbedding_split_partial_input0_2769"});
    env.set("_3137", tmp[0]);
    env.set("_3138", tmp[1]);
  }
  env.set("_3139", builder.concat([env.get("_3138"), env.get("_3137")], 3, {"label":"/model/layers.21/attn/k_rotary/RotaryEmbedding_concat_partial_input0_2770"}));
  env.set("InsertedPrecisionFreeCast__3139", builder.cast(env.get("_3139"), "float32"));
  env.set("InsertedPrecisionFreeCast__3140", builder["mul"](env.get("InsertedPrecisionFreeCast__3139"), env.get("InsertedPrecisionFreeCast__3127"), {"label":"/model/layers.21/attn/k_rotary/RotaryEmbedding_mul_sin_2771"}));
  env.set("InsertedPrecisionFreeCast__3142", builder["mul"](env.get("InsertedPrecisionFreeCast__3140"), env.get("InsertedPrecisionFreeCast__3141"), {"label":"/model/layers.21/attn/k_rotary/RotaryEmbedding_mul_sign_2772"}));
  env.set("InsertedPrecisionFreeCast__3143", builder.reshape(env.get("InsertedPrecisionFreeCast__3142"), [1,128,4,64]));
  env.set("InsertedPrecisionFreeCast__3148", builder["add"](env.get("InsertedPrecisionFreeCast__3147"), env.get("InsertedPrecisionFreeCast__3143"), {"label":"/model/layers.21/attn/k_rotary/RotaryEmbedding_add_mul_cos_sin_2779"}));
  env.set("_3148", builder.cast(env.get("InsertedPrecisionFreeCast__3148"), "float16"));
  env.set("_3149", builder.reshape(env.get("_3148"), [1,128,256]));
  env.set("_3150", builder.reshape(env.get("_3149"), [1,128,4,64]));
  env.set("present_21_key_42", builder["scatterND"](env.get("past_key_values_21_key_3151"), env.get("Inserted_1391"), env.get("_3150"), {"label":"/model/layers.21/attn/GroupQueryAttention_/GQA/present_key/ScatterND_2782"}));
  env.set("_3152", builder.reshape(env.get("present_21_key_42"), [1,4,1,512,64]));
  env.set("_3153", builder.expand(env.get("_3152"), [1,4,8,512,64], {"label":"/model/layers.21/attn/GroupQueryAttention_/GQA/true_present_key/expand_2787"}));
  env.set("_3154", builder.reshape(env.get("_3153"), [1,32,512,64]));
  env.set("_3155", builder["transpose"](env.get("_3154"), {"label":"/model/layers.21/attn/GroupQueryAttention_/GQA/present_key/transpose_2789","permutation":[0,1,3,2]}));
  env.set("InsertedPrecisionFreeCast__3155", builder.cast(env.get("_3155"), "float32"));
  env.set("Inserted_1404", builder["clamp"](env.get("position_ids_625"), {"label":"Inserted_Clip_2805","maxValue":2047,"minValue":-2048}));
  env.set("_3174", builder["gather"](env.get("_644"), env.get("Inserted_1404"), {"axis":0,"label":"/model/layers.21/attn/q_rotary/RotaryEmbedding_gather_cos_2804"}));
  env.set("_3175", builder.reshape(env.get("_3174"), [1,128,1,1,32]));
  env.set("InsertedPrecisionFreeCast__3175", builder.cast(env.get("_3175"), "float32"));
  env.set("_3161", builder.dequantizeLinear(env.get("_3158"), env.get("_3159"), env.get("_3160"), {"axis":2,"blockSize":32,"label":"/model/layers.21/attn/q_proj/MatMul_Q4_dequantizeLinear_2793"}));
  env.set("_3162", builder.reshape(env.get("_3161"), [2048,2048]));
  env.set("_3163", builder["transpose"](env.get("_3162"), {"label":"/model/layers.21/attn/q_proj/MatMul_Q4_transpose_dequantizeLinear_2795","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__3163", builder.cast(env.get("_3163"), "float32"));
  env.set("InsertedPrecisionFreeCast__3164", builder["matmul"](env.get("_3107"), env.get("InsertedPrecisionFreeCast__3163"), {"label":"/model/layers.21/attn/q_proj/MatMul_Q4_matmul_2796"}));
  env.set("_3164", builder.cast(env.get("InsertedPrecisionFreeCast__3164"), "float16"));
  env.set("_3165", builder.reshape(env.get("_3164"), [1,128,32,64]));
  env.set("_3166", builder.reshape(env.get("_3165"), [1,128,32,2,32]));
  env.set("InsertedPrecisionFreeCast__3166", builder.cast(env.get("_3166"), "float32"));
  env.set("InsertedPrecisionFreeCast__3176", builder["mul"](env.get("InsertedPrecisionFreeCast__3166"), env.get("InsertedPrecisionFreeCast__3175"), {"label":"/model/layers.21/attn/q_rotary/RotaryEmbedding_mul_cos_2807"}));
  env.set("InsertedPrecisionFreeCast__3177", builder.reshape(env.get("InsertedPrecisionFreeCast__3176"), [1,128,32,64]));
  env.set("InsertedPrecisionFreeCast__3171", builder.cast(env.get("_3171"), "float32"));
  env.set("Inserted_1395", builder["clamp"](env.get("position_ids_625"), {"label":"Inserted_Clip_2791","maxValue":2047,"minValue":-2048}));
  env.set("_3156", builder["gather"](env.get("_624"), env.get("Inserted_1395"), {"axis":0,"label":"/model/layers.21/attn/q_rotary/RotaryEmbedding_gather_sin_2790"}));
  env.set("_3157", builder.reshape(env.get("_3156"), [1,128,1,1,32]));
  env.set("InsertedPrecisionFreeCast__3157", builder.cast(env.get("_3157"), "float32"));
  {
    const tmp = builder.split(env.get("_3166"), 2, {"axis":3,"label":"/model/layers.21/attn/q_rotary/RotaryEmbedding_split_partial_input0_2799"});
    env.set("_3167", tmp[0]);
    env.set("_3168", tmp[1]);
  }
  env.set("_3169", builder.concat([env.get("_3168"), env.get("_3167")], 3, {"label":"/model/layers.21/attn/q_rotary/RotaryEmbedding_concat_partial_input0_2800"}));
  env.set("InsertedPrecisionFreeCast__3169", builder.cast(env.get("_3169"), "float32"));
  env.set("InsertedPrecisionFreeCast__3170", builder["mul"](env.get("InsertedPrecisionFreeCast__3169"), env.get("InsertedPrecisionFreeCast__3157"), {"label":"/model/layers.21/attn/q_rotary/RotaryEmbedding_mul_sin_2801"}));
  env.set("InsertedPrecisionFreeCast__3172", builder["mul"](env.get("InsertedPrecisionFreeCast__3170"), env.get("InsertedPrecisionFreeCast__3171"), {"label":"/model/layers.21/attn/q_rotary/RotaryEmbedding_mul_sign_2802"}));
  env.set("InsertedPrecisionFreeCast__3173", builder.reshape(env.get("InsertedPrecisionFreeCast__3172"), [1,128,32,64]));
  env.set("InsertedPrecisionFreeCast__3178", builder["add"](env.get("InsertedPrecisionFreeCast__3177"), env.get("InsertedPrecisionFreeCast__3173"), {"label":"/model/layers.21/attn/q_rotary/RotaryEmbedding_add_mul_cos_sin_2809"}));
  env.set("_3178", builder.cast(env.get("InsertedPrecisionFreeCast__3178"), "float16"));
  env.set("_3179", builder.reshape(env.get("_3178"), [1,128,2048]));
  env.set("_3180", builder.reshape(env.get("_3179"), [1,128,32,64]));
  env.set("_3181", builder["transpose"](env.get("_3180"), {"label":"/model/layers.21/attn/GroupQueryAttention_/GQA/query/transpose_2812","permutation":[0,2,1,3]}));
  env.set("InsertedPrecisionFreeCast__3181", builder.cast(env.get("_3181"), "float32"));
  env.set("InsertedPrecisionFreeCast__3182", builder["matmul"](env.get("InsertedPrecisionFreeCast__3181"), env.get("InsertedPrecisionFreeCast__3155"), {"label":"/model/layers.21/attn/GroupQueryAttention_/Attention/qkv/matmul_1_2813"}));
  env.set("InsertedPrecisionFreeCast__3183", builder["mul"](env.get("InsertedPrecisionFreeCast__3182"), env.get("InsertedPrecisionFreeCast__684"), {"label":"/model/layers.21/attn/GroupQueryAttention_/Attention/qkv/div_2814"}));
  env.set("_3122", builder.expand(env.get("_617"), [1,32,128,512], {"label":"/model/layers.21/attn/GroupQueryAttention_/GQA/GQA_mask_shape_ones/expand_2754"}));
  env.set("_3123", builder.cumulativeSum(env.get("_3122"), 3, {"exclusive":true,"label":"/model/layers.21/attn/GroupQueryAttention_range_of_mask_shape_2755"}));
  env.set("_3119", builder["add"](env.get("_613"), env.get("_3111"), {"label":"/model/layers.21/attn/GroupQueryAttention_/GQA/attn_mask/add_2751"}));
  env.set("_3120", builder.expand(env.get("_3119"), [512,128], {"label":"/model/layers.21/attn/GroupQueryAttention_/GQA/expand_neq_right_2752"}));
  env.set("_3121", builder["transpose"](env.get("_3120"), {"label":"/model/layers.21/attn/GroupQueryAttention_/GQA/neq_right/transpose_2753","permutation":[1,0]}));
  env.set("Inserted_1369", builder["lesser"](env.get("_3123"), env.get("_3121"), {"label":"/model/layers.21/attn/GroupQueryAttention_/GQA/attn_mask/condition_1_2756"}));
  env.set("InsertedPrecisionFreeCast__3125", builder["where"](env.get("Inserted_1369"), env.get("InsertedPrecisionFreeCast__621"), env.get("InsertedPrecisionFreeCast__622"), {"label":"/model/layers.21/attn/GroupQueryAttention_/GQA/attn_mask/where_2758"}));
  env.set("InsertedPrecisionFreeCast__3184", builder["add"](env.get("InsertedPrecisionFreeCast__3183"), env.get("InsertedPrecisionFreeCast__3125"), {"label":"/model/layers.21/attn/GroupQueryAttention_/Attention/attn_mask/softmax_input_2815"}));
  env.set("InsertedPrecisionFreeCast__inlfunc_Softmax_token_20_X_ReduceMax", builder["reduceMax"](env.get("InsertedPrecisionFreeCast__3184"), {"axes":[3],"keepDimensions":true}));
  env.set("InsertedPrecisionFreeCast__inlfunc_Softmax_token_20_X_Sub", builder["sub"](env.get("InsertedPrecisionFreeCast__3184"), env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_20_X_ReduceMax"), {}));
  env.set("InsertedPrecisionFreeCast__inlfunc_Softmax_token_20_X_Exp", builder["exp"](env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_20_X_Sub"), {}));
  env.set("InsertedPrecisionFreeCast__inlfunc_Softmax_token_20_X_ReduceSum", builder["reduceSum"](env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_20_X_Exp"), {"axes":[3],"keepDimensions":true}));
  env.set("InsertedPrecisionFreeCast__3185", builder["div"](env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_20_X_Exp"), env.get("InsertedPrecisionFreeCast__inlfunc_Softmax_token_20_X_ReduceSum"), {}));
  env.set("InsertedPrecisionFreeCast__3186", builder["matmul"](env.get("InsertedPrecisionFreeCast__3185"), env.get("InsertedPrecisionFreeCast__3118"), {"label":"/model/layers.21/attn/GroupQueryAttention_/Attention/qkv/matmul_2_2817"}));
  env.set("_3186", builder.cast(env.get("InsertedPrecisionFreeCast__3186"), "float16"));
  env.set("_3187", builder["transpose"](env.get("_3186"), {"label":"/model/layers.21/attn/GroupQueryAttention_/Attention/qkv/transpose_2818","permutation":[0,2,1,3]}));
  env.set("_3188", builder.reshape(env.get("_3187"), [1,128,2048]));
  env.set("InsertedPrecisionFreeCast__3188", builder.cast(env.get("_3188"), "float32"));
  env.set("InsertedPrecisionFreeCast__3189", builder["matmul"](env.get("InsertedPrecisionFreeCast__3188"), env.get("InsertedPrecisionFreeCast__68"), {"label":"/model/layers.21/attn/o_proj/MatMul_Q4_matmul_2820"}));
  env.set("_3193", builder["add"](env.get("_3192"), env.get("InsertedPrecisionFreeCast__3189"), {"label":"/model/layers.21/post_attention_layernorm/SkipLayerNorm_add_skip_2824"}));
  env.set("_3216", builder.cast(env.get("_3193"), "float16"));
  env.set("_3217", builder.cast(env.get("_3216"), "float32"));
  env.set("_54", builder.dequantizeLinear(env.get("_51"), env.get("_52"), env.get("_53"), {"axis":2,"blockSize":32,"label":"/model/layers.21/mlp/down_proj/MatMul_Q4_dequantizeLinear_3"}));
  env.set("_55", builder.reshape(env.get("_54"), [2048,5632]));
  env.set("_56", builder["transpose"](env.get("_55"), {"label":"/model/layers.21/mlp/down_proj/MatMul_Q4_transpose_dequantizeLinear_5","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__56", builder.cast(env.get("_56"), "float32"));
  env.set("_3207", builder.dequantizeLinear(env.get("_3204"), env.get("_3205"), env.get("_3206"), {"axis":2,"blockSize":32,"label":"/model/layers.21/mlp/gate_proj/MatMul_Q4_dequantizeLinear_2834"}));
  env.set("_3208", builder.reshape(env.get("_3207"), [5632,2048]));
  env.set("_3209", builder["transpose"](env.get("_3208"), {"label":"/model/layers.21/mlp/gate_proj/MatMul_Q4_transpose_dequantizeLinear_2836","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__3209", builder.cast(env.get("_3209"), "float32"));
  env.set("_3200", builder.cast(env.get("_3199"), "float32"));
  env.set("_3194", builder["pow"](env.get("_3193"), env.get("_583"), {"label":"/model/layers.21/post_attention_layernorm/SkipLayerNorm_pow_2825"}));
  env.set("_3195", builder["reduceMean"](env.get("_3194"), {"axes":[2],"keepDimensions":true,"label":"/model/layers.21/post_attention_layernorm/SkipLayerNorm_reduceMean_2826"}));
  env.set("_3196", builder["add"](env.get("_3195"), env.get("_586"), {"label":"/model/layers.21/post_attention_layernorm/SkipLayerNorm_add_2827"}));
  env.set("_3197", builder["sqrt"](env.get("_3196"), {"label":"/model/layers.21/post_attention_layernorm/SkipLayerNorm_sqrt_2828"}));
  env.set("_3198", builder["div"](env.get("_3193"), env.get("_3197"), {"label":"/model/layers.21/post_attention_layernorm/SkipLayerNorm_div_2829"}));
  env.set("_3201", builder["mul"](env.get("_3200"), env.get("_3198"), {"label":"/model/layers.21/post_attention_layernorm/SkipLayerNorm_mul_2831"}));
  env.set("InsertedPrecisionFreeCast__3210", builder["matmul"](env.get("_3201"), env.get("InsertedPrecisionFreeCast__3209"), {"label":"/model/layers.21/mlp/gate_proj/MatMul_Q4_matmul_2837"}));
  env.set("InsertedPrecisionFreeCast__3211", builder["sigmoid"](env.get("InsertedPrecisionFreeCast__3210"), {"label":"/model/layers.21/mlp/act_fn/Sigmoid_2838"}));
  env.set("InsertedPrecisionFreeCast__3212", builder["mul"](env.get("InsertedPrecisionFreeCast__3210"), env.get("InsertedPrecisionFreeCast__3211"), {"label":"/model/layers.21/mlp/act_fn/Mul_2839"}));
  env.set("_60", builder.dequantizeLinear(env.get("_57"), env.get("_58"), env.get("_59"), {"axis":2,"blockSize":32,"label":"/model/layers.21/mlp/up_proj/MatMul_Q4_dequantizeLinear_6"}));
  env.set("_61", builder.reshape(env.get("_60"), [5632,2048]));
  env.set("_62", builder["transpose"](env.get("_61"), {"label":"/model/layers.21/mlp/up_proj/MatMul_Q4_transpose_dequantizeLinear_8","permutation":[1,0]}));
  env.set("InsertedPrecisionFreeCast__62", builder.cast(env.get("_62"), "float32"));
  env.set("InsertedPrecisionFreeCast__3203", builder["matmul"](env.get("_3201"), env.get("InsertedPrecisionFreeCast__62"), {"label":"/model/layers.21/mlp/up_proj/MatMul_Q4_matmul_2833"}));
  env.set("InsertedPrecisionFreeCast__3213", builder["mul"](env.get("InsertedPrecisionFreeCast__3212"), env.get("InsertedPrecisionFreeCast__3203"), {"label":"/model/layers.21/mlp/Mul_2840"}));
  env.set("InsertedPrecisionFreeCast__3214", builder["matmul"](env.get("InsertedPrecisionFreeCast__3213"), env.get("InsertedPrecisionFreeCast__56"), {"label":"/model/layers.21/mlp/down_proj/MatMul_Q4_matmul_2841"}));
  env.set("_3218", builder["add"](env.get("_3217"), env.get("InsertedPrecisionFreeCast__3214"), {"label":"/model/layers.22/final_norm_layernorm/SkipLayerNorm_add_skip_2845"}));
  env.set("_3219", builder["pow"](env.get("_3218"), env.get("_583"), {"label":"/model/layers.22/final_norm_layernorm/SkipLayerNorm_pow_2846"}));
  env.set("_3220", builder["reduceMean"](env.get("_3219"), {"axes":[2],"keepDimensions":true,"label":"/model/layers.22/final_norm_layernorm/SkipLayerNorm_reduceMean_2847"}));
  env.set("_3221", builder["add"](env.get("_3220"), env.get("_586"), {"label":"/model/layers.22/final_norm_layernorm/SkipLayerNorm_add_2848"}));
  env.set("_3222", builder["sqrt"](env.get("_3221"), {"label":"/model/layers.22/final_norm_layernorm/SkipLayerNorm_sqrt_2849"}));
  env.set("_3223", builder["div"](env.get("_3218"), env.get("_3222"), {"label":"/model/layers.22/final_norm_layernorm/SkipLayerNorm_div_2850"}));
  env.set("_3226", builder["mul"](env.get("_3225"), env.get("_3223"), {"label":"/model/layers.22/final_norm_layernorm/SkipLayerNorm_mul_2852"}));
  env.set("InsertedPrecisionFreeCast_logits_44", builder["matmul"](env.get("_3226"), env.get("InsertedPrecisionFreeCast__50"), {"label":"/lm_head/MatMul_Q4_matmul_2854"}));
  env.set("logits_44", builder.cast(env.get("InsertedPrecisionFreeCast_logits_44"), "float16"));

  const outputs = {};
  outputs["logits_44"] = env.get("logits_44");
  outputs["present_0_key_0"] = env.get("present_0_key_0");
  outputs["present_0_value_1"] = env.get("present_0_value_1");
  outputs["present_10_key_20"] = env.get("present_10_key_20");
  outputs["present_10_value_21"] = env.get("present_10_value_21");
  outputs["present_11_key_22"] = env.get("present_11_key_22");
  outputs["present_11_value_23"] = env.get("present_11_value_23");
  outputs["present_12_key_24"] = env.get("present_12_key_24");
  outputs["present_12_value_25"] = env.get("present_12_value_25");
  outputs["present_13_key_26"] = env.get("present_13_key_26");
  outputs["present_13_value_27"] = env.get("present_13_value_27");
  outputs["present_14_key_28"] = env.get("present_14_key_28");
  outputs["present_14_value_29"] = env.get("present_14_value_29");
  outputs["present_15_key_30"] = env.get("present_15_key_30");
  outputs["present_15_value_31"] = env.get("present_15_value_31");
  outputs["present_16_key_32"] = env.get("present_16_key_32");
  outputs["present_16_value_33"] = env.get("present_16_value_33");
  outputs["present_17_key_34"] = env.get("present_17_key_34");
  outputs["present_17_value_35"] = env.get("present_17_value_35");
  outputs["present_18_key_36"] = env.get("present_18_key_36");
  outputs["present_18_value_37"] = env.get("present_18_value_37");
  outputs["present_19_key_38"] = env.get("present_19_key_38");
  outputs["present_19_value_39"] = env.get("present_19_value_39");
  outputs["present_1_key_2"] = env.get("present_1_key_2");
  outputs["present_1_value_3"] = env.get("present_1_value_3");
  outputs["present_20_key_40"] = env.get("present_20_key_40");
  outputs["present_20_value_41"] = env.get("present_20_value_41");
  outputs["present_21_key_42"] = env.get("present_21_key_42");
  outputs["present_21_value_43"] = env.get("present_21_value_43");
  outputs["present_2_key_4"] = env.get("present_2_key_4");
  outputs["present_2_value_5"] = env.get("present_2_value_5");
  outputs["present_3_key_6"] = env.get("present_3_key_6");
  outputs["present_3_value_7"] = env.get("present_3_value_7");
  outputs["present_4_key_8"] = env.get("present_4_key_8");
  outputs["present_4_value_9"] = env.get("present_4_value_9");
  outputs["present_5_key_10"] = env.get("present_5_key_10");
  outputs["present_5_value_11"] = env.get("present_5_value_11");
  outputs["present_6_key_12"] = env.get("present_6_key_12");
  outputs["present_6_value_13"] = env.get("present_6_value_13");
  outputs["present_7_key_14"] = env.get("present_7_key_14");
  outputs["present_7_value_15"] = env.get("present_7_value_15");
  outputs["present_8_key_16"] = env.get("present_8_key_16");
  outputs["present_8_value_17"] = env.get("present_8_value_17");
  outputs["present_9_key_18"] = env.get("present_9_key_18");
  outputs["present_9_value_19"] = env.get("present_9_value_19");
  return await builder.build(outputs);
}
