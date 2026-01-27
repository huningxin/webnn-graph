# TinyLlama WebNN Conversion Guide

This directory contains the converted TinyLlama (unfused) ONNX model in WebNN format.

## Environment Setup

### Install Rust and build project

1. **Windows**:
   - Visit https://rust-lang.org/learn/get-started/
   - Download and run  `rustup-init.exe`
   - Follow the prompts to complete installation (default options recommended)
   - Restart your terminal to apply environment variables

2. **Verify Installation**:
   ```bash
   rustc --version
   cargo --version
   ```

3. **Build project**:
   ```bash
   cargo build
   ```

## Conversion Workflow

### 1. ONNX → WebNN Text Format (.webnn)

Run from project root directory:

```bash
# Basic conversion with constant folding optimization
cargo run convert-onnx --input model.onnx

# Output files:
# - model.webnn          (WebNN graph definition)
# - model.weights        (binary weights file)
# - model.manifest.json  (weights manifest)
```

**Parameters**:
- `--optimize`: Enable constant folding to compute dynamic Shape operations at conversion time
- `--override-dim`: Override dynamic dimension values (`--override-dim batch_size=1`) 
- `--inline-weights`: Inline small weights into graph file (useful for small models)

### 2. WebNN Text Format (.webnn) → JavaScript Code

```bash
# Generate WebNN builder code
cargo run emit-js model.webnn > buildGraph.js
```

Generated JavaScript file contains:
- `WeightsFile` class: Load and manage binary weights
- `buildGraph()` function: Build WebNN computation graph


## Browser Usage

```javascript
import { WeightsFile, buildGraph } from './TinyLlama_unfuse.js';

// Load weights and build graph
const weights = await WeightsFile.load(
  'TinyLlama_unfuse.weights',
  'TinyLlama_unfuse.manifest.json'
);

const context = await navigator.ml.createContext();
const graph = await buildGraph(context, weights);

// Prepare input data
const inputIds = new Int32Array([1, 2, 3, ...]);
const attentionMask = new Int32Array([1, 1, 1, ...]);

// Run inference
const outputs = await context.compute(graph, {
  input_ids: inputIds,
  attention_mask: attentionMask
});

console.log('Logits:', outputs.logits);
```
