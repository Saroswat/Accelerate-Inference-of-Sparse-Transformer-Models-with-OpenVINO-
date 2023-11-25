# Accelerate Inference of Sparse Transformer Models with OpenVINO™ and 4th Gen Intel&reg; Xeon&reg; Scalable Processors

This tutorial illustrates how to enhance the inference performance of sparse Transformer models using [OpenVINO](https://docs.openvino.ai/) on 4th Gen Intel® Xeon® Scalable Processors.

The tutorial involves downloading a BERT-base model, which has been quantized, sparsified, and fine-tuned for [SST2 datasets](https://huggingface.co/datasets/sst2) using [Optimum-Intel](https://github.com/huggingface/optimum-intel). The performance advantage is demonstrated on Scalable Processors by employing [Sparse Weight Decompression](https://docs.openvino.ai/2023.0/openvino_docs_OV_UG_supported_plugins_CPU.html#sparse-weights-decompression), a runtime option that leverages model sparsity for efficiency. The tutorial is structured as follows:

## Prerequisites [$\Uparrow$](#Table-of-content:)

```bash
%pip install -q "openvino>=2023.1.0"
%pip install -q "git+https://github.com/huggingface/optimum-intel.git" datasets onnx onnxruntime
```

## Imports [$\Uparrow$](#Table-of-content:)

```python
import shutil
from pathlib import Path

from optimum.intel.openvino import OVModelForSequenceClassification
from transformers import AutoTokenizer, pipeline
from huggingface_hub import hf_hub_download
```

### Download, Quantize, and Sparsify the Model, Using Hugging Face Optimum API [$\Uparrow$](#Table-of-content:)

The first step involves downloading a quantized sparse transformer model translated to OpenVINO IR. Subsequently, the model is subjected to a simple classification to validate its functionality. For details on the quantization and sparsification process, refer to the [OpenVINO/bert-base-uncased-sst2-int8-unstructured80](https://huggingface.co/OpenVINO/bert-base-uncased-sst2-int8-unstructured80) model card on Hugging Face.

```python
# Specify the quantized, sparsified model ID
model_id = "OpenVINO/bert-base-uncased-sst2-int8-unstructured80"

# Set up and download the model to HF Cache folder
ov_model = OVModelForSequenceClassification.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Initialize the sentiment classifier pipeline
sentiment_classifier = pipeline("text-classification", model=ov_model, tokenizer=tokenizer)

# Sample text for inference
text = "He's a dreadful magician."
outputs = sentiment_classifier(text)

print(outputs)
```

For benchmarking, the OpenVINO benchmark application is utilized, and the IRs are organized into a single folder.

```python
# Create a folder for the quantized sparse model
quantized_sparse_dir = Path("bert_80pc_sparse_quantized_ir")
quantized_sparse_dir.mkdir(parents=True, exist_ok=True)

# Download IRs to the folder
ov_ir_xml_path = hf_hub_download(repo_id=model_id, filename="openvino_model.xml")
ov_ir_bin_path = hf_hub_download(repo_id=model_id, filename="openvino_model.bin")

# Copy IRs to the specified folder
shutil.copy(ov_ir_xml_path, quantized_sparse_dir)
shutil.copy(ov_ir_bin_path, quantized_sparse_dir)
```

## Benchmark Quantized Dense Inference Performance [$\Uparrow$](#Table-of-content:)

Benchmark the dense inference performance using parallel execution on four CPU cores to simulate a small instance in a cloud infrastructure. The sequence length is set to 64 as an example, but it is recommended to tune based on specific applications.

```python
# Dump benchmarking config for dense inference
with (quantized_sparse_dir / "perf_config.json").open("w") as outfile:
    outfile.write(
        """
        {
            "CPU": {"NUM_STREAMS": 4, "INFERENCE_NUM_THREADS": 4}
        }
        """
    )

# Run the benchmarking application
!benchmark_app -m $quantized_sparse_dir/openvino_model.xml -shape "input_ids[1,64],attention_mask[1,64],token_type_ids[1,64]" -load_config $quantized_sparse_dir/perf_config.json
```

## Benchmark Quantized Sparse Inference Performance [$\Uparrow$](#Table-of-content:)

To enable the sparse weight decompression feature, users can add it to the runtime config as shown below. `CPU_SPARSE_WEIGHTS_DECOMPRESSION_RATE` takes values between 0.5 and 1.0, acting as a layer-level sparsity threshold for enabling a layer.

```python
# Dump benchmarking config for sparse inference
with (quantized_sparse_dir / "perf_config_sparse.json").open("w") as outfile:
    outfile.write(
        """
        {
            "CPU": {"NUM_STREAMS": 4, "INFERENCE_NUM_THREADS": 4, "CPU_SPARSE_WEIGHTS_DECOMPRESSION_RATE": 0.75}
        }
        """
    )

# Run the benchmarking application with sparse inference configuration
!benchmark_app -m $quantized_sparse_dir/openvino_model.xml -shape "input_ids[1,64],attention_mask[1,64],token_type_ids[1,64]" -load_config $quantized_sparse_dir/perf_config_sparse.json
```

## When This Might Be Helpful [$\Uparrow$](#Table-of-content:)

This feature is beneficial for improving inference performance when handling multiple requests in parallel asynchronously, particularly with small sequence lengths (e.g., 32 and lower). For more details on asynchronous inference with OpenVINO, refer to the following documentation:

- [Deployment Optimization Guide](https://docs.openvino.ai/2023.0/openvino_docs_deployment_optimization_guide_common.html#doxid-openvino-docs-deployment-optimization-guide-common-1async-api)
- [Inference Request API](https://docs.openvino.ai/2023.0/openvino_docs_OV_UG_Infer_request.html#doxid-openvino-docs-o-v-u-g-infer-request-1in-out-tensors)
