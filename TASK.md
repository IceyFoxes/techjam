# Task 3: Implement a GPU Kernel for a Transformer Layer

## Source Status

- Source: Problem statement supplied by the repository owner.
- Last updated by the organizer: 27 August 2026, 6:25 PM.
- Previous source status (superseded 29 August 2026): "The referenced image and
  Feishu-only Appendix test-shape content were not included in the supplied text.
  Do not infer the missing shapes; add them with their official source when
  available."
- Organizer update: Appendix: Test Shapes added and `torch_transformer_benchmark.py` updated.
- Organizer Appendix screenshot supplied by the repository owner as
  [`task_shapes.png`](task_shapes.png) on 29 August 2026. The table is transcribed
  in Section 3.7 below.
- Technical Workshop Webinar with Q&A: 28 August 2026, 3:00 PM to 3:45 PM.
- Webinar recording: stated to be uploaded by 29 August 2026, 12:00 PM.

## 3.1 Background

Transformer is a widely used neural network architecture in modern AI. It is the core structure behind many natural language processing, computer vision, speech, recommendation, and large language model systems.

The main idea of Transformer is self-attention. Self-attention allows each token in a sequence to interact with other tokens directly. Compared with recurrent models, Transformer can process tokens in parallel, which makes it suitable for GPU acceleration.

Given an input sequence represented as a matrix:

$$
X \in \mathbb{R}^{N \times d}
$$

where $N$ is the sequence length and $d$ is the hidden dimension, the Transformer first projects the input into Query, Key, and Value matrices:

$$
Q = XW_Q
$$

$$
K = XW_K
$$

$$
V = XW_V
$$

The scaled dot-product attention is computed as:

$$
\operatorname{Attention}(Q,K,V) =
\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

where $d_k$ is the dimension of each attention head. The scaling factor $\sqrt{d_k}$ is used to prevent the dot-product values from becoming too large, which could make the softmax distribution unstable.

However, the computation of Transformer is expensive. Important operations include matrix multiplication, attention score calculation, softmax, normalization, and feed-forward layers. These operations may be limited by GPU compute throughput, memory bandwidth, cache efficiency, kernel launch overhead, and tensor core utilization.

In this competition, participants are asked to use AI-assisted methods to optimize the runtime efficiency of a Transformer structure on a given GPU model. The optimized implementation should improve performance while keeping the output numerically correct compared with the reference implementation.

Participants may consider optimization methods such as operator fusion, memory layout optimization, reduced-precision computation, tensor core usage, softmax optimization, and custom CUDA, Triton, or PyTorch implementations.

The goal of this task is to explore how AI can help developers analyze Transformer workloads, identify bottlenecks, and generate more efficient implementations for specific GPU hardware.

## 3.2 Problem Statement

- Given a fixed formula of a Transformer layer, participants need to submit one or several GPU kernels that implement the layers and pass the given test cases.
- The test cases are written in PyTorch. Participants can modify the layer implementation and decide which parts of the layers should be fused into one kernel.
- The test case will compare the participant implementation with the original PyTorch implementation. As implemented by [`torch_transformer_benchmark.py`](torch_transformer_benchmark.py), each output element passes when its absolute error is at most `0.002` or its error is at most `0.02 * abs(reference)`.
- Test cases will contain different input shapes, including large and small batch sizes, sequence lengths, and dimensions. Participants can select implementations by shape checks. All input-shape combinations are intended to be disclosed to participants.
- AI tools are encouraged so participants can implement different kernels for different input shapes in limited time.
- Optimize and test code on your own machine. Different optimization methods may be appropriate for different GPU models.
- Provide a clear technical report, including details of the AI skills and tools used, for bonus points.

Participants need to:

1. Use the repository's [`torch_transformer_benchmark.py`](torch_transformer_benchmark.py) benchmark script.
2. Implement and optimize the customized-implementation section using AI assistance or by hand.
3. Run the script on their own machine.
4. Provide a clear technical report covering the environment, including CPU, GPU, and disk; optimizations performed; and final test results.

The originally supplied text referenced an image after the
customized-implementation instruction but did not embed it. The Appendix image is
now available separately as [`task_shapes.png`](task_shapes.png).

## 3.3 Constraints and Scope

| Category | Constraints and scope |
| --- | --- |
| In scope | AI-based code generation, GPU kernel fusion, profiling tools, and related optimization work. |
| Out of scope | Production-ready deployment. |

## 3.4 Available Resources and Data

Use the repository's PyTorch benchmark: [`torch_transformer_benchmark.py`](torch_transformer_benchmark.py).

Treat this root benchmark as the immutable reference. Place working implementations under `src/`; if benchmark changes are needed for experimentation, modify a copy under `src/` rather than the original file.

## 3.5 Deliverables

### 1. Written Project Description on Devpost

Provide a clear written description that includes:

- How the solution addresses the problem statement.
- Development tools used, such as VS Code, Colab, or Jupyter.
- APIs used, such as OpenAI GPT-4o or Google Maps API.
- Libraries and frameworks used, such as Hugging Face Transformers, PyTorch, scikit-learn, or pandas.
- Datasets and assets used, such as Google Local Reviews or manually labelled data.

### 2. Public Code or GitHub Repository

Submit a link to a public repository containing well-structured, commented code for all solution components and a README that includes:

- Project overview.
- Setup and installation instructions.
- Steps to reproduce results.
- A brief reflection on limitations and potential improvements.
- Team member contributions, if applicable.

### 3. Demo Video

Submit a short video that:

- Demonstrates the solution working end to end, such as inference results, a dashboard, or model predictions.
- Is uploaded to YouTube with public visibility.
- Is linked from the Devpost description.
- Does not include third-party trademarks or copyrighted content without permission.

For backend or NLP tracks where a front-end interface is not applicable, a walkthrough showing API usage, inference examples, or result analysis is accepted.

## 3.6 Judging Criteria

| Criterion | Definition | Weight |
| --- | --- | ---: |
| Technical Execution | Strong engineering fundamentals, well-structured code, thoughtful architecture, effective API or model usage, reliable demonstration, and deliberate technical complexity. | 35% |
| Innovation and Problem Insight | Originality, sharp problem framing, clear importance, and a direct solution. | 20% |
| Impact and Relevance | Potential value to real users or stakeholders, meaningful reach, tangible benefit, and relevance beyond the prompt. | 20% |
| Feasibility and Practicality | A realistic path beyond a prototype, proportionate resource use, sustainable architecture, and grounded implementation. | 15% |
| Presentation and Communication (Final Event Only) | Clear communication and a coherent problem-to-solution story, with depth when answering questions. | 10% |

## 3.7 Appendix: Test Shapes

### Current status (29 August 2026)

The organizer's Appendix table is available in [`task_shapes.png`](task_shapes.png).
The table uses `QKV Dim` for the benchmark's `d_model` value. All disclosed cases
are causal.

| # | Batch Size | QKV Dim (`d_model`) | Heads | Seq Len | Layers | Causal | FFN Dim |
| ---: | ---: | ---: | ---: | ---: | ---: | :---: | ---: |
| 1 | 64 | 128 | 4 | 128 | 4 | TRUE | 128 |
| 2 | 1 | 128 | 4 | 128 | 4 | TRUE | 128 |
| 3 | 4 | 128 | 4 | 128 | 4 | TRUE | 128 |
| 4 | 16 | 128 | 4 | 128 | 4 | TRUE | 128 |
| 5 | 128 | 128 | 4 | 128 | 4 | TRUE | 128 |
| 6 | 10000 | 128 | 4 | 128 | 4 | TRUE | 128 |
| 7 | 64 | 32 | 4 | 128 | 4 | TRUE | 32 |
| 8 | 64 | 1024 | 4 | 128 | 4 | TRUE | 1024 |
| 9 | 64 | 128 | 1 | 128 | 4 | TRUE | 128 |
| 10 | 64 | 128 | 2 | 128 | 4 | TRUE | 128 |
| 11 | 64 | 128 | 16 | 128 | 4 | TRUE | 128 |
| 12 | 64 | 128 | 4 | 32 | 4 | TRUE | 128 |
| 13 | 64 | 128 | 4 | 1024 | 4 | TRUE | 128 |
| 14 | 32 | 1024 | 16 | 100000 | 2 | TRUE | 1024 |

### Previous status (superseded 29 August 2026)

> The supplied statement says this content is only supported in Feishu Docs, but
> it does not contain the test-shape table. The official shape list remains
> required before shape-specific optimization and complete benchmark validation
> can be planned.

This status was superseded when the repository owner supplied the organizer's
Appendix screenshot. The original wording is retained here as required by the
repository's non-destructive research policy.
