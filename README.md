## STInterLLM: Causal Intervention is Large Language Models Need for Large-scale Spatio-temporal Forecasting
## Abstract
> Spatio-temporal forecasting plays a crucial role in the dynamic perception of smart cities, such as traffic flow prediction, renewable energy forecasting, and load prediction. It aims to understand the trends of spatio-temporal changes under the interaction of different factors. Sufficient high-quality data and powerful models are the foundation for accurate spatio-temporal forecasting. Unfortunately, data sparsity and privacy protection result in a lack of adequate prior information (e.g., spatial structures and high-resolution data) in spatio-temporal forecasting. This directly affects the ability of the model to learn and infer cross spatio-temporal and cross-scale features. To address this issue, we propose a novel model named Spatio-temporal Intervention Large Language Model (STInterLLM). This model employs a newly designed causal intervention encoder and Chain-of-Action instruction text to update the spurious correlations in the spatio-temporal adaptive graph and enforce the decomposition of the prediction process, thereby enhancing the causal representation of features while mitigating hallucinations in LLMs. Finally, a lightweight marker alignment module ensures the consistency of the encoder, instruction text, and LLM, enabling accurate forecasting of large-scale spatio-temporal evolution patterns. Extensive experiments are conducted on power distribution systems and urban traffic systems, and the proposed STInterLLM consistently achieves state-of-the-art performance across significantly different scenarios.
> 
![image](https://github.com/lishijie15/DEFMN/blob/ba745b7380de1ae4a9ee5819471696b75e07d402/pictures/DEFMN.png)
> 
### Introduction of Our Model

* To better adapt to various scenarios lacking high-quality data, we propose the STInterLLM, which integrates a novel encoder and instruction-tuning to achieve efficient and accurate large-scale spatio-temporal prediction while minimizing double hallucinations.
* To precisely eliminate spurious spatio-temporal associations caused by adaptive graphs, we specifically design the lightweight causal inference and intervention mechanism named causal intervention encoder. This is integrated with multi-scale temporal feature extraction modules to help the encoder enable the LLM to understand and learn complex spatio-temporal evolution patterns.
* To effectively mitigate hallucinations in the LLM, we construct a novel Chains-of-Action instruction-tuning and alignment module. By enforcing step-by-step decomposition and examples of predictive actions, ensuring that the LLM aligns with and comprehends the complex information provided by the proposed encoder.
* To fully evaluate the performance of the model, we conduct extensive experiments based on the PDS (Scenario 1) and NYC-taxi (Scenario 2) datasets. The proposed STInterLLM achieves SOTA performance across various spatio-temporal scenarios.


### Installation and Run

#### Datasets

The complete load and DG datasets will be made publicly available later.

#### Requirements

STInterLLM is compatible with PyTorch==2.0.1 versions.

All neural networks are implemented using PyTorch and trained on 8 NVIDIA A800 80GB GPUs / 8 NVIDIA H100 80GB GPUs.

Please code as below to install some nesessary libraries.

```
pip install 
```



#### Run Our Model

To run our model, please execute the following code in the directory as `./`.

```
bash causalgpt_train.sh
```

