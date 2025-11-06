# Frequency-guided composite infrared-visible image fusion for progressive visual structure and detail learning.
<img width="2339" height="694" alt="image" src="https://github.com/user-attachments/assets/d10c7020-fb39-4f49-8c7d-102366ca31a3" />

## abstract
Infrared-Visible image fusion aims to integrate complementary information from both imaging modalities to produce images with enhanced visual and semantic details. However, an effective feature fusion for simultaneously modeling visual low-frequency structure and high-frequency details still remain challenging.

To address this issue, we propose a frequency-guided feature composite network in this paper for infrared-visible image fusion. The network begins with a shared feature encoder to extract modality- irrelevant representations, and a dual-branch architecture is designed to separately encode low-frequency structural features and high-frequency detail features. A Two-stage Composite Fusion Module (TCFM) is then elaborated to progressively fuse intra-frequency and inter-frequency features through a combination of intra-block connections and inter-block alignment, enabling more effective feature interaction across frequency domains. In addition, a Pixel-Gradient Hybrid Loss (PGHL) is introduced to jointly constrain pixel-level and gradient-level information, thereby preserving structural features while enhancing visual details.

Extensive experiments on infrared-visible datasets show that our method achieves superior visual quality and quantitative performance improvement when compared with state-of-the-art (SOTA) fusion methods, demonstrating the effectiveness of the proposed frequency-guided feature composite network to progressively strengthen visual structure and detail learning.
## Project Structure
```plaintext
|-- core
|   |-- block.py
|   |-- dataset.py
|   |-- high_extractor.py
|   |-- loss.py
|   |-- low_extractor.py
|   |-- net.py
|-- fig
|-- log
|-- models
|   |-- model_loss_group_epoch100.pth
|-- test_log
|-- tools
|   |-- evaluator.py
|   |-- utils.py
|-- config.py
|-- README.md
|-- test.py
|-- train.py
|-- requirements.txt
```
**explain**：
- **core**:
  - Core Function Code：
      - **block.py**:Blocks that may be used in the network
      - **dataset.py**:Dataset processing code
      - **high_extractor.py**:High-frequency processing network module
      - **loss.py**:Loss function module
      - **low_extractor.py**:Low-frequency processing network module
      - **net.py**:Network module
- **fig**:
  - The architecture of our network
- **log**：
    - Save the trained model file. After the training is completed, the model will be saved in this directory for subsequent testing and use
- **models**：
    - The model in the article
- **test_log**：
    - Test result storage location
- **tools**：
    - Tool modules that may be used in the project:
        - **evaluator.py**：Evaluation metrics calculation code
        - **utils.py**：Utils code
- **config.py**：
    - Configuration file
- **test.py**：
    - test file
- **train.py**：
    - train file
- **requirements.txt**：
    - List the Python libraries required by the project and their versions. You can install them quickly using `pip install -r requirements.txt` .

## Instructions for Use
### train
Enter the main directory and use the following command to start the training:
```bash
python train.py
```
### test
After the training is completed, use the following command for testing:
```bash
python test.py
```
