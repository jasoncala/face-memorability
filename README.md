# Controlling Memorability of Face Images

Vector Internship 2023, Exploratory Work

### **TODO** - Create a new notebook containing only important code snippets and instructions.

The contents of this Github Repo include:

1. Summary slides containing the results of this exploratory work.
2. A summary report giving more context to the slides.
3. A combined Jupyter notebook meant to be used as a reference on how vectors/latents were manipulated.
   - **(DO NOT RUN DIRECTLY, REQUIRES FILES FROM PAPER)**
4. Various direction vectors that can be used to modify semantic traits.
5. Training and evaluation scripts for exploring memorability prediction using deep learning models.

If one would like to begin exploring this topic, it is recommended to read the related works mentioned in the report and start with the supplementary materials provided at: [OpenReview](https://openreview.net/forum?id=tm9-r3-O2lt).

---

## How to Modify Memorability (or Any Trait)

The direction vector, multiplied by a magnitude in either direction, must be added to the image vector following the formula:
**w-edit = w + αw∗**

In the supplementary materials, the memorability direction vector is represented by the variable `w_sq`, as shown in the screenshot below:

![Memorability modification example](doc_images/mem_modify.png)

---

## How to Conditionally Modify Memorability (or Any Trait)

To modify the memorability of an image while keeping another trait constant, subspace projection must be done.

![Subspace Projection Formula](doc_images/subspace_projection.png)

A new direction vector is created through subspace projection of two traits as shown below:

![Conditional smile example](doc_images/conditional_smile.png)

This new direction vector (`cond_smile`) can be applied just as the original. For example, this will modify memorability (`w_sq`) without modifying smile.

---

## Fine tuning files

### **`faces_test.py`**

- Evaluates pretrained models on facial memorability tasks.
- Calculates Spearman’s correlation coefficient to assess the relationship between true and predicted memorability scores.
- Outputs errors and performance statistics for analysis.

### **`vgg_hf_drop.py`**

- Fine-tunes a modified VGGFace model with dropout layers for memorability prediction.
- Includes a custom Euclidean distance loss function for regression.
- Uses callbacks for saving models, logging metrics, and reducing the learning rate on plateau.

### **`vgg_hf.py`**

- Similar to `vgg_hf_drop.py` but without dropout layers.
- Focuses on fine-tuning a simpler version of the VGGFace model for memorability tasks.

### **`vggtrain.py`**

- Another variation of training for VGGFace models.
- Includes slight differences in logging and learning rate configurations compared to `vgg_hf.py`.

### **`res_hf_drop.py`**

- Uses the ResNet50 backbone of VGGFace for memorability prediction.
- Adds a dropout layer and modifies the output layer for regression.
- Implements training with the same pipeline but fine-tuned on the ResNet architecture.

### **`res_train_rf.py`**

- Fine-tunes the ResNet50-based VGGFace model for memorability prediction.
- Similar to `res_hf_drop.py` but without dropout layers, focusing on standard fine-tuning.

### **`restrain.py`**

- Fine-tunes the ResNet50 backbone with adjustments for memorability prediction.
- Adds mechanisms for training and validation, leveraging callbacks for monitoring performance.

### **`sen_hf_drop.py`**

- Fine-tunes the SENet50 backbone of VGGFace with added dropout for regularization.
- Optimized for memorability prediction using a custom loss function.
- Implements training with a focus on using the SENet architecture effectively.

### **`sentrain.py`**

- Similar to `sen_hf_drop.py` but excludes dropout layers.
- Fine-tunes the SENet50 backbone for memorability tasks with simpler configurations.
