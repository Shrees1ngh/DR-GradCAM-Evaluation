# Project Understanding Document: Evaluating Grad-CAM Explanations for Diabetic Retinopathy Grading

> **Project:** B.Tech CSE Minor Project  
> **Institution:** HMR Institute of Technology and Management  
> **Team:** Shree Bhagwan, Shivam Tiwari, Sushant Kumar  
> **Supervisor:** Dr. Pradeep  
> **Status:** Draft for Review  

---

## 1. About This Document

This document explains our whole project in simple English. It starts from the basics of the eye and diabetes. Then it explains the problem and the idea. After that it explains what we will build and how. The last parts help in the viva with questions and answers and a glossary of difficult words.

The format follows a simple SRS (Software Requirements Specification) style. An SRS tells what a system is for, what it must do and what it needs.

---

## 2. Where the Idea Came From

The idea came from a Smart India Hackathon problem statement called *Explainable AI for Diabetic Retinopathy Screening in Rural India*. That problem was very big. It asked for image quality checks, structure segmentation, grading, explanation reports and a Simulink simulation of a screening program.

We did not take all of that. We took one important part from it: the explanation part. Our minor project asks a focused research question: **When an AI shows a heatmap to explain its decision, can we trust that heatmap?**

Our base paper is:  
*Biswas et al., "Interpreting Deep Neural Networks in Diabetic Retinopathy Grading: A Comparison with Human Decision Criteria", Life 2025.*

---

## 3. Medical Background

This part is important for the viva. If you know these words well you will sound confident.

### 3.1 Parts of the Eye We Need
* **Retina:** A thin layer at the back of the eye. It senses light and sends signals to the brain. Think of it as the film of a camera.
* **Fundus:** The inside back surface of the eye. It includes the retina, the blood vessels, the optic disc and the macula. When doctors say fundus they mean the part of the eye they can see through the pupil.
* **Fundus image:** A colour photo of the fundus. A special camera called a fundus camera takes it. The patient sits in front of the camera and the photo takes a few seconds.
* **Optic disc:** The bright round spot where the optic nerve leaves the eye. It carries signals to the brain. It is a normal part of the eye and not a disease.
* **Macula:** The small central area of the retina. It gives sharp central vision for reading and recognising faces.
* **Fovea:** The very centre of the macula. It gives the sharpest vision.
* **Blood vessels:** Tiny arteries and veins spread over the retina. They bring oxygen and food to it.

### 3.2 Diabetes and the Eye
Diabetes means the sugar level in the blood stays too high. Over many years this high sugar damages the walls of small blood vessels in the body. The retina has many very small vessels. So the retina gets damaged early.

This eye disease is called **Diabetic Retinopathy (DR)**. In DR the vessel walls become weak. They can leak fluid and blood. They can also get blocked. When blood flow drops the retina becomes short of oxygen. The eye then tries to grow new vessels. These new vessels are weak and cause more trouble.

DR is dangerous because it is usually silent in the early stages. The patient sees fine and feels no pain. By the time vision becomes bad the damage can be serious. DR is one of the leading causes of blindness in working age adults. The good news is that most vision loss can be prevented if DR is found early. This is why regular eye screening for every diabetic person is so important.

The base paper quotes the International Diabetes Federation. It says about 103 million of 537 million diabetic people had DR in 2021. India has one of the largest diabetic populations in the world. India also has very few eye specialists compared to its population and most of them are in cities. So many people in villages never get screened.

### 3.3 Signs of DR in a Fundus Image (Lesions)
A lesion means an abnormal spot or damage. These are the lesions that matter for us:
* **Microaneurysms (MA):** Tiny round bulges in small vessel walls. They look like very small red dots. They are the earliest sign of DR. They are so small that even doctors can miss them.
* **Haemorrhages (HE):** Bleeding in the retina. They look like red dots or blots of different sizes.
* **Hard exudates (EX):** Yellow fatty spots that leak out of weak vessels. They have sharp edges.
* **Soft exudates (SE):** White fluffy patches. They are also called cotton wool spots. They appear when a part of the retina does not get enough blood.
* **IRMA:** Intraretinal microvascular abnormalities; abnormal tiny vessels inside the retina. A sign of severe disease.
* **Venous beading:** Veins that look like a string of beads. A sign of severe disease.
* **Neovascularisation (NV):** Growth of new weak vessels. These bleed easily. This is the sign of the most dangerous stage.
* **Vitreous haemorrhage:** Bleeding into the jelly-like fluid in the middle of the eye. It can cause sudden loss of vision.

There is also **Diabetic Macular Oedema (DME)**. It is swelling in the macula. It is a separate condition but it also causes vision loss. Our project focuses on DR grades only.

### 3.4 The Five DR Grades (ICDR Scale)
Doctors use the International Clinical Diabetic Retinopathy (ICDR) scale. It has five levels and the IDRiD dataset also uses it:
* **Grade 0, No DR:** No signs of disease.
* **Grade 1, Mild:** Only microaneurysms.
* **Grade 2, Moderate:** More than just microaneurysms. Haemorrhages and exudates can be present. Not yet severe.
* **Grade 3, Severe:** Many haemorrhages in all four areas of the retina or venous beading or IRMA. No new vessels yet.
* **Grade 4, Proliferative DR (PDR):** New vessels or vitreous bleeding. Most dangerous.

Grades 0 to 3 together are called **NPDR (non-proliferative DR)**. Grade 4 is **PDR (proliferative DR)**. A patient with Grade 2 or above is called **referable DR**. This means the patient should be sent to an eye specialist.

### 3.5 Treatment (Only for Understanding)
Treatment depends on the stage. Early stages need good control of blood sugar and blood pressure and regular check-ups. Advanced stages can need laser treatment or injections in the eye or surgery. We are not building anything for treatment. We only deal with screening images.

---

## 4. The Problem

### 4.1 Medical Side
Doctors must look at fundus images one by one to grade DR. This takes time and skill. There are too few specialists for the number of diabetic patients. So screening is slow and many patients are missed.

### 4.2 AI Side
Deep learning models can now grade DR from fundus images with good accuracy. The model takes an image and gives a grade from 0 to 4. But the model is a black box. It does not tell us why it chose that grade. A doctor will not fully trust a decision without a reason.

To solve this researchers use a method called **Grad-CAM**. It draws a coloured heatmap on the image. Red areas show where the model paid the most attention. This looks very convincing.

### 4.3 The Real Problem We Study
A heatmap that looks good may still be misleading. Here is the catch:
* If the red area sits on a real lesion this does not prove the model used that lesion for its decision. The model may have decided from something else.
* Also the red area may sit on a place with no lesion and yet that place may be truly important for the model.

So there are two different things to check:
1. **Localization:** Does the heatmap point to the right place? That means the place where the doctor marked a lesion.
2. **Faithfulness:** Did that highlighted place really matter for the model's decision?

Many people measure only localization and assume faithfulness comes with it. We want to measure both separately and see how they relate.

### 4.4 Simple Analogy
Imagine a doctor says you have fever because your throat is red. Your throat may really be red. That is correct localization. But the doctor may have actually decided from your temperature chart and only pointed to the throat to look convincing. Then the explanation was not faithful. A good explanation must be both.

---

## 5. Aim and Objectives

### 5.1 Aim
To study how reliable Grad-CAM explanations are for diabetic retinopathy grading by measuring localization and faithfulness separately and comparing them across all DR grades.

### 5.2 Objectives
* Train a CNN model that grades DR from 0 to 4 using fundus images.
* Generate Grad-CAM heatmaps for its predictions.
* Measure localization by comparing heatmaps with expert marked lesion masks using IoU and Dice score.
* Measure faithfulness using the Deletion Test.
* Compare the two results statistically and check how the relation changes from Grade 0 to Grade 4.
* Write a research paper on the findings.

---

## 6. Existing Work and Our Difference

### 6.1 What Exists Today
* **Manual screening:** Specialists grade images by eye. Accurate but slow and costly.
* **AI grading systems:** Many research models and a few approved products such as IDx-DR and EyeArt grade DR automatically. Most of them focus on accuracy.
* **Heatmap explanations:** Many papers show Grad-CAM pictures only as a visual proof. They do not test if the pictures are reliable.
* **Localization studies:** Some papers compare heatmaps with lesion masks. Our base paper does this on IDRiD. Another study (arXiv 2608.28207) does it with Grad-CAM and HiResCAM. These studies find very low overlap scores. They also say overlap alone does not prove the explanation is faithful.
* **Faithfulness studies:** Some DR papers test deletion and insertion. One of them found that Integrated Gradients was more faithful than Grad-CAM. But those studies do not connect the result with lesion overlap.
* **Grade-wise study:** One paper studies how Grad-CAM attention changes with grade. But it does not run a deletion test.
* **Other medical fields:** A brain tumour MRI study argues that localization and faithfulness are separate properties and should be measured separately.

### 6.2 Our Unique Contribution
To the best of our knowledge we are not aware of a DR study that does all of these together:
* It measures localization and faithfulness in the same pipeline.
* It does this on the IDRiD dataset.
* It compares both results grade by grade from 0 to 4.
* It checks statistically whether good localization goes with good faithfulness.

We should never say that nobody has done this. We say that we found no study doing exactly this combination and that our work fills this gap.

---

## 7. What We Will Build

We will not build a website or a mobile app as the main output. This is a research project. The main output is a research pipeline. It is a set of Python notebooks and scripts that run on Google Colab or Kaggle with a free GPU.

### 7.1 Main Outputs
* A trained DR grading model.
* A pipeline that creates Grad-CAM heatmaps and calculates localization and faithfulness scores for every image.
* Result tables and graphs for each grade.
* A research paper.
* A minor project report and a presentation.

### 7.2 Optional Output
A small demo web page built with Streamlit or Gradio. A user uploads a fundus image. The page shows the predicted grade with confidence the Grad-CAM heatmap and the deletion test graph. This is only for a good demo in the viva. It is not required.

### 7.3 How It Will Be Used
The user here is a researcher or a student like us. The steps are simple:
* Open the notebook on Colab.
* Load the dataset.
* Run the training cell to get the model.
* Run the explanation cells to get heatmaps and scores.
* Run the analysis cells to get tables and graphs.

If we build the demo page then any person can upload one image and see the output without touching code.

---

## 8. How the System Works

The flow from start to end is as follows:
* **Step 1. Input:** A fundus image.
* **Step 2. Preprocessing:** Resize and normalise the image. Augment training images by flipping or rotating.
* **Step 3. Model:** A CNN predicts a grade from 0 to 4 with a confidence value.
* **Step 4. Grad-CAM:** It creates a heatmap for the predicted grade.
* **Step 5. Localization test:** Convert the heatmap into a black and white mask. Compare it with the doctor's lesion mask using IoU and Dice.
* **Step 6. Faithfulness test:** Hide the highlighted region in the image. Run the model again. Note how much the confidence falls.
* **Step 7. Analysis:** Group the scores by grade. Find the correlation between the two scores. Draw graphs.

### 8.1 A Worked Example
The numbers below are made up. They only help you picture the process. Imagine one fundus image whose true grade is 3.
1. The model predicts Grade 3 with a confidence of 0.80.
2. Grad-CAM draws a heatmap. We keep only the hottest 10 percent of the heatmap and turn it into a black and white mask. The base paper also used a high percentile cut like this.
3. The doctor's lesion mask for this image is already available. We compare the two masks. The overlap is tiny. IoU is 0.03 and Dice is 0.06. This is low localization.
4. We now hide the highlighted area in the image and run the model again. The confidence falls from 0.80 to 0.35. The drop is 0.45. This is a big drop.
5. As a control we hide a random area of the same size. The confidence falls only from 0.80 to 0.72. The drop is 0.08.

**How to read this:** The heatmap covers only a small part of the lesions so localization is low. But hiding it hurts the model a lot more than hiding a random area. So the highlighted area truly mattered. The explanation is faithful even though the overlap is low. This is exactly the kind of case where looking at overlap alone would mislead us.

We repeat this for every image. Then we average the scores grade by grade and check the correlation between the two scores.

---

## 9. Technical Details Explained Simply

### 9.1 CNN
A Convolutional Neural Network is a type of AI model made for images. It learns to find patterns like edges and spots and shapes in layers. Early layers find simple patterns. Later layers find complex patterns.

### 9.2 Transfer Learning and Fine Tuning
Training a CNN from nothing needs a huge number of images. We have only a few hundred. So we take a model that is already trained on millions of normal images such as ImageNet. We then train it a little more on fundus images. This is called fine tuning and the whole idea is called transfer learning. We plan to use ResNet50 or EfficientNet.

### 9.3 Confidence
The model gives a probability for each grade. The probability of the predicted grade is its confidence. For example 0.85 means the model is 85 percent sure.

### 9.4 Grad-CAM
Grad-CAM stands for Gradient weighted Class Activation Mapping. It looks at the last convolution layer of the model and uses gradients to find which image areas pushed the model toward its chosen grade. It then draws these areas as a heatmap. The heatmap is rough and low resolution. So it cannot trace very tiny lesions exactly. This is a known weakness and we will mention it in the paper.

### 9.5 IoU and Dice
Both measure how much two masks overlap. Value 0 means no overlap. Value 1 means perfect overlap.
* **IoU (Intersection over Union):** The overlapping area divided by the total area covered by both masks together.
* **Dice:** Two times the overlapping area divided by the sum of the sizes of both masks.

Previous papers got very low scores such as 0.005 to 0.03. So low values are normal and not a sign that our work is wrong.

### 9.6 Deletion Test
This test checks if the highlighted area really mattered. We hide the red area of the heatmap for example by filling it with black or blur. We then run the model again. If the confidence falls a lot then the area was important to the model. That means the explanation is faithful. If the confidence barely changes the explanation did not reflect what the model used.

A good practice is to also hide a random area of the same size as a control. If hiding the heatmap area hurts more than hiding a random area then the heatmap is truly meaningful. We should add this control in our experiments.

### 9.7 Statistical Analysis
We will calculate a correlation such as Spearman correlation between the localization score and the confidence drop. A value near 0 means no link. A positive value means better localization goes with better faithfulness. We will do this for each grade separately. Because we have few images we should also report confidence intervals or p values and be careful about strong claims.

### 9.8 Model Evaluation Terms
In the viva someone may ask how good your model is. These words help you answer:
* **Train, validation and test split:** We divide the images into three groups. The model learns from the training group. The validation group is used to tune settings. The test group is used only at the end to check the model on images it never saw.
* **Data leakage:** When test images accidentally get used in training. This makes results look better than they are. We must avoid it.
* **Accuracy:** The share of images where the predicted grade is exactly right.
* **Confusion matrix:** A table that shows which grades the model mixes up. Mild and Moderate are usually mixed up the most.
* **Sensitivity:** Out of the truly diseased cases how many the model catches.
* **Specificity:** Out of the truly healthy cases how many the model correctly calls healthy.
* **Quadratic weighted kappa (QWK):** A score for ordered grades. Predicting Grade 4 for a Grade 3 image is a smaller mistake than predicting Grade 0. QWK gives less penalty for near misses and more for far misses. It is a popular score for DR grading.

Our project is not about getting the best accuracy. A decent model is enough because our real goal is to test its explanations. But we should still report these scores honestly so that readers know how good the model is.

---

## 10. Dataset: IDRiD

IDRiD means Indian Diabetic Retinopathy Image Dataset. It comes from an eye clinic in India and is free for academic use. Images are very large at 4288 by 2848 pixels. It has three parts:
* **Disease Grading:** 516 images with DR grade labels from 0 to 4. 413 are for training and 103 for testing. We use it to train and test the model.
* **Segmentation:** 81 images with pixel level lesion masks drawn by experts. The masks cover microaneurysms and haemorrhages and hard exudates and soft exudates. The optic disc is marked too. We use it for localization.
* **Localization:** Marks for the optic disc and fovea centres. We may not need this part.

### Important Data Limits
* Lesion masks exist only for the 81 segmentation images. So localization can be done only on those images.
* Grade 0 images have no lesions. So IoU and Dice have no meaning there. We need to decide how to handle Grade 0. One option is to report only faithfulness for Grade 0.
* The grade labels are unequal. There are many Grade 0 and Grade 2 images and fewer of others.
* We need to check how many of the 81 images fall in each grade before we promise any grade-wise result.

---

## 11. Technology Stack

* **Language:** Python
* **Deep learning:** PyTorch with torchvision for ResNet50 or EfficientNet
* **Grad-CAM:** The pytorch-grad-cam library / native PyTorch hook implementation
* **Image work:** OpenCV and NumPy
* **Metrics and statistics:** scikit-learn and SciPy
* **Graphs:** Matplotlib and Seaborn
* **Environment:** Google Colab or Kaggle notebooks with free GPU / local Jupyter Notebook
* **Version control:** GitHub
* **Writing:** MS Word or LaTeX
* **Optional demo:** Streamlit or Gradio and Hugging Face Spaces for free hosting

---

## 12. Requirements (SRS Style)

### 12.1 Functional Requirements
* The system shall load fundus images and their grade labels.
* The system shall preprocess and augment the images.
* The system shall train a model that predicts DR grade from 0 to 4.
* The system shall report model accuracy and a confusion matrix.
* The system shall generate a Grad-CAM heatmap for any image.
* The system shall convert a heatmap into a binary mask using a threshold.
* The system shall calculate IoU and Dice against lesion masks.
* The system shall run the Deletion Test and record the confidence drop.
* The system shall group all scores by grade and calculate correlation.
* The system shall save tables and graphs.

### 12.2 Non Functional Requirements
* **Reproducibility:** Fix random seeds. Save settings so another person can repeat our results.
* **Clarity:** Code must be well organised and commented.
* **Performance:** The full pipeline must run on a free Colab or Kaggle GPU.
* **Honesty:** Results must be reported as they come even if they are negative.

### 12.3 Hardware and Software Needs
* A laptop with at least 8 GB RAM.
* Free GPU on Colab or Kaggle.
* Python and the libraries listed above.
* IDRiD dataset download.

### 12.4 Assumptions and Constraints
* The expert lesion masks are treated as the ground truth.
* The dataset is small so findings may not apply to every population.
* This system is for research. It is not a medical device and must not be used to diagnose real patients.

---

## 13. Expected Results

We do not know the results in advance. These are all valid findings:
* Localization is low but faithfulness is high: The model uses small parts of lesions and not whole lesions.
* Localization is good but faithfulness is low: The model looks at lesions but decides from something else.
* Both are good or both are poor.
* The relation changes with grade: For example it may be stronger in severe grades where lesions are bigger.

Any of these is a real finding and can go in the paper. A negative result is also a result.

---

## 14. Who Benefits

* **Doctors:** They learn when a heatmap can be trusted and when it should not be.
* **AI developers:** They get a method to test explanations beyond just looking at them.
* **Screening programmes:** Safer AI tools can help screen more people in places with few specialists.
* **Patients:** In the long run better trust in AI helps earlier detection.
* **Us:** We get a complete research experience with a paper.

---

## 15. Risks and Limitations

* Very small number of lesion mask images.
* Grad-CAM is coarse so IoU will be low by nature.
* The result depends on the threshold used to make the binary mask. We should test a few thresholds.
* Results depend on the model and one architecture may not represent all.
* Class imbalance can make the model weak on some grades.
* Hiding a region can make the image look unnatural so the model may react to the black patch itself. The random region control helps here.

---

## 16. Viva Questions and Simple Answers

* **Q1. What is your project in one line?**  
  We check whether Grad-CAM heatmaps for diabetic retinopathy grading are really reliable by testing both where they point and whether that place truly matters to the model.

* **Q2. What is diabetic retinopathy?**  
  It is an eye disease caused by long term high blood sugar. It damages the small blood vessels of the retina. It can lead to blindness if not found early.

* **Q3. What is a fundus image?**  
  It is a colour photo of the inside back of the eye taken with a fundus camera. It shows the retina and its blood vessels.

* **Q4. What are the five grades?**  
  No DR, Mild, Moderate, Severe, and Proliferative. They are numbered 0 to 4.

* **Q5. What is a black box model?**  
  A model that gives an answer without telling why.

* **Q6. What is Grad-CAM?**  
  A method that shows which image areas influenced the model's decision as a heatmap.

* **Q7. What is the difference between localization and faithfulness?**  
  Localization asks if the heatmap is at the right place. Faithfulness asks if that place really influenced the decision.

* **Q8. How do you measure localization?**  
  By comparing the heatmap mask with expert lesion masks using IoU and Dice.

* **Q9. How do you measure faithfulness?**  
  By the Deletion Test. We hide the highlighted region and see how much the confidence drops.

* **Q10. Why are your IoU values so low?**  
  Grad-CAM is coarse and lesions are tiny. Earlier papers also report very low values. Also the model may use only a small part of a lesion.

* **Q11. Has this been done before?**  
  Parts of it have. Localization studies and faithfulness studies exist. We found no study that does both together grade by grade on IDRiD. We say this to the best of our knowledge.

* **Q12. Why did you choose IDRiD?**  
  It is from an Indian clinic and it has both grade labels and expert lesion masks. Both are needed for our two tests.

* **Q13. Why transfer learning?**  
  Our dataset is small. A model already trained on millions of images learns faster and better with little data.

* **Q14. Is this a website or an app?**  
  The main work is a research pipeline in Python notebooks. A small demo page is optional.

* **Q15. What are your limitations?**  
  Small number of mask images and coarse heatmaps and no lesions in Grade 0. We state these openly in the paper.

* **Q16. How is this useful in real life?**  
  It helps decide how much doctors can trust AI explanations in DR screening.

* **Q17. What if good localization does not mean good faithfulness?**  
  That is a valid and useful finding. It would show that looking at overlap alone is not enough to trust an explanation.

---

## 17. Glossary

* **Retina:** Light sensing layer at the back of the eye.
* **Fundus:** Inner back surface of the eye.
* **Fundus camera:** Camera that photographs the fundus.
* **Optic disc:** Spot where the optic nerve leaves the eye.
* **Macula:** Central retina for sharp vision.
* **Fovea:** Centre of the macula.
* **Lesion:** An abnormal spot or damage.
* **Microaneurysm:** Tiny bulge in a small vessel. Earliest sign of DR.
* **Haemorrhage:** Bleeding.
* **Exudate:** Fluid or fat leaking out of vessels. Hard ones are yellow and soft ones are white.
* **Neovascularisation:** Growth of new weak vessels.
* **NPDR:** Non-proliferative DR. Grades 0 to 3.
* **PDR:** Proliferative DR. Grade 4.
* **Referable DR:** Grade 2 or above. Needs a specialist.
* **ICDR:** The international five level DR grading scale.
* **CNN:** AI model type for images.
* **Transfer learning:** Reusing a model trained on other data.
* **Fine tuning:** Training a pretrained model a little more on our data.
* **Confidence:** The probability the model gives to its chosen grade.
* **Black box:** A model whose reasons are not visible.
* **XAI:** Explainable AI.
* **Grad-CAM:** Method that makes heatmaps to explain a CNN.
* **Heatmap:** Coloured map showing attention strength.
* **Localization:** Heatmap points at the true lesion area.
* **Faithfulness:** Highlighted area truly drove the decision.
* **Mask:** A black and white image marking an area.
* **IoU:** Overlap score between two masks.
* **Dice:** Another overlap score between two masks.
* **Deletion Test:** Hide highlighted area and measure the confidence drop.
* **Correlation:** A number showing how two things move together.
* **Ground truth:** The trusted correct answer. Here it is the expert marks.

---

## 18. Final Scope and Completion Checklist

This section was added after we checked a longer project blueprint. It keeps our work close to our synopsis. It also adds a few good practices that make the results more trustworthy. The work is split into Must Do and Should Do and Optional.

### 18.1 The Main Question We Must Answer
Our synopsis asks one main question: **does good localization mean good faithfulness?** So for every image we must calculate both scores and then check the correlation between them. This is the heart of the project. Everything else only supports it.

### 18.2 Must Do (Core Work)
* Check the IDRiD files and labels and masks before writing any code.
* Train one good baseline model such as ResNet50 or EfficientNet.
* Generate Grad-CAM heatmaps.
* Calculate IoU and Dice on the images that have lesion masks. For Grade 0 mark the result as not applicable. Do not treat it as zero.
* Run the Deletion Test with a random region control.
* Find the correlation between localization and faithfulness grade by grade.
* Make honest tables and graphs and write a clear limitations part.

### 18.3 Should Do (Makes the Research Stronger)
* **Threshold sensitivity:** A heatmap becomes a mask only after we choose a cut value. For example we keep the top 5 or 10 or 20 percent of the heat. Try a few cut values and report how IoU and Dice change. Do not show only the best looking one.
* **Deletion curve:** Do not hide the area only once. Hide it in steps such as 10 and 20 and 30 percent of the hottest area. Write down the confidence after each step and draw a curve. A faster fall means the area matters more to the model. Do the same with random areas and compare the two curves.
* **Fill check:** A black patch can confuse the model by itself. So also try a blurred fill once and see if the conclusion stays the same.
* **Confidence intervals:** With few images the average can be unstable. A bootstrap picks images again and again at random and recalculates the average many times. This shows how much the average can move.
* **Paired comparison:** Salient deletion and random deletion use the same images. So compare them as pairs and not as two separate groups.
* **More model scores:** Report macro F1 and QWK along with accuracy because the grades are uneven and ordered.

### 18.4 Optional (Only If Time Is Left)
* Compare Grad-CAM with Grad-CAM++.
* Train a second model such as another CNN and check if the conclusion stays the same.
* Test on another dataset such as APTOS or Messidor-2. These datasets have no lesion masks so they can only help with classification.
* Build a nice demo page.

These are not in our submitted synopsis. So we should not promise them in the report. If we finish them we can mention them as extra work.

### 18.5 Things to Confirm Before Coding
* The exact IDRiD folder layout and file names and the licence rules.
* Whether patient IDs are given. This is needed for a safe split so the same patient is never in both training and testing.
* Whether we combine all lesion types into one mask or check each type separately.
* Which model we use and where we run it.
* Which Grad-CAM layer we use and how we normalise the heatmap.
* Fix the cut values and the deletion steps before we look at the final results. If we choose them after seeing results we may pick the ones that look best without meaning to.

### 18.6 When Is the Project Complete
* [x] Dataset files are checked and the split is written down.
* [x] The baseline model is trained and tested on held out images (pipeline ready & verified).
* [x] Grad-CAM heatmaps can be made again with the same result.
* [ ] IoU and Dice are calculated on images with lesion masks and Grade 0 is handled properly.
* [ ] Deletion results exist for both salient and random regions.
* [ ] Correlation between localization and faithfulness is calculated grade by grade where the sample size allows.
* [ ] Tables and graphs are ready.
* [ ] The report and paper explain the method and results and limitations.
* [ ] The report clearly says this is research only and not for real patient diagnosis.

---

## 19. Nine-Phase Implementation Roadmap

### Phase 1 — Project Setup & Dataset Audit *(Completed ✅)*
* Project folder structure and GitHub setup.
* Python environment and dependencies.
* IDRiD dataset download and extraction.
* Dataset paths configure karna.
* Images, labels, masks aur grade distribution verify karna.
* Dataset audit report generate karna.  
**Deliverable:** Working project structure + dataset audit report (`reports/dataset_audit_report.md`).

### Phase 2 — Data Preparation & Preprocessing *(Completed ✅)*
* Fundus images load karna.
* Image-label mapping verify karna.
* Image resizing and normalization ($512 \times 512$, aspect-ratio preserving).
* Training augmentation.
* Train, validation and test split strategy.
* Segmentation masks ko corresponding images se map karna.
* Data leakage checks.  
**Deliverable:** Clean and reproducible dataset pipeline (`reports/phase_2_completion_report.md`).

### Phase 3 — CNN Model Development *(Pipeline Implemented & Verified ✅)*
* ResNet50 / EfficientNet architecture select karna.
* Transfer learning and fine-tuning.
* Five-class DR grading (Grade 0–4).
* Model training on Google Colab GPU.
* Validation and hyperparameter tuning.
* Model checkpoint save karna.  
**Deliverable:** Trained DR classification model (`src/model.py`, `scripts/train.py`).

### Phase 4 — Model Evaluation
* Accuracy, precision, recall and F1-score.
* Confusion matrix.
* Quadratic Weighted Kappa (QWK).
* Per-grade performance.
* Test set evaluation.  
**Deliverable:** Model performance report.

### Phase 5 — Grad-CAM Explainability *(Pipeline Implemented & Verified ✅)*
* Trained model ke liye Grad-CAM implement karna.
* Predicted class ke heatmaps generate karna.
* Heatmaps ko original fundus images par overlay karna.
* Different grades ke explanations visualize karna.
* Heatmaps save and organize karna.  
**Deliverable:** Grad-CAM visualization pipeline (`src/explainability.py`, `tests/test_explainability.py`).

### Phase 6 — Localization Evaluation
* Grad-CAM heatmaps ko binary masks mein convert karna.
* Multiple threshold values evaluate karna (top 5%, 10%, 20%).
* Expert lesion masks ke saath comparison.
* IoU and Dice scores calculate karna.
* Grade-wise localization analysis.  
**Deliverable:** Localization scores and visualizations.

### Phase 7 — Faithfulness Evaluation
* Deletion Test implement karna.
* Important heatmap regions progressively remove karna (10%, 20%, 30%).
* Confidence drop measure karna.
* Random-region control implement karna.
* Deletion curves generate karna.
* Faithfulness scores calculate karna.  
**Deliverable:** Faithfulness evaluation pipeline.

### Phase 8 — Statistical Analysis & Research Findings
* Localization vs faithfulness comparison.
* Spearman correlation.
* Grade-wise statistical analysis.
* Confidence intervals and p-values.
* Tables, graphs and interpretation.
* Findings and limitations document karna.  
**Deliverable:** Complete experimental results.

### Phase 9 — Finalization & Presentation
* Complete research paper.
* Minor project report.
* Results and methodology presentation.
* Viva preparation.
* Optional Streamlit/Gradio demo.  
**Deliverable:** Final research project and presentation.

---

## 20. References

* **Biswas S. et al.** *Interpreting Deep Neural Networks in Diabetic Retinopathy Grading: A Comparison with Human Decision Criteria.* Life 2025, 15, 1473.
* *Explainable Diabetic Retinopathy Classification Using Vision Foundation Models.* arXiv:2608.28207.
* *Systematic Comparison of Heatmapping Techniques in Deep Learning in the Context of Diabetic Retinopathy Lesion Detection.* PMC7774113.
* *Toward Reliable Diabetic Retinopathy Screening.* MDPI Sensors 2026.
* *Localization Gains from Explanation Regularization Should Not Be Interpreted as Faithfulness Gains.* arXiv:2610.26292.
* **Selvaraju R. R. et al.** *Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization.* ICCV 2017.
* **Petsiuk V., Das A., Saenko K.** *RISE: Randomized Input Sampling for Explanation of Black-box Models.* BMVC 2018.
* **Porwal P. et al.** *Indian Diabetic Retinopathy Image Dataset (IDRiD).* Data 2018.

