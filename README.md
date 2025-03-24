# Automated Resume Parsing Using Named Entity Recognition (NER) with NLP

## Introduction
A crucial Natural Language Processing (NLP) method for identifying and categorizing significant information in text is Named Entity Recognition (NER). One of its most useful applications is automated resume parsing, which assists businesses in processing a large number of resumes quickly by locating important information like:

- **Personal Information** (Name, contact details)
- **Education** (Degrees, institutions)
- **Work Experience** (Job titles, company names)
- **Skills** (Technical proficiencies, languages)

The **Resume NER Dataset**, which offers resumes annotated with 36 entity categories, facilitates the building of highly accurate NER models. Businesses can improve efficiency and recruitment decision-making by automating talent acquisition using NER for resume processing.

**[Dataset Link](https://github.com/vrundag91/Resume-Corpus-Dataset)**

## Methodology
Several important steps in the structured pipeline are used to achieve automatic resume parsing utilizing Named Entity Recognition (NER) and NLP:

### 1. Data Collection and Preprocessing
The **Resume NER Dataset** from GitHub, which includes resumes labeled with 36 entity types like names, skills, education, work experience, organizations, and locations, is used.

- Named entity recognition (NER) algorithms are trained using this dataset to extract structured data from unstructured resume texts.
- **Preprocessing Steps:**

  - **Tokenization:** Breaking text into individual words, subwords, or phrases.
    - **Methods:**
      - Whitespace Tokenization (splitting by spaces)
      - Punctuation-Based Tokenization (removing punctuation)
      - Subword Tokenization (used in Transformer models like BERT)
    - **Libraries:** `NLTK`, `spaCy`

  - **Lowercasing & Stopword Removal:**
    - Lowercasing standardizes content while eliminating words that don't add value.
    - Stopword removal filters out common words like *is, the, and*.
    - **Example:**
      - Original: "She has experience in data science and AI."
      - After stopword removal: "experience data science AI."
    - **Libraries:** `Scikit-learn`, `NLTK`, `spaCy`

  - **Lemmatization & Stemming:**
    - **Lemmatization:** Converts words to dictionary forms based on meaning.
      - *Example:* "better" → "good", "running" → "run"
    - **Stemming:** Removes suffixes to get word roots.
      - *Example:* "flies" → "fli"
    - **Libraries:** `NLTK`, `spaCy`

  - **Handling Special Characters:**
    - Removes unnecessary symbols (`#`, `@`, `_`, `*`) unless relevant for entity recognition.
    - Retains meaningful symbols like (`.` in *Ph.D.*)
    - **Tool:** Regular expressions (`re` in Python)

### 2. Named Entity Recognition (NER) Model Selection
To extract important resume elements, various NER models are considered:

- **Rule-Based Approaches:** Using handcrafted regex patterns for entity extraction.
- **Machine Learning-Based Approaches:**
  - Conditional Random Fields (CRF) or Hidden Markov Models (HMM) for sequence labeling.
- **Deep Learning-Based Approaches:**
  - **BiLSTM-CRF:** Combines Bidirectional Long Short-Term Memory (BiLSTM) with CRF.
  - **Transformer-based Models:** `BERT`, `RoBERTa`, or `SpaCy’s` pre-trained NER models fine-tuned on the dataset.

### 3. Model Training and Evaluation
- Dataset is split into **Training, Validation, and Test Sets** (e.g., 80-10-10).
- Model is trained using optimization methods like **Adam Optimizer** and **categorical cross-entropy loss**.
- **Evaluation Metrics:**
  - Precision
  - Recall
  - F1-score

### 4. Post-Processing and Data Structuring
- Extracted entities are assigned to pre-established categories like Name, Education, and Skills.
- Data is structured in **Database, CSV, or JSON** format for easy HR system integration.

### 5. Deployment and Integration
- The trained model is integrated with **Applicant Tracking Systems (ATS)** via APIs.
- **Deployment Frameworks:** `Flask`, `FastAPI`, `Django`
- The system is tested on actual resumes for performance improvement.

### 6. Continuous Learning and Optimization
- The model is periodically **retrained** using updated resume datasets.
- **Optimization Techniques:**
  - Hyperparameter tuning
  - Data augmentation
  - Active learning
# Analysis
# Model Performance Summary

Based on classification performance on the test dataset, we contrast the spaCy deep learning NER model with the CRF-based model.

| Metric                | CRF Model | spaCy NER Model |
|-----------------------|-----------|----------------|
| **Overall Accuracy**  | 90%       | 93%            |
| **Weighted F1 Score** | 0.8957    | 0.91           |
| **Best Performing Entities** | Name, Email, Degree | Name, Email, Location |
| **Worst Performing Entities** | Years of Experience, Graduation Year, Skills | Years of Experience, Graduation Year, Skills |

Although the accuracy of both models was good, there are notable differences in entity-specific performance.  On structured items such as Name and Degree, the CRF model performs better than the spaCy model, which has a little higher accuracy.

## Key Observations from Model Evaluations

### A. CRF Model Analysis

#### Strengths:
- High precision for non-entity terms was attained (F1-score of 0.95), suggesting that the model distinguishes entity tokens from non-entity tokens with effectiveness.
- Performed well on structured entities such as **Name (F1-score: 0.75), Email Address (0.63), and Degree (0.73).**
- Higher recall for **College Name and Degree** compared to spaCy's NER model.

#### Weaknesses:
- Low recall for **Graduation Year (0.12) and Years of Experience (0.00),** suggesting the model struggles with numerical entity extraction.
- **Companies Worked At and Designation** categories have low recall (~0.35), meaning the model misses many valid cases.
- **Skills entity detection (F1-score: 0.50)** is suboptimal, likely due to the broad and diverse nature of skill-related terms.

### B. spaCy NER Model Analysis

#### Strengths:
- Higher recall on **Location and Designation** entities compared to the CRF model.
- Overall higher accuracy **(93%)** and better contextual entity recognition.
- Captures **Name, Email, and Location** entities with high accuracy, making it suitable for general entity extraction.

#### Weaknesses:
- Extremely low recall for **Skills (F1-score: 0.19)** → Many skill-related terms are being missed.
- **Graduation Year and Years of Experience show F1 scores of 0.00,** indicating the model completely fails to recognize these entities.
- Fluctuations in training loss (as seen in the loss curve) indicate instability in early training iterations.

## Training Loss Reduction (Deep Learning Model)
The training loss curve for the spaCy NER model shows:
- An initial steep drop in loss (~85% reduction after the first iteration), indicating rapid learning in early stages.
- Fluctuations in loss values between **iterations 2-5,** suggesting instability or underfitting.
- Loss stabilizes after **iteration 6,** meaning the model starts to converge.
- ![image](https://github.com/user-attachments/assets/6d09feb4-ad3a-4488-8f8f-2166993d1a5e)


## Entity-Level Performance Comparison

| Entity Type            | CRF Model F1-score | spaCy NER Model F1-score |
|------------------------|--------------------|--------------------------|
| **College Name**       | 0.63               | 0.54                     |
| **Companies Worked At**| 0.48               | 0.36                     |
| **Degree**            | 0.73               | 0.62                     |
| **Designation**       | 0.41               | 0.41                     |
| **Email Address**     | 0.63               | 0.67                     |
| **Graduation Year**   | 0.18               | 0.00                     |
| **Location**         | 0.38               | 0.44                     |
| **Name**             | 0.75               | 0.94                     |
| **Skills**           | 0.50               | 0.32                     |
| **Years of Experience** | 0.00             | 0.00                     |

## Findings:
- **CRF** performs better on **Degree, College Name, and Skills.**
- **spaCy model** is significantly better at detecting **Names, Emails, and Locations.**
- **Both models struggle with numerical entities** (Graduation Year, Years of Experience).

## References

- Dash, A., Darshana, S., Yadav, D. K., & Gupta, V. (2024). A clinical named entity recognition model using pretrained word embedding and deep neural networks. *Decision Analytics Journal, 10*, 100426.
- Li, M., Zhou, H., Yang, H., & Zhang, R. (2024). RT: a Retrieving and Chain-of-Thought framework for few-shot medical named entity recognition. *Journal of the American Medical Informatics Association, 31(9)*, 1929-1938.
- Tikayat Ray, A., Pinon Fischer, O. J., White, R. T., Cole, B. F., & Mavris, D. N. (2024). Development of a language model for named-entity-recognition in aerospace requirements. *Journal of Aerospace Information Systems, 21(6)*, 489-499.
- Zhang, Y., & Xiao, G. (2024). Named entity recognition datasets: a classification framework. *International Journal of Computational Intelligence Systems, 17(1)*, 71.
