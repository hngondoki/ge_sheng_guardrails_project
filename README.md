
# Sheng Guardrails Project

## Introduction

In Kenya, Sheng, an evolving informal language that fuses Swahili and English, is widely used, especially among the youth. While this linguistic hybrid fosters cultural expression and social cohesion, it poses significant challenges for Natural Language Processing (NLP) systems trained predominantly on standard English.

Current chatbot models, and other service providers provide In-Topic (IT) guardrails to determine whether user queries fall within predefined subject boundaries. However, these models often have limited linguistic comprehension for low resource languages such as Sheng due to limited training data and insufficient linguistic representation. This leads to poor topic classification, decreased user trust, and reduced overall utility of AI-driven services in underrepresented communities.

## Objective

The goal of this project is to explore model options that can classify text and ensure that generative AI systems respond only with vetted, contextually relevant content, thereby enhancing both the accuracy and safety of conversational AI in with low resource languages.

## Data
WARNING: The information contained in the datasets is of a sensitive nature. 

⚠️ Content Advisory:
Please be advised that the datasets referenced in this presentation contain sensitive and potentially explicit material, particularly related to mental, sexual, and reproductive health. This content is included strictly for academic and research purposes, and is presented with the intent to foster informed, respectful, and evidence-based discussion. Viewer discretion is advised.

wz_qa.csv: This is a file with 308 sheng data (questions and answers) that has been vetted and labelled already. The whole data set is IT and the questions + Answers will be combined to provide the vocabulary within the sentence embeddings. It has 2 columns, question and answer
wz_eng_labeled.csv: This contains 117 rows of data in english with labelled classifications of IT/OOT.
human_generated_training_data.csv: This is a 78 Row  list of manually generated and labelled sheng data
wz_inference_data.csv: This is a 25455 rows of sheng data that is unlabelled. 
human_labeled_inference_results.csv: A subset of 450 rows of sheng data generated and manually labelled as a subset
small_test_data: This is a manual generated and labelled sheng data listing 10 items to allow quick evaluation of accuracy through visual inspection (eyeballing).
<img width="3721" height="315" alt="image" src="https://github.com/user-attachments/assets/91ef0f35-1195-464f-a1d0-13026f4f96ac" />


## Approach
There are 5 main areas in this document:

1.   Pre-processing: This is where the imports, and helper functions are domiciled.The purpose is to optimize reusable code for the sections that follow next.
2.   Understanding the data set: This contains the description of the datasets being used across the experiments. It is predominantly what has been availed by Girl effect, however, in this section you can swap to your intended low resource languages.
3. Experiments: This section will be further sub divided into 2 large experiments.
This section will be divided into two main experiments.
The first experiment focuses on small datasets, tested across four different approaches (tracks), with the aim of identifying which approach achieves the highest accuracy and F1 score. The tracks are:
*   Track 1: The use of cosine similarity and traditional classifiers for a decision boundary using kmeans, logistic regression and Random Forest.
* Track 2: Multilingual Embedding Model Evaluation which explored the applicability of multilingual and cross-lingual models to the classification task.
* Track 3: Few-Shot Learning with OpenAI by Providing Labeled Examples in the Prompt for Direct Classification
* Track 4: Translation and Embedding via OpenAI, followed by classification with logistic regression.

The second experiment builds on Few-Shot Learning with OpenAI using 2 different prompts and 2 different openAI models () and comparing results to benchmark our first experiment. The objective is to assess whether the traditional models can perform as well as the modern OpenAI models.

## Results and Analysis
### Track 1:
In the first track, several sentence embedding models were used -  all-mpnet-base-v2, all-MiniLM-L6-v2 and text-embedding-3-large. The objectives are: 1) the effect of embedding model choice, and (2) the efficacy of different classifiers (logistic regression, k-means, and random forest) in establishing decision boundaries.

On the embedding comparison, the embedding models,all-mpnet-base-v2 and all-MiniLM-L6-v2, yielded similar cosine similarity distributions, with hardly any impact observed on the subsequent classification task. However, the distribution of text-embedding-3-large seemed more with no display of normality. However, it produced the highest accuracy and F1-scores across classifiers, especially with Logistic Regression and Random Forest.There was significant overlap in prediction classes classes. This indicates that the semantic meanings in Sheng across both IT and OOT classes are close to each other that it limits the ability of the classification models to determine a concise decision boundary.
| Classifier              | Accuracy (mpnet / MiniLM / 3-large) | F1-score (mpnet / MiniLM / 3-large) | ROC-AUC (mpnet / MiniLM / 3-large) |
| ----------------------- | ----------------------------------- | ----------------------------------- | ---------------------------------- |
| **Logistic Regression** | 0.79 / 0.78 / 0.84                  | 0.88 / 0.88 / 0.90                  | 0.55 / 0.55 / 0.65                 |
| **K-means**             | 0.33 / 0.44 / 0.73                  | 0.50 / 0.29 / 0.80                  | 0.78 / 0.80 / 0.21                 |
| **Random Forest**       | 0.83 / 0.81 / 0.84                  | 0.89 / 0.88 / 0.90                  | 0.81 / 0.73 / 0.75                 |



On the classification task performance, Random Forest performs consistently well across all embeddings, especially with F1-score (up to 0.90) and ROC-AUC (up to 0.81) indicating a strong balance between precision and recall and a good ability to distinguish classes. While K-means with text-embedding-3-large surprisingly shows decent F1-score (0.80) and accuracy (0.73), its ROC-AUC is very poor (0.21) highlighting that it's not consistently distinguishing between classes, despite some surface-level performance. Thus Random forest with AUC of >70% and accuracy/f1 scores >81%  implies a good useable model, while K-means and LR need more improvement.

### Track 2:

This track explored an additional NLP multi-lingual models ( Davlan/afro-xlmr-mini, facebook/contriever, facebook/msmarco, sentence-transformers/LaBSE, xlm-roberta-base). Despite trying several of these models the perfomance remained low at between 43% and 56% on the higher side. While these models are well trained for multilingual tasks, their performance on this dataset suggests limitations in effectively capturing the nuances of Sheng.facebook/contriever is the only model showing meaningful learning in this setup. There was no fine-tuning used to durther investigate the efficacy of these models to the classification task, hence the results are limited and should not be used to infer any performance.

Comparison table:

| Model    | Validation loss | Accuracy | F1 score |
| -------- | ------- | ------- | ------- |
| Davlan/afro-xlmr-mini  |0.740749  | 0.500000 | 0.333333
| facebook/msmarco | 1.381521   | 0.500000 | 0.333333 |
| facebook/contriever | 0.651121 | 0.562500 | 0.458937 |
| xlm-roberta-base| 0.707880 | 0.500000 | 0.333333 |
| sentence-transformers/LaBSE| 0.959065 | 0.500000 | 0.333333 |

### Track 3:

This track used OpenAI's gpt-3.5-turbo-instruct and gpt-4o-mini with a few shot prompting approach where some samples were included in the prompt as examples for the classification task. gpt-4o-mini Outperforms across all metrics. It shows a better balance between precision and recall (F1 = 0.835) and higher AUC score(0.86), making it more reliable overall..

| Metric        | gpt-4o-mini | gpt-3.5-turbo-instruct |
| ------------- | ----------- | ---------------------- |
| **Accuracy**  | 0.833       | 0.7430                 |
| **F1 Score**  | 0.835       | 0.7436                 |
| **AUC Score** | 0.86        | 0.73                   |

Comparing gpt-4o-mini vs Random Forest using text-embedding-3-large
Accuracy is nearly identical, with Random Forest clearly exceeding in the F1 score at 90% thus it is better in addressing class imbalance. However, GPT-4o-mini is better at distinguishing between classes across thresholds with the AUC score at 86%. This is further supported because gpt-4o-mini has better text understanding, handling of ambiguity and is highly adaptable given the large copra of data it is trained on. Random Forest on the other hand can be useful where there is structured data, constrains by cost or infrastructure, and there is desire for a transparent, and interpretable model.

| Metric   | **gpt-4o-mini** | **Random Forest (3-large)** |
| -------- | --------------- | --------------------------- |
| Accuracy | 0.833           | 0.84                        |
| F1 Score | 0.835           | 0.90                        |
| ROC-AUC  | 0.86            | 0.75                        |


### Track 4:

This track explored whether translating Sheng text to English could improve classification outcomes. Openai was used for translation and embeddings using it's text-embedding-3-large model. Notably, the embeddings were more skewed to the left, differentiating the openAI embeddings from the previous embedding models. Despite the classification inconsitencies experienced in track 1, logistic regression was used to define a decision boundary. The translated dataset were used as vocabulary embeddings, and logistic regression applied. However, performance was poor with an accuracy of 0.4375 and F1 score of 0.4705. Given the previous track results, further translation experiments on translations were not pursued.


## Conclusion
Random Forest emerges as a highly effective alternative to the classification task given its performance was comparable to modern models frrom OpenAI.Although the sentence embedding and classifier pipelines demonstrated that the underlying data is not linearly separable, Random Forest was able to define a decision boundary through the ambiguity with casualties in False negatives and false positives, hence there is a trade off. A Large language model like GPT-4 can generate flexible, non-linear reasoning paths within current context making them more perfoemant for noisy, overlapping datasets of which Random Forest cannot beat.
