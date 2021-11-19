# Welcome to your Jupyter Book

# Project Idea

Reference/Inspiration:
Alfini, A., Albert, M., Faria, A. V., Soldan, A., Pettigrew, C., Wanigatunga, S., ... & Spira, A. P. (2021). Associations of actigraphic sleep and circadian rest/activity rhythms with cognition in the early phase of Alzheimer’s disease. Sleep Advances, 2(1), zpab007. doi: 10.1093/sleepadvances/zpab007

## 1) what topic/idea: assuming that I don't know much about the topic you propose, explain why it is important and why the results from studying that topic matters, and what has been done before and what hasn't been yet

Older adults commonly experience changes in both the stability and timing of their rest-activity rhythms as well as changes in neurocognitive function. My research is focused on the link between these two facets of aging. In clinical and research settings, rest-activity patterns can be captured objectively and non-invasively through the use of actigraphy, which involves an individual wearing a wrist-based accelerometer as they go on about their daily lives. This data can then be summarized using measures which reflect different aspects of the circadian rhythm (e.g., amplitude, acrophase, intradaily variability, etc).

While a number of studies report that rest-activity rhythm measures are associated with pathological aging and cognitive decline, there is little agreement in the field as to which rest-activity measures best capture these patterns. Studies using two common approaches, the cosinor model and non-parametric approach, often report disparate results as to which measures capture the variance in cognitive outcomes. Not only can these standard measures be calculated in slightly different ways, the measures themselves are also not independent (e.g., rhythm amplitude is reflective of both day-to-day stability and mean activity levels, rhythm amplitude and rhythm minimum (how much movement occurs during the sleep pattern) are correlated, etc.). This constrains the interpretability and replicability of research findings.

For this reason, I am interested in using a data-driven approach to identify which aspects of rest-activity rhythms (based on actigraphy recordings) are important for cognition across the lifespan. 

1.	PCA - Rather than relying on standard rest-activity rhythm measures, I would like to apply PCA to from preprocessed actigraphy data (7 days, SR=1/30 Hz, 20160 values per subject), and then examine cross-sectional relationships among these components, cognitive performance, and brain biomarkers (e.g., hippocampal volume). 
2.	Decision Tree - If there are components that are significantly correlated with neurocognitive measures, I could then qualitatively describe the component (apparent bed time, wake time, etc) as well as see how it compares to standard rest-activity and sleep measures. For this, I could use a decision tree analysis, to see which (if any) combinations of rest-activity measures capture the information in the component. For this step, there are 11 rest-activity measures I would include. I could maybe do a separate analysis with the sleep measures (total sleep time, sleep efficiency, wake after sleep onset, sleep latency) to see whether that tree achieves a similar accuracy level. 

## 2) data: what type of data you will work on: amount of the data, dimensionality, and whether you have an access to the data ideal to answer your question or not

Predictors: I plan to use data collected by our lab under an NIH NIA grant. This consists of actigraphy, MRI, and neuropsychological assessment data from healthy young and older adults. The maximum sample size I have to work with is 118 individuals, with roughly equal sample size per age group. 

Outcomes: I would use age-normed z-scores from neuropsychological assessments of memory and executive function as cognitive measures. I would use neural features which are implicated in cognitive aging (which I have already extracted), such as hippocampal volume, corpus callosum fractional anisotropy, and retrieval-related functional connectivity.

I am not sure whether I should apply a classification algorithm (e.g., SVM) to predict a discrete outcome (e.g., split cognitive performance into high vs low, predict old vs young, etc.) or use k-fold cross validated regression and keep the outcomes continuous.

![](consort.png)

## 3) implementation: are you going to implement from scratch or you will rely on some existing packages; do you have all the resources needed, what help you may need.

I am not sure whether this meets the scope of the class project? And whether there are any specific evaluation metrics I should use to evaluate model performance. 

I think that I can use the scikit-learn library well enough to do this project.

I will need to do a good bit of data cleaning to get the sleep measures together.



