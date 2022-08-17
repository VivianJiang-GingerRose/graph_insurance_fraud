# Testing graph features using insurance fraud data

**Data Source**: Insurance fraud data from https://www.kaggle.com/code/yubi45/healthcare-fraud-detection/data <br>
For the first iteration, I only used inpatient data (claims filed for those patients who are admitted in the hospitals), with the modelling target provided in the data file. <br>

**Purpose**: Testing how features generated from various graphs contributes to the prediction of insurance fraud

## Graph Features
Graphs are formed by three types of relationships:
1. ClaimID to BeneficiaryID. One beneficiary can have many claims
2. ClaimID to PhysicianID. One physician can be related to many claims
3. ClaimID to ProviderID. One provider can be related to many claims

Three graph features are extracted from each of the three relationships:
- **Degree** i.e. number of edges for each node. For example, degree measures the number of claims a Physician is attached to
- **Closeness** In a connected graph, closeness centrality (or closeness) of a node is a measure of centrality in a network, calculated as the reciprocal of the sum of the length of the shortest paths between the node and all other nodes in the graph. For example, a number of claims can be connected together because they share the same physician
- **Infomap** Infomap is a graph clustering algorithm capable of achieving high-quality communities.

## Modelling considerations
- Train-test split: claims made prior to Oct-19 are allocated as the training set (31,780 rows), and claims made after Oct-19 are put into the test set (8,694 rows)
- For ethical considerations, features such as Gender, Race, State and County are excluded from the feature space
- Total number of features: **42** with graph features, **33** when graph features are excluded

## Results 
Interesting... <br>

model gini WITHOUT graph feature: **0.0112** 🙀 <br>
model gini WITH graph feature:    **0.7921** 🧐 <br>

Top four features from the model with graph features by shapley value are:
| Feature Name | Shap Value |
| :---        |    :----:   |
| g_degree_Provider      | 4021.962646       |
| g_closeness_AttendingPhysician   | 456.506958        |
| g_communities_infomap_Provider   | 305.000275        | 
| g_communities_infomap_AttendingPhysician   | 51.691402        | 

What the What??? <br>

Then I read this line provided in the description of insurance fraud data: <br>
>  Healthcare fraud is an organized crime which involves peers of providers, physicians, beneficiaries acting together to make fraud claims.

Ah, it all makes sense. No wonder graph features are super relavant for this business case!! <br>
I will dig deeper in the next iteration 🛠️👩‍🔬 Watch this space.


