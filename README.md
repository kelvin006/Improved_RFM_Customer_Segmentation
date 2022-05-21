
# Improving the RFM model in customer segmentation (MSc Dissertation)

The purpose of this project is to develop a new RFM model called Recency, Frequency, Monetary value and Product count (RFMC) to classify customers in a grocery online industry. The new variable, product count (C) measures the number of unique products bought by customers over the observation period. 
The project also compares various clustering techniques to validate the one suitable for this new model.

## The Task
Within the grocery ecommerce space, is there an alternate clustering technique to K-means that will better utilize the improved RFMC model? The Clustering techniques to be compared are K-means, Mini batch K-means, BIRCH, DBSCAN and Meanshift.
## My  Approach

Figure 1 shows the steps followed in this project.

<img width="454" alt="phases" src="https://user-images.githubusercontent.com/20168639/169622587-45ab184e-559b-442c-9c4b-f7a30e07824a.png">

Fig 1 - Steps

### 1. Data Gathering
Since the project is focused on the grocery ecommerce industry, using the publicly available grocery dataset published by Instacart ([kaggle.com](https://www.kaggle.com/c/instacart-market-basket-analysis/data)) was a good idea. The dataset consists of five relational tables, figure 2 shows the data schema.
It is anonymized (i.e., identifying details such as names were removed) and consists of over 3 million real orders made of over 30 million ordered products from more than 200,000 Instacart customers for a period of one year (01/01/2018 to 31/12/2018). The minimum and maximum number of orders ever made by a customer in that period are 4 and 100 respectively.

![data schema](https://user-images.githubusercontent.com/20168639/169671386-5f061aed-fe2f-4659-9f0f-0cb2c82e0343.png)

Fig 2 - Instacart Data Schema
- Note: The Instacart dataset lacks selling price and order date, those variables were created in this project.

### 2. Data Cleaning
The Instacart dataset was fairly clean when it was downloaded and required little or no cleaning. However, the various tables had to be joined using primary and foreign keys, for example dimension tables (such as ‘department’, ‘products’, ‘aisle’) had to be joined with the fact tables (‘orders’ and ‘ordered’). The essence of joining dimension to fact tables is to retrieve data from different tables that would be used for exploratory data analysis. After joining the fact and dimension tables, the dataset now has one large table with 17 columns and 33,819,106 rows.
Prior to joining the tables, the data type of some columns was changed, for example the ‘user_id’, ‘product_id’, ‘order_id’, ‘department_id’, and ‘aisle_id’ columns were converted from numerical variable to categorical variable. This step is crucial so as to get accurate insights when carrying out the exploratory data analysis.

### 3. Exploratory Data Analysis (EDA)
It is necessary to understand the data you are working with. Exploratory data analysis answers the ‘what’, ‘how’, and ‘when’, management of an organisation might have about any dataset. It is a crucial step in the data science lifecycle. The major reason for carrying out EDA in this project, is to understand how Instacart performed in the year 2018 and also to decide the machine learning algorithms to use in the model building phase.  

#### 3.1 Numerical Variable Analysis
Frequency distribution plot was used to inspect the distribution of numerical variables within the dataset. Numerical variables such as ‘Sales per User’ (figure 3) and ‘Orders per Hour’ (figure 4) were analysed.

![0008](https://user-images.githubusercontent.com/20168639/169624806-44f5541f-c1b7-41d1-afbc-871b8595dbf4.jpg)

Fig 3 - Frequency distribution of sales by users

The frequency distribution plot of sales by users shows a massive range of sales figure, with 15.7 being the lowest and 75,240 being the highest sales by user encountered during that period.  The figure also reveals that most customers (approximately 65%) spent only between 15.7 and about 3,000.

![0001](https://user-images.githubusercontent.com/20168639/169625026-c3d51cb4-d6bf-4f11-a8cd-dc27d2cda657.jpg)

Fig 4 - Frequency distribution of total order count by hour

Since Instacart is a grocery ecommerce business, it operates 24/7, hence customers can shop at any time of the day. The plot in figure 4 shows that most transactions (about 71%) occurred between 9am and 5pm, with the lowest being at 3am (0.16%) and highest being at 10am (8.50%). The bars in the plot are colour-coded in such a way that shows how the figures vary. 

#### 3.2 Categorical Variable Analysis
The Instacart dataset has lots of categorical variables and plots such as bar chart, line chart, area chart, donut chart and pie chart were used to visualise the information contained in the data. The most intriguing categorical variables were only considered in this project. 

![0002](https://user-images.githubusercontent.com/20168639/169625335-784ac274-2d9c-4837-9fb0-5559d881503b.jpg)

Fig 5 - Total order number and sales by month

Figure 5 focuses on how the business performed in terms of sales and number of orders from January to December. The bar chart shows the total number of orders by month. It is observed that the numbers do not vary so much from each other with the highest number of orders being approximately 286,000 orders in the month of January and October, while the lowest number of orders being approximately 257,000 occurred in the month of February. The trend is almost similar for ‘Sales by Month’ line chart, it shows that highest amount of total sales occurred in August, while the lowest occurred in February.

![0003](https://user-images.githubusercontent.com/20168639/169625499-3fb1bf2f-7e98-4ab6-9362-53967b218fde.jpg)

Fig 6 - Total sales by Department

Instacart has 21 departments, where all of their products belong to. The area chart in figure 6 shows that the department ‘Produce’ accounts for the highest total sales (about 27.31%) and ‘Bulk’ being the department with the smallest total sales figure of about 530,000 accounts for just 0.08% of the total sales for that period.

![0004](https://user-images.githubusercontent.com/20168639/169625718-d412d575-f742-4d9c-b335-87fdabec5bb0.jpg)

Fig 7 - Top 10 sales by products and aisles

Another way of visualising categorical data is the use of bar charts. Figure 7 shows the top 10 best products and aisles by sales. The data reveals that the customers spend most of their money on organic products, especially fruits. The ‘Sales by aisle (Top 10)’ chart shows that apart from fruits and vegetables, the next big category of products the customers spend money on is dairy products, such as yogurt, cheese and milk.

![0006](https://user-images.githubusercontent.com/20168639/169626664-42f8304b-e7ef-4f11-9086-b24447ef659f.jpg)

Fig 8 - Re-ordered products and Sales by weekday/weekend

The ‘Count of reordered products’ pie chart reveals the 59% of the products ordered in the year 2018 were being reordered by same customers. This tells us that there is a form of brand loyalty with Instacart’s customers. This insight can form the basis for a product recommendation system for the marketing team. The ‘Sales by Weekday/Weekend’ donut chart is a simple chart that provides information about the days customers prefer to shop with Instacart. 71.47% of the total sales occurred on a weekday, this insight says a lot about the shopping behaviour of Instacart’s customers.

#### 3.3 Descriptive Statistical Analysis
Some common types of descriptive statistics are; measures of central tendency, variation and quantiles. In this project, only measures of central tendency was considered, figure 9 shows figures for ‘Average No. of Orders by User’, ‘Average Spend by User’, ‘Average basket size per order’ and ‘Average Sales per Day’.

![0005](https://user-images.githubusercontent.com/20168639/169626762-2bc952fc-a475-425f-9579-a95c1bf3b4ba.jpg)

Fig 9 - Measures of central tendency

### 4. Data Pre-processing
This is a crucial phase before model building. It involves converting data into machine readable format. It comprises of feature engineering, scaling and transformation. It is necessary for the dataset to be properly pre-processed before feeding it to the clustering algorithms being considered in this project.

#### 4.1 Feature engineering
The RFMC model being developed in project is an improvement from the traditional RFM, where C is the number of unique products bought by the customer over the observation period. The larger the number, the more diverse in terms of product choice, the customer is said to be. Feature engineering was used to extract the Recency, Frequency, Monetary Value and Unique product count for each customer. Figure 10 shows what the new dataset looks like after undergoing feature engineering.

<img width="325" alt="dataset feature" src="https://user-images.githubusercontent.com/20168639/169627267-9a64e666-4c11-479f-b6bf-5bcbfd9c4586.png">

Fig 10 - Dataset after undergoing feature engineering

#### 4.2 Scaling 
The newly derived variables have different units, hence different ranges, for example recency is measured in days, monetary value is measured in U.S dollars. In order to reduce bias from the clustering algorithm, the variables need to be scaled, i.e., the mean of each variable becomes 0 and the standard deviation gets a value of 1. This process is called scaling. 

#### 4.3 transformation
Most unsupervised machine learning algorithms like k-means make the assumption that the input data is normally distributed. This assumption should be put into consideration when pre-processing the dataset. Figure 11 shows that the RFMC variables are not normally distributed. To handle this issue, a transformation method known as box-cox transformation was applied and the newly formed variables became normally distributed, see figure 12 below.

![not normal](https://user-images.githubusercontent.com/20168639/169627965-1a1e84bc-f090-4ca1-9aab-191653852644.png)

Fig 11 - Frequency distribution of RFMC variables

![normal](https://user-images.githubusercontent.com/20168639/169627990-efba493e-16cc-4965-8c5d-3fa4e75b0380.png)

Fig 12 - Frequency distribution of transformed & scaled RFMC variables

### 5. Model building
In this phase, five models were built with the newly scaled and transformed RFMC variables. The models used the algorithms being compared; K-means, Mini batch K-means, DBSCAN, and BIRCH.

#### 5.1 Choosing optimal number clusters
In order to use the K-means and Mini batch K-means algorithms, the initial number of clusters need to be determined. There are various ways to determine optimum number of clusters, namely; ‘Elbow method’, ‘Calinski-Harabasz score method’, and ‘Silhouette score method’. 
All three methods (see figures 13, 14 & 15) were used and 3 was the optimum number of clusters chosen for the K-means and Mini batch K-means algorithms as identified by the ‘Elbow method’.

<img width="334" alt="elbow method" src="https://user-images.githubusercontent.com/20168639/169628795-4494e698-5f8b-48b5-86a5-090121d64ec2.png">

Fig 13 - Elbow method

<img width="333" alt="calinski" src="https://user-images.githubusercontent.com/20168639/169628860-31511672-0647-442a-808b-5d4ded8b3fb4.png">

Fig 14 - Calinski-Harabasz score method

<img width="310" alt="silhouette" src="https://user-images.githubusercontent.com/20168639/169629022-ce5ec484-f7a4-4f4e-9b31-8bd0c34b3f2c.png">

Fig 15 - Silhouette score method

#### 5.2 K-means
K-means algorithm uses centroids to assign data points to respective groups, where they share similar characteristics. While using this clustering algorithm, three clusters was used for the ‘K’ parameter as determined in section 5.1. The table below shows the average RFMC values across the three clusters. Figure 16 also shows how each cluster compares to one another. Customers in cluster 2 were the best performing customers, while those in cluster 1 were the least performing customers.

<img width="444" alt="cluster table" src="https://user-images.githubusercontent.com/20168639/169669144-711d176d-0cc8-4670-893a-da95583135fd.png">

![relative plot Kmeans](https://user-images.githubusercontent.com/20168639/169669281-d64d2d37-97ee-42b3-b280-c55cbdbc413a.png)

Fig 16 - Relative Plot for clusters of the K-means model

#### 5.3 Mini Batch K-means
Mini Batch K-means clustering is similar to the traditional K-means algorithm, but instead of assigning all the data points to the closest centroids, it uses random subsets of data called mini batches for each iteration. This makes it more efficient than the traditional K-means algorithm, it requires less time and computing resources. Also, three clusters were used for the ‘K’ parameter when configuring this algorithm. The table below shows the average RFMC values across the three clusters, while figure 17 shows how each cluster compares to one another. Customers in cluster 1 were the best performing customers, while those in cluster 2 were the least performing customers. 

![relative plot mini batch Kmeans](https://user-images.githubusercontent.com/20168639/169669516-e8a9bc3c-68b0-4f5c-9cc4-918568bec272.png)

![relative plot mini batch Kmeans real](https://user-images.githubusercontent.com/20168639/169669600-a122c37d-5fd8-4d85-9c28-c5a52d19c80b.png)

Fig 17 - Relative Plot for clusters of the Mini Batch K-means model

#### 5.4 BIRCH
BIRCH clustering uses an in-memory data structure which is a representation of the entire dataset to create the various segments. This makes BIRCH clustering one of the fastest clustering algorithms. It also does not require the user to predetermine the number of clusters unlike K-means and Mini Batch K-means algorithm. The table below shows that this clustering technique created five customer segments, with cluster 3 having the most valuable customers and cluster 1 with the least performing customers. Figure 18 shows how each cluster compares relatively to one another.

![birch table](https://user-images.githubusercontent.com/20168639/169669817-b887e71c-774c-44c9-963e-a65265e09ad7.png)

![relative plot birch](https://user-images.githubusercontent.com/20168639/169669862-199ae4a7-69d7-4892-9ae3-acba7cec8a24.png)

Fig 18 - Relative Plot for clusters of the BIRCH model

#### 5.5 DBSCAN 
Just like BIRCH clustering, this clustering algorithm does not require the user to predetermine the number of clusters. However, the epsilon value needs to be predetermined. Epsilon denoted with ‘Eps’ is used by the algorithm to classify data points as a core point, border point or noise point. The K-distance graph is used to determine the best epsilon value for the DBSCAN algorithm, the epsilon value is where the line has the most curve, which is about 0.18 in this case, see figure 19.

<img width="825" alt="k-distance" src="https://user-images.githubusercontent.com/20168639/169669989-36ef99c4-abd2-4268-b3ce-a57537854f5b.png">

Fig 19 - K-distance graph

Apart from the Eps, another important value to be predetermined is the minimum number of points (‘MinPts’). The rule of thumb is to use two times the number of dimensions for MinPts, in this project, there are four dimensions, hence the value 8 was used as MinPts. Figure 20 shows the clusters determined by the DBSCAN algorithm, a total of 96 clusters were determined.

<img width="326" alt="DBSNAC" src="https://user-images.githubusercontent.com/20168639/169670038-60590fde-cb65-48e5-8c78-183b489529f4.png">

Fig 20 - Clustering with DBSCAN 

#### 5.6 Meanshift
Similar to DBSCAN and BIRCH, this type of clustering algorithm does not require the user to predetermine the number of clusters. Meanshift clustering automatically determine the number of clusters from the dataset. Unfortunately, this algorithm determined only one cluster from the Instacart dataset, and also took the longest time to run making the Meanshift clustering algorithm utilize lots of computing resources.

### 6. Model Evaluation
Three evaluation metrics were considered, they are; Davies-Bouldin index, Silhouette score and Calinski-Harabasz index. These evaluation metrics are known to be capable of measuring an algorithm’s intra-cluster homogeneity and inter-cluster separation.
Figure 21 shows a summary of the results gotten from the experiment, where K-means, Mini Batch K-means, BIRCH, DBSCAN and Meanshift clustering techniques were used with the newly devised RFMC model. These are the following highlights:

- For Davies-Bouldin index, BIRCH clustering had the best performance with a score of 1.124, followed by K-means with a score of 1.146. The lower the score the better.
- For Silhouette score, K-means clustering had the best performance with a score of 0.299, followed by Mini Batch K-means with a score of 0.293. The higher the score the better.
- For Calinski-Harabasz index, K-means clustering had the best performance with a score of 163088, followed by Mini Batch K-means with a score of 159839. The higher the score the better.
- DBSCAN performed the worst in all three evaluations.
- Results for Meanshift clustering were invalid because only one cluster was formed from this clustering technique. Also, Meanshift clustering technique took the longest to run, with a runtime of 2 hours 3 minutes.
- K-means was the fastest, with a runtime of 978 milliseconds, followed by BIRCH, with a runtime of 2.08 seconds.
- DBSCAN had the highest number of clusters, a total of 96 clusters.

![evalut](https://user-images.githubusercontent.com/20168639/169670358-4d4d4c4f-aa6f-4f42-973c-e815e3ebc377.png)

Fig 21 - Results from experiment

Figure 22, 23 and 24 show graphically how the algorithms perform against each other across the various clustering evaluation metrics. 

![davies](https://user-images.githubusercontent.com/20168639/169670410-7e5f1c7a-e6fa-4758-b342-4ae01a8cb65b.png)

Fig 22 - Davies-Bouldin Index

![silhee](https://user-images.githubusercontent.com/20168639/169670680-0536258e-6d04-4ca0-9878-1f1620935439.png)

Fig 23 - Silhouette Score

![calinskifgdg](https://user-images.githubusercontent.com/20168639/169670744-25ae36f2-ce50-4c84-823a-22f3e2c59c00.png)

Fig 24 - Calinski – Harabasz Index

## Conclusion
This project has provided a new model called the RFMC, which does not only segment customers based on their shopping behaviour but also gives insight into the diversity of products being shopped by the customers. It answers questions such as ‘Are the best performing customers shopping a wide range of products or they are loyal to a unique number of products?’ or ‘Is diversity in product being shopped responsible for the poor performance of certain customers?’. Grouping customers using this model provides foundation for the use of product recommendation. For example, after the segmentation process, the products shopped by the best performing cluster can be analysed and product range discovered can be recommended to members of the same of similar cluster. 

Also, since most of the clustering techniques we have today were only developed few years ago, they have not really been used to perform customer segmentation, very few researchers have done customer segmentation projects with new clustering techniques such as Meanshift, DBSCAN, Mini Batch and BIRCH. This research attempted to use these clustering techniques alongside with the new RFMC model to compare their performances with the traditional and popular K-means clustering technique. The metrics used to evaluate their performances are the Davies-Bouldin index, Silhouette score and Calinski – Harabasz Index. The K-means clustering technique was considered best by the, Silhouette score and Calinski – Harabasz Index with values 0.2993 and 163088 respectively, while the Davies-Bouldin index considered BIRCH as the best with a score of 1.124, Judging with the metrics and speed, the K-means clustering technique is considered the best among the five clustering techniques tested for online grocery businesses. The BIRCH clustering technique shows great potential and may improve with a different dataset. 

The project went further to evaluate the clusters discovered by the K-means algorithm. The customer clusters were given the following names; ‘Average customers’, ‘Premium customers’ and ‘Likely to Churn customers. This newly developed model will be useful not only in a retail environment but also in industries that offer a range of services like the telecommunications industry.

## Project Setup
The project was performed with a computer with the following characteristics:

•	Windows 11 Home 64-bit, Version 21H2, OS build 22000.613 

•	AMD Ryzen 5 5600H with Radeon Graphics 3.30 GHz

•	16GB RAM

•	1.5 TB SSD hard disk

•	NVIDIA GeForce RTX 3060 GPU

 The following software implementation were also used to perform the experiment:

•	Python: The primary programming language used to run the experiment.

•	Pandas: A python library used in manipulating and cleaning datasets.

•	Scikit-learn: A python library used in machine learning tasks. It was used to build the various clustering models used in this experiment. Scikit-learn was also used in data pre-processing to scale and transform the dataset.

•	Yellow Brick: This python library was used to determine the optimum number of clusters required for the K-means and Mini Batch K-means.

•	Datetime: This python library was used to work with date-time variables.

•	Power BI: A business intelligence software that was used in carrying out exploratory data analysis on the Instacart dataset. 







