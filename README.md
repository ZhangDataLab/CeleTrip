## CeleTrip (Celebrity Trip Detection Framework)
This is the source code for paper 'Where Did the President Visit Last Week? Detecting Celebrity Trips from News Articles'.

## Abstract
Celebrities’ whereabouts are of pervasive importance. For instance, where politicians go, how often they visit, and who they meet, come with profound geopolitical and economic implications. Although news articles contain travel information of celebrities, it is not possible to perform large-scale and network-wise analysis due to the lack of automatic itinerary detection tools. To design such tools, we have to overcome difficulties from the heterogeneity among news articles: 1) One single article can be noisy, with irrelevant people and locations, especially when the articles are long. 2) Though it may be helpful if we consider multiple articles together to determine a particular trip, the key semantics are still scattered across different articles intertwined with various noises, making it hard to aggregate them effectively. 3) Over 20\% of the articles refer to the celebrities' trips indirectly, instead of using the exact celebrity names or location names, leading to large portions of trips escaping regular detecting algorithms.
We model text content across articles related to each candidate location as a graph to better associate essential information and cancel out the noises. Besides, we design a special pooling layer based on attention mechanism and node similarity, reducing irrelevant information from longer articles. To make up the missing information resulted from indirect mentions, we construct knowledge sub-graphs for named entities (person, organization, facility, etc.). Specifically, we dynamically update embeddings of event entities like the G7 summit from news descriptions since the properties (date and location) of the event change each time, which is not captured by the pre-trained event representations. The proposed CeleTrip jointly trains these modules, which outperforms all baseline models and achieves 82.53\% in the F1 metric.
By open-sourcing the first tool and a carefully curated dataset for such a new task, we hope to facilitate relevant research in celebrity itinerary mining as well as the social and political analysis built upon the extracted trips.

## Celebrity Trip Dataset
We provide a real-word celebrity trip dataset in file `Celebrity Trip Dataset`. We collecte the trips of 26 politicians and 24 artists from January 2016 to February 2021 from [Wikipedia](https://www.wikipedia.org/), and obtain the date and locations. Afterwards, we crawl 2,617,548 news URLs from 01/2016 to 02/2021 from [GDELT](https://www.gdeltproject.org/), and get news articles using URLs from [Newspaper3k](https://github.com/codelucas/newspaper). We label trip locations and non-trip locations from the news articles, using the ground truth trip locations of celebrities provided by Wikipedia.


## CeleTrip Model

### Prerequisites

The code has been successfully tested in the following environment. (For older dgl versions, you may need to modify the code)

 - Python 3.8.1
 - PyTorch 1.11.0
 - dgl 0.9.0
 - Sklearn 1.1.2
 - SpaCy 3.2.1
 - SpaCy en-core-web-trf 3.2.0
 - Gensim 3.8.3
 - nltk 3.6.5

## Getting Started

### Prepare data

You can download the dataset from the link [google driver](https://drive.google.com/drive/folders/1bdD3hkuTm2Z92pNX5_ryX2IZsiLPuV-5?usp=sharing). We provide a sample of our dataset.

| Celebrity | Location | Date | Article | Label |
| --- | --- | --- | --- | --- |
| Donald Trump | Langley | 2017-01-21 | ['Trump bring politic to the CIA .', "WASHINGTON — member of the national security community react with shock on Saturday after President Donald Trump ’s inaugural visit to CIA headquarters in which he use a speech in front of the agency 's memorial to attack the medium and his critic .", ... ] | True |

<!-- | label	| name |	date |	location |	clean_sen_list |	ent_list |	url_list	| article |
| --- | --- | --- | --- | --- | --- | --- | --- |
| True | Donald Trump | 2017-01-21 | Langley | Trump also speak of the crowd size on Saturday while speak at CIA headquarters in Langley , Va. Todd on Sunday say tell falsehood " undermine the credibility " of the White House press shop . | [['Trump', 'PERSON'], ['Saturday', 'DATE'], ['CIA', 'ORG'], ['Langley', 'GPE'], ['Va.', 'GPE'], ['Todd', 'PERSON'], ['Sunday', 'DATE'], ['White House', 'ORG']] | ['https://www.buzzfeednews.com/article/nancyyoussef/trump-brings-politics-to-the-cia',...] | ['Trump bring politic to the CIA .', "WASHINGTON — member of the national security community react with shock on Saturday after President Donald Trump ’s inaugural visit to CIA headquarters in which he use a speech in front of the agency 's memorial to attack the medium and his critic .", ... ]|  -->

### Training CeleTrip

Please run following commands for building graph.
```python
python build_multilocation_graph.py
```

Please run following commands for training.
```python
python main_celetrip.py
```

### Preprocessing Tools

We open source our preprocessing tool (Time Detection and Location Extraction) in ```Preprocessing Tools/```.

## Cite
Please cite our paper if you find this code useful for your research:
