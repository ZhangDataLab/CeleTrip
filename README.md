## CeleTrip (Celebrity Trip Detection Framework)
This is the source code for paper 'Where Did the President Visit Last Week? Detecting Celebrity Trips from News Articles'.

## Celebrity Trip Dataset
We provide a real-word celebrity trip dataset in file `Celebrity Trip Dataset`. We collecte the trips of 26 politicians and 24 artists from January 2016 to February 2021 from [Wikipedia](https://www.wikipedia.org/), and obtain the date and locations. Afterwards, we crawl 2,617,548 news URLs from 01/2016 to 02/2021 from [GDELT](https://www.gdeltproject.org/), and get news articles using URLs from [Newspaper3k](https://github.com/codelucas/newspaper). 

We label trip locations and non-trip locations from the news articles, using the ground truth trip locations of celebrities provided by Wikipedia. Each row provides the celebrity name, location name, date, articles, and label. You can download the dataset from the link [google driver](www.bing.com).



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

We open source our preprocessing tool using in our framework, these tools can parse dates and extract locations from numerous articles. 

## Trips Visualization

We provide a [tool](http://itin.joycez.xyz/) to visualize the result of our trip-extraction model CeleTrip.


## Cite
Please cite our paper if you find this code useful for your research:




<!-- Description of this project



- Celebrity Trip Dataset
- TKGAT model
- trips_visualization

## Dataset

This is the dataset of celebrity trips.



## TKGAT Model

This is the implementation of TKGAT.



## Trips Visualization

We provide a [tool](http://itin.joycez.xyz/) to visualize the result of our trip-extraction model TKGAT. -->
