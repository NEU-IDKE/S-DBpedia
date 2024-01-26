# Assessment of the benchmark quality

Our benchmark is designed to target spatial knowledge graph completion tasks. Spatial coordinates and spatial relations are two important types of spatial information related to tasks in the benchmark, which will affect the performance of models. Therefore, we assess the quality of the benchmark from two perspectives, i.e., the quality of spatial coordinates and the quality of spatial relations. 

The former assesses whether the spatial coordinates in the benchmark are consistent with the coordinates in other existing spatial information data sources (such as WorldKG [1]), and the latter assesses the closeness of relations in the benchmark to spatial information.

## Quality of the spatial coordinates

Inspired by the quality evaluation of WorldKG (which is a large-scale data source containing spatial information, but was not specifically proposed for knowledge graph completion tasks), we also evaluated the quality of the benchmark. Our benchmark is designed to target spatial knowledge graph completion tasks, and the quality of spatial information will affect the performance of the benchmark. Consider that spatial information is represented by latitude and longitude. The latitude and longitude information is a two-dimensional value, so we use [Geohash](https://en.wikipedia.org/wiki/Geohash) to explore the quality of S-DBpedia's spatial information. 

Geohash is a method of encoding geographic coordinates into a string. It encodes the latitude and longitude coordinates of the earth's surface into a string. The length of this string depends on the accuracy. The longer the string, the higher the location accuracy. Geohash is often used in spatial indexing, geographical search, geographical data aggregation and other applications, such as map services, location services, social networks, advertising positioning and other fields. We will perform multi-source spatial information evaluation by using Geohash. In this part of the evaluation, we will evaluate the quality of spatial information in S-DBpedia. Specifically, we compare the entity latitude and longitude information in S-DBpedia with data in WorldKG and Wikidata.

For this purpose, we randomly select 1000 S-DBpedia entity samples from S-DBpedia. We use Geohash with five precisions to evaluate the quality of spatial location information. The evaluation results are shown in the table below. The relevant code can be viewed at [Evaluate code](https://github.com/NEU-IDKE/S-DBpedia/tree/master/Evaluate%20code).

| S-DBpeida geohash precision | Consistency |
| :-------------------------: | :---------: |
|         1_precision         |   100.0%    |
|         2_precision         |   100.0%    |
|         3_precision         |   100.0%    |
|         4_precision         |    98.8%    |
|         5_precision         |    98.2%    |
|         6_precision         |    96.3%    |

Note：`6_precision` represents the geohash encoding with a string length of 6 characters, and the others are similar. The longer the string, the more precisely the position is represented.

As we observe, S-DBpedia has very high consistency (between 96.3% and 100%) with the spatial information in WorldKG and Wikidata. Those inconsistencies are due to geographical entities such as mountainous areas that cover large areas. This situation is normal. Overall, the high-quality spatial information of S-DBpedia lays the foundation for subsequent spatial knowledge graph completion research.



## Quality of the spatial relation

Our knowledge graph completion dataset focuses on spatial knowledge information, and the closeness of relations to spatial information will also reflect the potential of the benchmark.

To ensure relation quality, we perform data processing on DBpedia source data using some semi-automated manually defined rules. Just like the data processing process in section 3.2 of the paper. First, the relationships with the prefix http://dbpedia.org/ontology/ are extracted to retain higher quality data. Then, by limiting the head and tail entities of the triplet to contain latitude and longitude information, relations that are not related to space are further removed. Although some relationships unrelated to space have been removed semi-automatically, there are still some irrelevant relationships that cannot be removed, such as <https://dbpedia.org/ontology/wikiPageWikiLink> and so on. The <https://dbpedia.org/ontology/wikiPageWikiLink> relation represents the link from a Wikipage to another Wikipage.  This relation does not represent a spatial relation. The emergence of such relationships will have potential impacts on the embedding of knowledge graphs.   We verified the impact of these relationships on the embedding model by using the TransE model. The experimental results showed that hits@10 decreased by 23.84% after adding these relationships. Similar phenomena occur with other baseline models.

As mentioned in Section 3.2, through semi-automated processing, there are 317 relationships in total. We manually access every relationship in S-DBpedia and discard spatially irrelevant relations to ensure the quality of relations in S-DBpedia. By doing this, we obtain 100% spatially dependent relations.



Overall, DBpedia's spatial information provides a reliable data source for the spatial knowledge graph S-DBpedia. Although DBpedia is not specific to the spatial knowledge domain, S-DBpedia solves this problem through semi-automated extraction as well as manual verification of spatial relations. This provides a large-scale and reliable high-quality spatial knowledge graph S-DBpedia for related research work on spatial knowledge graph completion.



## References  

 [1] Dsouza, Alishiba, et al. "Worldkg: A world-scale geographic knowledge graph." *Proceedings of the 30th ACM International Conference on Information & Knowledge Management*. 2021.

