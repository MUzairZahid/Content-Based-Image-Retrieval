# **Content Based Image Retrieval Using MATLAB**


---

## Problem statement:
With the vast popularity of embedded camera devices and the continuous development of the internet technology, a noticeable growth in web sharing and photo browsing has been witnessed in the last decade. Hence came the emergence of great number of applications based on image search. Traditionally image search can rely on text search techniques as search engines index multimedia data based on its surrounding metadata information around a photo on the web such as titles and tags. Such indexing can be highly inefficient since text words can be inconsistent with the visual content. Attention is hence drawn to content based image retrieval (CBIR) which have achieved great advance in the last few years.
    
    
## Existing approaches:    
Content based visual retrieval, which is also known by query by image content (QBIC) and content based visual information retrieval (CBVIR) is a simple application of computer vision to the problem of search of visual content in large databases. The application, as defined, has to deal with two main challenges: the intention gap and the semantic gap. The intention gap can be defined as the difficulty to express the expected visual content that the user faces when using the query at hand, while the semantic gap refers to the difficulty of expressing rather complicated semantic concepts with rudimentary visual feature.

Three main issues must addressed when developing a CBIR algorithm:



1.   Image representation
2.   Image organization 
3.   And similarity measurement between images

Image representation will mainly depend on the platform used by the developer of the algorithm. It has however to satisfy the condition of being both descriptive and discriminative as it is expected to be as the algorithm will be only as efficient as its ability to distinguish similar and dissimilar images. 
The large visual database presents organization as a nontrivial but rather laborious task to address. Fortunately, CBIR has information retrieval successful approaches to look up to as several CBIR algorithms rely on the commonly used inverted file structure to index large scale visual database allowing scalable retrieval. On the other hand, hashing based techniques can also present an alternative approach for indexing.

Similarity measurement between images is also critical in developing a solid code. It should reflect the relevance in semantics which is highly contradicted by the semantic gap previously discussed. And though highly relying on both image representation and organization, image similarity measurement is based on the visual matching results with the aid of weighing schemes.

A general formulation of the basic framework of a visual search algorithm can be divided into two main stages: the off line stage and the online stage. The flowchart of a general CBIR algorithm is shown in figure 1.

![Fig1](fig1.png)
**Fig.1: The general framework of content based image retrieval extracted from W. Zhou, H. Li and Q. Tian,” Recent Advances in Content-based Image Retrieval: A literature Survey”**

In this project, a CBIR algorithm will be developed using MATLAB as a platform where the program’s input will be a query image taken from the user to retrieve similar to the given photo as an output. The dataset where the output images will be retrieves is a local database with a number of 1000 photos. As previously discussed, the similarity measurement criteria can widely vary, in the project at hand however, the features that will be extracted from the image will be color based where the similarity will be histogram similarity. The developed code will be discussed in details along with the testing and evaluation of the results along with the comparison of the work at hand with competing systems and approaches.


## The project idea:
 
As previously mentioned, the user will input an image for the program to retrieve similar images to the input from a local database of photos. The representation of the user input will be performed according to the color histogram feature where the HSV space is chosen. Each H (Hue), S (Saturation) and V (Value) component is uniformly quantized into 8, 2 and 2 bins with resulting dimensions of 32. 
 
HSV space was created in the seventies by graphic design researchers as an alternative representation of the RGB color model in the aim of having a closer model to the way the human eye perceive color making attributes. HSV is a cylindrical geometry with the Hue as their angular dimension. Red is at 0° passing through the green primary at 120°, the blue primary at 240°and merging black to red at 360°. The neutral or gray colors represent the vertical axis with a range starting at Value = 0 representing black at the bottom to Value = 1 representing white at the top. The additive primary and secondary colors as well as the linear mixtures between adjacent pairs of them, which are commonly called pure colors, represent the outer edge of the cylinder with saturation 1. Mixing these colors with either black or white separately will leave the saturation unchanged, but mixing them with the combination of both black and white will alter the saturation to values less than 1. The HSV is demonstrated in figure 2 and 3. 


