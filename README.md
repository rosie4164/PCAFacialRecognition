# PCA Analysis Facial Recognition 

This was a project completed in the CS251 course at Colby College. The main goal
of the project was to use PCA as dimensionality reduction for 
images. However, I also used PCA as a method of simple facial recognition. 

## Datasets

The dataset used was the LFWcrop Face Dataset (https://conradsanderson.id.au/lfwcrop/), which is a cropped version of the Labeled Faces in the Wild Dataset (http://vis-www.cs.umass.edu/lfw/). The data set "contains more than 13,000 images of faces collected from the web" and was originally "designed for studying the problem of unconstrained face recognition". Due to file size, this dataset was not uploaded, but can be found at the above link. 
![alt text](http://vis-www.cs.umass.edu/lfw/Six_Face_Panels_sm.jpg) 
## Methodology
 * Implement the Eigenface Algorithm on the LFWCrop
        face dataset
    * PCA uses a number of "principal components" 
        to explain variance in data in order to reduce 
        dimensionality
    * Steps of PCA consist of standardizing input, 
        computing the covariance matrix, using eigenvectors
        and eigenvalues to compute principal components,
        creating a feature vector from the top principal
        components, and projecting principal components 
        back onto original dataspace. 
        -For this dataset, the top 200 principal components 
        were chosen
* Use PCA for Facial Recognition
    * A image of an indivual from the dataset is chosen 
        (a different image that is not in the dataset). 
    * project this image into the PCA space using the
        feature
    * compute distance between the new image vector and
        each of the original image vectors
    * return name of image with smallest distance to new
        image
