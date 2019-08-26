# audio-steganography-cnn
An audio steganalysis method based on CNN in the time domain.

## Abstract
In recent years, deep learning has achieved breakthrough results in various areas, such as computer vision, audio recog-nition, and natural language processing. However, just several related works have been investigated for digital multi-media forensics and steganalysis. In this paper, we design a novel CNN (convolutional neural networks) to detect audio steganography in the time domain. Unlike most existing CNN based methods which try to capture media contents, we carefully design the network layers to suppress audio content and adaptively capture the minor modiﬁcations introduced by ±1 LSB based steganography. Besides, we use a mix of convolutional layer and max pooling to perform subsampling to achieve good abstraction and prevent over-ﬁtting. In our experiments, we compared our network with six similar network architectures and two traditional methods using hand-crafted features. Extensive experimental results evaluated on 40,000 speech audio clips have shown the eﬀectiveness of the proposed convolutional network.


## Citation
Please cite the following paper if the code helps your research.

B. Chen, W. Luo and H. Li, “Audio Steganalysis With Convolutional Neural Network,” in Proc. 5th ACM Workshop Inf. Hiding Multimedia Secur. (IH&MMSec), 2017.
