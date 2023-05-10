## VGG Image Classifier

Table with model comparision between models:-

|Model                       |Training time|Training loss|Training accuracy|Testing accuracy|Number of model parameter|
|----------------------------|-------------|-------------|-----------------|----------------|-------------------------|
|VGG(1 block)                |195.099      |0.0206       |1                |0.725           |40961153                 |
|VGG(3 block)                |205.74       |0.1331       |0.9821           |0.725           |10333505                 |
|VGG(3 block) -Data augmented|261.7763922  |0.3806       |0.8125           |0.8             |10333505                 |
|Tranfer Learning(VGG16)     |743.9707832  |1.28E-14     |1                |3.05E-13        |17926209                 |
|MLP Model                   |172.1257512  |0.1665       |0.9625           |0.725           |17292049                 | 

### VGG 1

![vgg1_accuracy](https://user-images.githubusercontent.com/62815174/232404540-445df7d8-8f49-4aba-86d6-2500d9a85c41.png)

![vgg1_loss](https://user-images.githubusercontent.com/62815174/232404570-5e2fb515-6df8-44af-baf6-d62d76ed6f26.png)

### VGG 3

![vgg3_accuracy](https://user-images.githubusercontent.com/62815174/232404751-b5bc40bc-220a-4049-ba64-73fa8c284577.png)

![vgg3_loss](https://user-images.githubusercontent.com/62815174/232404780-ebb67e64-d603-4a9d-a4b0-570ba88241d6.png)


### VGG 3 - Data Augmentation

![vgg3da_accuracy](https://user-images.githubusercontent.com/62815174/232404906-d52315c9-2206-45bd-a488-e33b1b963e1b.png)

![vgg3da_loss](https://user-images.githubusercontent.com/62815174/232404923-4eefe50f-ab37-412b-a96b-918ec6096e7c.png)

### VGG 16

![vgg16_accuracy](https://user-images.githubusercontent.com/62815174/232405044-b210ec92-0052-43a7-bd18-bfceaefe226b.png)

![vgg16_loss](https://user-images.githubusercontent.com/62815174/232405062-0f894ae8-c48c-4948-8ccf-9531b8d876e5.png)


## Subjective Questions

### Are the results as expected? Why or why not?

The observed results align with expectations. For VGG1, VGG3 (without data augmentation), and the MLP model, the training accuracy is higher, but the testing accuracy is lower, indicating overfitting. This outcome was anticipated, as training for 50 epochs may not be sufficient for these models to generalize well to unseen data. On the other hand, VGG16 with transfer learning shows very low losses and perfect accuracy, which was expected as pretrained models can learn quickly from the dataset. Data augmentation proved beneficial for VGG3, as it improved testing accuracy by providing a more diverse set of images during training, preventing the model from memorizing the training data and promoting better generalization.

### Does data augmentation help? Why or why not?

Data augmentation proved beneficial for VGG3, as it improved testing accuracy by providing a more diverse set of images during training, preventing the model from memorizing the training data and promoting better generalization.

### Does it matter how many epochs you fine tune the model? Why or why not?

It does not matter much how many epochs we choose because after 5 epochs the losses are low . Uptill that point the model learns very quickly. 

### Are there any particular images that the model is confused about? Why or why not?

Yes there are some particular images that the model is confused about. The images where dogs have worn sunglasses or other things they are classified as parrot. Parrot images where there are close up shots and images contained multiple parrots, those images were misclassified as dogs.

### Create a MLP model with comparable number of parameters as VGG16 and compare your performance with the other models in the table. You can choose the distribution of number of neurons and number of layers. What can you conclude? 

We can conclude that the architecture plays a signifcant role in determining the performance on a given task. Even though they have similar number of paramters, the use of conolutional layers, allows VGG16 to learn much better from image data. 