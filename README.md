# Absorbing Knowledge from Multiple Domains by Generative Adversarial Networks (AMGAN)

  This project is under research. Based on DCGAN, the aim is to generate images belonging to a new domain different from existed domains.
  
  This model's novel points: 1. Two discriminators "teach" one generator. 2. A sobel operator is put in front of first layers of one discriminator, so the discriminator' ability is limited so that they just penalize difference of shape style between real samples and fake samples of one datasets. Another discriminator is called patch discriminator which just care about difference between real samples and fake samples of one datasets at the scale of small patches. So the model can generate images with shape style of one dataset and color style of another dataset. Obviously, generated images belong to a new domain different from existed domains.
  
  ## Results on SVHN and MNIST
   The model can learn to generate black-and-white type-script digits. The effect is not ideal. I am trying kinds of method to improve it.
   ![1](https://github.com/GuangyuanHao/AMGAN/raw/master/results/test1.jpg) 
   ![1](https://github.com/GuangyuanHao/AMGAN/raw/master/results/test2.jpg)
