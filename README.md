# Absorbing Knowledge from Multiple Domains by Generative Adversarial Networks
Based on DCGAN, the aim is to generate images belonging to a new domain different from existed domains.
This model's novel points: 1. Two discriminators "teach" one generator. 2. A high frequency filter and a low frequency filer are put in front of first layers of two discriminators respectively, so the discriminators' ability is limited so that they just penalize difference of  the high or low frequency part between real samples and  fake samples.
