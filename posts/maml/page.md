<title>AI Blog</title>
<script type="text/javascript" charset="utf-8" 
src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML,
https://vincenttam.github.io/javascripts/MathJaxLocal.js"></script>

# Model-Agnostic Meta-Learning (MAML)

## Motivation 
Deep neural networks have been shown to be extremely successful, yielding human- or even super-human level performance on various tasks ranging from image recognition to playing games (chess, go, ...).
Despite these amazing achievements, deep neural networks are very limited in their ability in to learn new tasks quickly (from little data). This makes them inapplicable in many real-world domains where little data and computational resources are available [(Hospedales et al. (2020))](https://arxiv.org/pdf/2004.05439.pdf). Naturally, this raises the question how we can enable these networks to learn *quickly* (from less data).

As mentioned in our [primer](https://mikehuisman.github.io/aiblog/posts/intro-metalearning/page.md), meta-learning is an approach to do precisely that! Meta-learning is inspired by ideas in evolution where individuals may have been selected for fast learning ability (as they would be able to gain a natural advantage over slower learners). Thus, evolution may have imprinted some *prior* in our brains that allows for fast learning on a life-time basis. Note that learning happens on two different levels: at the *outer-level*, evolution "searches" for a prior that allows for fast learning at the *inner-level* (on a lifetime basis). This double-loop learning process is visualized in the figure below (image from our [youtube video](https://www.youtube.com/watch?v=2Ipb3F4GlL4)). 

<p style="text-align:center;">
<figure>
    <img src="doubleloop.jpg" width="650" alt="Visualization of the double-loop learning process. At the outer-level, we try to find some prior that allows for faster learning on an individual lifetime basis."/>
</figure>
</p>

Meta-learning approaches mimic this double-loop learning process. In this blog post, we cover (arguably) the most influential work in the field of deep meta-learning, namely model-agnostic meta-learning, or MAML, created by [Finn et al. (2017)](https://arxiv.org/pdf/1703.03400.pdf). 

## Intuition behind MAML

The key idea of MAML is equivalent to that of the double-loop learning process in nature, with the only difference that we assume that the network architecture is fixed. "Well, if the network architecture is fixed then what kind of prior is there to learn?" you may ask. The answer is very simple. We want to find an initialization of the network parameters (weights) from which we can quickly learn new tasks. This idea is captured in the figure below. That is, suppose we have a network with only two parameters: a and b, and 4 tasks that we want to be able to learn quickly (A, B, C, and D). Naturally, we will want our initial weights in a centralized position which allows us to quickly move towards the optimal parameters for the different tasks. In this case, that prior corresponds to the center of the square imposed by the points A, B, C, and D.


<p style="text-align:center;">
<figure>
    <img src="intuition.jpg" width="400" alt="Intuition of having a good initialization."/>
</figure>
</p>

Also note that neural network optimization landscapes are not bowl-shaped. Thus, there may be tons of local minima in which you can get trapped by performing regular gradient descent on the loss function. The initialization parameters of your network thus influence the final point that you will arrive at after learning for some time steps T. Furthermore, the closer your initialization to the right solution, the faster the learning process will be! 


## Tasks 
In order to properly understand MAML, it is crucial to understand what tasks are and how they are composed. MAML was developed for the setting where tasks consist of two parts: a *support set* and a *query set*. Using this setup, our network can learn new tasks by making some updates on the support set. The success of this inner-level learning can then be measured in the query set. Note that the support and query sets correspond to regular train and test sets of examples (in our case). For more information on the tasks, please refer to our [previous blog post](https://mikehuisman.github.io/aiblog/posts/intro-metalearning/page.html).

## Formalizing MAML

Let us denote a task $j$ as $\mathcal{T}_j = (D^{tr}_j, D^{te}_j)$, consisting of a support set $D^{tr}_j$ and query set $D^{te}_j$. Next, suppose we have a fixed base-learner (neural) network with parameters $\theta$. Then, given a new task $\mathcal{T}_j$, our goal is to learn the task as well as possible within $s$ gradient update steps on the support set $D^{tr}_j$. As mentioned earlier, the success of learning on the support set is measured on the query set $D^{te}_j$.

Thus, given the task $\mathcal{T}_j$, our network updates its parameters using gradient descent for $s$ steps:
$$\theta_j^(1) := \theta - \alpha \nabla_{\theta} \mathcal{L}_{D^{tr}_j}(\theta),$$
$$\theta_j^(2) := \theta^(1) - \alpha \nabla_{\theta^(1)} \mathcal{L}_{D^{tr}_j}(\theta^(1)),$$
$$...$$
$$\theta_j^{(s)} := \theta^{(s-1)} - \alpha \nabla_{\theta^(s-1)} \mathcal{L}_{D^{tr}_j}(\theta^(s-1)).$$


Thus, we with to minimize the loss on the query set after making some updates on the support set: $\text{min} \mathcal{L}_{D^{te}_j}(\theta'_j)$, where $\theta'_j$ are the updated parameters from the support set $D^{tr}_j$. That is, $\theta'_j = \theta - \alpha \nabla_{\theta}\mathcal{L}$
