<title>AI Blog</title>

# Model-Agnostic Meta-Learning (MAML)

Deep neural networks have been shown to be extremely successful, yielding human- or even super-human level performance on various tasks ranging from image recognition to playing games (chess, go, ...).
Despite these amazing achievements, deep neural networks are very limited in their ability in to learn new tasks quickly (from little data). This makes them inapplicable in many real-world domains where little data and computational resources are available [(Hospedales et al. (2020))](https://arxiv.org/pdf/2004.05439.pdf). Naturally, this raises the question how we can enable these networks to learn *quickly* (from less data).

As mentioned in our [primer](https://mikehuisman.github.io/aiblog/posts/intro-metalearning/page.md), meta-learning is an approach to do precisely that! Meta-learning is inspired by ideas in evolution where individuals may have been selected for fast learning ability (as they would be able to gain a natural advantage over slower learners). Thus, evolution may have imprinted some *prior* in our brains that allows for fast learning on a life-time basis. Note that learning happens on two different levels: at the *outer-level*, evolution "searches" for a prior that allows for fast learning at the *inner-level* (on a lifetime basis). 

Meta-learning approaches mimic this double-loop learning process.    
