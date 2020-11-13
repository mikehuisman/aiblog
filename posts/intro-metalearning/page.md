<title>AI Blog</title>

## Introduction to Meta-Learning

The wish to build *Artificial Intelligence* (AI) dates back hundreds of years ago. With the rise of computers, this wish has become absolute reality, although one could philosophize about when we can call something intelligent. AI researchers are notoriously pessimistic: after solving a difficult problem that seemed to require intelligence, they suddenly seem to change their minds. "I solved this problem! But the algorithm does simply this and that... That is not intelligent!". We are less pessimistic, and argue that any program that produces behavior that would otherwise require human intelligence, can be called an AI, following one of the definitions in [Russel and Norvig](http://aima.cs.berkeley.edu/).

Looking back, it seems like the field of AI has grown in distinct stages. At the start, we explicitly wrote instructions to computers to perform certain tasks (e.g., write a search algorithm to play chess). This approach was successful to some extent (e.g., deep blue won from Kasparov at chess), but is not very flexible, as we can only solve tasks for which we can develop explicit procedures. As it turns out, there are many tasks for which it is extremely difficult to write these procedures! Think about all the things that humans do on a daily basis: recognizing faces, moving and navigating in the world, and making complex decisions. 

Due to these limitations, the field of AI has evolved. Instead of explicitly programming computers to do tasks, the idea is to write programs that can learn how to do tasks themselves! In modern times, we do this very successfully, as can be seen by the huge accomplishments that have been achieved due to deep neural networks! Despite these great advances, there are some drawbacks too. That is, neural networks often require lots and lots of data, and large computational resources. 

At this point in time, we are slowly transitioning into the next paradigm of AI. Moving away from writing explicit computer instructions, or programs that learn, we now attempt to write programs that *learn to learn*. In this way, the programs can learn how they should learn properly from less data points and with smaller computational budgets. The field that studies such algorithms is called *meta-learning*.   

