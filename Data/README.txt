# AtNE-Trust: Attributed Trust Network Embedding for Trust Predicrion in Online Social Networks

Datasets: Epinions and Ciao datasets are used in this paper. The raw datasets are available at: http://www.cse.msu.edu/~tangjili/trust.html

Epinions and Ciao contain several information：
1. trust relationships (trustor, trustee)
2. rating values (u, v, rating)
3. reviews (u, v, reviews)

u represents users, v represents items. There are also detail information about items including contents, categories.

Note that in our code, we read trust relationships data directly from the database. And we output these data in ".txt" form. Code needs to be
easily revised if the ".txt" form files are directly fed into the code.
