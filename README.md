**Studying Symbolic/Discrete Language Modeling**

A language model where token IDs are mapped up to symbolic embeddings and later residuals, and where all rules are readable and operate at the symbolic level, would be human interpretable and safe and controllable. A symbolic language model may be more parameter/memory/compute efficient and have better performance (better OOD performance including length-generalization).

Optimizing a model like this is probably intractable using hill climbing, genetic, annealing, or other classical techniques, as the loss landscape is too ill conditioned.

Modern LLMs make optimization possible because they can make intelligent and targeted edits to discrete structures (such as a knowledge graph).

Once a symbolic language model is bootstrapped with modern LLMs, it could itself handle its own optimization.

I have set up a looping structure to optimize a symbolic language model by repeatedly having Claude (or Grok) work on it. The structure is based on a classic Transformer:

Token IDs are mapped to categorical embeddings from a lookup table (for example, the token for "cat" may map to the embedding {"is_animal": true, "word_part_of_speech": "noun", ... })

Embeddings are transformed (to residuals) from an attention approximation and a FFN approximation.

Attention approximation: attention heads can retrieve first matching token, last matching token, a fixed offset, or most likely/average (Q/K approximation) and then transform the embedding/residual based on the result.

FFN approximation: the residual may be transformed based on itself.

Unembedding to predict next token (rules based).

My email: akemeklis@protonmail.com
